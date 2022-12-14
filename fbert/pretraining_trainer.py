from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import tensorflow as tf

from absl import flags, app

from modeling import FBertMainLayer, FBertMLMHead, FBertNSPHead
from modeling_configs import FBertConfig
from modeling_utils import compute_pretraining_loss, compute_pretraining_loss_for_distribute
from optimization import create_optimizer


FLAGS = flags.FLAGS

# Defines the needed files and saved directory.
flags.DEFINE_string("input_files", None, "The dataset of model, one or more files.")

flags.DEFINE_string("checkpoint_dir", None, "The directory of checkpoint of model and optimizer.")
flags.DEFINE_string("config_file", None, "The configuration file of model.If none, will use the default configuration.")

# Defines the number of parallel processed files.
flags.DEFINE_integer("num_proc", 2, "The number of processed cpus.")

# Defines the optimizer hyperparameter
flags.DEFINE_float("init_lr", 1e-4, "The learning rate of optimizer.")
flags.DEFINE_integer("num_train_steps", 40000, "Number of training step.")
flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup step.")
flags.DEFINE_float("weight_decay_rate", 0.01, "The weight decay rate of optimizer AdamW.")

# Defines the training and evaluation hyperparameter.
flags.DEFINE_integer("train_batch_size", 16, "Batch size of training dataset.")
flags.DEFINE_integer("eval_batch_size", 8, "Batch size of evaluation dataset.")
flags.DEFINE_bool("use_tpu", False, "Whether to use tpu(true) or cpu/gpu(false) training the model.")
flags.DEFINE_bool("is_distributed", True, "Whether to use distributed strategy training the model.")
flags.DEFINE_bool("is_training", True, "Whether is training or evaluating the model.")
flags.DEFINE_integer("num_print_steps", 10, "The steps in which print any train details, like loss, accuracy.")
flags.DEFINE_integer("num_saved_steps", 1000, "The number of saved steps the training the model.")
flags.DEFINE_integer("epochs", 10, "The total train epochs of model.")


class FBertPretrainingTrainer(object):
    def __init__(self, config, is_training, num_proc, init_lr, num_train_steps, num_warmup_steps, weight_decay_rate,
                 input_files, train_batch_size, eval_batch_size, checkpoint_dir, use_tpu, is_distributed,
                 num_print_steps, num_saved_steps, epochs):
        self.config = config
        self.is_training = is_training
        # model configuration hyperparameter
        self.max_seq_length = config.max_seq_length

        # file process hyperparameter
        self.num_proc = num_proc

        # optimizer hyperparameter
        self.init_lr = init_lr
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.weight_decay_rate = weight_decay_rate

        self.num_print_steps = num_print_steps
        self.num_saved_steps = num_saved_steps
        self.epochs = epochs
        # data files
        self.input_files = input_files
        # batch size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        #
        self.checkpoint_dir = checkpoint_dir

        self.use_tpu = use_tpu
        self.is_distributed = is_distributed
        # tpu strategy
        self.strategy = None

        # model, optimizer and metrics
        self.model = None
        self.mlm_head = None
        self.nsp_head = None
        self.optimizer = None
        self.metrics = []

        self.checkpoint = None
        self.checkpoint_manager = None

    @staticmethod
    def _decode_dataset(data, name_to_features):
        data = tf.io.parse_single_example(data, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(data.keys()):
            t = data[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            data[name] = t
        return data

    def load_dataset(self, batch_size, is_training=False):
        # Notes that when use TPU nodes train the model, the input files is not any local file
        # rather than any remote file from Google Cloud Storage, which start with the prefix "gs:".
        input_files = self.input_files.split(",")
        input_files = [tf.io.gfile.glob(input_file) for input_file in input_files]

        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([self.config.max_seq_length], tf.int64),
            "attention_mask": tf.io.FixedLenFeature([self.config.max_seq_length], tf.int64),
            "token_type_ids": tf.io.FixedLenFeature([self.config.max_seq_length], tf.int64),
            "mlm_labels": tf.io.FixedLenFeature([self.config.max_seq_length], tf.int64),
            "nsp_labels": tf.io.FixedLenFeature([1], tf.int64),
        }

        if is_training:
            dataset = tf.data.Dataset.from_tensor_slices(input_files)
            dataset = dataset.repeat(2)
            dataset = dataset.shuffle(len(input_files))

            dataset = dataset.interleave(
                lambda data: tf.data.TFRecordDataset(data),
                cycle_length=min(self.num_proc, len(input_files)),
                num_parallel_calls=min(self.num_proc, len(input_files))
            )
            dataset = dataset.shuffle(100)
        else:
            dataset = tf.data.TFRecordDataset(input_files)
            dataset = dataset.repeat(2)
        dataset = dataset.map(
            lambda data: self._decode_dataset(data, name_to_features),
            num_parallel_calls=self.num_proc
        )
        dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=self.num_proc)
        return dataset

    @staticmethod
    def _init_tpu_strategy():
        # In colab, there is no parameter in function.
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        # Starting initialize tpu.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        return strategy

    @staticmethod
    def _init_gpu_strategy():
        strategy = tf.distribute.MirroredStrategy()
        return strategy

    @staticmethod
    def _init_model_and_optimizer(config, init_lr, num_train_steps, num_warmup_steps, weight_decay_rate):
        model = FBertMainLayer(config)
        mlm_head = FBertMLMHead(config, model.embedding)
        nsp_head = FBertNSPHead(config)
        optimizer, lr_schedule = create_optimizer(
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            weight_decay_rate=weight_decay_rate
        )
        return model, mlm_head, nsp_head, optimizer

    @staticmethod
    def _init_checkpoint_and_manager(model, mlm_head, nsp_head, optimizer, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint = tf.train.Checkpoint(model=model, mlm_head=mlm_head, nsp_head=nsp_head, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_dir,
            max_to_keep=3
        )
        return checkpoint, checkpoint_manager

    @tf.function
    def train_step_for_distribute(self, distributed_inputs):
        def step_fn(inputs):
            if len(inputs) == 5:
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                token_type_ids = inputs["token_type_ids"]
                mlm_labels = inputs["mlm_labels"]
                nsp_labels = inputs["nsp_labels"]
            else:
                raise ValueError("The input must be a tuple of length 5.")
            labels = (mlm_labels, nsp_labels)

            with tf.GradientTape() as tape:
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    training=True
                )
                mlm_logits = self.mlm_head(logits[0])
                nsp_logits = self.nsp_head(logits[1])
                logits = (mlm_logits, nsp_logits)
                loss = compute_pretraining_loss_for_distribute(labels, logits)
                loss = tf.nn.compute_average_loss(loss, global_batch_size=self.train_batch_size)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
            # update metrics
            self.metrics[0].update_state(loss * self.strategy.num_replicas_in_sync)

        self.strategy.run(step_fn, args=(distributed_inputs,))

    @tf.function
    def train_step(self, inputs):
        if len(inputs) == 5:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]
            mlm_labels = inputs["mlm_labels"]
            nsp_labels = inputs["nsp_labels"]
        else:
            raise ValueError("The input must be a tuple of length 5.")
        labels = (mlm_labels, nsp_labels)
        with tf.GradientTape() as tape:
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                training=True
            )
            mlm_logits = self.mlm_head(logits[0])
            nsp_logits = self.nsp_head(logits[1])
            logits = (mlm_logits, nsp_logits)
            loss = compute_pretraining_loss(labels, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.metrics[0].update_state(loss)

    def test_step(self, inputs):
        if len(inputs) == 5:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]
            mlm_labels = inputs["mlm_labels"]
            nsp_labels = inputs["nsp_labels"]
        else:
            raise ValueError("The input must be a tuple of length 5.")
        labels = (mlm_labels, nsp_labels)
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=True
        )
        mlm_logits = self.mlm_head(logits[0])
        nsp_logits = self.nsp_head(logits[1])
        logits = (mlm_logits, nsp_logits)
        loss = compute_pretraining_loss(labels, logits)
        self.metrics[0].update_state(loss)

    def do_training(self):
        if self.is_distributed:
            if self.use_tpu:
                self.strategy = self._init_tpu_strategy()
            else:
                self.strategy = self._init_gpu_strategy()
            with self.strategy.scope():
                self.model, self.mlm_head, self.nsp_head, self.optimizer = self._init_model_and_optimizer(
                    self.config,
                    self.init_lr,
                    self.num_train_steps,
                    self.num_warmup_steps,
                    self.weight_decay_rate
                )
                self.metrics.extend([tf.keras.metrics.Mean()])
            self.checkpoint, self.checkpoint_manager = self._init_checkpoint_and_manager(
                self.model, self.mlm_head, self.nsp_head, self.optimizer, self.checkpoint_dir
            )
            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                logging.info("Restoring model and optimizer completely from checkpoint manager.")
            else:
                logging.info("Creating a new model and optimizer completely.")

            logging.info("Start loading the dataset.")
            per_replica_batch_size = self.train_batch_size // self.strategy.num_replicas_in_sync
            train_dataset = self.strategy.distribute_datasets_from_function(
                lambda _: self.load_dataset(per_replica_batch_size, is_training=True)
            )
            logging.info("Loaded dataset completely.")

            logging.info("Starting training the model.")
            for epoch in range(self.epochs):
                epoch_start_time = time.time()
                for step, distributed_inputs in enumerate(train_dataset):
                    self.train_step_for_distribute(distributed_inputs)
                    if step % self.num_print_steps == 0:
                        logging.info(
                            "epoch: {}, step: {}, loss: {:.4f}.".format(epoch, step, self.metrics[0].result())
                        )
                    if step % self.num_saved_steps == 0:
                        self.checkpoint_manager.save()
                        logging.info(
                            "saved model and optimizer at epoch {} step {}.".format(epoch, step)
                        )
                epoch_end_time = time.time()
                logging.info(
                    "epoch: {}, loss: {:.4f}.".format(epoch, self.metrics[0].result())
                )
                logging.info(
                    "times {} in 1 epoch.".format(epoch_end_time - epoch_start_time)
                )
                self.metrics[0].reset_states()
            else:
                raise ValueError(
                    "When use tpu to train the model, you must set both use_tpu and is_distribute to true."
                )
        else:
            self.model, self.mlm_head, self.nsp_head, self.optimizer = self._init_model_and_optimizer(
                self.config,
                self.init_lr,
                self.num_train_steps,
                self.num_warmup_steps,
                self.weight_decay_rate
            )
            self.metrics.extend([tf.keras.metrics.Mean()])
            self.checkpoint, self.checkpoint_manager = self._init_checkpoint_and_manager(
                self.model, self.mlm_head, self.nsp_head, self.optimizer, self.checkpoint_dir
            )

            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                logging.info("Restoring model and optimizer completely from checkpoint manager.")
            else:
                logging.info("Create a new model and optimizer completely.")

            logging.info("Start loading the dataset.")
            train_dataset = self.load_dataset(self.train_batch_size, is_training=True)
            logging.info("Loaded dataset completely.")

            logging.info("Start training the model.")
            logging.info("Batch size: {}".format(self.train_batch_size))
            for epoch in range(self.epochs):
                epoch_start_time = time.time()
                for step, inputs in enumerate(train_dataset):
                    self.train_step(inputs)
                    if step % self.num_print_steps == 0:
                        logging.info(
                            "epoch: {}, step: {}, loss: {:.4f}.".format(epoch, step, self.metrics[0].result())
                        )
                    if step % self.num_saved_steps == 0:
                        self.checkpoint_manager.save()
                        logging.info(
                            "saved model and optimizer at epoch {} step {}.".format(epoch, step)
                        )
                epoch_end_time = time.time()
                logging.info(
                    "epoch: {}, loss: {:.4f}.".format(epoch, self.metrics[0].result(), self.metrics[1].result())
                )
                logging.info(
                    "times {} in 1 epoch.".format(epoch_end_time - epoch_start_time)
                )
                self.metrics[0].reset_states()

    def do_evaluating(self):
        self.model, self.mlm_head, self.nsp_head, self.optimizer = self._init_model_and_optimizer(
            self.config,
            self.init_lr,
            self.num_train_steps,
            self.num_warmup_steps,
            self.weight_decay_rate
        )
        self.metrics.extend([tf.keras.metrics.Mean()])
        self.checkpoint, self.checkpoint_manager = self._init_checkpoint_and_manager(
            self.model, self.mlm_head, self.nsp_head, self.optimizer, self.checkpoint_dir
        )
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            logging.info("Restoring model and optimizer completely from checkpoint manager.")
        else:
            logging.info("Creating a new model and optimizer completely.")

        logging.info("Start loading the dataset.")
        train_dataset = self.load_dataset(self.eval_batch_size, is_training=False)
        logging.info("Loaded dataset completely.")

        logging.info("Start evaluating the model.")
        logging.info("Batch size: {}".format(self.eval_batch_size))
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            for step, inputs in enumerate(train_dataset):
                self.test_step(inputs)
                if step % self.num_print_steps == 0:
                    logging.info(
                        "epoch: {}, step: {}, loss: {:.4f}.".format(epoch, step, self.metrics[0].result())
                    )
            epoch_end_time = time.time()
            logging.info(
                "epoch: {}, loss: {:.4f}.".format(epoch, self.metrics[0].result(), self.metrics[1].result())
            )
            logging.info(
                "times {} in 1 epoch.".format(epoch_end_time - epoch_start_time)
            )
            self.metrics[0].reset_states()


def main(_argv):
    if FLAGS.config_file:
        config = FBertConfig.from_json(FLAGS.config_file)
    else:
        config = FBertConfig()
    trainer = FBertPretrainingTrainer(
        config=config,
        is_training=FLAGS.is_training,
        num_proc=FLAGS.num_proc,
        init_lr=FLAGS.init_lr,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        weight_decay_rate=FLAGS.weight_decay_rate,
        input_files=FLAGS.input_files,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        checkpoint_dir=FLAGS.checkpoint_dir,
        use_tpu=FLAGS.use_tpu,
        is_distributed=FLAGS.is_distributed,
        num_print_steps=FLAGS.num_print_steps,
        num_saved_steps=FLAGS.num_saved_steps,
        epochs=FLAGS.epochs,
    )
    #
    if FLAGS.is_training:
        trainer.do_training()
    else:
        trainer.do_evaluating()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_files")
    flags.mark_flag_as_required("checkpoint_dir")
    app.run(main)
