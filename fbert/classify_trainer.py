from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time

import tensorflow as tf

from absl import flags, app

from modeling_configs import FBertConfig
from modeling_utils import compute_sequence_classification_loss, compute_sequence_classification_loss_for_distribute
from modeling import FBertForSequenceClassification
from optimization import create_optimizer

FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "The input directory of glue data.")
flags.DEFINE_string("config_file", None, "The configuration file of model.")
flags.DEFINE_string("checkpoint_dir", None, "The saved directory of model and optimizer.")

flags.DEFINE_string("task_name", None, "The concrete name of glue.")
flags.DEFINE_bool("is_matched", False, "Whether to use the matched data in mnli.")

flags.DEFINE_bool("use_tpu", False, "Whether to use tpu training the model.")
flags.DEFINE_bool("is_distributed", True, "Whether to use the distributed strategy training the model.")

flags.DEFINE_float("init_lr", 3e-4, "The initial learning rate of optimizer.(3e-4, 1e-4, 5e-5, 3e-5)")
flags.DEFINE_integer("num_train_steps", 1000, "The number of training steps of optimizer.")
flags.DEFINE_integer("num_warmup_steps", 0, "The number of warmup steps of optimizer.")
flags.DEFINE_float("weight_decay_rate", 0.01, "The number of decay rate of optimizer, specifically, AdamW.")

flags.DEFINE_bool("is_training", True, "Whether to training or evaluating the model.")
flags.DEFINE_integer("train_batch_size", 16, "The batch size of dataset in training the model.")
flags.DEFINE_integer("eval_batch_size", 8, "The batch size of dataset in evaluating the model.")

flags.DEFINE_integer("epochs", 4, "The total epochs when training the model.")
flags.DEFINE_integer("num_print_steps", 10, "The number of print steps when training or evaluating the model.")
flags.DEFINE_integer("num_save_steps", 100, "The number of saved steps the training the model.")


class FBertClassifyTrainer(object):
    def __init__(self, config, input_dir, task_name, is_matched, checkpoint_dir,
                 use_tpu, is_distributed, init_lr, num_train_steps, num_warmup_steps, weight_decay_rate,
                 is_training, train_batch_size, eval_batch_size, epochs, num_print_steps, num_save_steps):
        self.config = config

        self.input_dir = input_dir

        self.task_name = task_name
        self.is_matched = is_matched

        self.model = None
        self.optimizer = None
        self.metrics = []

        self.init_lr = init_lr
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.weight_decay_rate = weight_decay_rate

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = None
        self.checkpoint_manager = None

        self.use_tpu = use_tpu
        self.is_distributed = is_distributed
        self.strategy = None

        self.is_training = is_training
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.epochs = epochs
        self.num_print_steps = num_print_steps
        self.num_save_steps = num_save_steps

    @staticmethod
    def _init_tpu_strategy():
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        return strategy

    @staticmethod
    def _init_gpu_strategy():
        strategy = tf.distribute.MirroredStrategy()
        return strategy

    def _init_model_and_optimizer(self):
        if self.task_name == "stsb":
            num_labels = 1
        elif self.task_name == "mnli":
            num_labels = 3
        else:
            num_labels = 2
        model = FBertForSequenceClassification(self.config, num_labels)
        optimizer, schedule = create_optimizer(
            init_lr=self.init_lr,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            weight_decay_rate=self.weight_decay_rate
        )
        return model, optimizer

    def _init_metrics(self):
        metrics = [tf.keras.metrics.Mean()]
        if self.task_name == "stsb":
            metrics.append(tf.keras.metrics.MeanSquaredError())
        else:
            metrics.append(tf.keras.metrics.SparseCategoricalAccuracy())
        return metrics

    @staticmethod
    def _init_checkpoint_and_manager(model, optimizer, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_dir,
            max_to_keep=3
        )
        return checkpoint, checkpoint_manager

    @staticmethod
    def _decode_record(data, name_to_features):
        data = tf.io.parse_single_example(data, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(data.keys()):
            t = data[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            data[name] = t
        return data

    def load_dataset(self, batch_size):
        input_dir = os.path.join(self.input_dir, self.task_name)
        if self.is_training:
            input_file = os.path.join(input_dir, "train_example.bin")
        else:
            if self.task_name == "mnli" and self.is_matched is True:
                input_file = os.path.join(input_dir, "test_matched_example.bin")
            elif self.task_name == "mnli":
                input_file = os.path.join(input_dir, "test_mismatched_example.bin")
            else:
                input_file = os.path.join(input_dir, "test_example.bin")

        dataset = tf.data.TFRecordDataset(input_file)

        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([self.config.max_seq_length], tf.int64),
            "attention_mask": tf.io.FixedLenFeature([self.config.max_seq_length], tf.int64),
            "token_type_ids": tf.io.FixedLenFeature([self.config.max_seq_length], tf.int64),
            "labels": tf.io.FixedLenFeature([1], tf.int64)
        }
        dataset = dataset.map(
            lambda data: self._decode_record(data, name_to_features)
        )
        if self.is_training:
            dataset.repeat(2)
            dataset.shuffle(1000)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset

    @tf.function
    def train_step_for_distribute(self, distributed_inputs):
        def step_fn(inputs):
            if len(inputs) == 4:
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                token_type_ids = inputs["token_type_ids"]
                labels = inputs["labels"]
            else:
                raise ValueError("The length of inputs must be a integer of 4.")
            with tf.GradientTape() as tape:
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    training=True
                )
                loss = compute_sequence_classification_loss_for_distribute(labels, logits)
                loss = tf.nn.compute_average_loss(loss, global_batch_size=self.train_batch_size)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))

            self.metrics[0].update_state(loss * self.strategy.num_replicas_in_sync)
            self.metrics[1].update_state(labels, logits)

        self.strategy.run(step_fn, args=(distributed_inputs,))

    @tf.function
    def train_step(self, inputs):
        if len(inputs) == 4:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]
            labels = inputs["labels"]
        else:
            raise ValueError("The length of inputs must be a integer of 4.")
        with tf.GradientTape() as tape:
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                training=True
            )
            loss = compute_sequence_classification_loss(labels, logits)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(labels, logits)

    def test_step(self, inputs):
        if len(inputs) == 4:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]
            labels = inputs["labels"]
        else:
            raise ValueError("The length of inputs must be a integer of 4.")
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=False
        )
        loss = compute_sequence_classification_loss(labels, logits)
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(labels, logits)
        self.metrics[2].update_state(labels, logits)

    def do_training(self):
        if self.is_distributed:
            if self.use_tpu:
                self.strategy = self._init_tpu_strategy()
            else:
                self.strategy = self._init_gpu_strategy()
            with self.strategy.scope():
                self.model, self.optimizer = self._init_model_and_optimizer()
                self.metrics = self._init_metrics()

            self.checkpoint, self.checkpoint_manager = self._init_checkpoint_and_manager(
                self.model, self.optimizer, self.checkpoint_dir
            )
            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                logging.info("Restore from latest model and optimizer checkpoint.")
            else:
                logging.info("Create a new model and optimizer.")

            logging.info("Start loading the dataset.")
            per_replica_batch_size = self.train_batch_size // self.strategy.num_replicas_in_sync
            dataset = self.strategy.distribute_datasets_from_function(
                lambda _: self.load_dataset(per_replica_batch_size)
            )
            logging.info("Loaded dataset completely.")

            logging.info("Start training the model.")
            for epoch in range(self.epochs):
                start = time.time()
                for step, distributed_inputs in enumerate(dataset):
                    self.train_step_for_distribute(distributed_inputs)
                    if step % self.num_print_steps == 0:
                        logging.info(
                            "Training epoch: {}, step: {}, loss: {:.2f}, accuracy: {:.2f}".format(
                                epoch, step, self.metrics[0].result(), self.metrics[1].result()
                            )
                        )
                    if step % self.num_save_steps == 0:
                        self.checkpoint_manager.save()
                        logging.info("Saved model and optimizer in epoch: {} step: {}".format(epoch, step))
                end = time.time()
                logging.info("Times {:.2f} in one epoch.".format(end - start))
                self.metrics[0].reset_states()
                self.metrics[1].reset_states()
            logging.info("Training the model completely.")
        else:
            self.model, self.optimizer = self._init_model_and_optimizer()
            self.metrics = self._init_metrics()

            self.checkpoint, self.checkpoint_manager = self._init_checkpoint_and_manager(
                self.model, self.optimizer, self.checkpoint_dir
            )

            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                logging.info("Restore from latest model and optimizer checkpoint.")
            else:
                logging.info("Create a new model and optimizer.")

            logging.info("Start loading the dataset.")
            dataset = self.load_dataset(self.train_batch_size)
            logging.info("Loaded dataset completely.")

            logging.info("Starting training the model.")
            for epoch in range(self.epochs):
                start = time.time()
                for step, inputs in enumerate(dataset):
                    self.train_step(inputs)
                    if step % self.num_print_steps == 0:
                        logging.info(
                            "Training epoch: {}, step: {}, loss: {:.2f}, accuracy: {:.2f}".format(
                                epoch, step, self.metrics[0].result(), self.metrics[1].result()
                            )
                        )
                    if step % self.num_save_steps == 0:
                        self.checkpoint_manager.save()
                        logging.info("Saved model and optimizer in epoch: {} step: {}".format(epoch, step))
                end = time.time()
                logging.info("Times {:.2f} in one epoch.".format(end - start))
                self.metrics[0].reset_states()
                self.metrics[1].reset_states()
            logging.info("Training the model completely.")

    def do_evaluating(self):
        self.model, self.optimizer = self._init_model_and_optimizer()
        self.metrics = self._init_metrics()

        self.checkpoint, self.checkpoint_manager = self._init_checkpoint_and_manager(
            self.model, self.optimizer, self.checkpoint_dir
        )

        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            logging.info("Restore from latest model and optimizer checkpoint.")
        else:
            logging.info("Create a new model and optimizer.")

        logging.info("Start loading the dataset.")
        dataset = self.load_dataset(self.eval_batch_size)
        logging.info("Loaded dataset completely.")

        logging.info("Starting evaluating the model.")
        for epoch in range(self.epochs):
            start = time.time()
            for step, inputs in enumerate(dataset):
                self.test_step(inputs)
                if step % self.num_print_steps == 0:
                    if self.task_name == "cola":
                        logging.info(
                            "Evaluating epoch: {}, step: {}, loss: {:.2f}, accuracy: {:.2f}, matthew: {:.2f}"
                            .format(epoch, step, self.metrics[0].result(), self.metrics[1].result(),
                                    self.metrics[2].result())
                        )
                    elif self.task_name == "stsb":
                        logging.info(
                            "Evaluating epoch: {}, step: {}, loss: {:.2f}, accuracy: {:.2f}, pearson: {:.2f}"
                            .format(epoch, step, self.metrics[0].result(), self.metrics[1].result(),
                                    self.metrics[2].result())
                        )
                    else:
                        logging.info(
                            "Evaluating epoch: {}, step: {}, loss: {:.2f}, accuracy: {:.2f}"
                            .format(epoch, step, self.metrics[0].result(), self.metrics[1].result())
                        )
                if step % self.num_save_steps == 0:
                    self.checkpoint_manager.save()
                    logging.info("Saved model and optimizer in epoch: {} step: {}".format(epoch, step))
            if self.task_name == "cola":
                logging.info(
                    "Evaluating epoch: {}, loss: {:.2f}, accuracy: {:.2f}, matthew: {:.2f}"
                    .format(epoch, self.metrics[0].result(), self.metrics[1].result(), self.metrics[2].result())
                )
            elif self.task_name == "stsb":
                logging.info(
                    "Evaluating epoch: {}, loss: {:.2f}, accuracy: {:.2f}, pearson: {:.2f}"
                    .format(epoch, self.metrics[0].result(), self.metrics[1].result(), self.metrics[2].result())
                )
            else:
                logging.info(
                    "Evaluating epoch: {}, loss: {:.2f}, accuracy: {:.2f}"
                    .format(epoch, self.metrics[0].result(), self.metrics[1].result())
                )
            end = time.time()
            logging.info("Times {:.2f} in one epoch.".format(end - start))
            self.metrics[0].reset_states()
            self.metrics[1].reset_states()
        logging.info("Evaluating the model completely.")


def main(_argv):
    if FLAGS.config_file is None:
        config = FBertConfig()
    else:
        config = FBertConfig.from_json(FLAGS.config_file)

    trainer = FBertClassifyTrainer(
        config=config,
        input_dir=FLAGS.input_dir,
        task_name=FLAGS.task_name,
        is_matched=FLAGS.is_matched,
        checkpoint_dir=FLAGS.checkpoint_dir,
        use_tpu=FLAGS.use_tpu,
        is_distributed=FLAGS.is_distributed,
        init_lr=FLAGS.init_lr,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        weight_decay_rate=FLAGS.weight_decay_rate,
        is_training=FLAGS.is_training,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        epochs=FLAGS.epochs,
        num_print_steps=FLAGS.num_print_steps,
        num_save_steps=FLAGS.num_save_steps
    )

    if trainer.is_training:
        trainer.do_training()
    else:
        trainer.do_evaluating()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_dir")
    flags.mark_flag_as_required("checkpoint_dir")
    flags.mark_flag_as_required("task_name")
    app.run(main)
