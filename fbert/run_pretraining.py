from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from fbert import modeling


def model_fn_builder(
        model_config,
        init_checkpoint,
        learning_rate,
        num_train_steps,
        num_warmup_steps,
        use_tpu,
        use_one_hot_embeddings
):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""
        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = None

        tvars = tf.trainable

    return model_fn


def get_masked_lm_output(
        model_config,
        input_tensor,
        output_weights,
        positions,
        label_ids,
        label_weights
):
    input_tensor = gather_indexes(input_tensor, positions)
    with tf.name_scope("cls/predictions"):
        with tf.name_scope("transform"):
            input_tensor = tf.keras.layers.Dense(
                model_config.hidden_size,
                modeling.get_activation(model_config.hidden_act),
                kernel_initializer=modeling.get_activation()
            )


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])

    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files, max_seq_length, max_predictions_per_seq,
                     is_training, num_cpu_threads=4):
    def input_fn(params):
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.io.FixedLenFeature([1], tf.int64),
        }
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            cycle_length = min(num_cpu_threads, len(input_files))

            d = d.apply(
                tf.data.experimental.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length
                )
            )
        else:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True
            )
        )
        return d
    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


