from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tempfile

import numpy as np
import tensorflow as tf

from fbert.modeling_utils import shape_list
from modeling_configs import FBertConfig
from modeling import FBertEmbedding, FBertAttention, FBertFourierTransform, FBertEncoder, FBertModel


class FBertModelTest(tf.test.TestCase):
    def test_config_to_dict(self):
        config = FBertConfig(embed_size=256, hidden_size=1024)
        config_dict = config.to_dict()
        self.assertAllEqual(config_dict["embed_size"], 256)
        self.assertAllEqual(config_dict["hidden_size"], 1024)
        self.assertAllEqual(config_dict["max_seq_length"], 4096)

    def test_config_to_string(self):
        config = FBertConfig(embed_size=256, hidden_size=1024)
        config_string = config.to_string()
        config_dict = json.loads(config_string)
        self.assertAllEqual(config_dict["embed_size"], 256)
        self.assertAllEqual(config_dict["hidden_size"], 1024)
        self.assertAllEqual(config_dict["num_hidden_layers"], 12)
        self.assertAllEqual(config_dict["num_hidden_groups"], 1)

    def test_config_to_json(self):
        config = FBertConfig(embed_size=256, hidden_size=1024)
        with tempfile.NamedTemporaryFile("w", delete=False) as file_obj:
            config.to_json(file_obj)
        with tf.io.gfile.GFile(file_obj.name, "r") as file_obj:
            config_dict = json.load(file_obj)
        self.assertAllEqual(config_dict["embed_size"], 256)
        self.assertAllEqual(config_dict["hidden_size"], 1024)

    def test_embedding_layer(self):
        config = FBertConfig()
        embedding_layer = FBertEmbedding(config)
        train_dataset = self.generate_dataset()[0]
        inputs = next(iter(train_dataset))[0]
        batch_size, seq_length = inputs.shape.as_list()
        outputs = embedding_layer(inputs)
        self.assertAllEqual(outputs.shape.as_list(), [batch_size, seq_length, config.embed_size])

    def test_attention_layer(self):
        config = FBertConfig()
        attention_layer = FBertAttention(config)
        random_inputs = tf.random.normal([64, 128, 512], 0.0, 0.2)
        input_shape = tf.shape(random_inputs)
        outputs = attention_layer(random_inputs)
        self.assertAllEqual(tf.shape(outputs), input_shape)

    def test_fourier_transform_layer(self):
        config = FBertConfig()
        fourier_transform_layer = FBertFourierTransform(config)
        inputs = tf.random.normal([64, 128, 512], 0.0, 0.2)
        input_shape = tf.shape(inputs)
        outputs = fourier_transform_layer(inputs)
        self.assertAllEqual(tf.shape(outputs), input_shape)

    def test_encoder(self):
        config = FBertConfig()
        encoder = FBertEncoder(config)
        input_ids = tf.random.normal([64, 128, 128], 0.0, 0.2)
        input_shape = shape_list(input_ids)
        attention_mask = tf.ones([input_shape[0], 1, 1, input_shape[1]], tf.int32)
        outputs = encoder(input_ids, attention_mask=attention_mask)
        self.assertAllEqual(shape_list(outputs), [input_shape[0], input_shape[1], config.hidden_size])

    def test_model(self):
        config = FBertConfig()
        model = FBertModel(config)
        random_input_ids = np.random.randint(0, 30000, [64, 256])
        random_input_ids = tf.convert_to_tensor(random_input_ids)
        input_shape = shape_list(random_input_ids)
        outputs = model(random_input_ids)
        self.assertAllEqual(shape_list(outputs), [input_shape[0], input_shape[1], config.hidden_size])

    @staticmethod
    def generate_dataset(train_batch_size=64, test_batch_size=32, mode="train"):
        dataset = tf.keras.datasets.imdb.load_data()
        train_dataset, test_dataset = dataset
        x_train, y_train = train_dataset
        x_test, y_test = test_dataset

        x_val, y_val = x_train[20000:], y_train[20000:]
        x_train, y_train = x_train[:20000], y_train[:20000]

        tf_x_train = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(x_train))
        tf_x_train = tf_x_train.batch(train_batch_size).map(lambda x: x.to_tensor())
        tf_y_train = tf.data.Dataset.from_tensor_slices(tf.constant(y_train))
        tf_y_train = tf_y_train.batch(train_batch_size)
        train_dataset = tf.data.Dataset.zip((tf_x_train, tf_y_train)).shuffle(128)

        tf_x_val = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(x_val))
        tf_x_val = tf_x_val.batch(train_batch_size).map(lambda x: x.to_tensor())
        tf_y_val = tf.data.Dataset.from_tensor_slices(tf.constant(y_val))
        tf_y_val = tf_y_val.batch(train_batch_size)
        val_dataset = tf.data.Dataset.zip((tf_x_val, tf_y_val))

        tf_x_test = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(x_test))
        tf_x_test = tf_x_test.batch(test_batch_size).map(lambda x: x.to_tensor())
        tf_y_test = tf.data.Dataset.from_tensor_slices(tf.constant(y_test))
        tf_y_test = tf_y_test.batch(test_batch_size)
        test_dataset = tf.data.Dataset.zip((tf_x_test, tf_y_test))

        if mode == "train":
            return train_dataset, val_dataset
        elif mode == "eval":
            return test_dataset
        else:
            raise ValueError("No extra mode are supported.")
