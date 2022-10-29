from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json

import tensorflow as tf


class FBertConfig(object):
    def __init__(
            self,
            vocab_size=30522,
            type_vocab_size=2,
            embed_size=128,
            hidden_size=512,
            intermediate_size=4096,
            num_hidden_layers=12,
            num_hidden_groups=1,
            num_inner_layers=1,
            num_heads=4,
            hidden_act="gelu",
            intermediate_act="gelu",
            hidden_dropout_rate=0.1,
            attention_probs_dropout_rate=0.1,
            layer_norm_epsilon=1e-12,
            max_seq_length=4096,
            initializer_range=0.1,
            use_fft=False
    ):
        """
        This is a configuration class for a modeling. It is used to instantiate a
        modeling according to specified arguments, defining the modeling architecture.

        Args:
            vocab_size:
                Vocabulary size of modeling. Defines the numbers of different tokens
                that can used to represent "input_ids".
            type_vocab_size:
                The vocabulary size of "token_type_ids".
            embed_size:
                Dimensionality of embedding layer.
            hidden_size:
                Dimensionality of inner encoder layer.
            intermediate_size:
                The dimensionality of "intermediate"(often name feed-forward) layer.
            num_hidden_layers:
                The number of hidden layers in a group.
            num_hidden_groups:
                The number of groups for the hidden layers, parameters are shared in
                same group.
            num_inner_layers:
                The number of inner repetition of encoder layer(e.g., attention, ff).
            num_heads:
                The number of attention heads for each attention layer.
            hidden_act:
                The non-linear activation function(function or string) in the encoder.
            intermediate_act:
                The non-linear activation function used in "intermediate"(often named
                feed-forward) layer.
            hidden_dropout_rate:
                The dropout probability for all fully connected layers in the embeddings,
                encoder.
            attention_probs_dropout_rate:
                The dropout ratio for the attention probabilities.
            layer_norm_epsilon:
                The epsilon used in layer normalization layers.
            max_seq_length:
                The maximum sequence length that the modeling might ever be used.
                Typically, set it to something large in case(e.g., 512, 1024, 2048).
            initializer_range:
                The standard deviation of the truncated normal initializer or random normal
                initializer for initial all weight matrices. In random uniform initializer,
                standing for a range in both maximum and minimum value(-initializer_range,
                initializer_range).
        """
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_inner_layers = num_inner_layers
        self.num_heads = num_heads
        self.hidden_act = hidden_act
        self.intermediate_act = intermediate_act
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_probs_dropout_rate = attention_probs_dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.max_seq_length = max_seq_length
        self.initializer_range = initializer_range
        self.use_fft = use_fft

    @classmethod
    def from_dict(cls, json_object):
        """Initializes configuration of modeling according to a json object(python dict)."""
        config = FBertConfig()
        keys = config.__dict__.keys()
        for key, value in json_object.items():
            if key not in keys:
                raise KeyError(f"{key} matches a incorrect key in modeling configuration dictionary.")
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json(cls, json_file):
        """Initializes configuration of modeling from a json file."""
        with tf.io.gfile.GFile(json_file, "r") as file_object:
            json_object = json.load(file_object)
        return cls.from_dict(json_object)

    @classmethod
    def from_string(cls, json_string):
        """Initializes configuration of modeling from a json string."""
        json_object = json.loads(json_string)
        return cls.from_dict(json_object)

    def to_dict(self):
        """Returns a dictionary of overall modeling structure. This is a deep copy version."""
        return copy.deepcopy(self.__dict__)

    def to_json(self, json_file):
        """Restores in a json file."""
        json.dump(self.to_dict(), json_file)

    def to_string(self):
        """Returns a string in a graceful style."""
        return json.dumps(self.to_dict(), indent=4)
