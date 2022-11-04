from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from functools import partial

import tensorflow as tf
from scipy import linalg

from modeling_utils import (
    shape_list,
    get_initializer,
    get_activation,
    stable_softmax,
    get_sinusoidal_position_embeddings
)


class FBertEmbedding(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        """Constructs a FBertEmbedding with an optional configuration."""
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.type_vocab_size = config.type_vocab_size
        self.embed_size = config.embed_size
        self.initializer_range = config.initializer_range

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate, name="dropout")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="layer_norm")

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="word_embedding",
            shape=[self.vocab_size, self.embed_size],
            initializer=get_initializer(self.initializer_range)
        )
        self.token_type_embeddings = self.add_weight(
            name="token_type_embedding",
            shape=[self.type_vocab_size, self.embed_size],
            initializer=get_initializer(self.initializer_range)
        )
        super().build(input_shape)

    def call(
            self,
            input_ids,
            token_type_ids=None,
            training=False
    ):
        input_shape = shape_list(input_ids)

        if len(input_shape) == 2:
            batch_size, seq_length = input_shape
        else:
            raise ValueError("input_ids must be a valid shape of [batch_size, seq_length].")

        if token_type_ids is None:
            token_type_ids = tf.zeros([batch_size, seq_length])
        token_type_ids = tf.cast(token_type_ids, tf.int32)

        word_embeddings = tf.gather(self.weight, input_ids)
        token_type_embeddings = tf.gather(self.token_type_embeddings, token_type_ids)
        final_embeddings = word_embeddings + token_type_embeddings

        final_embeddings = self.dropout(final_embeddings, training=training)
        final_embeddings = self.layer_norm(final_embeddings)
        return final_embeddings


class FBertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        """Constructs a FBertAttention with optional configuration."""
        super().__init__(**kwargs)

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                "The hidden size should be a integral multiple of the number of heads."
            )

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_size = int(config.hidden_size / config.num_heads)
        self.all_head_size = self.head_size * config.num_heads

        self.q_dense = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="q_dense"
        )
        self.k_dense = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="k_dense"
        )
        self.v_dense = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="v_dense"
        )

        self.h_dense = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="h_dense"
        )
        self.attention_dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_rate, name="attention_dropout")

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate, name="dropout")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="layer_norm")

    def split_to_multiple_heads(self, hidden_states):
        batch_size = shape_list(hidden_states)[0]
        hidden_states = tf.reshape(hidden_states, [batch_size, -1, self.num_heads, self.head_size])
        hidden_states = tf.transpose(hidden_states, perm=[0, 2, 1, 3])
        return hidden_states

    @staticmethod
    def apply_rotary_position_embeddings(query_layer, key_layer, value_layer, position_embeddings):
        """
        Apply rotary position embedding for query, key and value layer. For more detail about rotary position
        encoding, please refer to https://arxiv.org/abs/2104.09864.
        """
        if position_embeddings is not None:
            # sin [batch_size, num_heads, seq_len, hidden_size // 2]
            # cos [batch_size, num_heads, seq_len, hidden_size // 2]
            sin, cos = tf.split(position_embeddings, 2, -1)
            # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
            # sin_pos [batch_size, num_heads, seq_len, hidden_size // 2]
            sin_pos = tf.stack([sin, sin], axis=-1)
            # [batch_size, num_heads, seq_len, hidden_size // 2, 2] -> [batch_size, num_heads, seq_len, hidden_size]
            sin_pos = tf.reshape(sin_pos, shape_list(position_embeddings))
            # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
            # cos_pos [batch_size, num_heads, seq_len, hidden_size // 2]
            cos_pos = tf.stack([cos, cos], axis=-1)
            # [batch_size, num_heads, seq_len, hidden_size // 2, 2] -> [batch_size, num_heads, seq_len, hidden_size]
            cos_pos = tf.reshape(cos_pos, shape_list(position_embeddings))

            # assume q,k,v is a token vector, and qi is an element is this token vector
            # q [q0,q1,q2,q3......qd-2,qd-1] -> [-q1,q0,-q3,q2......-qd-1,qd-2]
            # query_layer1 [batch_size, num_heads, seq_len, hidden_size // 2, 2]
            query_layer1 = tf.stack([-query_layer[:, :, :, 1::2], query_layer[:, :, :, ::2]], axis=-1)
            # query_layer1
            # [batch_size, num_heads, seq_len, hidden_size // 2, 2] -> [batch_size, num_heads, seq_len, hidden_size]
            query_layer1 = tf.reshape(query_layer1, shape_list(query_layer))
            # query layer with position information.
            query_layer = query_layer * cos_pos + query_layer1 * sin_pos

            # key_layer1
            key_layer1 = tf.stack([-key_layer[:, :, :, 1::2], key_layer[:, :, :, ::2]], axis=-1)
            key_layer1 = tf.reshape(key_layer1, tf.shape(key_layer))
            key_layer = key_layer * cos_pos + key_layer1 * sin_pos

            # value_layer1
            value_layer1 = tf.stack([-value_layer[:, :, :, 1::2], value_layer[:, :, :, ::2]], axis=-1)
            value_layer1 = tf.reshape(value_layer1, shape_list(value_layer))
            value_layer = value_layer * cos_pos + value_layer1 * sin_pos
        return query_layer, key_layer, value_layer

    def call(
            self,
            hidden_states,
            attention_mask=None,
            position_embeddings=None,
            training=False
    ):
        mixed_query_layer = self.q_dense(hidden_states)
        mixed_key_layer = self.k_dense(hidden_states)
        mixed_value_layer = self.v_dense(hidden_states)

        # Splits query, key, value layer to multiple heads.
        # [batch_size, seq_length, all_head_size] -> [batch_size, num_heads, seq_length, head_size]
        query_layer = self.split_to_multiple_heads(mixed_query_layer)
        key_layer = self.split_to_multiple_heads(mixed_key_layer)
        value_layer = self.split_to_multiple_heads(mixed_value_layer)

        query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
            query_layer, key_layer, value_layer, position_embeddings
        )

        attention_scores = tf.einsum("bnih,bnjh->bnij", query_layer, key_layer)

        attention_probs = stable_softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might seem a bit
        # unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.einsum("bnij,bnjk->bnik", attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, [shape_list(hidden_states)[0], -1, self.all_head_size])

        # [..., all_head_size] -> [..., hidden_size]
        attention_outputs = self.h_dense(context_layer)
        attention_outputs = self.dropout(attention_outputs, training=training)
        attention_outputs = self.layer_norm(attention_outputs + hidden_states)
        return attention_outputs


class FBertFourierTransform(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.use_fft = config.use_fft
        self.max_seq_length = config.max_seq_length
        self.hidden_size = config.hidden_size

        self.fourier_transform = self.init_fourier_transform()

        self.f_dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="f_dense"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate, name="dropout")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="layer_nrom")

    @classmethod
    def _two_dim_matmul(cls, x, matrix_dim_one, matrix_dim_two):
        seq_length = x.shape[1]
        matrix_dim_one = matrix_dim_one[:seq_length, :seq_length]
        x = tf.cast(x, tf.complex64)
        return tf.einsum("bij,jk,ni->bnk", x, matrix_dim_two, matrix_dim_one)

    @classmethod
    def two_dim_matmul(cls, x, matrix_dim_one, matrix_dim_two):
        return cls._two_dim_matmul(x, matrix_dim_one, matrix_dim_two)

    def init_fourier_transform(self):
        if self.use_fft:
            if self.max_seq_length > 4096 and math.log2(self.max_seq_length).is_integer():
                raise ValueError(
                    "For large input sequence lengths (>4096), the maximum input "
                    "sequence length must be a power of 2 to take advantage of FFT "
                    "optimizations. We encourage the same for the model hidden "
                    "dimension. config.max_seq_length: %d. config.d_model: $d" %
                    self.config.max_seq_length, self.config.d_model
                )
            else:
                return tf.signal.fft2d
        else:
            dft_mat_hidden = tf.cast(linalg.dft(self.hidden_size), tf.complex64)
            dft_mat_seq = tf.cast(linalg.dft(self.max_seq_length), tf.complex64)
            return partial(
                self.two_dim_matmul, matrix_dim_one=dft_mat_seq, matrix_dim_two=dft_mat_hidden
            )

    def call(self, hidden_states, training=False, **kwargs):
        # Converts the tensor type of hidden_states to a "tf.complex64".
        fourier_outputs = self.fourier_transform(tf.cast(hidden_states, tf.complex64))
        fourier_outputs = tf.math.real(fourier_outputs)

        fourier_outputs = self.f_dense(fourier_outputs)
        fourier_outputs = self.dropout(fourier_outputs, training=training)
        fourier_outputs = self.layer_norm(fourier_outputs + hidden_states)
        return fourier_outputs


class FBertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attention = FBertAttention(config, name="attention")
        self.fourier_transform = FBertFourierTransform(config, name="fourier_transform")

    def call(
            self,
            hidden_states,
            attention_mask=None,
            position_embeddings=None,
            training=False
    ):
        fourier_transform_outputs = self.fourier_transform(
            hidden_states,
            training=training
        )
        attention_outputs = self.attention(
            fourier_transform_outputs,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            training=training
        )
        return attention_outputs


class FBertGroup(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.layers = [FBertLayer(config, name="layer_{}".format(i)) for i in range(config.num_inner_layers)]

    def call(
            self,
            hidden_states,
            attention_mask=None,
            position_embeddings=None,
            training=False
    ):
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                training=training
            )
            hidden_states = layer_outputs

        layer_outputs = hidden_states
        return layer_outputs


class FBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        """Constructs a FBertEncoder with optional configuration."""
        super().__init__(**kwargs)

        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        assert config.hidden_size % config.num_heads == 0

        self.head_size = int(config.hidden_size / config.num_heads)

        self.mapping_to_hidden_size = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="mapping_to_hidden_size"
        )
        self.groups = [FBertGroup(config, name="group_{}".format(i)) for i in range(config.num_hidden_groups)]

        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation=get_activation(config.hidden_act),
            name="dense"
        )

    def call(
            self,
            hidden_states,
            attention_mask=None,
            training=False
    ):
        # Mapping the final dimensions of hidden_states to hidden_size.
        # This is very useful when the input dimension is considerable large.
        # [..., embed_size] -> [..., hidden_size], embed_size << hidden_size.
        hidden_states = self.mapping_to_hidden_size(hidden_states)

        input_shape = shape_list(hidden_states)

        assert len(input_shape) == 3

        # The raw total dimensions of 'position_embeddings' is 2, which is [seq_length, hidden_size].
        # We need to transform it and broadcast to [batch_size, num_heads, seq_length, head_size] to adapt
        # the inner attention computation.
        position_embeddings = get_sinusoidal_position_embeddings(shape_list(hidden_states)[1], self.hidden_size)
        position_embeddings = tf.reshape(position_embeddings, (1, input_shape[1], self.num_heads, self.head_size))
        position_embeddings = tf.tile(position_embeddings, [input_shape[0], 1, 1, 1])
        position_embeddings = tf.transpose(position_embeddings, [0, 2, 1, 3])

        for idx in range(self.num_hidden_layers):
            group_idx = int(idx / (self.num_hidden_layers / self.num_hidden_groups))
            group_outputs = self.groups[group_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                training=training
            )
            hidden_states = group_outputs

        group_outputs = hidden_states

        group_outputs = self.dense(group_outputs)
        return group_outputs


class FBertMainLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.embedding = FBertEmbedding(config, name="embedding")
        self.encoder = FBertEncoder(config, name="encoder")

        self.pooling = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation=get_activation("tanh"),
            name="pooling"
        )

    def get_embedding(self):
        return self.embedding

    def call(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            training=False
    ):
        assert input_ids is not None

        input_shape = shape_list(input_ids)

        if len(input_shape) != 2:
            raise ValueError("The 'input_ids' should be a dimension of 2, not {}".format(len(input_shape)))

        if attention_mask is None:
            attention_mask = tf.ones(input_shape, tf.int32)

        one_const = tf.constant(1, dtype=tf.int32)
        attention_mask = tf.subtract(one_const, attention_mask)
        # Broadcast to [input_shape[0], 1, 1, input_shape[1]].
        attention_mask = tf.reshape(attention_mask, [input_shape[0], 1, 1, input_shape[1]])

        embedding_outputs = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            training=training
        )

        encoder_outputs = self.encoder(
            embedding_outputs,
            attention_mask=attention_mask,
            training=training
        )
        pooling_output = self.pooling(encoder_outputs[:, 0])
        return encoder_outputs, pooling_output


class FBertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config, embedding: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size

        self.dense = tf.keras.layers.Dense(
            config.embed_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense"
        )
        self.activation = get_activation("gelu")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon)

        self.bias = None

        self.embedding = embedding

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    def call(self, hidden_states, **kwargs):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embed_size])

        hidden_states = tf.matmul(a=hidden_states, b=self.embedding.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
        return hidden_states


class FBertNSPHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            2,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense"
        )

    def call(self, hidden_states, **kwargs):
        hidden_states = self.dense(hidden_states)
        return hidden_states


class FBertForPreTraining(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.fbert = FBertMainLayer(config, name="fbert")

        self.mlm = FBertMLMHead(config, self.fbert.embedding, name="mlm")
        self.nsp = FBertNSPHead(config, name="nsp")

    def call(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            training=False
    ):
        outputs = self.fbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training
        )
        sequence_outputs, pooling_outputs = outputs[:2]

        mlm_predictions = self.mlm(sequence_outputs)
        nsp_predictions = self.nsp(pooling_outputs)
        return mlm_predictions, nsp_predictions


class FBertForSequenceClassification(tf.keras.Model):
    def __init__(self, config, num_labels, **kwargs):
        super().__init__(**kwargs)

        self.num_labels = num_labels

        self.fbert = FBertMainLayer(config, name="fbert")

        self.dense = tf.keras.layers.Dense(
            self.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense"
        )

    def call(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            training=False
    ):
        outputs = self.fbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training
        )
        pooling_outputs = self.dense(outputs[1])

        return pooling_outputs
