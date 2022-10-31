from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def shape_list(tensor):
    """
    Deal with tensor use dynamic shape in tensorflow. All static dimensions will be returned as python integers, and
    dynamic dimensions will be returned as tf.Tensor scalars.
    """
    dynamic = tf.shape(tensor)
    static = tensor.shape.as_list()
    final = [
        static[n_dim] if static[n_dim] else dynamic[n_dim] for n_dim in range(len(static))
    ]
    return final


def get_initializer(initializer_range, initializer_string="truncated_normal"):
    """
    Creates an initializer used the given range. A few options are truncated_normal,
    random_normal and uniform_normal. The default is truncated normal initializer.
    """
    if initializer_string == "truncated_normal":
        return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
    elif initializer_string == "random_normal":
        return tf.keras.initializers.RandomNormal(stddev=initializer_range)
    elif initializer_string == "random_uniform":
        return tf.keras.initializers.RandomUniform(minval=-initializer_range, maxval=initializer_range)
    else:
        raise ValueError("Unsupported initializer {}.".format(initializer_string))


def get_activation(activation_string):
    """Returns a non-linear activation according to its name(string)."""
    # Clip the range of possible GeLU outputs between [-10, 10]. For more information on this trick,
    # please refer to https://arxiv.org/abs/2004.09602
    def gelu_10(x):
        return tf.clip_by_value(x, -10, 10)

    # The smoother version of the GELU. For more information,
    # please refer to the original paper https://arxiv.org/abs/1606.0841.
    def gelu_new(x):
        return tf.nn.gelu(x, approximate=True)

    def gelu_fast(x):
        x = tf.convert_to_tensor(x)
        coeff1 = tf.cast(0.044715, x.dtype)
        coeff2 = tf.cast(0.7978845608, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(x * coeff2 * (1.0 + coeff1 * x * x)))

    def quick_gelu(x):
        x = tf.convert_to_tensor(x)
        coeff = tf.cast(1.702, x.dtype)
        return x * tf.math.sigmoid(coeff * x)

    # Gated Linear Unit. Split the input x into two halves a and b, and return a * sigmoid(b).
    # For more detail, please refer to https://arxiv.org/abs/1612.08083.
    def glu(x):
        x = tf.convert_to_tensor(x)
        a, b = tf.split(x, 2, axis=-1)
        return a * tf.nn.sigmoid(b)

    string2func = {
        "tanh": tf.nn.tanh,
        "relu": tf.nn.relu,
        "relu6": tf.nn.relu6,
        "leaky_relu": tf.nn.leaky_relu,
        "gelu": tf.nn.gelu,
        "gelu_10": gelu_10,
        "gelu_new": gelu_new,
        "gelu_fast": gelu_fast,
        "quick_gelu": quick_gelu,
        "glu": glu,
        "elu": tf.nn.elu,
        "selu": tf.nn.selu,
        "softsign": tf.nn.softsign,
        "softplus": tf.nn.softplus,
        "silu": tf.nn.silu,  # A special case of swish which beta is equal to 1.
        "swish": tf.nn.swish
    }

    if string2func[activation_string] is not None:
        return string2func[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in {list(string2func.keys())}")


def stable_softmax(logits, axis=None, name=None):
    """
    Stable wrapper that returns the same output as `tf.nn.softmax`, but that works reliably with XLA on CPU.
    The arguments and outputs are the same as `tf.nn.softmax`, and relies on the fact that `softmax(x) = softmax(x + c)`
    """
    return tf.nn.softmax(logits + 1e-9, axis=axis, name=name)


def get_sinusoidal_position_embeddings(seq_len, hidden_size):
    pos_seq = tf.range(0, seq_len)
    pos_seq = tf.cast(pos_seq, dtype=tf.float32)

    inp_freq = 1 / 10000 ** (tf.range(0, hidden_size, 2) / hidden_size)
    inp_freq = tf.cast(inp_freq, dtype=pos_seq.dtype)

    pos_emb = tf.einsum("i,j->ij", pos_seq, inp_freq)
    pos_emb = tf.concat([tf.sin(pos_emb), tf.cos(pos_emb)], axis=-1)

    return pos_emb  # [seq_len, hidden_size]


def compute_pretraining_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    # mlm loss
    mlm_logits, mlm_labels = logits[0], labels[0]
    unmasked_loss = loss_fn(tf.nn.relu(mlm_labels), mlm_logits)
    loss_mask = tf.cast(mlm_labels != -100, dtype=unmasked_loss.dtype)
    masked_loss = unmasked_loss * loss_mask
    mlm_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
    # nsp loss
    nsp_logits, nsp_labels = logits[1], labels[1]
    nsp_loss = loss_fn(nsp_labels, nsp_logits)
    nsp_loss = tf.reduce_mean(nsp_loss)

    total_loss = nsp_loss + mlm_loss
    return total_loss


class FBertPretrainingAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name="accuracy", initializer="zero")

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels, logits = y_true, y_pred
        mlm_logits, mlm_labels = logits[0], labels[0]
        unmasked_accuracy = tf.keras.metrics.sparse_categorical_accuracy(tf.nn.relu(mlm_labels), mlm_logits)
        accuracy_mask = tf.cast(mlm_labels != -100, unmasked_accuracy.dtype)
        masked_accuracy = unmasked_accuracy * accuracy_mask
        mlm_accuracy = tf.reduce_sum(masked_accuracy) / tf.reduce_sum(accuracy_mask)
        # nsp accuracy
        nsp_logits, nsp_labels = logits[1], labels[1]
        nsp_accuracy = tf.keras.metrics.sparse_categorical_accuracy(nsp_labels, nsp_logits)
        nsp_accuracy = tf.reduce_mean(nsp_accuracy)

        total_accuracy = (mlm_accuracy + nsp_accuracy) / 2
        self.accuracy.assign(total_accuracy)

    def result(self):
        return self.accuracy

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.accuracy.assign(0.0)

