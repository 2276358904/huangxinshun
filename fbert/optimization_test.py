from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
from six.moves import zip

import tensorflow as tf

from fbert import optimization


class OptimizationTest(tf.test.TestCase):
    def test_adam(self):
        tf.executing_eagerly()

        with self.test_session() as sess:
            w = tf.compat.v1.get_variable(
                "w",
                shape=[3],
                initializer=tf.constant_initializer([0.1, -0.2, -0.1]))
            x = tf.constant([0.4, 0.2, -0.5])
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(x - w))
            tvars = tf.compat.v1.trainable_variables()
            grads = tape.gradient(loss, tvars)
            optimizer = optimization.AdamWeightDecay(learning_rate=0.2)
            train_op = optimizer.apply_gradients(zip(grads, tvars))
            init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                               tf.compat.v1.local_variables_initializer())
            sess.run(init_op)
            for _ in range(100):
                sess.run(train_op)
            w_np = sess.run(w)
            self.assertAllClose(w_np.flat, [0.4, 0.2, -0.5], rtol=1e-2, atol=1e-2)
