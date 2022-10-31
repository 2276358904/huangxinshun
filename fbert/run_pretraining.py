from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modeling import FBertForPreTraining
from optimization import create_optimizer


class FBertPretrainingTrainer(object):
    def __init__(self, num_proc):
        self.num_proc = num_proc

    def load_data(self, input_files, is_training=False):
        input_files = input_files.split(",")
        input_files = [tf.io.gfile.glob(input_file) for input_file in input_files]

        if is_training:
            dataset = tf.data.Dataset.from_tensor_slices(input_files)
            cycle_length = min(self.num_proc, len(input_files))
