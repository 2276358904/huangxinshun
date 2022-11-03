from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

import tensorflow as tf

from dataclasses import dataclass
from typing import List

from tokenization import FullTokenizer


@dataclass
class FBertClassifyData:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    labels: List[int]


class FBertClassifyDataGenerator(object):
    def __init__(self, config, input_files, output_files, vocab_file, do_lower_case):
        # Defines the config
        self.config = config
        # Defines the files.
        self.input_files = input_files
        self.output_files = output_files
        # Defines the tokenizer hyperparameter
        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case
        self.tokenizer = self._init_tokenizer()

        self.items = []

    def _init_tokenizer(self):
        return FullTokenizer(self.vocab_file, self.do_lower_case)

    def create_data(self):
        raise NotImplementedError()


class FBertClassifyDataGeneratorForCola(FBertClassifyDataGenerator):
    def __init__(self, config, input_files, output_files, vocab_file, do_lower_case):
        super().__init__(config, input_files, output_files, vocab_file, do_lower_case)

    def create_data(self):
        input_files = self.input_files.split(",")

        for input_file in input_files:
            with tf.io.gfile.GFile(input_file, "r") as reader:
                items = csv.reader(reader, delimiter="\t")
                for item in items:
                    self.items.append(item)

