from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dataclasses import dataclass
from typing import List
from absl import flags, app

from tokenization import FullTokenizer
from tokenization import convert_to_unicode

# Defines the global variables.
FLAGS = flags.FLAGS

# Defines the input and output files.
flags.DEFINE_string("input_files", None, "The model pretraining raw file, which could be one and more files.")
flags.DEFINE_string(
    "output_files",
    None,
    "The model pretraining serialized data file, including multiple data instances."
)

# Defines the tokenizer hyperparameter.
flags.DEFINE_string("vocab_file", None, "The vocabulary table of tokenizer.")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the text.")



@dataclass
class FBertData(object):
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    masked_lm_labels: List[int]
    next_sentence_labels: List[int]


class FBertDataBuilder(object):
    def __init__(self, vocab_file):
        self.tokenizer = self.init_tokenizer(vocab_file)

    @staticmethod
    def init_tokenizer(vocab_file):
        return FullTokenizer(vocab_file)

    @staticmethod
    def create_data_from_files():
        pass

    @staticmethod
    def create_data_from_documents(self):
        pass

    def load_data(self, input_files: str):
        documents = []
        input_files = input_files.split(",")
        for input_file in input_files:
            with tf.io.gfile.GFile(input_file, "r") as reader:
                while True:
                    text = convert_to_unicode(reader.readline())
                    if text is None:
                        break
                    text = text.strip()
                    if text == "":
                        documents.append([])
                    else:
                        if len(documents) == 0:
                            documents.append([])
                        tokens = self.tokenizer.tokenize(text)
                        documents[-1].extend(tokens)
        print(documents)

    def save_data(self, output_files):
        pass


builder = FBertDataBuilder("vocab.txt")
builder.load_data("test.txt")
