from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os

import tensorflow as tf

from dataclasses import dataclass
from typing import List
from absl import flags, app

from tokenization import FullTokenizer, convert_to_unicode
from modeling_configs import FBertConfig

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", None, "The input directory of glue data.")
flags.DEFINE_string("output_dir", None, "The output directory of glue data.")

flags.DEFINE_string("config_file", None, "The file that include the configuration of model.")

flags.DEFINE_string("vocab_file", None, "The vocabulary file used to initialize the tokenizer.")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text.")

flags.DEFINE_string("task_name", None, "The concreate task name of glue data.")
flags.DEFINE_string("split_name", "train", "The split name of task in glue. The options is 'train', 'dev', 'test'.")


@dataclass
class FBertClassifyData:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    labels: List[int]


class FBertClassifyDataGenerator(object):
    def __init__(self, config, vocab_file, do_lower_case, input_dir, output_dir):
        # Defines the config.
        self.config = config
        # Defines the tokenizer hyperparameter.
        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case
        self.tokenizer = self._init_tokenizer()
        # Defines the files.
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.items = []
        self.instances = []

    def _init_tokenizer(self):
        return FullTokenizer(self.vocab_file, self.do_lower_case)

    def load_data(self, split="train"):
        raise NotImplementedError()

    def save_data(self, split="train"):
        raise NotImplementedError()

    def _load_data(self, text_a, text_b=None, labels=None):
        text = convert_to_unicode(text_a)

        tokens_a = self.tokenizer.tokenize(text)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
        else:
            tokens_b = None

        input_ids, attention_mask, token_type_ids = self._create_data_from_tokens(tokens_a, tokens_b)

        while len(input_ids) < self.config.max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        self.instances.append(
            FBertClassifyData(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
        )

    def _save_data(self, output_file):
        writer = tf.io.TFRecordWriter(output_file)

        for instance in self.instances:
            feature = collections.OrderedDict()
            feature["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=instance.input_ids))
            feature["attention_mask"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=instance.attention_mask)
            )
            feature["token_type_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=instance.token_type_ids)
            )
            feature["labels"] = tf.train.Feature(int64_list=tf.train.Int64List(value=instance.labels))

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized_example = example.SerializeToString()

            writer.write(serialized_example)

        writer.close()

    def _truncate_sequence_pair(self, tokens_a, tokens_b):
        while True:
            if len(tokens_a) + len(tokens_b) <= self.config.max_seq_length - 3:
                break
            truncated_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            truncated_tokens.pop()

    def _create_data_from_tokens(self, tokens_a, tokens_b=None):
        if tokens_b:
            self._truncate_sequence_pair(tokens_a, tokens_b)

        tokens = []
        token_type_ids = []

        tokens.append("[CLS]")
        token_type_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)

        tokens.append("[SEP]")
        token_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                token_type_ids.append(1)

            tokens.append("[SEP]")
            token_type_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask, token_type_ids


class FBertClassifyDataGeneratorForCola(FBertClassifyDataGenerator):
    def __init__(self, config, vocab_file, do_lower_case, input_dir, output_dir):
        super().__init__(config, vocab_file, do_lower_case, input_dir, output_dir)

    def load_data(self, split="train"):
        input_file = os.path.join(self.input_dir, "cola", split + ".tsv")

        with tf.io.gfile.GFile(input_file, "r") as reader:
            items = csv.reader(reader, delimiter="\t")
            for item in items:
                self.items.append(item)

        for index, item in enumerate(self.items):
            if split == "test":
                text = item[3]
                labels = [0]
            else:
                text = item[3]
                labels = [int(item[1])]
            self._load_data(text, labels=labels)

    def save_data(self, split="train"):
        output_file = os.path.join(self.output_dir, "cola", split + "_example.bin")
        self._save_data(output_file)


class FBertClassifyDataGeneratorForMnli(FBertClassifyDataGenerator):
    def __init__(self, config, vocab_file, do_lower_case, input_dir, output_dir):
        super().__init__(config, vocab_file, do_lower_case, input_dir, output_dir)

    def load_data(self, split="train"):
        input_file = os.path.join(self.input_dir, "mnli", split + ".tsv")

        with tf.io.gfile.GFile(input_file, "r") as reader:
            items = csv.reader(reader, delimiter="\t")
            for item in items:
                self.items.append(item)

        for index, item in enumerate(self.items):
            if index == 0:
                continue
            if split == "test_matched" or split == "test_mismatched":
                text_a = item[8]
                text_b = item[9]
                labels = [0]
            else:
                text_a = item[8]
                text_b = item[9]
                labels = [int(item[-1])]
            self._load_data(text_a, text_b=text_b, labels=labels)

    def save_data(self, split="train"):
        output_file = os.path.join(self.output_dir, "mnli", split + "_example.bin")
        self._save_data(output_file)


class FBertClassifyDataGeneratorForMrpc(FBertClassifyDataGenerator):
    def __init__(self, config, vocab_file, do_lower_case, input_dir, output_dir):
        super().__init__(config, vocab_file, do_lower_case, input_dir, output_dir)

    def load_data(self, split="train"):
        input_file = os.path.join(self.input_dir, "mrpc", split + ".tsv")

        with tf.io.gfile.GFile(input_file, "r") as reader:
            items = csv.reader(reader, delimiter="\t")
            for item in items:
                self.items.append(item)

        for index, item in enumerate(self.items):
            if index == 0:
                continue
            if split == "test":
                text_a = item[3]
                text_b = item[4]
                labels = [0]
            else:
                text_a = item[3]
                text_b = item[4]
                labels = [int(item[0])]
            self._load_data(text_a, text_b=text_b, labels=labels)

    def save_data(self, split="train"):
        output_file = os.path.join(self.output_dir, "mrpc", split + "_example.bin")
        self._save_data(output_file)


class FBertClassifyDataGeneratorForQnli(FBertClassifyDataGenerator):
    def __init__(self, config, vocab_file, do_lower_case, input_dir, output_dir):
        super().__init__(config, vocab_file, do_lower_case, input_dir, output_dir)

    def load_data(self, split="train"):
        input_file = os.path.join(self.input_dir, "qnli", split + ".tsv")

        with tf.io.gfile.GFile(input_file, "r") as reader:
            items = csv.reader(reader, delimiter="\t")
            for item in items:
                self.items.append(item)

        for index, item in enumerate(self.items):
            if index == 0:
                continue
            if split == "test":
                text_a = item[1]
                text_b = item[2]
                labels = [0]
            else:
                text_a = item[1]
                text_b = item[2]
                if item[3] == "not_entailment":
                    labels = [0]
                else:
                    labels = [1]
            self._load_data(text_a, text_b=text_b, labels=labels)

    def save_data(self, split="train"):
        output_file = os.path.join(self.output_dir, "qnli", split + "_example.bin")
        self._save_data(output_file)


class FBertClassifyDataGeneratorForQqp(FBertClassifyDataGenerator):
    def __init__(self, config, vocab_file, do_lower_case, input_dir, output_dir):
        super().__init__(config, vocab_file, do_lower_case, input_dir, output_dir)

    def load_data(self, split="train"):
        input_file = os.path.join(self.input_dir, "qqp", split + ".tsv")

        with tf.io.gfile.GFile(input_file, "r") as reader:
            items = csv.reader(reader, delimiter="\t")
            for item in items:
                self.items.append(item)

        for index, item in enumerate(self.items):
            if index == 0:
                continue
            if split == "test":
                text_a = item[3]
                text_b = item[4]
                labels = [0]
            else:
                text_a = item[3]
                text_b = item[4]
                labels = [int(item[5])]
            self._load_data(text_a, text_b=text_b, labels=labels)

    def save_data(self, split="train"):
        output_file = os.path.join(self.output_dir, "qqp", split + "_example.bin")
        self._save_data(output_file)


class FBertClassifyDataGeneratorForRte(FBertClassifyDataGenerator):
    def __init__(self, config, vocab_file, do_lower_case, input_dir, output_dir):
        super().__init__(config, vocab_file, do_lower_case, input_dir, output_dir)

    def load_data(self, split="train"):
        input_file = os.path.join(self.input_dir, "rte", split + ".tsv")

        with tf.io.gfile.GFile(input_file, "r") as reader:
            items = csv.reader(reader, delimiter="\t")
            for item in items:
                self.items.append(item)

        for index, item in enumerate(self.items):
            if index == 0:
                continue
            if split == "test":
                text_a = item[1]
                text_b = item[2]
                labels = [0]
            else:
                text_a = item[1]
                text_b = item[2]
                if item[3] == "not_entailment":
                    labels = [0]
                else:
                    labels = [1]
            self._load_data(text_a, text_b=text_b, labels=labels)

    def save_data(self, split="train"):
        output_file = os.path.join(self.output_dir, "rte", split + "_example.bin")
        self._save_data(output_file)


class FBertClassifyDataGeneratorForSst2(FBertClassifyDataGenerator):
    def __init__(self, config, vocab_file, do_lower_case, input_dir, output_dir):
        super().__init__(config, vocab_file, do_lower_case, input_dir, output_dir)

    def load_data(self, split="train"):
        input_file = os.path.join(self.input_dir, "sst2", split + ".tsv")

        with tf.io.gfile.GFile(input_file, "r") as reader:
            items = csv.reader(reader, delimiter="\t")
            for item in items:
                self.items.append(item)

        for index, item in enumerate(self.items):
            if index == 0:
                continue
            if split == "test":
                text_a = item[0]
                labels = [0]
            else:
                text_a = item[0]
                labels = [int(item[1])]
            self._load_data(text_a, labels=labels)

    def save_data(self, split="train"):
        output_file = os.path.join(self.output_dir, "sst2", split + "_example.bin")
        self._save_data(output_file)


class FBertClassifyDataGeneratorForStsb(FBertClassifyDataGenerator):
    def __init__(self, config, vocab_file, do_lower_case, input_dir, output_dir):
        super().__init__(config, vocab_file, do_lower_case, input_dir, output_dir)

    def load_data(self, split="train"):
        input_file = os.path.join(self.input_dir, "stsb", split + ".tsv")

        with tf.io.gfile.GFile(input_file, "r") as reader:
            items = csv.reader(reader, delimiter="\t")
            for item in items:
                self.items.append(item)

        for index, item in enumerate(self.items):
            if index == 0:
                continue
            if split == "test":
                text_a = item[7]
                text_b = item[8]
                labels = [0]
            else:
                text_a = item[7]
                text_b = item[8]
                labels = [float(item[-1])]
            self._load_data(text_a, text_b=text_b, labels=labels)

    def save_data(self, split="train"):
        output_file = os.path.join(self.output_dir, "stsb", split + "_example.bin")
        self._save_data(output_file)


class FBertClassifyDataGeneratorForWnli(FBertClassifyDataGenerator):
    def __init__(self, config, vocab_file, do_lower_case, input_dir, output_dir):
        super().__init__(config, vocab_file, do_lower_case, input_dir, output_dir)

    def load_data(self, split="train"):
        input_file = os.path.join(self.input_dir, "wnli", split + ".tsv")

        with tf.io.gfile.GFile(input_file, "r") as reader:
            items = csv.reader(reader, delimiter="\t")
            for item in items:
                self.items.append(item)

        for index, item in enumerate(self.items):
            if index == 0:
                continue
            if split == "test":
                text_a = item[1]
                text_b = item[2]
                labels = [0]
            else:
                text_a = item[1]
                text_b = item[2]
                labels = [int(item[3])]
            self._load_data(text_a, text_b=text_b, labels=labels)

    def save_data(self, split="train"):
        output_file = os.path.join(self.output_dir, "wnli", split + "_example.bin")
        self._save_data(output_file)


def main(_argv):
    if FLAGS.config_file is None:
        config = FBertConfig()
    else:
        config = FBertConfig.from_json(FLAGS.config_file)
    # Init generator.
    if FLAGS.task_name == "cola":
        generator = FBertClassifyDataGeneratorForCola(config=config, vocab_file=FLAGS.vocab_file,
                                                      do_lower_case=FLAGS.do_lower_case, input_dir=FLAGS.input_dir,
                                                      output_dir=FLAGS.output_dir)
    elif FLAGS.task_name == "mnli":
        generator = FBertClassifyDataGeneratorForMnli(config=config, vocab_file=FLAGS.vocab_file,
                                                      do_lower_case=FLAGS.do_lower_case, input_dir=FLAGS.input_dir,
                                                      output_dir=FLAGS.output_dir)
    elif FLAGS.task_name == "mrpc":
        generator = FBertClassifyDataGeneratorForMrpc(config=config, vocab_file=FLAGS.vocab_file,
                                                      do_lower_case=FLAGS.do_lower_case, input_dir=FLAGS.input_dir,
                                                      output_dir=FLAGS.output_dir)
    elif FLAGS.task_name == "qnli":
        generator = FBertClassifyDataGeneratorForQnli(config=config, vocab_file=FLAGS.vocab_file,
                                                      do_lower_case=FLAGS.do_lower_case, input_dir=FLAGS.input_dir,
                                                      output_dir=FLAGS.output_dir)
    elif FLAGS.task_name == "qqp":
        generator = FBertClassifyDataGeneratorForQqp(config=config, vocab_file=FLAGS.vocab_file,
                                                     do_lower_case=FLAGS.do_lower_case, input_dir=FLAGS.input_dir,
                                                     output_dir=FLAGS.output_dir)
    elif FLAGS.task_name == "rte":
        generator = FBertClassifyDataGeneratorForRte(config=config, vocab_file=FLAGS.vocab_file,
                                                     do_lower_case=FLAGS.do_lower_case, input_dir=FLAGS.input_dir,
                                                     output_dir=FLAGS.output_dir)
    elif FLAGS.task_name == "sst2":
        generator = FBertClassifyDataGeneratorForSst2(config=config, vocab_file=FLAGS.vocab_file,
                                                      do_lower_case=FLAGS.do_lower_case, input_dir=FLAGS.input_dir,
                                                      output_dir=FLAGS.output_dir)
    else:  # FLAGS.task_name == "wnli":
        generator = FBertClassifyDataGeneratorForWnli(config=config, vocab_file=FLAGS.vocab_file,
                                                      do_lower_case=FLAGS.do_lower_case, input_dir=FLAGS.input_dir,
                                                      output_dir=FLAGS.output_dir)
    # Load and save data.
    generator.load_data(FLAGS.split_name)
    generator.save_data(FLAGS.split_name)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("task_name")
    app.run(main)
