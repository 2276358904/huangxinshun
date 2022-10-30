from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
import collections

import tensorflow as tf

from dataclasses import dataclass
from typing import List
from tqdm import tqdm
from absl import flags, app

from tokenization import FullTokenizer
from tokenization import convert_to_unicode
from modeling_configs import FBertConfig

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

# Defines the hyperparameter of generated data.
flags.DEFINE_bool("do_whole_word_mask", True, "Whether to use whole word masking rather than WordPiece masking.")
flags.DEFINE_float(
    "short_seq_prob",
    0.1,
    "Probability of creating sequences which are shorter than the maximum length."
)
flags.DEFINE_bool("is_dynamic_mask", True, "Whether to use dynamic mask strategy.")
flags.DEFINE_integer("dup_times", 10, "Number of times to duplicate the input data (with different masks).")
flags.DEFINE_integer("max_masked_word", 20, "Maximum length of masked word in a sequence.")
flags.DEFINE_float("masked_word_prob", 0.15, "Probability of masked word in a sequence.")


@dataclass
class FBertData(object):
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    mlm_labels: List[int]
    nsp_labels: List[int]


class FBertDataBuilder(object):
    def __init__(self, config, vocab_file, do_lower_case, do_whole_word_mask, short_seq_prob, is_dynamic_mask,
                 dup_times, max_masked_word, masked_word_prob):

        self.max_seq_length = config.max_seq_length

        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case

        self.do_whole_word_mask = do_whole_word_mask
        self.short_seq_prob = short_seq_prob

        self.is_dynamic_mask = is_dynamic_mask
        self.dup_times = dup_times

        self.max_masked_word = max_masked_word
        self.masked_word_prob = masked_word_prob

        self.random = random.Random()
        self.tokenizer = self.init_tokenizer()
        self.instances = []

    def get_instances(self):
        return self.instances

    def init_tokenizer(self):
        return FullTokenizer(self.vocab_file, self.do_lower_case)

    def _create_mlm_and_nsp_labels(self, tokens, is_random_next):
        indexes = []
        for i, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            if self.do_whole_word_mask and len(indexes) >= 1 and token.startswith("##"):
                indexes[-1].append(i)
            else:
                indexes.append([i])

        self.random.shuffle(indexes)

        num_masked_words = min(self.max_masked_word, int(round(len(tokens) * self.masked_word_prob)))

        masked_words = []
        masked_word_positions = []
        output_tokens = list(tokens)
        vocab_words = list(self.tokenizer.vocab.keys())
        for index_set in indexes:
            if len(masked_words) >= num_masked_words:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_words) + len(index_set) > num_masked_words:
                continue
            for index in index_set:
                # 80% of the time, replace with [MASK].
                if self.random.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original word.
                    if self.random.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word.
                    else:
                        masked_token = vocab_words[self.random.randint(0, len(self.tokenizer.vocab) - 1)]
                output_tokens[index] = masked_token
                masked_words.append(masked_token)
                masked_word_positions.append(index)

        assert len(masked_words) <= num_masked_words
        assert len(masked_words) == len(masked_word_positions)

        mlm_labels = [-100] * (len(tokens))

        for i in range(len(masked_word_positions)):
            mlm_labels[masked_word_positions[i]] = self.tokenizer.convert_token_to_id(masked_words[i])

        nsp_labels = []
        if is_random_next:
            nsp_labels.append(1)
        else:
            nsp_labels.append(0)
        return output_tokens, mlm_labels, nsp_labels

    def _truncate_sequence_pair(self, tokens_a, tokens_b):
        max_num_tokens = self.max_seq_length - 3

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break
            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if self.random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def _create_data_from_documents(self, documents):
        instances = []
        # The sequence includes [CLS], [SEP], [SEP].
        max_num_tokens = self.max_seq_length - 3

        for document_index, document in enumerate(documents):
            target_seq_length = max_num_tokens
            if self.random.random() < self.short_seq_prob:
                target_seq_length = max(2, self.random.randint(2, max_num_tokens))

            current_sequences = []
            current_length = 0
            i = 0
            while i < len(document):
                current_sequence = document[i]
                current_sequences.append(current_sequence)
                current_length += len(current_sequence)
                if i == len(document) - 1 or current_length >= target_seq_length:
                    if current_sequences:  # equal to 'if current_sequences != []'.
                        # Sequence A.
                        # The variable `a_end` is how many tokens from `current_sequences` go into the 'A' (first)
                        # sentence.
                        a_end = 1
                        if len(current_sequences) >= 2:
                            a_end = self.random.randint(1, len(current_sequences) - 1)
                        tokens_a = []
                        for j in range(a_end):
                            tokens_a.extend(current_sequences[j])

                        # Sequence B.
                        tokens_b = []
                        if len(current_sequences) == 1 or self.random.random() < 0.5:
                            is_random_next = True
                            target_b_length = target_seq_length - len(tokens_a)
                            random_document_index = 0
                            for _ in range(10):
                                random_document_index = self.random.randint(0, len(documents) - 1)
                                if random_document_index != document_index:
                                    break
                            random_document = documents[random_document_index]
                            random_start = self.random.randint(0, len(random_document) - 1)
                            for j in range(random_start, len(random_document)):
                                tokens_b.extend(random_document[j])
                                if len(tokens_b) >= target_b_length:
                                    break
                            num_unused_sequences = len(current_sequences) - a_end
                            i -= num_unused_sequences
                        else:
                            is_random_next = False
                            for j in range(a_end, len(current_sequences)):
                                tokens_b.extend(current_sequences[j])

                        self._truncate_sequence_pair(tokens_a, tokens_b)

                        assert len(tokens_a) >= 1
                        assert len(tokens_b) >= 1

                        tokens = []
                        token_type_ids = []
                        tokens.append("[CLS]")
                        token_type_ids.append(0)
                        for token in tokens_a:
                            tokens.append(token)
                            token_type_ids.append(0)

                        tokens.append("[SEP]")
                        token_type_ids.append(0)

                        for token in tokens_b:
                            tokens.append(token)
                            token_type_ids.append(1)

                        tokens.append("[SEP]")
                        token_type_ids.append(0)

                        tokens, mlm_labels, nsp_labels = self._create_mlm_and_nsp_labels(tokens, is_random_next)
                        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                        attention_mask = [1] * len(input_ids)
                        instance = FBertData(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            mlm_labels=mlm_labels,
                            nsp_labels=nsp_labels
                        )
                        instances.append(instance)
                    current_sequences = []
                    current_length = 0
                i += 1
        return instances

    def load_data(self, input_files: str, shuffle=True):
        # ***unpack input files***
        input_files = input_files.split(",")

        # ***create the documents***
        documents = [[]]

        for input_file in input_files:
            with tf.io.gfile.GFile(input_file, "r") as reader:
                while True:
                    text = convert_to_unicode(reader.readline())
                    # There, one document matches counterpart in documents(list).
                    if not text:
                        break
                    text = text.strip()
                    # If exists the blank line, ignores it. we just take the contents of non-blank line.
                    tokens = self.tokenizer.tokenize(text)
                    if not text:
                        documents.append([])
                    else:
                        documents[-1].append(tokens)

        # Removes the blank document.
        documents = [x for x in documents if x]
        # Shuffles the documents in a random sequence.
        if shuffle:
            self.random.shuffle(documents)

        # *** create the instances of input data. ***
        instances = []

        if self.is_dynamic_mask and self.dup_times >= 1 and isinstance(self.dup_times, int):
            for _ in tqdm(range(self.dup_times)):
                instances.extend(self._create_data_from_documents(documents))
        elif not self.is_dynamic_mask:
            instances.extend(self._create_data_from_documents(documents))
        else:
            raise ValueError("The dup_times should be a integer(>=1).")

        if shuffle:
            self.random.shuffle(instances)
        print(len(instances))
        self.instances = instances

    def save_data(self, output_files):
        output_files = output_files.split(",")
        writers = []
        writer_index = 0
        for output_file in output_files:
            writers.append(tf.io.TFRecordWriter(output_file))

        for index, instance in enumerate(self.instances):
            input_ids = instance.input_ids
            attention_mask = instance.attention_mask
            token_type_ids = instance.token_type_ids
            mlm_labels = instance.mlm_labels
            nsp_labels = instance.nsp_labels
            # Pads to max sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                attention_mask.append(0)
                token_type_ids.append(0)
                mlm_labels.append(-100)

            assert len(input_ids) == self.max_seq_length
            assert len(attention_mask) == self.max_seq_length
            assert len(token_type_ids) == self.max_seq_length
            assert len(mlm_labels) == self.max_seq_length
            assert len(nsp_labels) == 1

            feature = collections.OrderedDict()
            feature["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids))
            feature["attention_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=attention_mask))
            feature["token_type_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=token_type_ids))
            feature["mlm_labels"] = tf.train.Feature(int64_list=tf.train.Int64List(value=mlm_labels))
            feature["nsp_labels"] = tf.train.Feature(int64_list=tf.train.Int64List(value=nsp_labels))

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized_example = example.SerializeToString()

            writers[writer_index].write(serialized_example)
            writer_index = (writer_index + 1) % len(writers)

        for writer in writers:
            writer.close()


def main(_argv):
    # Initializes configuration.
    config = FBertConfig(max_seq_length=128)

    # Creates data builder.
    builder = FBertDataBuilder(
        config=config,
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case,
        do_whole_word_mask=FLAGS.do_whole_word_mask,
        short_seq_prob=FLAGS.short_seq_prob,
        is_dynamic_mask=FLAGS.is_dynamic_mask,
        dup_times=FLAGS.dup_times,
        max_masked_word=FLAGS.max_masked_word,
        masked_word_prob=FLAGS.masked_word_prob
    )
    # Loads and saves the data.
    logging.info("*****Loading from file...*****")
    builder.load_data(FLAGS.input_files)
    logging.info("*****Load completed.")
    logging.info("*****Print first 20 example.*****")
    instances = builder.get_instances()
    for i in range(20):
        logging.info(
            "input_ids: {} \n attention_mask: {} \n token_type_ids: {} \n mlm_labels: {} \n nsp_labels: {}".format(
                instances[i].input_ids, instances[i].attention_mask, instances[i].token_type_ids,
                instances[i].mlm_labels, instances[i].nsp_labels
            )
        )

    logging.info("*****Saving to file...*****")
    builder.save_data(FLAGS.output_files)
    logging.info("*****Save completed.*****")
    logging.info(
        "*****Total saved {} instance into {}, respectively.*****".format(len(builder.get_instances()),
                                                                          FLAGS.output_files)
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("input_files")
    flags.mark_flag_as_required("output_files")
    flags.mark_flag_as_required("vocab_file")
    app.run(main)
