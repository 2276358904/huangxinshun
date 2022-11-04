from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import tensorflow as tf

from modeling_configs import FBertConfig
from classify_generator import FBertClassifyDataGenerator


class FBertClassifyGeneratorTest(tf.test.TestCase):
    def test_classify_generator(self):
        config = FBertConfig()
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
                        "##ing", ","]
        with tempfile.NamedTemporaryFile("w+", delete=False) as file_wrap:
            file_wrap.write("".join([x + "\n" for x in vocab_tokens]))

        generator = FBertClassifyDataGenerator(
            config, vocab_file=file_wrap.name, do_lower_case=True, input_dir=None, output_dir=None
        )
        tokens_a = ["un", "##want", "##ed", ",", "runn", "##ing"]
        input_ids, attention_mask, token_type_ids = generator._create_data_from_tokens(tokens_a)
        self.assertAllEqual(input_ids, [1, 7, 4, 5, 10, 8, 9, 2])

        text = "unwanted,running"
        label = 0
        generator._load_data(text, label)
        actual_list = [1, 7, 4, 5, 10, 8, 9, 2]
        pad_list = [0] * (config.max_seq_length - len(actual_list))
        actual_list.extend(pad_list)
        self.assertAllEqual(
            generator.instances[0].input_ids, actual_list
        )
