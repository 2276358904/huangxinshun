from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import tempfile
import six

import tensorflow as tf

from . import tokenization


class TokenizationTest(tf.test.TestCase):
    def test_full_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing", ","
        ]
        with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
            if six.PY2:
                vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
            else:
                vocab_writer.write("".join(
                    [x + "\n" for x in vocab_tokens]).encode("utf-8"))

            vocab_file = vocab_writer.name

        tokenizer = tokenization.FullTokenizer(vocab_file)
        os.unlink(vocab_file)

        tokens = tokenizer.tokenize(u"UNwant\u00E9d,running")
        self.assertAllEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])

        self.assertAllEqual(
            tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_basic_tokenizer_chinese(self):
        tokenizer = tokenization.BasicTokenizer()

        self.assertAllEqual(
            tokenizer.tokenize(u"ah\u535A\u63A8zz"),
            [u"ah", u"\u535A", u"\u63A8", u"zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = tokenization.BasicTokenizer(do_lower_case=True)

        self.assertAllEqual(
            tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
            ["hello", "!", "how", "are", "you", "?"])
        self.assertAllEqual(tokenizer.tokenize(u"H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

        self.assertAllEqual(
            tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
            ["HeLLo", "!", "how", "Are", "yoU", "?"])

    def test_wordpiece_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

        self.assertAllEqual(tokenizer.tokenize(""), [])

        self.assertAllEqual(
            tokenizer.tokenize("unwanted running"),
            ["un", "##want", "##ed", "runn", "##ing"])

        self.assertAllEqual(
            tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])


if __name__ == "__main__":
    tf.test.main()
