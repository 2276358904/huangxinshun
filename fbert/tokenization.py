from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import collections
import six
import unicodedata

import tensorflow as tf

from tokenization_utils import _is_punctuation, _is_whitespace, _is_control, _is_chinese


def convert_to_unicode(text):
    """
    Converts the input text to unicode encoding if it is not already, assume the
    unicode encoding of input is utf-8.
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return six.ensure_text(text, "utf-8", "ignore")
        elif isinstance(text, six.text_type):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""
    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, six.text_type):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    # Splits text into tokens according to whitespace, and the result excludes
    # the whitespace.
    tokens = text.split()
    return tokens


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            tokens = convert_to_unicode(reader.readline())
            if not tokens:
                break
            # If two or more new tokens in one line, which split according
            # to whitespace, add it to vocabulary.
            tokens = tokens.strip().split() if tokens.strip() else " "
            for token in tokens:
                if token not in vocab:
                    vocab[token] = index
                    index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab.get(item))
    return output


class FullTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True):
        """
        Runs end-to-end tokenization.

        Args:
            vocab_file: File containing the vocabulary.
            do_lower_case: Whether to lowercase the input when tokenizing.
        """
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its wordpiece tokens. This should be passed through
        a BasicTokenizer and WordpieceTokenizer.

        Args:
            text: A long document text that should be tokenized.

        Returns:
            A list of wordpiece tokens.
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens (string) in a sequence of ids (int)."""
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids (int) in a sequence of tokens (string)."""
        return convert_by_vocab(self.inv_vocab, ids)

    def convert_text_to_ids(self, text):
        """Converts a single string in a sequence of ids (int)."""
        tokens = self.tokenize(text)
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_text(self, ids):
        """Converts a sequence of ids (int) in a single string."""
        tokens = self.convert_ids_to_tokens(ids)
        return " ".join(tokens).replace(" ##", "").strip()


class BasicTokenizer(object):
    def __init__(self, do_lower_case=True):
        """
        Constructs a BasicTokenizer that will run basic tokenization (punctuation
        splitting, lower casing, etc.).

        Args:
            do_lower_case: Whether to lowercase the input when tokenizing.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    @staticmethod
    def _clean_text(text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            # Gets a value of character, which is a hexadecimal or decimal number.
            # More information about unicode chart, please infer to
            # https://www.ssec.wisc.edu/~tomw/java/unicode.html.
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            if _is_chinese(char):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _run_strip_accents(text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    @staticmethod
    def _run_split_on_punc(text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]


class WordpieceTokenizer(object):
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        """
        Constructs a WordPieceTokenizer that will run WordPiece tokenization.
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy
        longest-match-first algorithm to perform tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """
        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start: end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
