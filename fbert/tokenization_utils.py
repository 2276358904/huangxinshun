from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata


def _is_control(char):
    """Checks whether "char" is a control character."""
    # This is technically control characters, but we count them as whitespace characters.
    if char == "\t" or char == "\r" or char == "\n":
        return False
    cat = unicodedata.category(char)
    # If char is a control or format character, return true.
    # For example, "\t" is control character with a unicode number 9,
    # and any format character with number range in 65520 to 65535.
    if cat in ["Cc", "Cf"]:
        return True
    return False


def _is_whitespace(char):
    """Checks whether "char" is a whitespace character."""
    # \t, \n, and \r are technically control characters, but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\r" or char == "\t" or char == "\n":
        return True
    cat = unicodedata.category(char)
    if cat in ["Zs"]:
        return True
    return False


def _is_chinese(char):
    """Checks whether "char" is the codepoint of a CJK character."""
    cp = ord(char)
    # This defines a "chinese character" as anything in the CJK Unicode block:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of other languages.
    if (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)  #
        or (0x20000 <= cp <= 0x2A6DF)  #
        or (0x2A700 <= cp <= 0x2B73F)  #
        or (0x2B740 <= cp <= 0x2B81F)  #
        or (0x2B820 <= cp <= 0x2CEAF)  #
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def _is_punctuation(char):
    """Checks whether "char" is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class, but we treat them as punctuation anyway, for
    # consistency.
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


