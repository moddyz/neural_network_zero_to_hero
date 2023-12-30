"""Utility to transform data between their human readible and machine readible representations."""

from typing import List


class AsciiTranscoder:
    """Encoding and decoding functionality for the ASCII character set."""

    def __init__(self):
        self.chars = [chr(i) for i in range(128)]
        self._char_to_int = {char: i for i, char in enumerate(self.chars)}
        self._int_to_char = {i: char for i, char in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, string):
        """Encode a string into its representative sequence of integers."""
        return [self._char_to_int[char] for char in string]

    def decode(self, int_seq: List[int]):
        """Decode a sequence of integers into its representative string"""
        return "".join([self._int_to_char[integer] for integer in int_seq])
