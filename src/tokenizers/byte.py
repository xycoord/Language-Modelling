from .base import Tokenizer, Token

class ByteTokenizer(Tokenizer):
    """
    A tokenizer that converts between strings and byte tokens.
    Each character is mapped to it's UTF-8 byte sequence.
    The vocabulary is the set of all UTF-8 bytes.
    """
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> list[Token]:
        """Convert a string to a list of tokens"""
        return list(text.encode('utf-8'))

    def decode(self, tokens: list[Token]) -> str:
        """Convert a list of tokens to a string"""
        return bytes(tokens).decode('utf-8', errors='replace')
        