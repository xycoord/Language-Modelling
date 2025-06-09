class BaseTokenizer:
    def __init__(self):
        self.vocab_size = 0

    def encode(self, text):
        """Convert a string to a list of tokens"""
        raise NotImplementedError

    def decode(self, tokens):
        """Convert a list of tokens to a string"""
        raise NotImplementedError
