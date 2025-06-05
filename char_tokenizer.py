class CharTokenizer:
    """
    A tokenizer that converts between strings and tokens.
    Each character is mapped to a unique token.
    Each token is mapped to a unique character.
    The vocabulary is minimal for the training text.
    """
    def __init__(self, text):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_token = {ch:i for i, ch in enumerate(self.chars)}
        self.token_to_char = {i:ch for i, ch in enumerate(self.chars)}

    def encode(self, string):
        """Convert a string to a list of tokens"""
        return [self.char_to_token[c] for c in string]

    def decode(self, tokens):
        """Convert a list of tokens to a string"""
        return ''.join([self.token_to_char[t] for t in tokens])