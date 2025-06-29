from .base import Tokenizer, Token

class CharTokenizer(Tokenizer):
    """
    A tokenizer that converts between strings and tokens.
    Each character is mapped to a unique token.
    Each token is mapped to a unique character.
    The vocabulary is minimal for the training text.
    """
    def __init__(self, text: str):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_token = {ch:i for i, ch in enumerate(self.chars)}
        self.token_to_char = {i:ch for i, ch in enumerate(self.chars)}

    def encode(self, string: str) -> list[Token]:
        """Convert a string to a list of tokens"""
        unknown_chars = set(string) - set(self.chars)
        if unknown_chars:
            raise ValueError(f"Characters not in vocabulary: {unknown_chars}")
        return [self.char_to_token[c] for c in string]

    def decode(self, tokens: list[Token]) -> str:
        """Convert a list of tokens to a string"""
        invalid_tokens = [t for t in tokens if t not in self.token_to_char]
        if invalid_tokens:
            raise ValueError(f"Invalid tokens: {invalid_tokens}")
        return ''.join([self.token_to_char[t] for t in tokens])