from abc import ABC, abstractmethod
from typing import TypeAlias

Token: TypeAlias = int

class Tokenizer(ABC):
    def __init__(self):
        self.vocab_size = 0
    
    @abstractmethod
    def encode(self, text: str) -> list[Token]:
        pass

    @abstractmethod
    def decode(self, tokens: list[Token]) -> str:
        pass