from dataclasses import dataclass

@dataclass
class BigramConfig:
    vocab_size: int

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")