from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int
    embed_dim: int = 32
    num_heads: int = 4
    head_size: Optional[int] = None
    n_layers: int = 4
    dropout: float = 0.0
    # FeedForward: hidden_dim = embed_dim * hidden_multiplier
    hidden_multiplier: int = 4
    parallel: bool = True
    flash: bool = True

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_multiplier <= 0:
            raise ValueError(f"hidden_multiplier must be positive, got {self.hidden_multiplier}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        
        if self.head_size is None:
            self.head_size = self.embed_dim // self.num_heads