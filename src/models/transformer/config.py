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
    parallel: bool = True
    flash: bool = True

    def __post_init__(self):
        if self.head_size is None:
            self.head_size = self.embed_dim // self.num_heads