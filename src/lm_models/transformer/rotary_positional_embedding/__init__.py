"""
Rotary Positional Embeddings for Transformers

Two implementations available:
- direct_application: All-in-one layer for ease of use
  This directly applies the embedding to the input tensor.
- factory: Factory pattern for efficiently reusing the same embedding across multiple layers
  This creates a reusable embedding of the position which is separately applied to the input tensor.
"""

from .direct_application import RotaryPositionalEmbedding
from .factory import RotaryEmbeddingFactory, apply_rotary_embedding

__all__ = [
    "RotaryPositionalEmbedding",
    "RotaryEmbeddingFactory", 
    "apply_rotary_embedding"
]