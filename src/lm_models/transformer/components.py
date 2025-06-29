import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import warnings

from .config import TransformerConfig

class ResidualProjection(nn.Linear):
    """Linear layer that projects back to the residual stream"""
    ...

class Attention(nn.Module):
    """
    Attention mechanism that can use flash attention if available.
    """
    def __init__(self, block_size: int, head_size: int, dropout: float = 0.0, flash: bool = True):
        super().__init__()
        self.dropout = dropout
        self.head_size = head_size
        self.block_size = block_size

        self.flash = flash and hasattr(F, 'scaled_dot_product_attention')
        if flash and not self.flash:
            warnings.warn(
                "Flash Attention is not available, using fallback implementation", 
                UserWarning,
                stacklevel=2
            )
        
        if not self.flash:
            self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size)))
            self.attention_dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        seq_len = q.shape[-2]

        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.block_size}")
        
        if self.flash:
            return F.scaled_dot_product_attention(q, k, v, 
                attn_mask = None, 
                dropout_p = self.dropout if self.training else 0,
                is_causal = True)
        
        # Not flash
        attention_scores = q @ k.transpose(-2, -1) * self.head_size**-0.5
        masked_scores = attention_scores.masked_fill(self.causal_mask[:seq_len, :seq_len] == 0, float('-inf'))
        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        return attention_weights @ v

class AttentionHead(nn.Module):
    """
    Single attention head that processes a head_size dimensional subspace.
    Learns Q/K/V projections and computes attention within that subspace.
    """
    def __init__(self, embed_dim: int, block_size: int, head_size: int, dropout: float = 0.0, flash: bool = True):
        super().__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.query_proj = nn.Linear(embed_dim, head_size, bias=False)
        self.key_proj = nn.Linear(embed_dim, head_size, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_size, bias=False)
        self.attention = Attention(block_size, head_size, dropout, flash)

    def forward(self, input: Tensor) -> Tensor:
        seq_len = input.shape[-2]
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.block_size}")
        
        q = self.query_proj(input)
        k = self.key_proj(input)
        v = self.value_proj(input)

        output = self.attention(q, k, v)

        return output

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention that uses separate AttentionHeads for each head.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([AttentionHead(config.embed_dim, config.block_size, config.head_size, config.dropout, config.flash) for _ in range(config.num_heads)])
        self.output_proj = ResidualProjection(config.num_heads * config.head_size, config.embed_dim)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, input: Tensor) -> Tensor:
        seq_len = input.shape[-2]
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")
        
        out = torch.cat([head(input) for head in self.heads], dim=-1)
        out = self.output_proj(out)
        out = self.residual_dropout(out)
        return out
    
class ParallelMultiHeadAttention(nn.Module):
    """
    Multi-head attention that uses a single linear projection to compute Q, K, V.

    - Reduces matrix multiplications from 3*num_heads to 1
    - Mathematically equivalent to the non-parallel version
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # extract from config for convenience
        self.num_heads = config.num_heads
        self.head_size = config.head_size

        self.qkv_proj = nn.Linear(config.embed_dim, config.num_heads * config.head_size * 3, bias=False)
        self.attention = Attention(config.block_size, self.head_size, config.dropout, config.flash)
        self.output_proj = ResidualProjection(config.num_heads * self.head_size, config.embed_dim)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, input: Tensor) -> Tensor:
        B, T, _ = input.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.config.block_size}")

        # Compute q, k, and v using a single linear projection
        # q, k, and v are concatenated along the last dimension
        qkv = self.qkv_proj(input) # (B, T, num_heads * head_size * 3)
        q, k, v = qkv.split(self.num_heads * self.head_size, dim=-1) # (B, T, num_heads * head_size)
        
        # Reshape for attention
        # Attention operates on the (T, head_size) dimensions
        # (B, num_heads) are batch dimensions processed in parallel
        # (B, T, num_heads * head_size) -> (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
         
        output = self.attention(q, k, v)

        # Concatenate heads
        # Transpose creates a non-contiguous tensor, so we need to call contiguous()
        # (B, num_heads, T, head_size) -> (B, T, num_heads * head_size)
        output = output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size)
        
        output = self.output_proj(output)
        output = self.residual_dropout(output)
        return output


class FeedForward(nn.Module):
    """
    Position-wise feedforward network.
    Provides non-linear transformations between attention layers, enabling
    the model to learn complex functions beyond linear attention.
    """
    def __init__(self, embed_dim: int, hidden_multiplier: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_multiplier * embed_dim),
            nn.ReLU(),
            ResidualProjection(hidden_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)
    
    
class TransformerBlock(nn.Module):
    """
    Standard transformer layer combining multi-head attention with feedforward network.
    Uses pre-norm architecture (LayerNorm before each sublayer) with residual connections.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = ParallelMultiHeadAttention(config) if config.parallel \
                         else MultiHeadAttention(config)
        self.feed_forward = FeedForward(config.embed_dim, config.hidden_multiplier, config.dropout)
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, residual_stream: Tensor) -> Tensor:
        residual_stream = residual_stream + self.self_attention(self.layer_norm1(residual_stream))
        residual_stream = residual_stream + self.feed_forward(self.layer_norm2(residual_stream))
        return residual_stream