import torch
import torch.nn as nn
from torch.nn import functional as F
import warnings
from .config import TransformerConfig

class ResidualProjection(nn.Linear):
    """Linear layer that projects back to the residual stream"""
    pass

class Attention(nn.Module):
    """
    Attention mechanism that can use flash attention if available.
    """
    def __init__(self, block_size, head_size, dropout=0.0, flash=True):
        super().__init__()
        self.dropout = dropout
        self.head_size = head_size

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

    def forward(self, q, k, v):
        if self.flash:
            return F.scaled_dot_product_attention(q, k, v, 
                attn_mask = None, 
                dropout_p = self.dropout if self.training else 0,
                is_causal = True)
        
        # Not flash
        seq_len = q.shape[-2]
        attention_scores = q @ k.transpose(-2, -1) * self.head_size**-0.5
        masked_scores = attention_scores.masked_fill(self.causal_mask[:seq_len, :seq_len] == 0, float('-inf'))
        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        return attention_weights @ v

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, block_size, head_size, dropout=0.0, flash=True):
        super().__init__()
        self.head_size = head_size
        self.query_proj = nn.Linear(embed_dim, head_size, bias=False)
        self.key_proj = nn.Linear(embed_dim, head_size, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_size, bias=False)
        self.attention = Attention(block_size, head_size, dropout, flash)

    def forward(self, input):
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
        self.heads = nn.ModuleList([AttentionHead(config.embed_dim, config.block_size, config.head_size, config.dropout, config.flash) for _ in range(config.num_heads)])
        self.proj = ResidualProjection(config.num_heads * config.head_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        out = torch.cat([head(input) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class ParallelMultiHeadAttention(nn.Module):
    """
    Multi-head attention that uses a single linear projection to compute Q, K, V.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # extract from config for convenience
        self.num_heads = config.num_heads
        self.head_size = config.head_size

        self.qkv_proj = nn.Linear(config.embed_dim, config.num_heads * config.head_size * 3, bias=False)
        self.attention = Attention(config.block_size, self.head_size, config.dropout, config.flash)
        self.proj = ResidualProjection(config.num_heads * self.head_size, config.embed_dim)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        B, T, _ = input.shape

        # Compute Q, K, V using a single linear projection
        qkv = self.qkv_proj(input) # (B, T, num_heads * head_size * 3)
        q, k, v = qkv.split(self.num_heads * self.head_size, dim=-1) # (B, T, num_heads * head_size)
        
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)

        output = self.attention(q, k, v)

        output = output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size) # (B, T, num_heads * head_size)
        output = self.proj(output)
        output = self.residual_dropout(output)
        return output


class FeedForward(nn.Module):

    def __init__(self, embed_dim, hidden_multiplier=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_multiplier * embed_dim),
            nn.ReLU(),
            ResidualProjection(hidden_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.net(input)
    
class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = ParallelMultiHeadAttention(config) if config.parallel \
                         else MultiHeadAttention(config)
        self.feed_forward = FeedForward(config.embed_dim, dropout=config.dropout)
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, residual_stream):
        residual_stream = residual_stream + self.self_attention(self.layer_norm1(residual_stream))
        residual_stream = residual_stream + self.feed_forward(self.layer_norm2(residual_stream))
        return residual_stream