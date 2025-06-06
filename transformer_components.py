import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualProjection(nn.Linear):
    """Linear layer that projects back to the residual stream"""
    pass

class AttentionHead(nn.Module):
    def __init__(self, head_size, embed_dim, block_size, dropout=0.0):
        super().__init__()
        self.head_size = head_size
        self.key_proj = nn.Linear(embed_dim, head_size, bias=False)
        self.query_proj = nn.Linear(embed_dim, head_size, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        _, seq_len, _ = input.shape

        k = self.key_proj(input)
        q = self.query_proj(input)
        v = self.value_proj(input)

        attention_scores = q @ k.transpose(-2, -1) * self.head_size**-0.5
        masked_scores = attention_scores.masked_fill(self.causal_mask[:seq_len, :seq_len] == 0, float('-inf'))
        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = attention_weights @ v

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_dim, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, embed_dim, block_size, dropout) for _ in range(num_heads)])
        self.proj = ResidualProjection(num_heads * head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        out = torch.cat([head(input) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            ResidualProjection(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.net(input)
    
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, head_size, block_size, dropout=0.0):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads, head_size, embed_dim, block_size, dropout)
        self.feed_forward = FeedForward(embed_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, residual_stream):
        residual_stream = residual_stream + self.self_attention(self.layer_norm1(residual_stream))
        residual_stream = residual_stream + self.feed_forward(self.layer_norm2(residual_stream))
        return residual_stream