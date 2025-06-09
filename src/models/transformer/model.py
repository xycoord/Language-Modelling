import torch
import torch.nn as nn
from torch.nn import functional as F
from .components import TransformerBlock, ResidualProjection
import math
from .config import TransformerConfig

class TransformerLanguageModel(nn.Module):
    """
    A transformer language model that predicts the next token in a sequence.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding_table = nn.Embedding(config.block_size, config.embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.embed_dim)
        self.to_logits = nn.Linear(config.embed_dim, config.vocab_size)

        # Explicitly initialize the weights as per GPT-2
        self.base_std = 0.02
        self.scaled_std = self.base_std / math.sqrt(2 * config.n_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights of the model as per GPT-2"""
        if isinstance(module, ResidualProjection):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.scaled_std)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.base_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.base_std)

    def forward(self, idx):
        _, T = idx.shape

        # Embed tokens and positions
        token_embeds = self.token_embedding_table(idx) # (B, T, embed_dim)
        position_embeds = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, embed_dim)
        combined_embeds = token_embeds + position_embeds # (B, T, embed_dim)

        # Apply blocks and final layer norm
        residual_stream = self.blocks(combined_embeds) # (B, T, embed_dim)
        residual_stream = self.final_layer_norm(residual_stream)

        # Project to logits
        logits = self.to_logits(residual_stream) # (B, T, vocab_size)
        
        return logits

    def generate(self, idx, max_new_tokens):
        """Generate a sequence of tokens from the model"""
        assert idx.shape[1] + max_new_tokens <= self.config.block_size, "Cannot generate more tokens than the block size"
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx