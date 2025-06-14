import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math

# Local imports
from .components import TransformerBlock, ResidualProjection
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

    def _init_weights(self, module: nn.Module):
        """Initialize the weights of the model as per GPT-2"""
        if isinstance(module, ResidualProjection):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.scaled_std)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.base_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.base_std)

    def forward(self, context: Tensor) -> Tensor:
        _, T = context.shape

        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.config.block_size}")
        if not T == 0 and context.max() >= self.config.vocab_size:
            raise ValueError("Token values must be between 0 and vocab_size")

        # Embed tokens and positions
        token_embeds = self.token_embedding_table(context) # (B, T, embed_dim)
        position_embeds = self.position_embedding_table(torch.arange(T, device=context.device)) # (T, embed_dim)
        combined_embeds = token_embeds + position_embeds # (B, T, embed_dim) - broadcasts position_embeds across batch

        # Apply blocks and final layer norm
        residual_stream = self.blocks(combined_embeds) # (B, T, embed_dim)
        residual_stream = self.final_layer_norm(residual_stream)

        # Project to logits
        logits = self.to_logits(residual_stream) # (B, T, vocab_size)
        
        return logits

    @torch.no_grad()
    def generate(self, context: Tensor, max_new_tokens: int) -> Tensor:
        """Generate tokens autoregressively using multinomial sampling."""
        if context.shape[1] + max_new_tokens > self.config.block_size:
            raise ValueError("Cannot generate more tokens than the block size")
        
        for _ in range(max_new_tokens):
            logits = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, new_token), dim=1)
        return context

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device