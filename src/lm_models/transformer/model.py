import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math

# Local imports
from .components import TransformerBlock, ResidualProjection
from .kv_cache import KVCache
from .config import TransformerConfig

class TransformerLanguageModel(nn.Module):
    """
    A transformer language model for next token prediction.

    Features:
    - KV cache support for efficient autoregressive generation
    - Weight tying between input and output embeddings
    - GPT-2 style weight initialization with residual scaling
    - Configurable architecture via TransformerConfig
    
    Args:
        config: TransformerConfig specifying model dimensions and hyperparameters
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding_table = nn.Embedding(config.block_size, config.embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.embed_dim)
        self.to_logits = nn.Linear(config.embed_dim, config.vocab_size)

        # Weight tying
        self.token_embedding_table.weight = self.to_logits.weight

        # Explicitly initialize the weights as per GPT-2
        self.base_std = 0.02
        self.scaled_std = self.base_std / math.sqrt(2 * config.n_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights following GPT-2 scheme:
        - Normal(0, 0.02) for embeddings and linear layers
        - Residual projections scaled by 1/sqrt(2*n_layers) for gradient flow
        """
        if isinstance(module, ResidualProjection):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.scaled_std)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.base_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.base_std)

    def forward(self, tokens: Tensor, kv_cache: KVCache | None = None) -> Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            tokens: Input token indices of shape (batch_size, seq_len)
            kv_cache: Optional KV cache for efficient autoregressive generation.
                    If provided, only processes new tokens and updates cache.
        
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
            
        Raises:
            ValueError: If sequence length exceeds block_size or tokens are out of vocab range
        """
        _, T = tokens.shape
        cache_len = len(kv_cache[0]) if kv_cache is not None else 0

        if T + cache_len > self.config.block_size:
            raise ValueError(f"Sequence length {T + cache_len} exceeds block size {self.config.block_size}")
        if T > 0 and tokens.max() >= self.config.vocab_size:
            raise ValueError("Token values must be between 0 and vocab_size")

        # Embed tokens and positions
        token_embeds = self.token_embedding_table(tokens) # (B, T, embed_dim)
        position_embeds = self.position_embedding_table(torch.arange(cache_len, cache_len + T, device=tokens.device)) # (T, embed_dim)
        combined_embeds = token_embeds + position_embeds # (B, T, embed_dim) - broadcasts position_embeds across batch

        # Apply blocks and final layer norm
        residual_stream = combined_embeds
        if kv_cache is None:
            for block in self.blocks:
                residual_stream = block(residual_stream) # (B, T, embed_dim)
        else:
            for block, cache_layer in zip(self.blocks, kv_cache):
                # cache_layer is implicitly updated by the block
                residual_stream = block(residual_stream, cache_layer) # (B, T, embed_dim)
        residual_stream = self.final_layer_norm(residual_stream)

        # Project to logits
        logits = self.to_logits(residual_stream) # (B, T, vocab_size)
        
        return logits

    @torch.no_grad()
    def generate(self, context: Tensor, max_new_tokens: int, kv_cache: KVCache) -> Tensor:
        """
        Generate tokens autoregressively using multinomial sampling.
        
        Processes the initial context through the model, then generates new tokens
        one at a time using the KV cache for efficiency. The cache is updated in-place
        and can be reused for subsequent generation calls.
        
        Args:
            context: Initial context tokens of shape (batch_size, context_len).
                    This is processed once and stored in the cache.
            max_new_tokens: Number of new tokens to generate.
            kv_cache: KV cache to use and update. Must be created with matching
                    batch_size, dtype, and device.
        
        Returns:
            Generated tokens of shape (batch_size, max_new_tokens).
            
        Raises:
            ValueError: If context_len + max_new_tokens exceeds the model's block_size.
        """
        cache_len = len(kv_cache[0])
        _, T = context.shape
        if cache_len + T + max_new_tokens > self.config.block_size:
            raise ValueError("Cannot generate more tokens than the block size")

        generated_tokens = []
        input_tokens = context
        for _ in range(max_new_tokens):
            logits = self(input_tokens, kv_cache)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(new_token)
            input_tokens = new_token
            
        if len(generated_tokens) == 0:
            return torch.empty(input_tokens.shape[0], 0, device=input_tokens.device)
        return torch.cat(generated_tokens, dim=1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype