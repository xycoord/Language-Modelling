import torch
import torch.nn as nn
from typing import Optional

class RotaryPositionalEmbedding(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    In this implementation we cache the embeddings for each position up to
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. Must be even. This is usually set to the dim 
            of each head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the model.
        base (int): The base for the geometric progression used to compute
            the rotation angles. Default: 10,000 (as in the original paper).
        interleaved (bool): Whether to use the interleaved or half-flipped method.
            True:  Rotate adjacent pairs of dimensions.
                   i.e. (0, 1), (2, 3), ... (dim - 2, dim - 1)
                   This is the implementation of RoPE in the original paper.
            False: Pair the first half of the dimensions with the second half.
                   i.e. (0, dim//2), (1, dim//2 + 1), ... (dim//2 - 1, dim - 1)
                   This is the implementation of RoPE in the Meta Llama repo.
                   It's faster than the interleaved method because it doesn't require 
                   reshaping the input.
            Model weights are dependent on this parameter so use the same mode for 
            training and inference.

    Note: if reusing the same embedding across multiple layers, it is recommended to
          use the factory module instead (see ``RotaryEmbeddingFactory``).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: int = 10_000,
        interleaved: bool = True,
    ) -> None:
        super().__init__()
        
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        if max_seq_len < 0:
            raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}")
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.interleaved = interleaved
        self.register_buffer("cache", torch.Tensor(), persistent=False)
        self._build_cache()

    @property
    def dtype(self):
        return self.cache.dtype
    
    @property
    def device(self):
        return self.cache.device

    def _build_cache(self):
        freq_indices = torch.arange(0, self.dim, 2, dtype=self.dtype, device=self.device)
        exponents = - freq_indices / self.dim
        theta = self.base ** exponents # [dim // 2]

        position_indices = torch.arange(self.max_seq_len, dtype=self.dtype, device=self.device)

        # Outer product: [max_seq_len, 1] * [1, dim // 2] -> [max_seq_len, dim // 2]
        angles = position_indices.unsqueeze(1) * theta.unsqueeze(0)
        angles = angles.float() 

        # cache cos and sin together for good cache locality
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Add a heads dimension for broadcasting
        cos = cos.unsqueeze(-2) # [..., seq_len, 1, dim//2]
        sin = sin.unsqueeze(-2) # [..., seq_len, 1, dim//2]

        # The cache is structured differently depending on the method.
        if self.interleaved:
            # Cache has shape [max_seq_len, 1, dim // 2, 2]
            self.cache = torch.stack([cos, sin], dim=-1)
        else:
            # Pre-double the cache.
            cos_doubled = torch.cat([cos, cos], dim=-1)
            sin_doubled = torch.cat([sin, sin], dim=-1)
            # Cache has shape [max_seq_len, 1, dim, 2]
            self.cache = torch.stack([cos_doubled, sin_doubled], dim=-1)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[batch_size, seq_len, num_heads, head_dim]``
            input_pos (Optional[torch.Tensor]): 
                Optional tensor which contains the position ids of each token.
                During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [batch_size, seq_len].
                During inference, this indicates the position of the current token. 
                If input_pos is not None, the cache is indexed with input_pos. 
                If None, assume the index of the token is its position id. 
                Default is None.

        Returns:
            torch.Tensor: rotated tensor with same shape as input

        """
        if x.ndim != 4:
            raise ValueError(f"x must be a 4D tensor, got {x.ndim}D")
        if x.size(-1) != self.dim:
            raise ValueError(f"x.size(-1) must match dim, got {x.size(-1)} and {self.dim}")

        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len out of bounds, must be less than or equal to max_seq_len, got {seq_len} and {self.max_seq_len}")
        if seq_len == 0: # explicitly handle empty sequences safely
            return x

        cos, sin = self._get_cached_cos_sin(input_pos, (x.shape[0], seq_len))

        x_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.interleaved:
                x_out = self._apply_rope_interleaved(x, cos, sin)
            else:
                x_out = self._apply_rope_half_flipped(x, cos, sin)

        return x_out.to(x_dtype)

    def _get_cached_cos_sin(self, input_pos: torch.Tensor | None, input_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the cached cos and sin values from the cache.
        """
        if input_pos is not None:
            if input_pos.dtype.is_floating_point:
                raise ValueError(f"input_pos must be an integer tensor, got {input_pos.dtype}")
            if not torch.compiler.is_compiling():
                if torch.any(input_pos < 0): # Expensive check
                    raise IndexError(f"Out of bounds: input_pos contains negative values: {input_pos[input_pos < 0]}")
            try:
                torch.broadcast_shapes(input_pos.shape, input_shape)
            except RuntimeError as e:
                raise RuntimeError(f"input_pos shape {input_pos.shape} not broadcastable with x batch/seq dims {input_shape}") from e
            
            # use specified positions to index the cache
            rope_cache = self.cache[input_pos]
        else:
            # default to starting from 0
            seq_len = input_shape[1]
            rope_cache = self.cache[:seq_len]

        cos, sin = rope_cache.unbind(dim=-1) # This is a zero-copy operation.

        return cos, sin

    def _apply_rope_interleaved(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply the rotary positional embeddings to the input tensor using the interleaved method.
        This is the original implementation of RoPE.
        """
        # Reshape input into odd and even indices
        x_paired = x.reshape(*x.shape[:-1], self.dim // 2, 2)
        x_odd, x_even = x_paired[..., 0], x_paired[..., 1]
        
        # Apply the rotation
        x_out_odd = x_odd * cos - x_even * sin
        x_out_even = x_odd * sin + x_even * cos

        # Combine back and flatten
        x_out = torch.stack([x_out_odd, x_out_even], dim=-1).flatten(3)

        return x_out

    def _apply_rope_half_flipped(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply the rotary positional embeddings to the input tensor using the half-flipped method.
        This is the implementation of RoPE in the Meta Llama repo.
        It's faster than the interleaved method because it doesn't require reshaping the input.
        """

        x_first_half, x_second_half = x.chunk(2, dim=-1)
        x_half_flipped = torch.cat([-x_second_half, x_first_half], dim=-1)

        # Apply the rotation
        x_out = (x * cos) + (x_half_flipped * sin)

        return x_out
