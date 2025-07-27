import torch
import torch.nn as nn

class RotaryEmbeddingFactory(nn.Module):
    """
    Factory for creating Rotary Positional Embeddings (RoPE).
    
    RoPE encodes positional information by rotating query and key embeddings based on 
    their sequence position. This factory precomputes rotation frequencies and generates
    reusable positional embeddings. This is the recommended approach for most use cases,
    where the embedding is reused across multiple layers.
    
    Usage Pattern:
        # Create factory once during model initialization
        rope_factory = RotaryEmbeddingFactory(head_dim)
        
        # At start of each forward pass, create positional embedding for full sequence
        pos_emb = rope_factory(torch.arange(seq_len, device=device))
        
        # Apply to query and key at each layer
        q = apply_rotary_embedding(q, pos_emb)
        k = apply_rotary_embedding(k, pos_emb)
    
    This implemenation pairs dimensions using the half-flipped method:
        dim i pairs with dim i + dim/2
        Each pair is rotated by angle: position × base^(-2i/dim)
    
    Args:
        dim: Head dimension (must be even for rotation pairs)
        base: Frequency base for position encoding (default: 10,000)
    """
    def __init__(
        self,
        dim: int,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self.dim = dim
        self.base = base
        self._build_theta_cache()
    
    def _build_theta_cache(self):
        """Precompute rotation frequencies: base^(-2i/dim) for each dimension pair."""
        freq_indices = torch.arange(0, self.dim, 2)
        exponents = - freq_indices / self.dim
        theta = self.base ** exponents # [dim // 2]

        self.register_buffer("theta", theta)

    def forward(self, input_positions: torch.Tensor) -> torch.Tensor:
        """
        Generate positional embeddings for given sequence positions.
    
        Creates cos/sin rotation values for each position that can be efficiently applied
        to query/key tensors using apply_rope().
        
        Args:
            input_positions: 1D tensor of position indices, typically torch.arange(seq_len)
        
        Returns:
            Positional embedding tensor [seq_len, 1, dim, 2] where:
            - seq_len matches input_positions length
            - 1 allows broadcasting across attention heads  
            - dim matches the head dimension
            - 2 contains [cos, sin] values for apply_rope()
        """
        if input_positions.ndim != 1:
            raise ValueError(f"input_positions must be a 1D tensor, got {input_positions.ndim}D")
        if input_positions.dtype.is_floating_point:
            raise ValueError(f"input_positions must be an integer tensor, got {input_positions.dtype}")
        if not torch.compiler.is_compiling():
            if torch.any(input_positions < 0): # Expensive check
                raise IndexError(f"Out of bounds: input_positions contains negative values: {input_positions[input_positions < 0]}")

        positions = input_positions.float() # [seq_len]

        # Outer product: [seq_len, 1] * [1, dim // 2] -> [seq_len, dim // 2]
        angles = positions.unsqueeze(1) * self.theta.unsqueeze(0)

        # cache cos and sin together for good cache locality
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Add a heads dimension for broadcasting
        cos = cos.unsqueeze(-2) # [..., seq_len, 1, dim//2]
        sin = sin.unsqueeze(-2) # [..., seq_len, 1, dim//2]

        # Repeat for half-flipped method
        cos_repeated = torch.cat([cos, cos], dim=-1)
        sin_repeated = torch.cat([sin, sin], dim=-1)
        positional_embedding = torch.stack([cos_repeated, sin_repeated], dim=-1)
        # [seq_len, 1, dim, 2]

        return positional_embedding

def apply_rotary_embedding(x: torch.Tensor, positional_embedding: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings using the half-flipped method:
        dim i pairs with dim i + dim/2
        Each pair is rotated by angle: position × base^(-2i/dim)

    Expects positional_embedding from RotaryEmbeddingFactory.forward() with cos/sin values repeated for 
    efficient half-flipped computation.

    Optimised for torch.compile - intermediate tensors will be fused away.

    Usage:
        Create a positional embedding at the start of the model:
            positional_embedding = RotaryEmbeddingFactory(dim).forward(torch.arange(seq_len))
        Apply the positional embedding to query and key on each layer:
            q = apply_rotary_embedding(q, positional_embedding)
            k = apply_rotary_embedding(k, positional_embedding)

    Args:
        x (torch.Tensor): 
            input tensor
            [batch_size, seq_len, num_heads, head_dim]
        positional_embedding (torch.Tensor): 
            positional embedding tensor
            [seq_len, 1, dim, 2]

    Returns:
        torch.Tensor: rotated tensor with same shape as input
    """
    if x.device != positional_embedding.device:
        raise RuntimeError(f"Device mismatch: x on {x.device}, positional_embedding on {positional_embedding.device}")
    if positional_embedding.ndim != 4:
        raise ValueError(f"positional_embedding must be 4D [seq_len, 1, head_dim, 2], got {positional_embedding.ndim}D tensor")
    if x.ndim != 4:
        raise ValueError(f"x must be 4D [batch_size, seq_len, num_heads, head_dim], got {x.ndim}D tensor")

    _, seq_len, _, dim = x.shape
    seq_len_pos, _, dim_pos, _ = positional_embedding.shape
    
    if dim_pos != dim:
        raise RuntimeError(f"Head dimension mismatch: x has {dim}D heads but positional_embedding expects {dim_pos}D")
    if seq_len_pos != seq_len:
        raise RuntimeError(f"Sequence length mismatch: input has {seq_len} tokens but positional embedding has {seq_len_pos} positions")

    if seq_len == 0: # explicitly handle empty sequences safely
        return x

    cos, sin = positional_embedding.unbind(dim=-1) # This is a zero-copy operation.

    x_dtype = x.dtype
    # Disable autocast to maintain precision for rotation calculations
    with torch.autocast(device_type=x.device.type, enabled=False):
        x_first_half, x_second_half = x.chunk(2, dim=-1)
        x_half_flipped = torch.cat([-x_second_half, x_first_half], dim=-1)

        # Apply the rotation
        x_out = (x * cos) + (x_half_flipped * sin)

    return x_out.to(x_dtype)