from torch import Tensor
import torch

from .config import TransformerConfig

class KVCacheLayer:
    """
    A cache layer for storing keys and values for a single layer of the transformer.
    The keys and values are stored in separate, pre-allocated tensors.

    They use the dimensions (B, num_heads, T, head_size) for the keys and values
    such that they can be used directly in the attention mechanism.

    Intended usage in MHA:
    1. Append new keys and values to the cache.
    2. Get the full keys and values tensors for use in the attention mechanism.
    """

    def __init__(self, keys: Tensor, values: Tensor, current_len: int):
        """
        The constructor is not intended to be used directly.
        Use the empty class method instead.
        """
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("Keys and values must have 4 dimensions: (B, num_heads, T, head_size)")
        if keys.shape != values.shape:
            raise ValueError("Keys and values must have the same shape")
        if current_len > keys.shape[2]:
            raise ValueError("Current length must be less than or equal to the block size in the keys and values")
        if current_len < 0:
            raise ValueError("Current length cannot be negative")
        self._keys = keys
        self._values = values
        self.max_len = keys.shape[2]
        self._current_len = current_len

    @classmethod
    def empty(cls, config: TransformerConfig, batch_size: int, dtype: torch.dtype, device: torch.device) -> 'KVCacheLayer':
        """
        Creates an empty cache layer.
        The keys and values tensors are pre-allocated to the full block size.
        Args:
            config: The transformer config.
            batch_size: The batch size.
            dtype: The dtype of the keys and values tensors.
            device: The device of the keys and values tensors.
        """
        if config.head_size is None:
            raise ValueError("head_size must be set in the config")
        keys = torch.empty(batch_size, config.num_heads, config.block_size, config.head_size, dtype=dtype, device=device)
        values = torch.empty(batch_size, config.num_heads, config.block_size, config.head_size, dtype=dtype, device=device)
        return cls(keys, values, 0)

    def __len__(self) -> int:
        """
        Returns the current length of the cache.
        """
        return self._current_len

    def clone(self) -> 'KVCacheLayer':
        """
        Returns a deep copy of the cache layer, including cloned tensors.
        Useful for creating independent cache states or checkpointing.
        """
        return KVCacheLayer(self._keys.clone(), self._values.clone(), self._current_len)

    def get_tensor_copies(self, length: int | None = None) -> tuple[Tensor, Tensor]:
        """
        Returns a cloned tuple of the keys and values tensors.
        Use this when you need independent tensors that won't be affected by further cache updates.
        Args:
            length: The length to truncate the tensors to. If None, it's truncated to the current length.
        """
        if length is None:
            length = self._current_len
        return self._keys[:,:,:length,:].clone(), self._values[:,:,:length,:].clone()

    def append(self, new_keys: Tensor, new_values: Tensor):
        """
        Appends new keys and values to the cache.
        Args:
            new_keys: The new keys to append. Must have shape (B, num_heads, new_len, head_size).
            new_values: The new values to append. Must have shape (B, num_heads, new_len, head_size).
        """
        # device validation
        if new_keys.device != self._keys.device:
            raise ValueError(f"Device mismatch: cache on {self._keys.device}, new keys on {new_keys.device}")
        if new_values.device != self._values.device:
            raise ValueError(f"Device mismatch: cache on {self._values.device}, new values on {new_values.device}")
        
        # dtype validation  
        if new_keys.dtype != self._keys.dtype:
            raise ValueError(f"Dtype mismatch: cache expects {self._keys.dtype}, got {new_keys.dtype}")
        if new_values.dtype != self._values.dtype:
            raise ValueError(f"Dtype mismatch: cache expects {self._values.dtype}, got {new_values.dtype}")

        if new_keys.shape[2] != new_values.shape[2]:
            raise ValueError("New keys and values must have the same number of tokens")

        new_keys_dims = new_keys.shape[:2] + (new_keys.shape[3],)
        cache_dims = self._keys.shape[:2] + (self._keys.shape[3],)
        if new_keys_dims != cache_dims:
            raise ValueError("New keys and values must match the dimensions of the cache")

        new_len = self._current_len + new_keys.shape[2]
        if new_len > self.max_len:
            raise ValueError("KVCacheLayer is full")

        self._keys[:,:,self._current_len:new_len,:] = new_keys
        self._values[:,:,self._current_len:new_len,:] = new_values
        self._current_len = new_len

    @property
    def keys(self) -> Tensor:
        """
        Returns the keys tensor truncated to the current length.
        """
        return self._keys[:,:,:self._current_len,:]

    @property
    def values(self) -> Tensor:
        """
        Returns the values tensor truncated to the current length.
        """
        return self._values[:,:,:self._current_len,:]

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the keys and values tensors.
        """
        return self._keys.dtype
    
    @property
    def device(self) -> torch.device:
        """
        Returns the device of the keys and values tensors.
        """
        return self._keys.device

    def to(self, device: torch.device) -> 'KVCacheLayer':
        """
        Moves the keys and values tensors to the given device.
        Returns self for chaining.
        """
        self._keys = self._keys.to(device)
        self._values = self._values.to(device)
        return self

KVCache = tuple[KVCacheLayer, ...]