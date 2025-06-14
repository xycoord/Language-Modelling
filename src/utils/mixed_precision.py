import torch
from contextlib import nullcontext
from typing import ContextManager

from .config_loader import Config

def setup_precision(config: Config) -> None:
    """
    One-time setup of precision settings. Call once at startup.
    """
    if config.device == 'cuda':
        torch.set_float32_matmul_precision('high')
        
        if config.mixed_precision and torch.cuda.is_bf16_supported():
            print('Using mixed precision: bfloat16 with TF32 matmul')
        else:
            print('Using precision: float32 with TF32 matmul')
    else:
        torch.set_float32_matmul_precision('highest')
        print('Using precision: float32 (CPU)')


def get_autocast_ctx(config: Config) -> ContextManager[None]:
    """
    Returns autocast context for mixed precision if configured and supported.
    """
    if config.mixed_precision and config.device == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.amp.autocast(device_type=config.device, dtype=torch.bfloat16)
    else:
        return nullcontext()