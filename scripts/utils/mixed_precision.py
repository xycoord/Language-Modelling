import torch
from contextlib import nullcontext, ContextManager

def get_autocast_ctx(device: str) -> ContextManager[None]:
    """
    Returns autocast context for bfloat16 mixed precision if supported, otherwise nullcontext.
    Prints the precision type being used.
    """
    if device == 'cuda' and torch.cuda.is_bf16_supported():
        mixed_precision_type = torch.bfloat16
        autocast_ctx = torch.amp.autocast(device_type=device, dtype=mixed_precision_type)
        print(f'Using mixed precision type: {mixed_precision_type}')
    else:
        autocast_ctx = nullcontext()
        print(f'Using full precision: float32')
    return autocast_ctx