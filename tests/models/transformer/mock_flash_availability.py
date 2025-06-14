# Test __init__

import torch.nn.functional as F
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


@contextmanager
def mock_flash_availability(available):
    if available:
        mock_fn = MagicMock()
        with patch.object(F, 'scaled_dot_product_attention', mock_fn):
            yield
    else:
        # Temporarily remove the attribute
        original_attr = getattr(F, 'scaled_dot_product_attention', 'MISSING')
        if hasattr(F, 'scaled_dot_product_attention'):
            delattr(F, 'scaled_dot_product_attention')
        try:
            yield
        finally:
            if original_attr != 'MISSING':
                setattr(F, 'scaled_dot_product_attention', original_attr)