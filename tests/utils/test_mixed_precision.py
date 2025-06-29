from src.script_utils.mixed_precision import setup_precision, get_autocast_ctx

import pytest
from unittest.mock import patch, MagicMock
from contextlib import nullcontext
import torch


@pytest.fixture
def config_factory():
    """Factory fixture to create config objects with different settings."""
    def _create_config(device, mixed_precision):
        config = MagicMock()
        config.device = device
        config.mixed_precision = mixed_precision
        return config
    return _create_config


# Comprehensive parametrised test for setup_precision
@pytest.mark.parametrize("device,mixed_precision,bf16_supported,expected_precision,expected_output,bf16_call_expected", [
    # CUDA with mixed precision cases
    ('cuda', True, True, 'high', 'Using mixed precision: bfloat16 with TF32 matmul', True),
    ('cuda', True, False, 'high', 'Using precision: float32 with TF32 matmul', True),
    # CUDA without mixed precision
    ('cuda', False, None, 'high', 'Using precision: float32 with TF32 matmul', False),
    # CPU cases (mixed_precision setting ignored)
    ('cpu', True, None, 'highest', 'Using precision: float32 (CPU)', False),
    ('cpu', False, None, 'highest', 'Using precision: float32 (CPU)', False),
])
@patch('torch.set_float32_matmul_precision')
@patch('torch.cuda.is_bf16_supported')
def test_setup_precision_all_scenarios(
    mock_bf16_supported, mock_set_precision, capsys, config_factory,
    device, mixed_precision, bf16_supported, expected_precision, expected_output, bf16_call_expected
):
    """Comprehensive test covering all setup_precision scenarios."""
    if bf16_supported is not None:
        mock_bf16_supported.return_value = bf16_supported
    
    config = config_factory(device, mixed_precision)
    setup_precision(config)
    
    mock_set_precision.assert_called_once_with(expected_precision)
    
    if bf16_call_expected:
        mock_bf16_supported.assert_called_once()
    else:
        mock_bf16_supported.assert_not_called()
    
    captured = capsys.readouterr()
    assert expected_output in captured.out


# get_autocast_ctx tests - comprehensive parametrised coverage
@pytest.mark.parametrize("device,mixed_precision,bf16_supported,should_autocast", [
    ('cuda', True, True, True),
    ('cuda', True, False, False),
    ('cuda', False, True, False),
    ('cuda', False, False, False),
    ('cpu', True, True, False),
    ('cpu', False, False, False),
])
@patch('torch.amp.autocast')
@patch('torch.cuda.is_bf16_supported')
def test_get_autocast_ctx_parametrised(
    mock_bf16_supported, mock_autocast, config_factory,
    device, mixed_precision, bf16_supported, should_autocast
):
    """Parametrised test covering all get_autocast_ctx scenarios."""
    mock_bf16_supported.return_value = bf16_supported
    mock_autocast_instance = MagicMock()
    mock_autocast.return_value = mock_autocast_instance
    
    config = config_factory(device, mixed_precision)
    
    result = get_autocast_ctx(config)
    
    if should_autocast:
        mock_autocast.assert_called_once_with(
            device_type='cuda', 
            dtype=torch.bfloat16
        )
        assert result == mock_autocast_instance
    else:
        if device == 'cuda' and mixed_precision:
            mock_bf16_supported.assert_called_once()
        mock_autocast.assert_not_called()
        assert isinstance(result, nullcontext)