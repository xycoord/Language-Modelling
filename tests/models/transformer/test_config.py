import pytest

from src.models.transformer.config import TransformerConfig


@pytest.mark.parametrize("param_name", [
    "vocab_size", "block_size", "embed_dim", "num_heads", "hidden_multiplier", "n_layers"
])
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_config_rejects_non_positive_parameters(param_name, invalid_value):
    """TransformerConfig should reject non-positive values for most parameters"""
    base_kwargs = {"vocab_size": 10, "block_size": 10}
    base_kwargs[param_name] = invalid_value
    
    with pytest.raises(ValueError, match=f"{param_name} must be positive"):
        TransformerConfig(**base_kwargs)


@pytest.mark.parametrize("dropout_value", [-1, -0.1, 1.1, 2])
def test_config_rejects_invalid_dropout(dropout_value):
    """TransformerConfig should reject dropout values outside [0, 1]"""
    with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
        TransformerConfig(
            vocab_size=10,
            block_size=10,
            dropout=dropout_value
        )
        

def test_config_head_size_is_calculated_correctly():
    """TransformerConfig should calculate head_size correctly"""
    config = TransformerConfig(vocab_size=10, block_size=10, embed_dim=32, num_heads=4)
    assert config.head_size == 32 // 4

    config = TransformerConfig(vocab_size=10, block_size=10, embed_dim=32, num_heads=8, head_size=8)
    assert config.head_size == 8
