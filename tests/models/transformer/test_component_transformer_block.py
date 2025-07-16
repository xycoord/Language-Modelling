import torch
import pytest
from src.lm_models.transformer.kv_cache import KVCacheLayer
from src.lm_models.transformer.components import TransformerBlock
from src.lm_models.transformer.config import TransformerConfig

# ================================ Test init ================================

def test_init():
    """Test that TransformerBlock initializes without crashing"""
    config = TransformerConfig(
        embed_dim=128,
        vocab_size=256,
        block_size=64,
        head_size=32,
        num_heads=4,
        hidden_multiplier=4,
        dropout=0.1,
        parallel=False
    )
    block = TransformerBlock(config)
    assert isinstance(block, torch.nn.Module)

# ================================ Test forward ================================

@pytest.fixture
def config():
    return TransformerConfig(
        embed_dim=128,
        vocab_size=256,
        block_size=64,
        head_size=32,
        num_heads=4,
        hidden_multiplier=4,
        dropout=0.1,
        parallel=False
    )



@pytest.fixture
def transformer_block(config):
    return TransformerBlock(config)

@pytest.fixture
def sample_input():
    return torch.randn(2, 10, 128)

def test_forward_shape_preservation(transformer_block, sample_input):
    """Test core contract: preserves input shape"""
    output = transformer_block(sample_input)
    assert output.shape == sample_input.shape

def test_forward_basic_functionality(transformer_block, sample_input):
    """Test basic functionality: doesn't crash, produces finite outputs"""
    output = transformer_block(sample_input)
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()

def test_forward_deterministic_given_weights(transformer_block, sample_input):
    """Test deterministic behavior with fixed weights"""
    transformer_block.eval()
    output1 = transformer_block(sample_input)
    output2 = transformer_block(sample_input)
    assert torch.allclose(output1, output2)


def test_forward_causal_behavior_preserved(transformer_block, sample_input):
    """Test that causal attention behavior is preserved through the block"""
    transformer_block.eval()
    
    baseline_output = transformer_block(sample_input)
    
    modified_input = sample_input.clone()
    modified_input[:, -1, :] += 1000  
    modified_output = transformer_block(modified_input)
    
    assert torch.allclose(
        baseline_output[:, :-1, :], 
        modified_output[:, :-1, :], 
        atol=1e-5  
    ), "Causal behavior should be preserved through TransformerBlock"

def test_forward_different_sequence_lengths(transformer_block):
    """Test with different sequence lengths"""
    for seq_len in [0, 1, 5, 32, 64]:  # Up to block_size
        input_tensor = torch.randn(1, seq_len, 128)
        output = transformer_block(input_tensor)
        assert output.shape == (1, seq_len, 128)

def test_forward_sequence_exceeds_block_size(transformer_block, config):
    """Test error handling for sequences exceeding block_size"""
    input_tensor = torch.randn(1, config.block_size + 1, config.embed_dim)
    with pytest.raises(ValueError, match="Sequence length .* exceeds block size"):
        transformer_block(input_tensor)

def test_forward_gradient_flow(transformer_block, sample_input):
    """Test that gradients flow through the entire block"""
    sample_input.requires_grad_(True)
    output = transformer_block(sample_input)
    loss = output.sum()
    loss.backward()
    
    assert sample_input.grad is not None, "Gradient should not be None"
    assert not torch.allclose(sample_input.grad, torch.zeros_like(sample_input.grad)), "Gradient should not be zero"

def test_forward_training_vs_eval_modes(transformer_block, sample_input):
    """Test dropout behavior in training vs eval"""
    transformer_block.eval()
    torch.manual_seed(42)
    out1 = transformer_block(sample_input)
    torch.manual_seed(123)
    out2 = transformer_block(sample_input)
    assert torch.allclose(out1, out2), "Output should be deterministic in eval mode"
    
    transformer_block.train()
    torch.manual_seed(42)
    train_out1 = transformer_block(sample_input)
    torch.manual_seed(123)
    train_out2 = transformer_block(sample_input)
    assert not torch.allclose(train_out1, train_out2, atol=1e-6), "Output should be different in training mode"

@pytest.mark.parametrize("parallel", [False, True])
def test_forward_parallel_vs_sequential_attention(config, sample_input, parallel):
    """Test that both attention implementations work"""
    config.parallel = parallel
    block = TransformerBlock(config)
    block.eval()
    output = block(sample_input)
    
    B, T, _ = sample_input.shape
    target_shape = (B, T, config.embed_dim)
    assert output.shape == target_shape
    assert torch.isfinite(output).all(), "Output should be finite"

def test_forward_rejects_kv_cache_for_non_parallel_attention(config, sample_input):
    """Test that kv_cache is only supported for ParallelMultiHeadAttention"""
    empty_kv_cache = KVCacheLayer.empty(config, batch_size=2, dtype=torch.float32, device=torch.device('cpu'))
    
    config.parallel = False
    block_sequential = TransformerBlock(config)
    with pytest.raises(ValueError, match="kv_cache is only supported for ParallelMultiHeadAttention"):
        block_sequential(sample_input, kv_cache=empty_kv_cache)
    
    config.parallel = True
    block_parallel = TransformerBlock(config)
    output = block_parallel(sample_input, kv_cache=empty_kv_cache)
    assert output.shape == sample_input.shape