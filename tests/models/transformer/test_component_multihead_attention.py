from src.lm_models.transformer.components import MultiHeadAttention, ParallelMultiHeadAttention
from src.lm_models.transformer.config import TransformerConfig
from src.lm_models.transformer.kv_cache import KVCacheLayer
import pytest
import torch

@pytest.fixture(params=[
    MultiHeadAttention,
    ParallelMultiHeadAttention
])
def multihead_attention_class(request):
    """Parameterized fixture providing all multihead attention classes."""
    return request.param

@pytest.fixture
def config():
    """Parameterized fixture providing all multihead attention classes."""
    return TransformerConfig(
        embed_dim=128,
        vocab_size=256,
        block_size=128,
        head_size=128,
        num_heads=4,
        dropout=0.1,
        flash=False
    )

# ================================ Test init ================================

def test_init(multihead_attention_class, config):
    """Test that the multihead attention class is initialized correctly"""
    multihead_attention = multihead_attention_class(config)
    assert isinstance(multihead_attention, torch.nn.Module)

# ================================ Test forward ================================

@pytest.fixture
def multihead_attention(multihead_attention_class, config):
    """Parameterized fixture providing all multihead attention classes."""
    return multihead_attention_class(config)

@pytest.fixture
def sample_inputs(config):
    """Sample attention head inputs for testing"""
    return torch.randn(1, config.block_size, config.embed_dim)


def test_forward_shape_transformation(multihead_attention, sample_inputs, config):
    """Test that the multihead attention class transforms the input shape correctly"""
    output = multihead_attention(sample_inputs)
    assert output.shape == (1, config.block_size, config.head_size)


def test_forward_deterministic_given_weights(multihead_attention, sample_inputs):
    """Test that the multihead attention class is deterministic given the same weights"""
    multihead_attention.eval()
    output1 = multihead_attention(sample_inputs)
    output2 = multihead_attention(sample_inputs)
    assert torch.allclose(output1, output2)


def test_forward_different_sequence_lengths(multihead_attention, config):
    """Test that the multihead attention class works with different sequence lengths"""
    for seq_len in [0, 1, 5, 32, 64]:
        input = torch.randn(1, seq_len, config.embed_dim)
        output = multihead_attention(input)
        assert output.shape == (1, seq_len, config.head_size)

def test_forward_sequence_exceeds_block_size(multihead_attention, config):
    """Test that sequence length exceeds block size raises error"""
    input = torch.randn(1, config.block_size + 1, 128)
    with pytest.raises(ValueError, match="Sequence length .* exceeds block size"):
        multihead_attention(input)


def test_forward_causal_behavior_preserved(multihead_attention, sample_inputs):
    """Test that causal masking works through the projections"""
    multihead_attention.eval()
    
    baseline_output = multihead_attention(sample_inputs)
    
    # Modify future positions - this should not affect past outputs
    modified_input = sample_inputs.clone()
    modified_input[:, -1, :] += 1000  # Large change to last token
    modified_output = multihead_attention(modified_input)
    
    # Past positions should be unaffected (causal property preserved)
    assert torch.allclose(
        baseline_output[:, :-1, :], 
        modified_output[:, :-1, :], 
        atol=1e-6
    ), "Causal masking should work through projections"


def test_forward_gradient_flow(multihead_attention, sample_inputs):
    """Test that the multihead attention class has gradient flow"""
    multihead_attention.train()
    sample_inputs.requires_grad = True
    output = multihead_attention(sample_inputs)
    loss = output.sum()
    loss.backward()
    assert sample_inputs.grad is not None
    assert not torch.allclose(sample_inputs.grad, torch.zeros_like(sample_inputs.grad))


def test_forward_numerical_stability(multihead_attention, sample_inputs):
    """Test that the multihead attention class is numerically stable"""
    multihead_attention.eval()
    output = multihead_attention(sample_inputs)
    assert torch.isfinite(output).all(), "Output should be finite"
    assert not torch.isnan(output).any(), "Output should not contain NaN"


def test_forward_training_vs_eval_modes(multihead_attention, sample_inputs):
    """Test that the multihead attention class works in training and eval modes"""
    multihead_attention.eval()
    torch.manual_seed(42)
    eval_out1 = multihead_attention(sample_inputs)
    torch.manual_seed(42)
    eval_out2 = multihead_attention(sample_inputs)
    assert torch.allclose(eval_out1, eval_out2), "Eval mode should be deterministic"
    
    multihead_attention.train()
    torch.manual_seed(42)
    train_out1 = multihead_attention(sample_inputs)
    torch.manual_seed(123)
    train_out2 = multihead_attention(sample_inputs)
    assert not torch.allclose(train_out1, train_out2, atol=1e-6), "Training mode should have dropout variation"


# ================================ KV Cache Tests ================================

@pytest.fixture
def parallel_multihead_attention(config):
    """ParallelMultiHeadAttention instance (KV cache only works with this class)"""
    return ParallelMultiHeadAttention(config)

@pytest.fixture  
def empty_kv_cache(config):
    """Empty KVCacheLayer for testing"""
    return KVCacheLayer.empty(config, batch_size=1, dtype=torch.float32, device=torch.device('cpu'))

@pytest.fixture
def populated_kv_cache(config):
    """KVCacheLayer with some existing keys/values"""
    cache = KVCacheLayer.empty(config, batch_size=1, dtype=torch.float32, device=torch.device('cpu'))
    # Add some dummy keys/values to simulate previous context
    dummy_keys = torch.randn(1, config.num_heads, 10, config.head_size)
    dummy_values = torch.randn(1, config.num_heads, 10, config.head_size)
    cache.append(dummy_keys, dummy_values)
    return cache

@pytest.fixture
def short_sample_inputs(config):
    """Short input sequences for incremental generation testing"""
    return torch.randn(1, 5, config.embed_dim)

@pytest.fixture(params=[1, 5, 32, 64])
def variable_length_inputs(request, config):
    """Parameterized inputs with different sequence lengths"""
    seq_len = request.param
    return torch.randn(1, seq_len, config.embed_dim)


def test_forward_accepts_none_kv_cache(parallel_multihead_attention, sample_inputs):
    """Test that forward() accepts None for kv_cache parameter (backward compatibility)"""
    output = parallel_multihead_attention(sample_inputs, kv_cache=None)
    assert output.shape == (1, sample_inputs.shape[1], sample_inputs.shape[2])


def test_forward_accepts_valid_kv_cache(parallel_multihead_attention, sample_inputs, empty_kv_cache):
    """Test that forward() accepts a valid KVCacheLayer instance"""
    output = parallel_multihead_attention(sample_inputs, kv_cache=empty_kv_cache)
    assert output.shape == (1, sample_inputs.shape[1], sample_inputs.shape[2])


def test_kv_cache_state_mutation(parallel_multihead_attention, sample_inputs, empty_kv_cache):
    """Test that providing kv_cache mutates the cache state"""
    initial_cache_len = len(empty_kv_cache)
    parallel_multihead_attention(sample_inputs, kv_cache=empty_kv_cache)
    final_cache_len = len(empty_kv_cache)
    
    assert final_cache_len > initial_cache_len, "Cache length should increase after forward pass"
    assert final_cache_len == initial_cache_len + sample_inputs.shape[1], "Cache should grow by input sequence length"


def test_multiple_forward_passes_accumulate_cache(parallel_multihead_attention, short_sample_inputs, empty_kv_cache):
    """Test that multiple forward passes accumulate state in the cache"""
    # First forward pass
    parallel_multihead_attention(short_sample_inputs, kv_cache=empty_kv_cache)
    first_cache_len = len(empty_kv_cache)
    
    # Second forward pass
    parallel_multihead_attention(short_sample_inputs, kv_cache=empty_kv_cache)
    second_cache_len = len(empty_kv_cache)
    
    assert second_cache_len == first_cache_len + short_sample_inputs.shape[1], "Cache should accumulate across forward passes"


def test_cache_vs_no_cache_output_differs(parallel_multihead_attention, short_sample_inputs, populated_kv_cache):
    """Test that output differs when using cache vs. not using cache for the same input"""
    parallel_multihead_attention.eval()
    
    output_no_cache = parallel_multihead_attention(short_sample_inputs)
    output_with_cache = parallel_multihead_attention(short_sample_inputs, kv_cache=populated_kv_cache)
    
    assert not torch.allclose(output_no_cache, output_with_cache, atol=1e-6), "Output should differ when using populated cache"


def test_causal_masking_with_cache(parallel_multihead_attention, short_sample_inputs, populated_kv_cache):
    """Test that causal masking still works correctly across cached and new tokens"""
    parallel_multihead_attention.eval()
    
    baseline_output = parallel_multihead_attention(short_sample_inputs, kv_cache=populated_kv_cache.clone())
    
    # Modify future positions in the cache (should not affect current output due to causal masking)
    modified_cache = populated_kv_cache.clone()
    modified_cache._keys[:, :, -1, :] += 1000  # Modify last cached position
    modified_output = parallel_multihead_attention(short_sample_inputs, kv_cache=modified_cache)
    
    assert torch.allclose(baseline_output, modified_output, atol=1e-6), "Causal masking should prevent future cache positions from affecting output"


def test_output_shape_with_cache(parallel_multihead_attention, variable_length_inputs, empty_kv_cache, config):
    """Test that output shape is correct regardless of cache usage and sequence length"""
    seq_len = variable_length_inputs.shape[1]
    output = parallel_multihead_attention(variable_length_inputs, kv_cache=empty_kv_cache)
    
    expected_shape = (1, seq_len, config.embed_dim)
    assert output.shape == expected_shape, f"Output shape should be {expected_shape} regardless of cache usage"


def test_incremental_vs_full_sequence_equivalence(parallel_multihead_attention, sample_inputs, empty_kv_cache):
    """Test that incremental generation produces same results as full sequence processing"""
    parallel_multihead_attention.eval()
    seq_len = sample_inputs.shape[1]
    
    full_output = parallel_multihead_attention(sample_inputs)
    
    incremental_outputs = []
    incremental_cache = empty_kv_cache.clone()
    
    for i in range(seq_len):
        single_token = sample_inputs[:, i:i+1, :]  # Single token input
        token_output = parallel_multihead_attention(single_token, kv_cache=incremental_cache)
        incremental_outputs.append(token_output)
    
    incremental_full = torch.cat(incremental_outputs, dim=1)
    
    assert torch.allclose(full_output, incremental_full, atol=1e-5), "Incremental generation should match full sequence processing"


@pytest.mark.parametrize("seq_len", [0, 1])
def test_edge_case_sequence_lengths(parallel_multihead_attention, config, empty_kv_cache, seq_len):
    """Test behaviour with edge case sequence lengths"""
    if seq_len == 0:
        edge_input = torch.randn(1, 0, config.embed_dim)
    else:
        edge_input = torch.randn(1, seq_len, config.embed_dim)
    
    output = parallel_multihead_attention(edge_input, kv_cache=empty_kv_cache)
    assert output.shape == (1, seq_len, config.embed_dim), f"Should handle sequence length {seq_len}"


def test_cache_at_capacity_boundary(parallel_multihead_attention, config, empty_kv_cache):
    """Test behaviour when cache approaches capacity"""
    # Fill cache to near capacity
    remaining_capacity = config.block_size - len(empty_kv_cache)
    if remaining_capacity > 0:
        near_full_input = torch.randn(1, remaining_capacity, config.embed_dim)
        output = parallel_multihead_attention(near_full_input, kv_cache=empty_kv_cache)
        assert output.shape == (1, remaining_capacity, config.embed_dim)
        assert len(empty_kv_cache) == config.block_size, "Cache should be at full capacity"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes_with_cache(parallel_multihead_attention, config, batch_size):
    """Test that cache works correctly with different batch sizes"""
    batch_cache = KVCacheLayer.empty(config, batch_size=batch_size, dtype=torch.float32, device=torch.device('cpu'))
    batch_input = torch.randn(batch_size, 10, config.embed_dim)
    
    output = parallel_multihead_attention(batch_input, kv_cache=batch_cache)
    assert output.shape == (batch_size, 10, config.embed_dim), f"Should handle batch size {batch_size}"