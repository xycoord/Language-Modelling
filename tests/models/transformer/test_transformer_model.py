import pytest
import torch

from src.lm_models.transformer.kv_cache import KVCacheLayer
from src.lm_models.transformer.model import TransformerLanguageModel
from src.lm_models.transformer.config import TransformerConfig

@pytest.fixture(
    params=[
        pytest.param("cpu", id="cpu_device"),
        pytest.param("cuda", id="cuda_device", 
                    marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                           reason="CUDA not available"))
    ]
)
def device(request):
    """Fixture providing available devices for testing"""
    return torch.device(request.param)

@pytest.fixture
def config():
    return TransformerConfig(
        vocab_size=1000,
        embed_dim=128,
        block_size=64,
        head_size=32,
        num_heads=4,
        n_layers=2,
        hidden_multiplier=4,
        dropout=0.0,
        parallel=True
    )

# KV Cache fixtures

@pytest.fixture
def empty_kv_cache(model, config):
    """Empty KV cache for the model"""
    batch_size = 2  # Match sample_tokens batch size
    return tuple([
        KVCacheLayer.empty(config, batch_size, model.dtype, model.device) 
        for _ in range(config.n_layers)
    ])

@pytest.fixture
def pre_populated_kv_cache(model, config, sample_tokens):
    """KV cache with some existing content"""
    batch_size = sample_tokens.shape[0]
    kv_cache = tuple([
        KVCacheLayer.empty(config, batch_size, model.dtype, model.device) 
        for _ in range(config.n_layers)
    ])
    # Populate with first few tokens
    _ = model(sample_tokens[:, :5], kv_cache)
    return kv_cache

@pytest.fixture(params=[
    "empty_cache",
    "pre_populated_cache"
])
def kv_cache_state(request, empty_kv_cache, pre_populated_kv_cache):
    """Parameterized fixture for different cache states"""
    if request.param == "empty_cache":
        return empty_kv_cache
    else:
        return pre_populated_kv_cache

@pytest.fixture
def different_batch_tokens(config):
    """Tokens with different batch size for mismatch testing"""
    return torch.randint(0, config.vocab_size, (3, 8))


# Test init
def test_init(config):
    model = TransformerLanguageModel(config)
    assert isinstance(model, TransformerLanguageModel)

@pytest.fixture
def model(config):
    return TransformerLanguageModel(config)

def test_weight_tying_identity(model):
    """Test that embedding and output weights are the same tensor object."""
    embedding_weights = model.token_embedding_table.weight
    output_weights = model.to_logits.weight
    
    assert embedding_weights is output_weights, (
        "Embedding and output layer weights should be the same tensor object"
    )


# Test forward

@pytest.fixture
def sample_tokens(config):
    """Sample token sequence (integers)"""
    return torch.randint(0, config.vocab_size, (2, 10))  # (batch=2, seq_len=10)

def test_forward_shape_transformation(model, sample_tokens, config):
    """Test core contract: (B, T) -> (B, T, vocab_size)"""
    logits = model(sample_tokens)
    
    batch_size, seq_len = sample_tokens.shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape

def test_forward_basic_functionality(model, sample_tokens):
    """Test basic functionality: doesn't crash, produces finite outputs"""
    logits = model(sample_tokens)
    
    assert torch.isfinite(logits).all(), "Logits should be finite"
    assert not torch.isnan(logits).any(), "Logits should not contain NaN"

def test_forward_deterministic_given_weights(model, sample_tokens):
    """Test deterministic behavior with fixed weights"""
    model.eval()
    logits1 = model(sample_tokens)
    logits2 = model(sample_tokens)
    assert torch.allclose(logits1, logits2), "Should be deterministic for fixed weights"

def test_forward_different_sequence_lengths(model, config):
    """Test with different sequence lengths"""
    for seq_len in [0, 1, 5, 32, config.block_size]:
        tokens = torch.randint(0, config.vocab_size, (1, seq_len))
        logits = model(tokens)
        assert logits.shape == (1, seq_len, config.vocab_size)

def test_forward_sequence_exceeds_block_size(model, config):
    """Test error handling when sequence exceeds block_size"""
    # This might not error in forward() but in the underlying blocks
    long_tokens = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
    
    with pytest.raises(ValueError, match="Sequence length .* exceeds block size"):
        model(long_tokens)

def test_forward_device_consistency(model, sample_tokens, device):
    """Test that position embeddings work correctly on different devices"""
    model_on_device = model.to(device)
    sample_tokens = sample_tokens.to(device)
    logits = model_on_device(sample_tokens)
    
    assert logits.device.type == device.type, f"Logits should be on {device}"
    assert torch.isfinite(logits).all(), f"Should work on {device}"

def test_forward_causal_behavior_preserved(model, sample_tokens):
    """Test that causal behavior is preserved through the full model"""
    model.eval()
    
    baseline_logits = model(sample_tokens)
    
    # Modify future tokens (last position)
    modified_tokens = sample_tokens.clone()
    modified_tokens[:, -1] = (modified_tokens[:, -1] + 100) % 1000  # Different token
    
    modified_logits = model(modified_tokens)
    
    # Past predictions should be unchanged (causal property)
    seq_len = sample_tokens.shape[1]
    if seq_len > 1:
        assert torch.allclose(
            baseline_logits[:, :-1, :], 
            modified_logits[:, :-1, :], 
            atol=1e-6
        ), "Causal behavior should be preserved through full model"

def test_forward_gradient_flow(model, sample_tokens):
    """Test that gradients flow through the entire model"""
    # Make sample_tokens require gradients isn't needed since they're integers
    # But we can test that model parameters get gradients
    logits = model(sample_tokens)
    loss = logits.sum()
    loss.backward()
    
    params_with_grads = [p for p in model.parameters() if p.grad is not None]
    total_params = list(model.parameters())
    assert len(params_with_grads) > 0, "Model should have trainable parameters with gradients"
    assert len(params_with_grads) == len(total_params), "All parameters should have gradients"

def test_forward_training_vs_eval_modes(config):
    """Test dropout behavior in training vs eval"""
    config_with_dropout = config
    config_with_dropout.dropout = 0.3
    model = TransformerLanguageModel(config_with_dropout)
    tokens = torch.randint(0, config_with_dropout.vocab_size, (1, 5))
    
    model.eval()
    torch.manual_seed(42)
    logits1 = model(tokens)
    torch.manual_seed(123)
    logits2 = model(tokens)
    assert torch.allclose(logits1, logits2), "Eval mode should be deterministic"
    
    model.train()
    torch.manual_seed(42)
    train_logits1 = model(tokens)
    torch.manual_seed(123)
    train_logits2 = model(tokens)
    assert not torch.allclose(train_logits1, train_logits2, atol=1e-6), \
        "Training mode should have dropout variation"

def test_forward_token_range(model, config):
    """Test handling of different token values"""
    # Valid tokens (within vocab_size)
    valid_tokens = torch.randint(0, config.vocab_size, (1, 5))
    logits = model(valid_tokens)
    assert torch.isfinite(logits).all()
    
    # Edge case: all tokens are 0
    zero_tokens = torch.zeros((1, 5), dtype=torch.long)
    logits = model(zero_tokens)
    assert torch.isfinite(logits).all()
    
    # Edge case: all tokens are max value
    max_tokens = torch.full((1, 5), config.vocab_size - 1, dtype=torch.long)
    logits = model(max_tokens)
    assert torch.isfinite(logits).all()

def test_forward_invalid_token_range(model, config):
    """Test handling of different token values"""
    # Valid tokens (within vocab_size)
    invalid_tokens = torch.randint(config.vocab_size, config.vocab_size + 1, (1, 5))
    with pytest.raises(ValueError, match="Token values must be between 0 and vocab_size"):
        model(invalid_tokens)

def test_forward_position_embedding_behavior(model, config):
    """Test that position embeddings create position-dependent outputs"""
    # Use same token repeated at different positions
    seq1 = torch.tensor([[42, 42, 42]])
    logits1 = model(seq1)
    
    assert not torch.allclose(logits1[0, 0, :], logits1[0, 1, :], atol=1e-3), \
        "Same token at positions 0 and 1 should differ due to position embeddings"
    assert not torch.allclose(logits1[0, 1, :], logits1[0, 2, :], atol=1e-3), \
        "Same token at positions 1 and 2 should differ due to position embeddings"

def test_forward_with_kv_cache_basic(model, sample_tokens, empty_kv_cache, device):
    """Test forward pass accepts and works with KV cache"""
    model = model.to(device)
    sample_tokens = sample_tokens.to(device)
    empty_kv_cache = tuple(layer.to(device) for layer in empty_kv_cache)
    
    logits = model(sample_tokens, empty_kv_cache)
    
    batch_size, seq_len = sample_tokens.shape
    expected_shape = (batch_size, seq_len, model.config.vocab_size)
    assert logits.shape == expected_shape, "Forward with KV cache should maintain shape contract"
    assert torch.isfinite(logits).all(), "Logits should be finite with KV cache"

def test_forward_position_embedding_with_kv_cache(model, config, pre_populated_kv_cache):
    """Test position embeddings start from cache length when using KV cache"""
    new_tokens = torch.randint(0, config.vocab_size, (2, 3))
    
    logits = model(new_tokens, pre_populated_kv_cache)
    
    assert logits.shape == (2, 3, config.vocab_size), "Should process new tokens correctly"
    assert torch.isfinite(logits).all(), "Should produce finite logits with pre-populated cache"



# Test generate

def test_generate_shape_transformation(model, sample_tokens, empty_kv_cache):
    """Test core contract: (B, T) -> (B, max_new_tokens)"""
    max_new_tokens = 10
    generated_tokens = model.generate(sample_tokens, max_new_tokens, empty_kv_cache)
    assert generated_tokens.shape == (2, max_new_tokens)

def test_generate_invalid_max_new_tokens(model, sample_tokens, empty_kv_cache):
    """Test error handling when max_new_tokens exceeds block_size"""
    max_new_tokens = model.config.block_size - sample_tokens.shape[1] + 1
    with pytest.raises(ValueError, match="Cannot generate more tokens than the block size"):
        model.generate(sample_tokens, max_new_tokens, empty_kv_cache)

def test_generate_token_range_validation(model, sample_tokens, empty_kv_cache):
    """Test that generated tokens are in valid vocabulary range"""
    max_new_tokens = model.config.block_size - sample_tokens.shape[1]
    generated_tokens = model.generate(sample_tokens, max_new_tokens, empty_kv_cache)
    
    assert generated_tokens.min() >= 0, "Generated tokens should be >= 0"
    assert generated_tokens.max() < model.config.vocab_size, \
        f"Generated tokens should be < vocab_size ({model.config.vocab_size})"

def test_generate_deterministic_with_seed(model, sample_tokens, empty_kv_cache):
    """Test that generation is deterministic with same seed"""
    max_new_tokens = 5
    
    torch.manual_seed(42)
    generated1 = model.generate(sample_tokens, max_new_tokens, empty_kv_cache)
    torch.manual_seed(42)
    generated2 = model.generate(sample_tokens, max_new_tokens, empty_kv_cache)
    
    assert torch.equal(generated1, generated2), \
        "Same seed should produce identical generation"

def test_generate_different_seeds_produce_different_results(model, sample_tokens, empty_kv_cache):
    """Test that different seeds produce different results (probabilistic)"""
    max_new_tokens = 10
    
    torch.manual_seed(42)
    generated1 = model.generate(sample_tokens, max_new_tokens, empty_kv_cache)
    torch.manual_seed(123)
    generated2 = model.generate(sample_tokens, max_new_tokens, empty_kv_cache)
    
    assert not torch.equal(generated1, generated2), \
        "Different seeds should produce different generation"

def test_generate_zero_new_tokens(model, sample_tokens, empty_kv_cache):
    """Test edge case of generating zero new tokens"""
    generated_tokens = model.generate(sample_tokens, max_new_tokens=0, kv_cache=empty_kv_cache)
    assert generated_tokens.shape == (2, 0), \
        "Zero new tokens should return empty tensor"

def test_generate_device_consistency(model, config, device, empty_kv_cache):
    """Test that generated tokens are on same device as input"""
    model = model.to(device)
    empty_kv_cache = tuple([cache_layer.to(device) for cache_layer in empty_kv_cache])
    context = torch.randint(0, config.vocab_size, (2, 5)).to(device)
    
    generated_tokens = model.generate(context, max_new_tokens=3, kv_cache=empty_kv_cache)
    
    assert generated_tokens.device.type == device.type, \
        f"Generated tokens should be on {device}"


def test_generate_no_gradients(model, sample_tokens, empty_kv_cache):
    """Test that generation doesn't compute gradients"""
    for param in model.parameters():
        param.requires_grad_(True)
    
    model.generate(sample_tokens, max_new_tokens=3, kv_cache=empty_kv_cache)
    
    for param in model.parameters():
        assert param.grad is None, \
            "Generation should not accumulate gradients (due to @torch.no_grad())"

def test_generate_with_provided_kv_cache(model, sample_tokens, empty_kv_cache, device):
    """Test generate accepts and works with provided KV cache"""
    model = model.to(device)
    sample_tokens = sample_tokens.to(device)
    empty_kv_cache = tuple(layer.to(device) for layer in empty_kv_cache)
    
    max_new_tokens = 5
    generated_tokens = model.generate(sample_tokens, max_new_tokens, empty_kv_cache)
    
    assert generated_tokens.shape == (2, max_new_tokens), "Should generate correct number of tokens"
    assert generated_tokens.min() >= 0, "Generated tokens should be >= 0"
    assert generated_tokens.max() < model.config.vocab_size, "Generated tokens should be < vocab_size"

def test_generate_with_various_cache_states(model, sample_tokens, kv_cache_state):
    """Test generation works with both empty and pre-populated caches"""
    max_new_tokens = 3
    generated_tokens = model.generate(sample_tokens, max_new_tokens, kv_cache_state)
    
    assert generated_tokens.shape == (2, max_new_tokens), "Should work with any cache state"
    assert generated_tokens.min() >= 0, "Should produce valid tokens"
    assert generated_tokens.max() < model.config.vocab_size, "Should produce valid tokens"

def test_generate_external_kv_cache_persistence(model, sample_tokens, empty_kv_cache):
    """Test that external KV cache persists state across generate calls"""
    initial_cache_length = len(empty_kv_cache[0])
    
    # First generation
    generated1 = model.generate(sample_tokens, 3, empty_kv_cache)
    cache_length_after_first = len(empty_kv_cache[0])
    
    # Second generation reusing same cache
    generated2 = model.generate(sample_tokens, 2, empty_kv_cache)
    cache_length_after_second = len(empty_kv_cache[0])
    
    assert cache_length_after_first > initial_cache_length, "Cache should be populated after first generation"
    assert cache_length_after_second > cache_length_after_first, "Cache should accumulate across calls"
    assert generated1.shape == (2, 3), "First generation should produce correct shape"
    assert generated2.shape == (2, 2), "Second generation should produce correct shape"

def test_multi_step_generation_with_shared_cache(model, sample_tokens, empty_kv_cache):
    """Test multiple generation steps sharing the same cache"""
    context1 = sample_tokens[:, :5]
    context2 = sample_tokens[:, 5:]
    
    # Generate from first context
    gen1 = model.generate(context1, 2, empty_kv_cache)
    
    # Generate from second context with same cache
    gen2 = model.generate(context2, 2, empty_kv_cache)
    
    assert gen1.shape == (2, 2), "First generation should work correctly"
    assert gen2.shape == (2, 2), "Second generation should work correctly"
    assert len(empty_kv_cache[0]) > 0, "Cache should contain accumulated state"

# Test error handling

def test_generate_kv_cache_device_mismatch(model, sample_tokens, empty_kv_cache):
    """Test error handling when cache is on wrong device"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for device mismatch test")
    
    model = model.to('cpu')
    sample_tokens = sample_tokens.to('cpu')
    empty_kv_cache = tuple(layer.to('cuda') for layer in empty_kv_cache)
    
    with pytest.raises(ValueError, match="Device mismatch"):
        model.generate(sample_tokens, 2, empty_kv_cache)

def test_generate_kv_cache_batch_size_mismatch(model, different_batch_tokens, empty_kv_cache):
    """Test error when cache batch size doesn't match context"""
    # empty_kv_cache has batch_size=2, different_batch_tokens has batch_size=3
    with pytest.raises(ValueError, match="New keys and values must match the dimensions of the cache"):
        model.generate(different_batch_tokens, 2, empty_kv_cache)

def test_kv_cache_capacity_limits(model, config, empty_kv_cache):
    """Test behavior when approaching block size limits with cache"""
    # Use a context that, when combined with max_new_tokens, exceeds block_size
    max_context_len = config.block_size - 2  # Leave room for only 2 new tokens
    long_context = torch.randint(0, config.vocab_size, (2, max_context_len))
    
    # This should work (within block_size)
    generated = model.generate(long_context, 2, empty_kv_cache)
    assert generated.shape == (2, 2), "Should generate within block_size limits"
    
    # This should fail (exceeds block_size)
    with pytest.raises(ValueError, match="Cannot generate more tokens than the block size"):
        model.generate(long_context, 3, empty_kv_cache)

# Test device
def test_device(model, device):
    model = model.to(device)
    assert model.device.type == device.type
