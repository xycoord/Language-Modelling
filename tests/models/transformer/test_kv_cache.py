import pytest
import torch

from src.lm_models.transformer.kv_cache import KVCacheLayer
from src.lm_models.transformer.config import TransformerConfig


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def basic_config():
    return TransformerConfig(
        vocab_size=1000,
        block_size=8,
        embed_dim=64,      # head_size=16 (auto-calculated)
        num_heads=4,
        n_layers=2
    )

@pytest.fixture  
def small_config():
    return TransformerConfig(
        vocab_size=100,
        block_size=4,
        embed_dim=32,      # head_size=8
        num_heads=4,
        n_layers=1
    )

@pytest.fixture
def explicit_head_size_config():
    return TransformerConfig(
        vocab_size=1000,
        block_size=8,
        embed_dim=64,
        num_heads=4,
        head_size=20,      # Explicitly set, overrides auto-calc
        n_layers=2
    )

@pytest.fixture
def single_head_config():
    return TransformerConfig(
        vocab_size=1000,
        block_size=6,
        embed_dim=16,      # head_size=16
        num_heads=1,
        n_layers=2
    )

@pytest.fixture
def empty_cache(basic_config):
    return KVCacheLayer.empty(basic_config, batch_size=2, dtype=torch.float32, device=torch.device('cpu'))

@pytest.fixture
def partial_cache(empty_cache, basic_config):
    seq_len = 3
    new_keys = torch.randn(2, basic_config.num_heads, seq_len, basic_config.head_size)
    new_values = torch.randn(2, basic_config.num_heads, seq_len, basic_config.head_size)
    empty_cache.append(new_keys, new_values)
    return empty_cache

@pytest.fixture
def nearly_full_cache(empty_cache, basic_config):
    seq_len = basic_config.block_size - 1  # 7 tokens
    new_keys = torch.randn(2, basic_config.num_heads, seq_len, basic_config.head_size)
    new_values = torch.randn(2, basic_config.num_heads, seq_len, basic_config.head_size)
    empty_cache.append(new_keys, new_values)
    return empty_cache

@pytest.fixture
def full_cache(empty_cache, basic_config):
    seq_len = basic_config.block_size  # 8 tokens
    new_keys = torch.randn(2, basic_config.num_heads, seq_len, basic_config.head_size)
    new_values = torch.randn(2, basic_config.num_heads, seq_len, basic_config.head_size)
    empty_cache.append(new_keys, new_values)
    return empty_cache

@pytest.fixture
def sample_single_token(basic_config):
    return (
        torch.randn(2, basic_config.num_heads, 1, basic_config.head_size),
        torch.randn(2, basic_config.num_heads, 1, basic_config.head_size)
    )

@pytest.fixture  
def sample_multi_token(basic_config):
    return (
        torch.randn(2, basic_config.num_heads, 3, basic_config.head_size),
        torch.randn(2, basic_config.num_heads, 3, basic_config.head_size)
    )

@pytest.fixture
def wrong_device_tensors(basic_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return (
        torch.randn(2, basic_config.num_heads, 2, basic_config.head_size, device='cuda'),
        torch.randn(2, basic_config.num_heads, 2, basic_config.head_size, device='cuda')
    )

@pytest.fixture
def wrong_dtype_tensors(basic_config):
    return (
        torch.randn(2, basic_config.num_heads, 2, basic_config.head_size, dtype=torch.float16),
        torch.randn(2, basic_config.num_heads, 2, basic_config.head_size, dtype=torch.float16)
    )


# ============================================================================
# CONSTRUCTOR TESTS
# ============================================================================

@pytest.mark.parametrize("current_len", [0, 3, 8])
def test_constructor_valid_inputs(basic_config, current_len):
    keys = torch.randn(2, basic_config.num_heads, basic_config.block_size, basic_config.head_size)
    values = torch.randn(2, basic_config.num_heads, basic_config.block_size, basic_config.head_size)
    
    cache = KVCacheLayer(keys, values, current_len)
    
    assert len(cache) == current_len, f"Expected current length {current_len}, got {len(cache)}"
    assert cache.max_len == basic_config.block_size, f"Expected max_len {basic_config.block_size}, got {cache.max_len}"
    assert cache.keys.shape == (2, basic_config.num_heads, current_len, basic_config.head_size), "Keys shape incorrect"
    assert cache.values.shape == (2, basic_config.num_heads, current_len, basic_config.head_size), "Values shape incorrect"

@pytest.mark.parametrize("keys_shape,values_shape,current_len,expected_error", [
    ((2, 4, 8), (2, 4, 8, 16), 0, "4 dimensions"),           # 3D keys
    ((2, 4, 8, 16), (2, 4, 8), 0, "4 dimensions"),           # 3D values  
    ((2, 4, 8, 16), (2, 5, 8, 16), 0, "same shape"),         # mismatched shapes
    ((2, 4, 8, 16), (2, 4, 8, 16), 10, "block size"),        # current_len too big
    ((2, 4, 8, 16), (2, 4, 8, 16), -1, "Current length"),    # negative current_len
])
def test_constructor_invalid_inputs(keys_shape, values_shape, current_len, expected_error):
    keys = torch.randn(*keys_shape)
    values = torch.randn(*values_shape)
    
    with pytest.raises(ValueError, match=expected_error):
        KVCacheLayer(keys, values, current_len)


# ============================================================================
# FACTORY METHOD TESTS
# ============================================================================

@pytest.mark.parametrize("config_fixture", [
    "basic_config", "small_config", "explicit_head_size_config", "single_head_config"
])
def test_empty_with_various_configs(config_fixture, request):
    config = request.getfixturevalue(config_fixture)
    cache = KVCacheLayer.empty(config, batch_size=2, dtype=torch.float32, device=torch.device('cpu'))
    
    assert len(cache) == 0, "Empty cache should have length 0"
    assert cache.max_len == config.block_size, f"Expected max_len {config.block_size}, got {cache.max_len}"
    assert cache.keys.shape == (2, config.num_heads, 0, config.head_size), "Keys shape incorrect for empty cache"
    assert cache.values.shape == (2, config.num_heads, 0, config.head_size), "Values shape incorrect for empty cache"
    assert cache.dtype == torch.float32, "Dtype not preserved"
    assert cache.device == torch.device('cpu'), "Device not preserved"

@pytest.mark.parametrize("dtype,device", [
    (torch.float32, torch.device('cpu')),
    (torch.float16, torch.device('cpu')),
    pytest.param(torch.float32, torch.device('cuda'), 
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
])
def test_empty_dtype_device_combinations(basic_config, dtype, device):
    cache = KVCacheLayer.empty(basic_config, batch_size=1, dtype=dtype, device=device)
    
    assert cache.dtype == dtype, f"Expected dtype {dtype}, got {cache.dtype}"
    assert cache.device == device, f"Expected device {device}, got {cache.device}"

def test_empty_missing_head_size():
    # Create config that bypasses validation to set head_size=None
    config = TransformerConfig.__new__(TransformerConfig)
    config.vocab_size = 1000
    config.block_size = 8
    config.embed_dim = 64
    config.num_heads = 4
    config.head_size = None
    config.n_layers = 2
    
    with pytest.raises(ValueError, match="head_size must be set"):
        KVCacheLayer.empty(config, batch_size=2, dtype=torch.float32, device=torch.device('cpu'))


# ============================================================================
# LENGTH TESTS
# ============================================================================

@pytest.mark.parametrize("cache_fixture,expected_len", [
    ("empty_cache", 0),
    ("partial_cache", 3), 
    ("full_cache", 8)
])
def test_len(cache_fixture, expected_len, request):
    cache = request.getfixturevalue(cache_fixture)
    assert len(cache) == expected_len, f"Expected length {expected_len}, got {len(cache)}"


# ============================================================================
# KEYS/VALUES PROPERTY TESTS
# ============================================================================

@pytest.mark.parametrize("cache_fixture,expected_seq_len", [
    ("empty_cache", 0),
    ("partial_cache", 3),
    ("full_cache", 8)
])
def test_keys_values_properties(cache_fixture, expected_seq_len, basic_config, request):
    cache = request.getfixturevalue(cache_fixture)
    
    expected_shape = (2, basic_config.num_heads, expected_seq_len, basic_config.head_size)
    assert cache.keys.shape == expected_shape, f"Keys shape expected {expected_shape}, got {cache.keys.shape}"
    assert cache.values.shape == expected_shape, f"Values shape expected {expected_shape}, got {cache.values.shape}"
    assert cache.keys.dtype == torch.float32, "Keys dtype incorrect"
    assert cache.values.dtype == torch.float32, "Values dtype incorrect"

def test_keys_values_are_views(partial_cache):
    original_keys = partial_cache.keys.clone()
    original_values = partial_cache.values.clone()
    
    # Modify underlying tensors directly
    partial_cache._keys[:, :, :partial_cache._current_len, :] *= 2
    partial_cache._values[:, :, :partial_cache._current_len, :] *= 3
    
    # Properties should reflect changes (proving they're views)
    assert not torch.equal(partial_cache.keys, original_keys), "Keys property should reflect underlying tensor changes"
    assert not torch.equal(partial_cache.values, original_values), "Values property should reflect underlying tensor changes"


# ============================================================================
# CLONE TESTS
# ============================================================================

@pytest.mark.parametrize("cache_fixture", [
    "empty_cache", "partial_cache", "full_cache"
])
def test_clone(cache_fixture, request):
    original = request.getfixturevalue(cache_fixture)
    clone = original.clone()
    
    # Verify identical state
    assert len(clone) == len(original), "Clone should have same length"
    assert clone.max_len == original.max_len, "Clone should have same max_len"
    assert clone.dtype == original.dtype, "Clone should have same dtype"
    assert torch.equal(clone.keys, original.keys), "Clone keys should be identical"
    assert torch.equal(clone.values, original.values), "Clone values should be identical"
    
    # Verify independence
    if len(original) > 0:
        original._keys[:, :, :original._current_len, :] *= 2
        assert not torch.equal(clone.keys, original.keys), "Clone should be independent of original"


# ============================================================================
# GET TENSOR COPIES TESTS
# ============================================================================

def test_get_tensor_copies_default_length(partial_cache):
    keys_copy, values_copy = partial_cache.get_tensor_copies()
    
    assert torch.equal(keys_copy, partial_cache.keys), "Default length should return current keys"
    assert torch.equal(values_copy, partial_cache.values), "Default length should return current values"
    
    # Verify independence
    partial_cache._keys[:, :, :partial_cache._current_len, :] *= 2
    assert not torch.equal(keys_copy, partial_cache.keys), "Copies should be independent"

@pytest.mark.parametrize("length", [0, 1, 3])
def test_get_tensor_copies_specific_length(partial_cache, basic_config, length):
    keys_copy, values_copy = partial_cache.get_tensor_copies(length)
    
    expected_shape = (2, basic_config.num_heads, length, basic_config.head_size)
    assert keys_copy.shape == expected_shape, f"Keys copy shape should be {expected_shape}"
    assert values_copy.shape == expected_shape, f"Values copy shape should be {expected_shape}"
    
    if length > 0:
        expected_keys = partial_cache._keys[:, :, :length, :]
        expected_values = partial_cache._values[:, :, :length, :]
        assert torch.equal(keys_copy, expected_keys), "Keys copy content incorrect"
        assert torch.equal(values_copy, expected_values), "Values copy content incorrect"

def test_get_tensor_copies_independence(partial_cache):
    keys_copy, values_copy = partial_cache.get_tensor_copies()
    
    # Modify copies
    keys_copy *= 2
    values_copy *= 3
    
    # Original should be unchanged
    original_keys = partial_cache._keys[:, :, :partial_cache._current_len, :]
    original_values = partial_cache._values[:, :, :partial_cache._current_len, :]
    assert not torch.equal(keys_copy, original_keys), "Original keys should be unchanged"
    assert not torch.equal(values_copy, original_values), "Original values should be unchanged"


# ============================================================================
# APPEND TESTS
# ============================================================================

def test_append_to_empty_cache(empty_cache, sample_multi_token, basic_config):
    new_keys, new_values = sample_multi_token
    
    empty_cache.append(new_keys, new_values)
    
    assert len(empty_cache) == 3, "Cache length should be 3 after appending 3 tokens"
    assert torch.equal(empty_cache.keys, new_keys), "Keys should match appended data"
    assert torch.equal(empty_cache.values, new_values), "Values should match appended data"

@pytest.mark.parametrize("seq_len", [0, 1, 3])
def test_append_multiple_tokens(empty_cache, basic_config, seq_len):
    new_keys = torch.randn(2, basic_config.num_heads, seq_len, basic_config.head_size)
    new_values = torch.randn(2, basic_config.num_heads, seq_len, basic_config.head_size)
    
    empty_cache.append(new_keys, new_values)
    
    assert len(empty_cache) == seq_len, f"Cache length should be {seq_len}"
    if seq_len > 0:
        assert torch.equal(empty_cache.keys, new_keys), "Keys should match appended data"
        assert torch.equal(empty_cache.values, new_values), "Values should match appended data"

def test_append_multiple_calls(empty_cache, sample_single_token):
    new_keys, new_values = sample_single_token
    
    empty_cache.append(new_keys, new_values)
    assert len(empty_cache) == 1, "Length should be 1 after first append"
    
    empty_cache.append(new_keys, new_values)
    assert len(empty_cache) == 2, "Length should be 2 after second append"
    
    # First token should still be there
    assert torch.equal(empty_cache.keys[:, :, 0:1, :], new_keys), "First token should be preserved"

def test_append_fill_exactly(nearly_full_cache, basic_config):
    # Add exactly one more token to fill cache
    new_keys = torch.randn(2, basic_config.num_heads, 1, basic_config.head_size)
    new_values = torch.randn(2, basic_config.num_heads, 1, basic_config.head_size)
    
    nearly_full_cache.append(new_keys, new_values)
    
    assert len(nearly_full_cache) == basic_config.block_size, "Cache should be exactly full"

def test_append_preserves_existing_data(partial_cache, sample_single_token, basic_config):
    original_keys = partial_cache.keys.clone()
    original_values = partial_cache.values.clone()
    new_keys, new_values = sample_single_token
    
    partial_cache.append(new_keys, new_values)
    
    # Original data should be preserved
    assert torch.equal(partial_cache.keys[:, :, :3, :], original_keys), "Existing keys should be preserved"
    assert torch.equal(partial_cache.values[:, :, :3, :], original_values), "Existing values should be preserved"
    # New data should be appended
    assert torch.equal(partial_cache.keys[:, :, 3:4, :], new_keys), "New keys should be appended"
    assert torch.equal(partial_cache.values[:, :, 3:4, :], new_values), "New values should be appended"


# ============================================================================
# ERROR CONDITION TESTS
# ============================================================================

@pytest.mark.parametrize("new_keys_shape,new_values_shape,expected_error", [
    ((2, 4, 3, 16), (2, 4, 2, 16), "same number of tokens"),  # seq len mismatch
    ((3, 4, 2, 16), (3, 4, 2, 16), "match the dimensions"),   # batch mismatch  
    ((2, 5, 2, 16), (2, 5, 2, 16), "match the dimensions"),   # heads mismatch
    ((2, 4, 2, 17), (2, 4, 2, 17), "match the dimensions"),   # head_size mismatch
])
def test_append_shape_errors(partial_cache, new_keys_shape, new_values_shape, expected_error):
    new_keys = torch.randn(*new_keys_shape)
    new_values = torch.randn(*new_values_shape)
    
    with pytest.raises(ValueError, match=expected_error):
        partial_cache.append(new_keys, new_values)

def test_append_cache_overflow(nearly_full_cache, basic_config):
    # Try to add 2 tokens when only 1 slot remains
    new_keys = torch.randn(2, basic_config.num_heads, 2, basic_config.head_size)
    new_values = torch.randn(2, basic_config.num_heads, 2, basic_config.head_size)
    
    with pytest.raises(ValueError, match="full"):
        nearly_full_cache.append(new_keys, new_values)

def test_append_wrong_device(partial_cache, sample_multi_token, wrong_device_tensors):
    right_keys, right_values = sample_multi_token
    wrong_keys, wrong_values = wrong_device_tensors

    with pytest.raises(ValueError, match="Device mismatch"):
        partial_cache.append(right_keys, wrong_values)
    with pytest.raises(ValueError, match="Device mismatch"):
        partial_cache.append(wrong_keys, right_values)

def test_append_wrong_dtype(partial_cache, sample_multi_token, wrong_dtype_tensors):
    right_keys, right_values = sample_multi_token
    wrong_keys, wrong_values = wrong_dtype_tensors
    
    with pytest.raises(ValueError, match="Dtype mismatch"):
        partial_cache.append(right_keys, wrong_values)
    with pytest.raises(ValueError, match="Dtype mismatch"):
        partial_cache.append(wrong_keys, right_values)


# ============================================================================
# MEMORY ALIASING TESTS
# ============================================================================

def test_memory_aliasing_behavior(partial_cache, basic_config):
    # Properties should return views
    keys_view = partial_cache.keys
    values_view = partial_cache.values
    
    # get_tensor_copies should return independent copies
    keys_copy, values_copy = partial_cache.get_tensor_copies()
    
    # clone should return independent cache
    cloned_cache = partial_cache.clone()
    
    # Modify original cache
    partial_cache._keys[:, :, :partial_cache._current_len, :] *= 2
    partial_cache._values[:, :, :partial_cache._current_len, :] *= 3
    
    # Views should reflect changes
    assert torch.equal(keys_view, partial_cache.keys), "Keys property should be a view"
    assert torch.equal(values_view, partial_cache.values), "Values property should be a view"
    
    # Copies should be independent
    assert not torch.equal(keys_copy, partial_cache.keys), "get_tensor_copies should return independent data"
    assert not torch.equal(values_copy, partial_cache.values), "get_tensor_copies should return independent data"
    
    # Clone should be independent
    assert not torch.equal(cloned_cache.keys, partial_cache.keys), "Clone should be independent"
    assert not torch.equal(cloned_cache.values, partial_cache.values), "Clone should be independent"


# ============================================================================
# DTYPE AND DEVICE PROPERTY TESTS
# ============================================================================

def test_dtype_property(empty_cache):
    assert empty_cache.dtype == torch.float32, "Dtype property should match tensor dtype"

def test_device_property(empty_cache):
    assert empty_cache.device == torch.device('cpu'), "Device property should match tensor device"

def test_to_method_chaining(empty_cache):
    result = empty_cache.to(torch.device('cpu'))
    assert result is empty_cache, "to() method should return self for chaining"