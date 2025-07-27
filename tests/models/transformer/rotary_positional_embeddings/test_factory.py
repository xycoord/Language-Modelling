import pytest
import torch
from src.lm_models.transformer.rotary_positional_embedding.factory import RotaryEmbeddingFactory, apply_rotary_embedding

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def standard_factory():
    """Standard RoPE_Factory instance for most tests."""
    return RotaryEmbeddingFactory(dim=64, base=10000)

@pytest.fixture
def minimal_factory():
    """Minimal valid RoPE_Factory for edge case testing."""
    return RotaryEmbeddingFactory(dim=2, base=10000)

@pytest.fixture
def standard_input():
    """Standard 4D input tensor [batch=2, seq=8, heads=4, dim=64]."""
    return torch.randn(2, 8, 4, 64)

@pytest.fixture
def empty_sequence_input():
    """Input tensor with an empty sequence [batch=1, seq=0, heads=1, dim=64]."""
    return torch.randn(1, 0, 1, 64)

@pytest.fixture
def standard_positions():
    """Standard sequential positions tensor [0, 1, ..., 7]."""
    return torch.arange(8)

@pytest.fixture
def empty_positions():
    """Empty positions tensor for zero-length sequences."""
    return torch.arange(0)

# ============================================================================
# PARAMETRIZED FIXTURES (DTYPE & DEVICE)
# ============================================================================

# Define parameter sets for dtypes and devices
dtypes = [torch.float32, torch.float16, torch.bfloat16]
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
test_configs = [(dtype, device) for dtype in dtypes for device in devices]

@pytest.fixture(params=test_configs)
def test_setup(request):
    """
    A parametrized fixture that yields a complete testing context for each
    combination of dtype and device.
    """
    dtype, device = request.param
    dim = 64
    seq_len = 8
    factory = RotaryEmbeddingFactory(dim=dim).to(device)
    positions = torch.arange(seq_len, device=device)
    pos_embedding = factory(positions)
    input_tensor = torch.randn(2, seq_len, 4, dim, dtype=dtype, device=device)

    return {
        "factory": factory,
        "input_tensor": input_tensor,
        "positions": positions,
        "pos_embedding": pos_embedding,
        "dtype": dtype,
        "device": device,
    }

# ============================================================================
# PART 1: RoPE_Factory.__init__ TESTS
# ============================================================================

def test_initialization_with_valid_parameters():
    """Tests that RoPE_Factory initializes without error for valid parameters."""
    RotaryEmbeddingFactory(dim=2, base=10000)
    RotaryEmbeddingFactory(dim=128, base=500)

@pytest.mark.parametrize(
    "kwargs, match_str",
    [
        ({"dim": 63}, "even"),
        ({"dim": 0}, "positive"),
        ({"dim": -2}, "positive"),
        ({"dim": 64, "base": 0}, "positive"),
        ({"dim": 64, "base": -10000}, "positive"),
    ],
)
def test_init_raises_error_for_invalid_parameters(kwargs, match_str):
    """Tests that __init__ raises ValueError for invalid constructor arguments."""
    with pytest.raises(ValueError, match=match_str):
        RotaryEmbeddingFactory(**kwargs)

# ============================================================================
# PART 2: RoPE_Factory.forward TESTS
# ============================================================================

def test_forward_returns_correct_shape(standard_factory, standard_positions):
    """Tests that forward() returns a tensor of the correct shape."""
    dim = standard_factory.dim
    seq_len = len(standard_positions)
    pos_embedding = standard_factory(standard_positions)
    expected_shape = (seq_len, 1, dim, 2)
    assert pos_embedding.shape == expected_shape, f"Expected shape {expected_shape}, but got {pos_embedding.shape}"

def test_forward_returns_correct_device(standard_positions):
    """Tests that the returned tensor is on the correct device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    factory = RotaryEmbeddingFactory(dim=64).to("cuda")
    positions = standard_positions.to("cuda")
    pos_embedding = factory(positions)
    assert pos_embedding.device.type == "cuda", "Tensor should be on the CUDA device"

def test_forward_handles_boundary_conditions(standard_factory, empty_positions):
    """Tests forward() with single-token and empty sequences."""
    dim = standard_factory.dim

    # Single token
    single_pos = torch.tensor([0])
    pos_embedding_single = standard_factory(single_pos)
    assert pos_embedding_single.shape == (1, 1, dim, 2), "Shape mismatch for single position"

    # Empty sequence
    pos_embedding_empty = standard_factory(empty_positions)
    assert pos_embedding_empty.shape == (0, 1, dim, 2), "Shape mismatch for empty positions"

def test_forward_output_is_deterministic(standard_factory, standard_positions):
    """Tests that forward() produces the same output for the same input."""
    output1 = standard_factory(standard_positions)
    output2 = standard_factory(standard_positions)
    assert torch.equal(output1, output2), "forward() should be deterministic"

def test_forward_output_values_are_correct(minimal_factory):
    """Tests that forward() computes mathematically correct cos/sin values."""
    # For position 0, angle is 0, so cos(0)=1 and sin(0)=0.
    pos_0 = torch.tensor([0])
    embedding_at_0 = minimal_factory(pos_0)
    cos_at_0, sin_at_0 = embedding_at_0.unbind(-1)
    
    assert torch.all(cos_at_0 == 1.0), "cos(0) must be 1 for all dimensions"
    assert torch.all(sin_at_0 == 0.0), "sin(0) must be 0 for all dimensions"

@pytest.mark.parametrize(
    "invalid_positions, match_str",
    [
        (torch.randn(8, 2), "1D tensor"),          # Invalid rank
        (torch.arange(8, dtype=torch.float32), "integer tensor"), # Invalid dtype
        (torch.tensor([0, 1, -1]), "negative"), # Invalid value
    ],
)
def test_forward_raises_error_for_invalid_input(standard_factory, invalid_positions, match_str):
    """Tests that forward() raises errors for invalid input_positions."""
    with pytest.raises((ValueError, IndexError, RuntimeError), match=match_str):
        standard_factory(invalid_positions)

# ============================================================================
# PART 3: apply_rope TESTS
# ============================================================================

def test_apply_rope_preserves_shape_dtype_device(test_setup):
    """Tests that apply_rope preserves the input tensor's properties."""
    input_tensor = test_setup["input_tensor"]
    pos_embedding = test_setup["pos_embedding"]
    
    output_tensor = apply_rotary_embedding(input_tensor, pos_embedding)
    
    assert output_tensor.shape == input_tensor.shape, "Output shape must match input shape"
    assert output_tensor.dtype == input_tensor.dtype, "Output dtype must match input dtype"
    assert output_tensor.device.type == input_tensor.device.type, "Output device must match input device"

def test_apply_rope_preserves_magnitude(test_setup):
    """Tests that RoPE rotation preserves the L2 norm of feature pairs."""
    input_tensor = test_setup["input_tensor"]
    pos_embedding = test_setup["pos_embedding"]
    dtype = test_setup["dtype"]
    
    output_tensor = apply_rotary_embedding(input_tensor, pos_embedding)
    
    # The half-flipped method pairs (x_i, x_{i + dim/2})
    def get_magnitudes(x):
        first_half, second_half = x.chunk(2, dim=-1)
        # Stack to create pairs: [batch, seq, heads, dim/2, 2]
        pairs = torch.stack([first_half, second_half], dim=-1)
        return torch.norm(pairs, p=2, dim=-1)

    input_magnitudes = get_magnitudes(input_tensor)
    output_magnitudes = get_magnitudes(output_tensor)

    # Use dtype-appropriate tolerances
    if dtype == torch.float16:
        atol, rtol = 5e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 2e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5

    assert torch.allclose(input_magnitudes, output_magnitudes, atol=atol, rtol=rtol), \
        "Rotation did not preserve vector pair magnitudes"

def test_apply_rope_handles_empty_sequence(empty_sequence_input, empty_positions, standard_factory):
    """Tests that apply_rope handles zero-length sequences gracefully."""
    pos_embedding = standard_factory(empty_positions)
    output = apply_rotary_embedding(empty_sequence_input, pos_embedding)
    assert output.shape == empty_sequence_input.shape, "Shape should be preserved for empty sequence"

def test_apply_rope_raises_error_for_invalid_ranks(standard_input, standard_factory, standard_positions):
    """Tests that apply_rope rejects tensors that are not 4-dimensional."""
    pos_embedding = standard_factory(standard_positions)
    
    # Invalid input rank
    with pytest.raises(ValueError, match="x must be 4D"):
        apply_rotary_embedding(standard_input[0], pos_embedding)
        
    # Invalid positional embedding rank
    with pytest.raises(ValueError, match="positional_embedding must be 4D"):
        apply_rotary_embedding(standard_input, pos_embedding.squeeze(1))

@pytest.mark.parametrize("mismatch_type", ["device", "dim", "seq_len"])
def test_apply_rope_raises_error_for_mismatched_properties(standard_input, standard_factory, standard_positions, mismatch_type):
    """Tests that apply_rope raises errors for mismatched tensor properties."""
    pos_embedding = standard_factory(standard_positions)
    
    if mismatch_type == "device":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")
        x = standard_input.to("cpu")
        pe = pos_embedding.to("cuda")
        match = "Device mismatch"
    elif mismatch_type == "dim":
        x = torch.randn(2, 8, 4, 32) # dim=32
        pe = pos_embedding # dim=64
        match = "Head dimension mismatch"
    elif mismatch_type == "seq_len":
        x = standard_input
        pe = standard_factory(torch.arange(4)) # seq_len=4
        match = "Sequence length mismatch"
        
    with pytest.raises(RuntimeError, match=match):
        apply_rotary_embedding(x, pe)

# ============================================================================
# PART 4: INTEGRATION & MATHEMATICAL VERIFICATION
# ============================================================================

def test_dot_product_is_invariant_to_relative_position(standard_factory):
    """Tests that the dot product depends on relative, not absolute, position."""
    dim = standard_factory.dim
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)
    
    # Calculate attention score for absolute positions m and n
    m, n = 8, 5
    pos_m = standard_factory(torch.tensor([m]))
    pos_n = standard_factory(torch.tensor([n]))
    q_at_m = apply_rotary_embedding(q, pos_m)
    k_at_n = apply_rotary_embedding(k, pos_n)
    dot_product_absolute = torch.sum(q_at_m * k_at_n)

    # Calculate attention score for relative position m-n
    relative_pos = m - n
    pos_relative = standard_factory(torch.tensor([relative_pos]))
    pos_zero = standard_factory(torch.tensor([0]))
    q_at_relative = apply_rotary_embedding(q, pos_relative)
    k_at_zero = apply_rotary_embedding(k, pos_zero) # k is at position 0
    dot_product_relative = torch.sum(q_at_relative * k_at_zero)

    assert torch.allclose(dot_product_absolute, dot_product_relative, atol=1e-5), \
        "Dot product must be invariant to absolute positions"

def test_rotation_is_applied_pairwise_without_bleeding(standard_factory):
    """Tests that rotation of one feature pair does not affect others."""
    dim = standard_factory.dim
    test_input = torch.zeros(1, 1, 1, dim)
    # Activate the first feature pair
    test_input[..., 0] = 5.0
    test_input[..., dim // 2] = 2.0
    
    pos_embedding = standard_factory(torch.tensor([1]))
    output = apply_rotary_embedding(test_input, pos_embedding)

    # Check that the activated pair was rotated
    assert not torch.isclose(output[..., 0], test_input[..., 0]), \
        "First element of the pair should be rotated"
    assert not torch.isclose(output[..., dim // 2], test_input[..., dim // 2]), \
        "Second element of the pair should be rotated"
        
    # Check that all other elements remain zero
    output[..., 0] = 0
    output[..., dim // 2] = 0
    assert torch.all(output == 0), "Rotation should not cause information to bleed between pairs"

def test_gradients_flow_correctly(test_setup):
    """Ensure gradients flow through RoPE operations."""
    input_tensor = test_setup["input_tensor"].requires_grad_(True)
    pos_embedding = test_setup["pos_embedding"]
    
    output = apply_rotary_embedding(input_tensor, pos_embedding)
    loss = output.sum()
    loss.backward()
    
    assert input_tensor.grad is not None
    assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))
