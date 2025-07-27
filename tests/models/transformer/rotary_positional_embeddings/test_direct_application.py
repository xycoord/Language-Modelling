import pytest
import torch
from src.lm_models.transformer.rotary_positional_embedding.direct_application import RotaryPositionalEmbedding

"""
Enhanced RoPE Test Suite covering both implementation variants:

1. interleaved=True: Rotates pairs of adjacent elements (0,1), (2,3), (4,5), ...
2. interleaved=False: Rotates first half against second half: (0,dim//2), (1,dim//2+1), etc.

Both variants implement valid RoPE rotations but along different axes, so they 
should produce DIFFERENT outputs while both preserving the fundamental mathematical 
properties of rotary position embeddings (magnitude preservation, position dependency, etc.).

Key testing considerations:
- At position 0, both variants are equivalent (no rotation applied)
- Cache shapes include an extra heads dimension for broadcasting: [seq_len, 1, ..., 2]
- Magnitude preservation must account for different pairing schemes
- Tests must use non-zero positions to see differences between variants
"""

def _get_theta(dim, base):
    return 1.0 / (
        base
        ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )

def test_variants_have_different_pairing_schemes():
    """Demonstrate the specific difference in how the two variants pair elements for rotation"""
    dim = 8
    rope_interleaved = RotaryPositionalEmbedding(dim=dim, max_seq_len=10, interleaved=True)
    rope_half_flipped = RotaryPositionalEmbedding(dim=dim, max_seq_len=10, interleaved=False)
    
    # Create input with distinct values for each position
    input_tensor = torch.zeros(1, 1, 1, dim)
    input_tensor[0, 0, 0, :] = torch.arange(dim, dtype=torch.float32)
    
    # Apply rotation at position 1 (not 0, to ensure rotation occurs)
    output_interleaved = rope_interleaved(input_tensor, input_pos=torch.tensor([1]))
    output_half_flipped = rope_half_flipped(input_tensor, input_pos=torch.tensor([1]))
    
    # For interleaved: pairs (0,1), (2,3), (4,5), (6,7) should be affected
    # For half-flipped: pairs (0,4), (1,5), (2,6), (3,7) should be affected
    
    # They should produce different results
    assert not torch.allclose(output_interleaved, output_half_flipped, atol=1e-6), \
        "Different pairing schemes should produce different outputs"
    
    # Both should be different from input (rotation applied)
    assert not torch.allclose(output_interleaved, input_tensor, atol=1e-6), \
        "Interleaved should modify the input"
    assert not torch.allclose(output_half_flipped, input_tensor, atol=1e-6), \
        "Half-flipped should modify the input"

# ============================================================================
# PARAMETRIZED FIXTURES FOR BOTH VARIANTS
# ============================================================================

@pytest.fixture(params=[True, False], ids=["interleaved", "half_flipped"])
def rope_variant(request):
    """Parameter for testing both RoPE variants"""
    return request.param

@pytest.fixture
def standard_rope(rope_variant):
    """Standard RoPE instance for most tests - now tests both variants"""
    return RotaryPositionalEmbedding(dim=64, max_seq_len=512, base=10000, interleaved=rope_variant)

@pytest.fixture
def minimal_rope(rope_variant):
    """Minimal valid RoPE for edge case testing - now tests both variants"""
    return RotaryPositionalEmbedding(dim=2, max_seq_len=4, base=2, interleaved=rope_variant)

@pytest.fixture
def large_rope(rope_variant):
    """Large RoPE for boundary testing - now tests both variants"""
    return RotaryPositionalEmbedding(dim=128, max_seq_len=2048, base=10000, interleaved=rope_variant)

@pytest.fixture
def standard_input():
    """Standard 4D input tensor [batch=2, seq=8, heads=4, dim=64]"""
    return torch.randn(2, 8, 4, 64)

@pytest.fixture
def single_token_input():
    """Single token input [batch=1, seq=1, heads=1, dim=64]"""
    return torch.randn(1, 1, 1, 64)

@pytest.fixture
def empty_sequence_input():
    """Empty sequence [batch=1, seq=0, heads=1, dim=64]"""
    return torch.randn(1, 0, 1, 64)

@pytest.fixture
def sequential_positions():
    """Standard sequential positions [0, 1, 2, ..., 7]"""
    return torch.arange(8)

@pytest.fixture
def non_sequential_positions():
    """Non-sequential positions for packed sequences"""
    return torch.tensor([0, 5, 10, 2, 7, 15, 3, 8])

@pytest.fixture
def broadcast_positions():
    """Positions that test broadcasting [batch=2, seq=8]"""
    return torch.arange(8).unsqueeze(0).expand(2, -1)


# ============================================================================
# DTYPE AND DEVICE TESTING WITH BOTH VARIANTS
# ============================================================================

# Define the parameter sets for the fixture
dtypes = [torch.float16, torch.float32, torch.bfloat16]
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
variants = [True, False]  # interleaved variants
test_configs = [(dtype, device, variant) for dtype in dtypes for device in devices for variant in variants]

@pytest.fixture(params=test_configs)
def forward_pass_data(request):
    """
    A parameterized fixture that yields a RoPE instance and an input tensor
    for each combination of dtype, device, and variant.
    """
    dtype, device, interleaved = request.param
    rope = RotaryPositionalEmbedding(dim=64, max_seq_len=512, base=10000, interleaved=interleaved).to(device)
    input_tensor = torch.randn(2, 8, 4, 64, dtype=dtype, device=device)
    output_tensor = rope(input_tensor)
    return {
        "rope": rope,
        "input": input_tensor,
        "output": output_tensor,
        "interleaved": interleaved
    }

# ============================================================================
# VARIANT-SPECIFIC TESTING
# ============================================================================

def test_variants_produce_different_results():
    """Test that the two RoPE variants produce different results (they rotate along different axes)"""
    dim, max_seq_len, base = 64, 512, 10000
    
    rope_interleaved = RotaryPositionalEmbedding(dim=dim, max_seq_len=max_seq_len, base=base, interleaved=True)
    rope_half_flipped = RotaryPositionalEmbedding(dim=dim, max_seq_len=max_seq_len, base=base, interleaved=False)
    
    # Use non-zero positions to avoid the case where no rotation is applied
    # At position 0, both variants are equivalent (no rotation)
    positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    
    # Test with various input configurations
    test_cases = [
        torch.randn(2, 8, 4, 64),  # standard
        torch.randn(1, 8, 1, 64),  # single batch, multiple tokens
        torch.randn(3, 8, 8, 64),  # larger batch
    ]
    
    for input_tensor in test_cases:
        output_interleaved = rope_interleaved(input_tensor, input_pos=positions)
        output_half_flipped = rope_half_flipped(input_tensor, input_pos=positions)
        
        # They should NOT be equivalent (different rotation axes)
        assert not torch.allclose(output_interleaved, output_half_flipped, atol=1e-5), \
            f"Interleaved and half-flipped variants should produce DIFFERENT results for input shape {input_tensor.shape}"

def test_variant_differences_with_custom_positions():
    """Test that variants produce different results with custom input positions"""
    rope_interleaved = RotaryPositionalEmbedding(dim=32, max_seq_len=100, interleaved=True)
    rope_half_flipped = RotaryPositionalEmbedding(dim=32, max_seq_len=100, interleaved=False)
    
    input_tensor = torch.randn(2, 6, 2, 32)
    # Use non-zero positions to ensure rotation is applied
    positions = torch.tensor([5, 10, 15, 20, 25, 30])
    
    output_interleaved = rope_interleaved(input_tensor, input_pos=positions)
    output_half_flipped = rope_half_flipped(input_tensor, input_pos=positions)
    
    assert not torch.allclose(output_interleaved, output_half_flipped, atol=1e-5), \
        "Both variants should produce DIFFERENT results with custom positions"

@pytest.mark.parametrize("interleaved", [True, False])
def test_variant_specific_cache_structure(interleaved):
    """Test that cache structure is correct for each variant"""
    dim, max_seq_len = 64, 100
    rope = RotaryPositionalEmbedding(dim=dim, max_seq_len=max_seq_len, interleaved=interleaved)
    
    if interleaved:
        # Cache should have shape [max_seq_len, 1, dim//2, 2] (note the heads dimension)
        expected_shape = (max_seq_len, 1, dim // 2, 2)
    else:
        # Cache should have shape [max_seq_len, 1, dim, 2] (note the heads dimension)
        expected_shape = (max_seq_len, 1, dim, 2)
    
    assert rope.cache.shape == expected_shape, \
        f"Cache shape mismatch for interleaved={interleaved}: expected {expected_shape}, got {rope.cache.shape}"

def test_both_variants_preserve_fundamental_rope_properties():
    """Test that both variants independently preserve the fundamental mathematical properties of RoPE"""
    for interleaved in [True, False]:
        rope = RotaryPositionalEmbedding(dim=32, max_seq_len=100, interleaved=interleaved)
        input_tensor = torch.randn(2, 8, 4, 32)
        
        # Test magnitude preservation using non-zero positions
        positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        output = rope(input_tensor, input_pos=positions)
        
        if interleaved:
            # For interleaved: pairs are (0,1), (2,3), (4,5), ...
            input_pairs = input_tensor.reshape(*input_tensor.shape[:-1], -1, 2)
            output_pairs = output.reshape(*output.shape[:-1], -1, 2)
        else:
            # For half-flipped: pairs are (0, dim//2), (1, dim//2+1), etc.
            dim = input_tensor.shape[-1]
            first_half = input_tensor[..., :dim//2]
            second_half = input_tensor[..., dim//2:]
            input_pairs = torch.stack([first_half, second_half], dim=-1)
            
            output_first_half = output[..., :dim//2]
            output_second_half = output[..., dim//2:]
            output_pairs = torch.stack([output_first_half, output_second_half], dim=-1)
        
        input_magnitudes = torch.norm(input_pairs, dim=-1)
        output_magnitudes = torch.norm(output_pairs, dim=-1)
        
        assert torch.allclose(input_magnitudes, output_magnitudes, atol=1e-5), \
            f"Variant interleaved={interleaved} should preserve magnitudes"
        
        # Test position dependency with different position sets
        pos_a = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12])
        pos_b = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
        
        output_a = rope(input_tensor, input_pos=pos_a)
        output_b = rope(input_tensor, input_pos=pos_b)
        
        assert not torch.allclose(output_a, output_b), \
            f"Variant interleaved={interleaved} should produce different outputs for different positions"

def test_rotation_axes_difference():
    """Demonstrate that the two variants rotate along different axes"""
    # Use a simple test case to show the difference clearly
    dim = 4
    rope_interleaved = RotaryPositionalEmbedding(dim=dim, max_seq_len=10, interleaved=True)
    rope_half_flipped = RotaryPositionalEmbedding(dim=dim, max_seq_len=10, interleaved=False)
    
    # Create a simple input where the difference will be clear
    input_tensor = torch.zeros(1, 1, 1, dim)
    input_tensor[0, 0, 0, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    # Use non-zero position to ensure rotation is applied
    output_interleaved = rope_interleaved(input_tensor, input_pos=torch.tensor([1]))
    output_half_flipped = rope_half_flipped(input_tensor, input_pos=torch.tensor([1]))
    
    # Verify they're different
    assert not torch.allclose(output_interleaved, output_half_flipped, atol=1e-6), \
        "The two variants should produce different outputs"
    
    # Verify both have rotated the input (not just copied it)
    assert not torch.allclose(output_interleaved, input_tensor, atol=1e-6), \
        "Interleaved variant should have rotated the input"
    assert not torch.allclose(output_half_flipped, input_tensor, atol=1e-6), \
        "Half-flipped variant should have rotated the input"

@pytest.mark.parametrize("interleaved", [True, False])
def test_implementation_chooses_correct_method(interleaved):
    """Test that the implementation correctly chooses the right rotation method based on interleaved parameter"""
    rope = RotaryPositionalEmbedding(dim=8, max_seq_len=10, interleaved=interleaved)
    input_tensor = torch.randn(1, 2, 1, 8)
    
    # Verify the interleaved attribute is set correctly
    assert rope.interleaved == interleaved
    
    # Verify the method works without errors using non-zero positions
    positions = torch.tensor([1, 2])
    output = rope(input_tensor, input_pos=positions)
    assert output.shape == input_tensor.shape
    
    # Test that the same configuration produces deterministic results
    output2 = rope(input_tensor, input_pos=positions)
    assert torch.allclose(output, output2, atol=1e-6)

# ============================================================================
# DTYPE AND DEVICE TESTING (NOW COVERS BOTH VARIANTS)
# ============================================================================

def test_shape_is_preserved(forward_pass_data):
    """Output shape must match the input shape."""
    assert forward_pass_data["output"].shape == forward_pass_data["input"].shape

def test_dtype_is_preserved(forward_pass_data):
    """Output dtype must match the input dtype."""
    assert forward_pass_data["output"].dtype == forward_pass_data["input"].dtype

def test_device_is_preserved(forward_pass_data):
    """Output device must match the input device."""
    assert forward_pass_data["output"].device.type == forward_pass_data["input"].device.type

def test_is_deterministic(forward_pass_data):
    """The same input must always produce the same output."""
    # Rerun the forward pass to test determinism
    output2 = forward_pass_data["rope"](forward_pass_data["input"])
    assert torch.allclose(forward_pass_data["output"], output2, atol=1e-6)

def test_magnitude_is_preserved(forward_pass_data):
    """The norm of each vector pair must be preserved after rotation."""
    input_tensor = forward_pass_data["input"]
    output_tensor = forward_pass_data["output"]
    interleaved = forward_pass_data["interleaved"]
    
    if interleaved:
        # For interleaved: pairs are (0,1), (2,3), (4,5), ...
        input_pairs = input_tensor.reshape(*input_tensor.shape[:-1], -1, 2)
        output_pairs = output_tensor.reshape(*output_tensor.shape[:-1], -1, 2)
    else:
        # For half-flipped: pairs are (0, dim//2), (1, dim//2+1), (2, dim//2+2), ...
        dim = input_tensor.shape[-1]
        first_half = input_tensor[..., :dim//2]
        second_half = input_tensor[..., dim//2:]
        input_pairs = torch.stack([first_half, second_half], dim=-1)
        
        output_first_half = output_tensor[..., :dim//2]
        output_second_half = output_tensor[..., dim//2:]
        output_pairs = torch.stack([output_first_half, output_second_half], dim=-1)

    input_magnitudes = torch.norm(input_pairs, dim=-1)
    output_magnitudes = torch.norm(output_pairs, dim=-1)

    # Use dtype-appropriate tolerances
    dtype = input_tensor.dtype
    if dtype == torch.float16:
        atol, rtol = 5e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 2e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5

    assert torch.allclose(input_magnitudes, output_magnitudes, atol=atol, rtol=rtol), \
        f"Rotation did not preserve magnitudes for interleaved={interleaved}."

# ============================================================================
# DIMENSION CONFIGURATION TESTING (NOW COVERS BOTH VARIANTS)
# ============================================================================

@pytest.mark.parametrize("dim,max_seq_len,base,interleaved", [
    (2, 4, 2, True),         # minimal interleaved
    (2, 4, 2, False),        # minimal half-flipped
    (32, 128, 1000, True),   # small interleaved
    (32, 128, 1000, False),  # small half-flipped
    (64, 512, 10000, True),  # standard interleaved
    (64, 512, 10000, False), # standard half-flipped
    (128, 2048, 10000, True), # large interleaved
    (128, 2048, 10000, False), # large half-flipped
])
def test_rope_initialization(dim, max_seq_len, base, interleaved):
    """Test initialization contracts across different configurations and variants"""
    RotaryPositionalEmbedding(dim=dim, max_seq_len=max_seq_len, base=base, interleaved=interleaved)

# ============================================================================
# SEQUENCE LENGTH BOUNDARY TESTING (NOW COVERS BOTH VARIANTS)
# ============================================================================

@pytest.mark.parametrize("seq_len,should_error", [
    (0, False),    # empty sequence
    (1, False),    # single token
    (512, False),  # max length
    (513, True),   # exceeds max length
])
def test_sequence_length_boundaries(standard_rope, seq_len, should_error):
    """Test sequence length boundary conditions"""
    input_tensor = torch.randn(1, seq_len, 1, 64)
    
    if should_error:
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            standard_rope(input_tensor)
    else:
        output = standard_rope(input_tensor)
        assert output.shape == input_tensor.shape, f"Output shape mismatch for seq_len={seq_len}"

# ============================================================================
# INPUT POSITION PATTERN TESTING (NOW COVERS BOTH VARIANTS)
# ============================================================================

def test_uses_default_sequential_positions(standard_rope, standard_input):
    """Tests that RoPE works correctly with no input_pos provided."""
    output = standard_rope(standard_input)
    assert output.shape == standard_input.shape

def test_explicit_sequential_positions_match_default(standard_rope, standard_input, sequential_positions):
    """Tests that explicitly passing sequential positions matches the default behavior."""
    output_default = standard_rope(standard_input)
    output_explicit = standard_rope(standard_input, input_pos=sequential_positions)
    assert torch.allclose(output_default, output_explicit, atol=1e-6)

def test_handles_non_sequential_positions(standard_rope, standard_input, non_sequential_positions):
    """Tests RoPE with non-sequential positions, as used in packed sequences."""
    output = standard_rope(standard_input, input_pos=non_sequential_positions)
    assert output.shape == standard_input.shape

def test_handles_broadcasted_positions(standard_rope, standard_input, broadcast_positions):
    """Tests that positions can be broadcasted across the batch dimension."""
    output = standard_rope(standard_input, input_pos=broadcast_positions)
    assert output.shape == standard_input.shape

def test_raises_error_for_out_of_bounds_positions(standard_rope, standard_input):
    """Tests that an IndexError is raised for positions exceeding max_seq_len."""
    # max_seq_len for standard_rope is 512
    invalid_positions = torch.tensor([0, 1, 2, 3, 512, 513, 6, 7])
    with pytest.raises((IndexError, RuntimeError)):
        standard_rope(standard_input, input_pos=invalid_positions)
        
# ============================================================================
# MATHEMATICAL PROPERTY VERIFICATION (NOW COVERS BOTH VARIANTS)
# ============================================================================

def test_position_dependency_produces_different_embeddings(standard_rope, standard_input):
    input_token = standard_input[:, :1]
    pos_a = torch.tensor([5])
    pos_b = torch.tensor([10])
    
    output_a = standard_rope(input_token, input_pos=pos_a)
    output_b = standard_rope(input_token, input_pos=pos_b)

    assert not torch.allclose(output_a, output_b), \
        "Outputs for different positions should not be identical."

def test_dot_product_is_invariant_to_relative_position(standard_rope):
    dim = standard_rope.dim
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)
    m, n = 8, 5
    
    # attention using absolute positions
    q_at_m = standard_rope(q, input_pos=torch.tensor([m]))
    k_at_n = standard_rope(k, input_pos=torch.tensor([n]))
    dot_product_absolute = torch.sum(q_at_m * k_at_n)

    # attention using relative positions
    relative_pos = m - n
    q_at_relative = standard_rope(q, input_pos=torch.tensor([relative_pos]))
    k_at_zero = standard_rope(k, input_pos=torch.tensor([0]))
    dot_product_relative = torch.sum(q_at_relative * k_at_zero)

    assert torch.allclose(dot_product_absolute, dot_product_relative, atol=1e-5), \
        "Dot product must be invariant to absolute positions, depending only on relative position."

def test_rotation_is_applied_pairwise_without_bleeding(standard_rope):
    dim = standard_rope.dim
    test_input = torch.zeros(1, 1, 1, dim)
    
    if standard_rope.interleaved:
        # For interleaved: test that only the first pair (0,1) is affected
        test_input[0, 0, 0, 0] = 5.0
        test_input[0, 0, 0, 1] = 2.0
        
        output = standard_rope(test_input, input_pos=torch.tensor([1]))

        assert not torch.isclose(output[0, 0, 0, 0], test_input[0, 0, 0, 0]), \
            "First element of the pair should be rotated."
        assert not torch.isclose(output[0, 0, 0, 1], test_input[0, 0, 0, 1]), \
            "Second element of the pair should be rotated."
            
        assert torch.allclose(output[0, 0, 0, 2:], torch.zeros(dim - 2)), \
            "Rotation should not cause information to bleed between pairs."
    else:
        # For half-flipped: test that pairs (0, dim//2) and (1, dim//2+1) are affected
        test_input[0, 0, 0, 0] = 5.0
        test_input[0, 0, 0, 1] = 2.0
        test_input[0, 0, 0, dim//2] = 3.0
        test_input[0, 0, 0, dim//2 + 1] = 4.0
        
        output = standard_rope(test_input, input_pos=torch.tensor([1]))

        # Elements 0 and dim//2 should be rotated (they form a pair)
        assert not torch.isclose(output[0, 0, 0, 0], test_input[0, 0, 0, 0]), \
            "Element 0 should be rotated."
        assert not torch.isclose(output[0, 0, 0, dim//2], test_input[0, 0, 0, dim//2]), \
            "Element dim//2 should be rotated."
            
        # Elements 1 and dim//2+1 should be rotated (they form a pair)
        assert not torch.isclose(output[0, 0, 0, 1], test_input[0, 0, 0, 1]), \
            "Element 1 should be rotated."
        assert not torch.isclose(output[0, 0, 0, dim//2 + 1], test_input[0, 0, 0, dim//2 + 1]), \
            "Element dim//2+1 should be rotated."
            
        # Other elements should remain zero
        other_indices = list(range(2, dim//2)) + list(range(dim//2 + 2, dim))
        if other_indices:  # Only check if there are other elements
            assert torch.allclose(output[0, 0, 0, other_indices], torch.zeros(len(other_indices))), \
                "Rotation should not affect unpaired elements."

# ============================================================================
# ERROR CONDITION TESTING
# ============================================================================

@pytest.mark.parametrize(
    "invalid_arg, invalid_value, match_str",
    [
        # Dimension `dim` validation
        ("dim", 63, "even"),
        ("dim", 0, "positive"),
        ("dim", -2, "positive"),
        # Max sequence length `max_seq_len` validation
        ("max_seq_len", -1, "non-negative"),
        # Base `base` validation
        ("base", 0, "positive"),
        ("base", -10000, "positive"),
    ],
)
def test_initialization_raises_error_for_invalid_parameters(invalid_arg, invalid_value, match_str):
    """Tests that __init__ raises ValueError for invalid constructor arguments."""
    valid_args = {"dim": 64, "max_seq_len": 512, "base": 10000}
    valid_args[invalid_arg] = invalid_value

    with pytest.raises(ValueError, match=match_str):
        RotaryPositionalEmbedding(**valid_args)

@pytest.mark.parametrize(
    "invalid_shape", [(2, 8, 64), (2, 8, 4, 64, 1)]  # 3D and 5D tensors
)
def test_forward_raises_error_for_invalid_tensor_rank(standard_rope, invalid_shape):
    """Tests that the forward pass rejects tensors that are not 4-dimensional."""
    invalid_input = torch.randn(invalid_shape)
    with pytest.raises(ValueError, match=r"4D"):
        standard_rope(invalid_input)

def test_forward_raises_error_for_feature_dimension_mismatch(standard_rope):
    """Tests error when the input's last dimension mismatches the rope's dim."""
    # standard_rope.dim is 64
    invalid_input = torch.randn(2, 8, 4, 32)  # Input dim is 32
    with pytest.raises(ValueError, match=r"match"):
        standard_rope(invalid_input)

def test_forward_raises_error_for_sequence_length_out_of_bounds(standard_rope):
    """Tests error when implicit sequence length exceeds max_seq_len."""
    # standard_rope.max_seq_len is 512
    invalid_input = torch.randn(1, 513, 1, 64)
    with pytest.raises((ValueError, IndexError, RuntimeError), match=r"bounds|size"):
        standard_rope(invalid_input)

@pytest.mark.parametrize(
    "pos_tensor_creator", [
        pytest.param(lambda msl: torch.tensor([0, 1, -1]), id="value_is_negative"),
    ],
)
def test_forward_raises_error_for_out_of_bounds_positions(
    standard_rope, standard_input, pos_tensor_creator
):
    """Tests that an error is raised for position values outside the valid range [0, max_seq_len - 1]."""
    max_seq_len = standard_rope.max_seq_len
    invalid_pos_tensor = pos_tensor_creator(max_seq_len)

    with pytest.raises((ValueError, IndexError, RuntimeError), match=r"bounds"):
        standard_rope(standard_input, input_pos=invalid_pos_tensor)

def test_forward_raises_error_for_unbroadcastable_position_shape(standard_rope, standard_input):
    """Tests that an error is raised if the position tensor's shape cannot be broadcast."""
    # Input shape is [2, 8, ...], so position shape [7] is not broadcastable
    invalid_pos_shape = torch.arange(7)
    with pytest.raises(RuntimeError, match=r"broadcast"):
        standard_rope(standard_input, input_pos=invalid_pos_shape)

def test_forward_raises_error_for_non_integer_position_dtype(standard_rope, standard_input):
    """Tests that an error is raised if the position tensor has a non-integer dtype."""
    invalid_pos_dtype = torch.tensor([0, 1, 2], dtype=torch.float32)
    with pytest.raises(ValueError, match=r"integer"):
        standard_rope(standard_input, input_pos=invalid_pos_dtype)