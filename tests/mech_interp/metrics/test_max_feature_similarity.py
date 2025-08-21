import pytest
import torch

from src.mech_interp.metrics.sae_feature_recovery_metrics import max_feature_similarity

@pytest.fixture
def basic_2d_tensor():
    """Simple 2D tensor with known maximum values."""
    return torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 1.0, 2.0],
        [2.0, 5.0, 1.0]
    ])


@pytest.fixture
def basic_3d_tensor():
    """3D tensor with batch dimension."""
    return torch.tensor([
        [[1.0, 2.0], [3.0, 1.0]],
        [[2.0, 1.0], [1.0, 4.0]]
    ])


@pytest.fixture
def tensor_with_negatives():
    """Tensor containing negative values."""
    return torch.tensor([
        [-1.0, 2.0, -3.0],
        [0.0, -1.0, 2.0],
        [2.0, -2.0, 1.0]
    ])


@pytest.fixture
def tensor_with_special_values():
    """Tensor containing NaN and infinity."""
    return torch.tensor([
        [1.0, float('inf'), 3.0],
        [float('nan'), 2.0, 1.0],
        [2.0, 1.0, float('-inf')]
    ])


@pytest.mark.parametrize("shape", [
    (3, 4),
    (2, 5, 3),
    (2, 3, 4, 5),
    (1, 2, 3, 4, 5)
])
def test_valid_shapes_produce_correct_output_shape(shape):
    """Test that valid input shapes produce correctly shaped outputs."""
    tensor = torch.randn(shape)
    result = max_feature_similarity(tensor)
    
    expected_shape = shape[:-2] + (shape[-1],)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


@pytest.mark.parametrize("invalid_shape", [
    (),
    (5,)
])
def test_invalid_shapes_raise_value_error(invalid_shape):
    """Test that tensors with fewer than 2 dimensions raise ValueError."""
    if invalid_shape == ():
        tensor = torch.tensor(1.0)
    else:
        tensor = torch.randn(invalid_shape)
    
    with pytest.raises(ValueError, match="Similarity matrix must be at least 2D"):
        max_feature_similarity(tensor)


@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64
])
def test_dtype_preservation(dtype):
    """Test that output preserves input dtype."""
    tensor = torch.randint(0, 10, (3, 4)).to(dtype)
    result = max_feature_similarity(tensor)
    
    assert result.dtype == dtype, f"Expected dtype {dtype}, got {result.dtype}"


def test_device_preservation():
    """Test that output preserves input device."""
    tensor = torch.randn(3, 4)
    result = max_feature_similarity(tensor)
    
    assert result.device == tensor.device, f"Expected device {tensor.device}, got {result.device}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_preservation():
    """Test that CUDA device is preserved."""
    tensor = torch.randn(3, 4).cuda()
    result = max_feature_similarity(tensor)
    
    assert result.device == tensor.device, f"Expected device {tensor.device}, got {result.device}"


def test_basic_2d_maximum_computation(basic_2d_tensor):
    """Test correct maximum computation for 2D input."""
    result = max_feature_similarity(basic_2d_tensor)
    expected = torch.tensor([4.0, 5.0, 3.0])
    
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_basic_3d_maximum_computation(basic_3d_tensor):
    """Test correct maximum computation for 3D input with batch dimension."""
    result = max_feature_similarity(basic_3d_tensor)
    expected = torch.tensor([[3.0, 2.0], [2.0, 4.0]])
    
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_handles_negative_values(tensor_with_negatives):
    """Test correct handling of negative values."""
    result = max_feature_similarity(tensor_with_negatives)
    expected = torch.tensor([2.0, 2.0, 2.0])
    
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_handles_special_values(tensor_with_special_values):
    """Test handling of NaN and infinity values."""
    result = max_feature_similarity(tensor_with_special_values)
    
    assert torch.isinf(result[1]), "Expected infinity in position 1"
    assert torch.isnan(result[0]), "Expected NaN in position 0"
    assert result[2] == 3.0, f"Expected 3.0 in position 2, got {result[2]}"


def test_single_element_matrices():
    """Test edge case with minimal valid tensor dimensions."""
    tensor = torch.tensor([[5.0]])
    result = max_feature_similarity(tensor)
    expected = torch.tensor([5.0])
    
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_single_feature_multiple_sae():
    """Test case with single true feature, multiple SAE features."""
    tensor = torch.tensor([[1.0], [3.0], [2.0]])
    result = max_feature_similarity(tensor)
    expected = torch.tensor([3.0])
    
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_single_sae_multiple_features():
    """Test case with single SAE feature, multiple true features."""
    tensor = torch.tensor([[1.0, 2.0, 3.0]])
    result = max_feature_similarity(tensor)
    expected = torch.tensor([1.0, 2.0, 3.0])
    
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_zero_tensor():
    """Test with tensor containing all zeros."""
    tensor = torch.zeros(3, 4)
    result = max_feature_similarity(tensor)
    expected = torch.zeros(4)
    
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_returns_tensor_type():
    """Test that function returns a torch.Tensor."""
    tensor = torch.randn(3, 4)
    result = max_feature_similarity(tensor)
    
    assert isinstance(result, torch.Tensor), f"Expected torch.Tensor, got {type(result)}"


def test_large_batch_dimensions():
    """Test with multiple batch dimensions."""
    tensor = torch.randn(2, 3, 4, 5, 6)
    result = max_feature_similarity(tensor)
    expected_shape = (2, 3, 4, 6)
    
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    
    # Verify each batch element is computed correctly
    for i in range(2):
        for j in range(3):
            for k in range(4):
                batch_slice = tensor[i, j, k]
                expected_slice = batch_slice.max(dim=-2)[0]
                actual_slice = result[i, j, k]
                assert torch.allclose(actual_slice, expected_slice), "Batch computation incorrect"