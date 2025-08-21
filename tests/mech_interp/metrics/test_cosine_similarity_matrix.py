import pytest
import torch

from src.mech_interp.metrics.sae_feature_recovery_metrics import cosine_similarity_matrix

@pytest.fixture
def available_devices():
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
    return devices


@pytest.mark.parametrize("w_sae_shape,w_true_shape,device_sae,device_true,expected_error", [
    # Dimension validation
    ((), (5, 10), "cpu", "cpu", ValueError),
    ((5,), (5, 10), "cpu", "cpu", ValueError),
    ((5, 10), (), "cpu", "cpu", ValueError),
    ((5, 10), (5,), "cpu", "cpu", ValueError),
    # Mismatched dimensions
    ((5, 10), (2, 5, 10), "cpu", "cpu", ValueError),
    ((2, 5, 10), (5, 10), "cpu", "cpu", ValueError),
    # Mismatched batch dimensions
    ((2, 5, 10), (3, 5, 10), "cpu", "cpu", ValueError),
    ((2, 3, 5, 10), (2, 4, 5, 10), "cpu", "cpu", ValueError),
    # Mismatched hidden dimensions
    ((5, 10), (3, 20), "cpu", "cpu", ValueError),
    # Device mismatch (only test if CUDA available)
    pytest.param((5, 10), (3, 10), "cpu", "cuda", ValueError, 
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
])
def test_error_conditions(w_sae_shape, w_true_shape, device_sae, device_true, expected_error):
    w_sae = torch.randn(w_sae_shape, device=device_sae) if w_sae_shape else torch.tensor(1.0, device=device_sae)
    w_true = torch.randn(w_true_shape, device=device_true) if w_true_shape else torch.tensor(1.0, device=device_true)
    
    with pytest.raises(expected_error):
        cosine_similarity_matrix(w_sae, w_true)


@pytest.mark.parametrize("vec1,vec2,expected_sim,description", [
    ([1.0, 0.0], [0.0, 1.0], 0.0, "orthogonal"),
    ([1.0, 0.0], [1.0, 0.0], 1.0, "identical"),
    ([1.0, 0.0], [-1.0, 0.0], -1.0, "opposite"),
    ([3.0, 4.0], [3.0, 4.0], 1.0, "identical_non_unit"),
    ([1.0, 1.0], [1.0, -1.0], 0.0, "orthogonal_non_unit"),
    ([2.0, 0.0], [4.0, 0.0], 1.0, "parallel_different_magnitudes"),
])
def test_known_mathematical_cases(vec1, vec2, expected_sim, description):
    w_sae = torch.tensor([vec1], dtype=torch.float32)
    w_true = torch.tensor([vec2], dtype=torch.float32)
    
    result = cosine_similarity_matrix(w_sae, w_true)
    
    assert result.shape == (1, 1), f"Expected shape (1, 1) for {description}"
    assert torch.allclose(result, torch.tensor([[expected_sim]]), atol=1e-6), \
        f"Expected similarity {expected_sim} for {description}, got {result.item()}"


@pytest.mark.parametrize("w_sae_shape,w_true_shape,expected_output_shape", [
    ((5, 10), (3, 10), (5, 3)),
    ((2, 5, 10), (2, 3, 10), (2, 5, 3)),
    ((2, 3, 5, 10), (2, 3, 3, 10), (2, 3, 5, 3)),
    ((1, 64), (1, 64), (1, 1)),
])
def test_basic_functionality_and_shapes(w_sae_shape, w_true_shape, expected_output_shape):
    w_sae = torch.randn(w_sae_shape)
    w_true = torch.randn(w_true_shape)
    
    result = cosine_similarity_matrix(w_sae, w_true)
    
    assert result.shape == expected_output_shape, \
        f"Expected output shape {expected_output_shape}, got {result.shape}"
    assert result.device == w_sae.device, "Output device should match input device"
    assert not torch.isnan(result).any(), "Result should not contain NaN values for random inputs"

def test_mathematical_properties():
    w_sae = torch.randn(5, 10)
    w_true = torch.randn(3, 10)
    
    result = cosine_similarity_matrix(w_sae, w_true)
    
    # Cosine similarity should be bounded in [-1, 1]
    assert torch.all(result >= -1.0), "Cosine similarity should be >= -1"
    assert torch.all(result <= 1.0), "Cosine similarity should be <= 1"


def test_scale_invariance():
    w_sae = torch.randn(3, 5)
    w_true = torch.randn(2, 5)
    
    result1 = cosine_similarity_matrix(w_sae, w_true)
    
    # Scale both tensors by different factors
    result2 = cosine_similarity_matrix(w_sae * 10, w_true * 0.1)
    
    assert torch.allclose(result1, result2, atol=1e-6), \
        "Cosine similarity should be invariant to scaling"


def test_self_similarity():
    w_sae = torch.randn(3, 5)
    
    result = cosine_similarity_matrix(w_sae, w_sae)
    expected_diagonal = torch.ones(3)
    
    assert torch.allclose(torch.diag(result), expected_diagonal, atol=1e-6), \
        "Self-similarity should be 1.0 for all vectors"


def test_device_consistency(available_devices):
    for device in available_devices:
        w_sae = torch.randn(3, 5, device=device)
        w_true = torch.randn(2, 5, device=device)
        
        result = cosine_similarity_matrix(w_sae, w_true)
        
        assert result.device == device, f"Output should be on {device}"


def test_batch_processing():
    batch_size = 4
    w_sae = torch.randn(batch_size, 5, 10)
    w_true = torch.randn(batch_size, 3, 10)
    
    result = cosine_similarity_matrix(w_sae, w_true)
    
    assert result.shape == (batch_size, 5, 3), \
        "Should correctly handle batched inputs"