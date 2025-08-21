import pytest
import torch

from src.mech_interp.metrics.sae_feature_recovery_metrics import injective_feature_matching

@pytest.fixture
def simple_similarity_matrix():
    """Basic 3x3 similarity matrix with clear matching preferences."""
    return torch.tensor([
        [0.9, 0.1, 0.2],  # SAE feature 0 strongly prefers true feature 0
        [0.3, 0.8, 0.1],  # SAE feature 1 strongly prefers true feature 1  
        [0.1, 0.2, 0.7],  # SAE feature 2 strongly prefers true feature 2
    ])

@pytest.fixture
def competing_similarity_matrix():
    """Matrix where multiple SAE features compete for same true feature."""
    return torch.tensor([
        [0.9, 0.1, 0.1],  # Both SAE 0 and SAE 1 want true feature 0
        [0.8, 0.2, 0.1],  # SAE 0 should win (0.9 > 0.8)
        [0.1, 0.1, 0.6],  # SAE 2 gets true feature 2
    ])

@pytest.fixture
def below_threshold_matrix():
    """Matrix with all similarities below typical thresholds."""
    return torch.tensor([
        [0.2, 0.1, 0.1],
        [0.1, 0.3, 0.1], 
        [0.1, 0.1, 0.2],
    ])

@pytest.fixture
def batched_similarity_matrix():
    """Batched version for testing batch independence."""
    batch1 = torch.tensor([
        [0.9, 0.1],
        [0.2, 0.8],
    ])
    batch2 = torch.tensor([
        [0.3, 0.7], 
        [0.6, 0.4],
    ])
    return torch.stack([batch1, batch2])


@pytest.fixture
def empty_matrices():
    """Various empty matrix configurations."""
    return {
        'no_sae': torch.zeros(0, 3),
        'no_true': torch.zeros(3, 0), 
        'both_empty': torch.zeros(0, 0),
    }


def test_input_validation_1d_matrix():
    """1D similarity matrix should raise ValueError."""
    with pytest.raises(ValueError, match="must be at least 2D"):
        injective_feature_matching(torch.tensor([0.5, 0.3, 0.1]))


@pytest.mark.parametrize("threshold", [-0.1, 1.1, -1.0, 2.0])
def test_input_validation_invalid_threshold(threshold, simple_similarity_matrix):
    """Invalid threshold values should raise ValueError."""
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        injective_feature_matching(simple_similarity_matrix, threshold)


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_input_validation_valid_threshold(threshold, simple_similarity_matrix):
    """Valid threshold values should not raise errors."""
    result = injective_feature_matching(simple_similarity_matrix, threshold)
    assert result.shape == (3,)


def test_output_shape_2d(simple_similarity_matrix):
    """Output shape should match (..., n_true) for 2D input."""
    result = injective_feature_matching(simple_similarity_matrix, 0.0)
    assert result.shape == (3,), "Output should have shape (n_true,)"


def test_output_shape_batched(batched_similarity_matrix):
    """Output shape should match (..., n_true) for batched input.""" 
    result = injective_feature_matching(batched_similarity_matrix, 0.0)
    expected_shape = (2, 2)  # (batch_size, n_true)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


def test_output_values_non_negative(simple_similarity_matrix):
    """All output values should be non-negative."""
    result = injective_feature_matching(simple_similarity_matrix, 0.0)
    assert (result >= 0).all(), "All output values should be non-negative"


def test_output_values_bounded(simple_similarity_matrix):
    """Output values should not exceed maximum input similarity."""
    result = injective_feature_matching(simple_similarity_matrix, 0.0)
    max_input = simple_similarity_matrix.max()
    assert (result <= max_input).all(), "Output values should not exceed input maximum"


def test_unmatched_positions_are_zero(below_threshold_matrix):
    """Unmatched positions should be exactly 0.0."""
    result = injective_feature_matching(below_threshold_matrix, 0.5)
    # All similarities are below 0.5, so all should be unmatched
    expected = torch.zeros(3)
    assert torch.equal(result, expected), "All positions should be 0.0 when below threshold"


def test_perfect_matches_selected(simple_similarity_matrix):
    """Perfect diagonal matches should be selected when threshold allows."""
    # Modify to have perfect diagonal
    perfect_matrix = torch.eye(3)
    result = injective_feature_matching(perfect_matrix, 0.9)
    expected = torch.ones(3)
    assert torch.allclose(result, expected), "Perfect matches should be selected"


def test_greedy_selection_highest_first(competing_similarity_matrix):
    """Highest similarity pairs should be chosen first in greedy selection."""
    result = injective_feature_matching(competing_similarity_matrix, 0.0)
    
    # SAE 0 should get feature 0 (0.9), 
    # SAE 1 gets feature 1 (0.2), despite having a higher similarity to feature 0 (0.8), 
    # SAE 2 gets feature 2 (0.6)
    expected = torch.tensor([0.9, 0.2, 0.6])
    assert torch.allclose(result, expected), "Highest similarity should win competition"


def test_injective_mapping_no_double_assignment():
    """Each SAE feature should match at most one true feature."""
    # Create matrix where SAE 0 could match multiple true features highly
    matrix = torch.tensor([
        [0.9, 0.8, 0.7],  # SAE 0 has high similarity to all
        [0.1, 0.1, 0.1],  # SAE 1 has low similarity to all
    ])
    result = injective_feature_matching(matrix, 0.0)
    
    # Only one true feature should get the 0.9, one should get 0.1 and the other should be 0
    high_count = (result > 0.1).sum().item()
    assert high_count == 1, "Only one true feature should be matched by SAE 0"
    no_match_count = (result == 0).sum().item()
    assert no_match_count == 1, "One true feature should not be matched"
    assert torch.allclose(result.max(), torch.tensor(0.9)), "Highest similarity should be preserved"


def test_threshold_filtering():
    """Only matches above threshold should be included."""
    matrix = torch.tensor([
        [0.8, 0.2],  # Only first match above 0.5 threshold
        [0.3, 0.9],  # Only second match above 0.5 threshold  
    ])
    result = injective_feature_matching(matrix, 0.5)
    
    # Should get 0.8 for true feature 0, 0.9 for true feature 1
    expected = torch.tensor([0.8, 0.9])
    assert torch.allclose(result, expected), "Only above-threshold matches should be included"


def test_batch_independence(batched_similarity_matrix):
    """Different batch elements should be processed independently."""
    result = injective_feature_matching(batched_similarity_matrix, 0.0)
    
    # Process each batch individually and compare
    batch1_result = injective_feature_matching(batched_similarity_matrix[0], 0.0)
    batch2_result = injective_feature_matching(batched_similarity_matrix[1], 0.0)
    
    expected = torch.stack([batch1_result, batch2_result])
    assert torch.allclose(result, expected), "Batched processing should match individual processing"


@pytest.mark.parametrize("n_sae,n_true", [
    (0, 3),
    (3, 0),
])
def test_empty_sae_features(n_sae, n_true):
    """Empty SAE features should return zeros."""
    result = injective_feature_matching(torch.zeros(n_sae, n_true), 0.0)
    expected = torch.zeros(n_true)
    assert torch.equal(result, expected), "No SAE features should result in all zeros"


def test_single_element_matrix():
    """Single element matrix should work correctly."""
    matrix = torch.tensor([[0.7]])
    
    # Above threshold
    result_above = injective_feature_matching(matrix, 0.5)
    assert torch.allclose(result_above, torch.tensor([0.7])), "Single element above threshold should be matched"
    
    # Below threshold  
    result_below = injective_feature_matching(matrix, 0.8)
    assert torch.allclose(result_below, torch.tensor([0.0])), "Single element below threshold should not be matched"


def test_pure_function_input_unchanged(simple_similarity_matrix):
    """Input similarity matrix should remain unchanged."""
    original = simple_similarity_matrix.clone()
    injective_feature_matching(simple_similarity_matrix, 0.5)
    assert torch.equal(simple_similarity_matrix, original), "Input matrix should be unchanged"


def test_device_preservation():
    """Output should be on same device as input."""
    matrix = torch.tensor([[0.5, 0.3], [0.2, 0.8]])
    result = injective_feature_matching(matrix, 0.0)
    assert result.device == matrix.device, "Output device should match input device"


def test_dtype_preservation():
    """Output should have same dtype as input."""
    matrix = torch.tensor([[0.5, 0.3], [0.2, 0.8]], dtype=torch.float64)
    result = injective_feature_matching(matrix, 0.0)
    assert result.dtype == matrix.dtype, "Output dtype should match input dtype"


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (1, 1, 1)])
def test_various_batch_shapes(batch_shape):
    """Function should handle various batch shapes correctly."""
    n_sae, n_true = 2, 3
    full_shape = batch_shape + (n_sae, n_true)
    matrix = torch.rand(full_shape)
    
    result = injective_feature_matching(matrix, 0.0)
    expected_shape = batch_shape + (n_true,)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


def test_high_dimensional_batch():
    """Function should work with high-dimensional batch shapes."""
    # 3D batch: (2, 2, 2) batch dimensions + (3, 4) feature dimensions
    matrix = torch.rand(2, 2, 2, 3, 4)
    result = injective_feature_matching(matrix, 0.0)
    
    assert result.shape == (2, 2, 2, 4), "High-dimensional batches should be handled correctly"
    assert (result >= 0).all(), "All values should be non-negative"