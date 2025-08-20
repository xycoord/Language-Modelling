import pytest
import torch
from src.mech_interp.data_generators.sparse_data_generator import create_uniform_sparsity, create_sparsity_range


@pytest.fixture
def assert_tensor_properties():
    def _assert_properties(tensor, expected_shape, expected_values=None):
        assert isinstance(tensor, torch.Tensor), "Return value must be a torch.Tensor"
        assert tensor.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {tensor.shape}"
        if expected_values is not None:
            assert torch.allclose(tensor, expected_values), "Tensor values do not match expected values"
    return _assert_properties


@pytest.fixture
def assert_sparsity_progression():
    def _assert_progression(tensor, min_sparsity, max_sparsity):
        sparsities = tensor[:, 0]
        
        assert torch.all(sparsities[1:] >= sparsities[:-1]), "Sparsity values must be non-decreasing across rows"
        assert torch.all(sparsities >= min_sparsity), f"All sparsity values must be >= {min_sparsity}"
        assert torch.all(sparsities <= max_sparsity), f"All sparsity values must be <= {max_sparsity}"
        
        for i, row in enumerate(tensor):
            assert torch.all(row == row[0]), f"All values in row {i} must be identical"
            
    return _assert_progression


@pytest.mark.parametrize("feature_dim,sparsity", [
    (1, 0.0),      # minimum sparsity
    (1, 1.0),      # maximum sparsity  
    (1, 0.5),      # minimum feature_dim
    (100, 0.5),    # typical case
    (50, 0.1),     # typical case
])
def test_create_uniform_sparsity_valid_inputs(feature_dim, sparsity, assert_tensor_properties):
    result = create_uniform_sparsity(feature_dim, sparsity)
    expected_values = torch.full((feature_dim,), sparsity)
    assert_tensor_properties(result, (feature_dim,), expected_values)


@pytest.mark.parametrize("feature_dim,sparsity", [
    (1, -0.1),     # sparsity below valid range
    (1, 1.1),      # sparsity above valid range
    (0, 0.5),      # feature_dim at boundary
    (-1, 0.5),     # feature_dim below valid range
])
def test_create_uniform_sparsity_error_conditions(feature_dim, sparsity):
    with pytest.raises(ValueError, match="must be"):
        create_uniform_sparsity(feature_dim, sparsity)


@pytest.mark.parametrize("min_sparsity,max_sparsity,num_models,feature_dim", [
    (0.0, 0.5, 1, 1),         # single model and feature
    (0.1, 0.9, 2, 1),         # two models
    (0.1, 0.9, 5, 100),       # typical case
])
def test_create_sparsity_range_valid_inputs(min_sparsity, max_sparsity, num_models, feature_dim, 
                                          assert_tensor_properties, assert_sparsity_progression):
    result = create_sparsity_range(min_sparsity, max_sparsity, num_models, feature_dim)
    assert_tensor_properties(result, (num_models, feature_dim))
    assert_sparsity_progression(result, min_sparsity, max_sparsity)


@pytest.mark.parametrize("min_sparsity,max_sparsity,num_models,feature_dim", [
    (0.6, 0.5, 1, 1),    # min greater than max
    (-0.1, 0.5, 1, 1),   # min_sparsity below valid range
    (0.1, 1.0, 1, 1),    # max_sparsity at invalid boundary
    (0.1, 1.1, 1, 1),    # max_sparsity above valid range
    (0.1, 0.5, 0, 1),    # num_models at boundary
    (0.1, 0.5, -1, 1),   # num_models below valid range
    (0.1, 0.5, 1, 0),    # feature_dim at boundary
    (0.1, 0.5, 1, -1),   # feature_dim below valid range
])
def test_create_sparsity_range_error_conditions(min_sparsity, max_sparsity, num_models, feature_dim):
    with pytest.raises(ValueError, match="must"):
        create_sparsity_range(min_sparsity, max_sparsity, num_models, feature_dim)