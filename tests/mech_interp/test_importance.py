import pytest
import torch
from src.mech_interp.importance import importance_decay_by_ratio, importance_decay_by_min

@pytest.mark.parametrize("func,feature_dim,param", [
    (importance_decay_by_ratio, 5, 0.8),
    (importance_decay_by_ratio, 10, 0.5),
    (importance_decay_by_ratio, 3, 1.0),
    (importance_decay_by_min, 5, 0.3),
    (importance_decay_by_min, 10, 0.1),
    (importance_decay_by_min, 3, 1.0),
])
def test_basic_tensor_properties(func, feature_dim, param):
    result = func(feature_dim, param)
    
    assert result.shape == (feature_dim,), f"Expected shape ({feature_dim},), got {result.shape}"
    assert result.dtype == torch.float, f"Expected dtype torch.float, got {result.dtype}"
    assert torch.allclose(result[0], torch.tensor(1.0)), "First element must equal 1.0"
    assert torch.all(result > 0), "All elements must be positive"


@pytest.mark.parametrize("func,feature_dim,param,expected_ratio", [
    (importance_decay_by_ratio, 5, 0.7, 0.7),
    (importance_decay_by_ratio, 8, 0.3, 0.3),
    (importance_decay_by_min, 5, 0.2, 0.2 ** (1/4)),
    (importance_decay_by_min, 8, 0.1, 0.1 ** (1/7)),
])
def test_geometric_progression_property(func, feature_dim, param, expected_ratio):
    result = func(feature_dim, param)
    
    if len(result) > 1:
        ratios = result[1:] / result[:-1]
        assert torch.allclose(ratios, torch.tensor(expected_ratio), rtol=1e-6), \
            "Consecutive elements must maintain geometric progression ratio"


@pytest.mark.parametrize("func,feature_dim,param,expected_sum", [
    (importance_decay_by_ratio, 5, 0.5, (1 - 0.5**5) / (1 - 0.5)),
    (importance_decay_by_ratio, 3, 1.0, 3.0),
    (importance_decay_by_min, 4, 0.125, (1 - 0.5**4) / (1 - 0.5)),  # 0.125**(1/3) = 0.5
])
def test_geometric_series_sum(func, feature_dim, param, expected_sum):
    result = func(feature_dim, param)
    
    assert torch.allclose(result.sum(), torch.tensor(expected_sum), rtol=1e-5), \
        "Sum must follow geometric series formula"


@pytest.mark.parametrize("feature_dim", [0, -1, -5])
@pytest.mark.parametrize("func,param", [
    (importance_decay_by_ratio, 0.5),
    (importance_decay_by_min, 0.3)
])
def test_error_invalid_feature_dimensions(func, feature_dim, param):
    with pytest.raises(ValueError):
        func(feature_dim, param)


@pytest.mark.parametrize("feature_dim,param", [
    (1.5, 0.5),
    ("invalid", 0.5),
    (None, 0.5),
    ([5], 0.5)
])
@pytest.mark.parametrize("func", [importance_decay_by_ratio, importance_decay_by_min])
def test_error_invalid_parameter_types(func, feature_dim, param):
    with pytest.raises((TypeError, ValueError)):
        func(feature_dim, param)


@pytest.mark.parametrize("feature_dim,decay_ratio", [
    (5, 0.5), (7, 0.3), (3, 0.1), (10, 0.8)
])
def test_decay_ratio_decreasing_sequence(feature_dim, decay_ratio):
    result = importance_decay_by_ratio(feature_dim, decay_ratio)
    
    assert torch.all(result[1:] <= result[:-1]), "Sequence must be monotonically decreasing"


@pytest.mark.parametrize("feature_dim,decay_ratio", [
    (5, 1.5), (3, 2.0), (4, 10.0)
])
def test_decay_ratio_increasing_sequence(feature_dim, decay_ratio):
    result = importance_decay_by_ratio(feature_dim, decay_ratio)
    
    assert torch.all(result[1:] >= result[:-1]), "Sequence must be monotonically increasing"


@pytest.mark.parametrize("feature_dim", [3, 5, 10])
def test_decay_ratio_constant_sequence(feature_dim):
    result = importance_decay_by_ratio(feature_dim, 1.0)
    
    assert torch.allclose(result, torch.ones(feature_dim)), "All elements must equal 1.0"


@pytest.mark.parametrize("feature_dim", [5, 8, 12])
def test_decay_ratio_zero_behavior(feature_dim):
    result = importance_decay_by_ratio(feature_dim, 0.0)
    
    assert torch.allclose(result[0], torch.tensor(1.0)), "First element must be 1.0"
    if feature_dim > 1:
        assert torch.allclose(result[1:], torch.zeros(feature_dim - 1)), "Remaining elements must be 0.0"


@pytest.mark.parametrize("feature_dim,decay_ratio", [
    (10, 1e-6), (8, 1e-8), (15, 1e-4)
])
def test_decay_ratio_very_small_values(feature_dim, decay_ratio):
    result = importance_decay_by_ratio(feature_dim, decay_ratio)
    
    assert result[0] == 1.0, "First element must be 1.0"
    if feature_dim > 1:
        assert torch.all(result[1:] < 1e-3), "Later elements must be very small"


@pytest.mark.parametrize("feature_dim,decay_ratio", [
    (3, 1e6), (4, 1e4), (2, 1000)
])
def test_decay_ratio_very_large_values(feature_dim, decay_ratio):
    result = importance_decay_by_ratio(feature_dim, decay_ratio)
    
    assert result[0] == 1.0, "First element must be 1.0"
    if feature_dim > 1:
        assert torch.all(result[1:] >= 1e3), "Later elements must be very large"


@pytest.mark.parametrize("feature_dim,min_importance", [
    (5, 0.3), (7, 0.1), (3, 0.5), (10, 0.05)
])
def test_min_importance_decreasing_sequence(feature_dim, min_importance):
    result = importance_decay_by_min(feature_dim, min_importance)
    
    assert torch.all(result[1:] <= result[:-1]), "Sequence must be monotonically decreasing"


@pytest.mark.parametrize("feature_dim,min_importance", [
    (5, 1.5), (3, 2.0), (4, 5.0)
])
def test_min_importance_increasing_sequence(feature_dim, min_importance):
    result = importance_decay_by_min(feature_dim, min_importance)
    
    assert torch.all(result[1:] >= result[:-1]), "Sequence must be monotonically increasing"


@pytest.mark.parametrize("feature_dim", [3, 5, 10])
def test_min_importance_constant_sequence(feature_dim):
    result = importance_decay_by_min(feature_dim, 1.0)
    
    assert torch.allclose(result, torch.ones(feature_dim)), "All elements must equal 1.0"


@pytest.mark.parametrize("feature_dim,min_importance", [
    (10, 1e-6), (8, 1e-8), (15, 1e-4)
])
def test_min_importance_very_small_last_element(feature_dim, min_importance):
    result = importance_decay_by_min(feature_dim, min_importance)
    
    assert torch.allclose(result[-1], torch.tensor(min_importance), rtol=1e-5), \
        "Last element must equal min_importance"


@pytest.mark.parametrize("param", [-0.5, "invalid", None, [0.5]])
def test_error_invalid_decay_ratio(param):
    with pytest.raises((ValueError, TypeError)):
        importance_decay_by_ratio(5, param)


@pytest.mark.parametrize("param", [-0.1, "invalid", None, [0.5]])
def test_error_invalid_min_importance(param):
    with pytest.raises((ValueError, TypeError)):
        importance_decay_by_min(5, param)


@pytest.mark.parametrize("func,feature_dim,param,expected", [
    (importance_decay_by_ratio, 3, 0.5, [1.0, 0.5, 0.25]),
    (importance_decay_by_ratio, 4, 0.1, [1.0, 0.1, 0.01, 0.001]),
    (importance_decay_by_ratio, 2, 2.0, [1.0, 2.0]),
    (importance_decay_by_ratio, 5, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0]),
    (importance_decay_by_min, 3, 0.25, [1.0, 0.5, 0.25]),
    (importance_decay_by_min, 2, 0.1, [1.0, 0.1]),
    (importance_decay_by_min, 4, 1.0, [1.0, 1.0, 1.0, 1.0]),
])
def test_known_value_regression(func, feature_dim, param, expected):
    result = func(feature_dim, param)
    expected_tensor = torch.tensor(expected, dtype=torch.float)
    
    assert torch.allclose(result, expected_tensor, rtol=1e-5), \
        f"Result {result.tolist()} must match expected {expected}"


@pytest.mark.parametrize("func", [importance_decay_by_ratio, importance_decay_by_min])
def test_single_feature_dimension(func):
    result = func(1, 0.5)
    
    assert result.shape == (1,), "Single feature must return tensor of shape (1,)"
    assert torch.allclose(result, torch.tensor([1.0])), "Single feature result must be [1.0]"


def test_two_feature_minimum_case():
    result = importance_decay_by_min(2, 0.25)
    
    assert result.shape == (2,), "Two features must return tensor of shape (2,)"
    assert torch.allclose(result[0], torch.tensor(1.0)), "First element must be 1.0"
    assert torch.allclose(result[1], torch.tensor(0.25)), "Second element must equal min_importance"


def test_exact_mathematical_formula():
    feature_dim = 5
    decay_ratio = 0.6
    result = importance_decay_by_ratio(feature_dim, decay_ratio)
    
    for i in range(feature_dim):
        expected_value = decay_ratio ** i
        assert torch.allclose(result[i], torch.tensor(expected_value), rtol=1e-6), \
            f"Element {i} must equal decay_ratio^{i} = {expected_value}"


def test_last_element_equals_min_importance():
    feature_dim = 7
    min_importance = 0.15
    result = importance_decay_by_min(feature_dim, min_importance)
    
    assert torch.allclose(result[-1], torch.tensor(min_importance), rtol=1e-5), \
        "Last element must equal min_importance parameter"


def test_equivalent_results_computed_decay_ratio():
    feature_dim = 6
    min_importance = 0.2
    
    result_min = importance_decay_by_min(feature_dim, min_importance)
    
    computed_decay_ratio = min_importance ** (1 / (feature_dim - 1))
    result_ratio = importance_decay_by_ratio(feature_dim, computed_decay_ratio)
    
    assert torch.allclose(result_min, result_ratio, rtol=1e-5), \
        "decay_by_min must produce same result as decay_by_ratio with computed decay_ratio"