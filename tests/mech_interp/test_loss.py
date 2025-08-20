import pytest
import torch
from mech_interp.loss import weighted_mse_loss

def test_basic_computation_verification():
    output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    weight = torch.tensor([2.0, 0.5])
    
    # Manual calculation: mean of all weighted squared errors
    # (target - output)^2 = [[1.0, 1.0], [1.0, 1.0]]
    # weight * (target - output)^2 = [[2.0, 0.5], [2.0, 0.5]]
    # mean of all elements = (2.0 + 0.5 + 2.0 + 0.5) / 4 = 1.25
    expected = 1.25
    
    result = weighted_mse_loss(output, target, weight)
    
    assert torch.allclose(result, torch.tensor(expected)), "Basic computation should match manual calculation"


def test_unweighted_mse_equivalence():
    output = torch.randn(3, 4)
    target = torch.randn(3, 4)
    weight = torch.ones(4)
    
    weighted_result = weighted_mse_loss(output, target, weight)
    unweighted_result = torch.mean((target - output) ** 2)
    
    assert torch.allclose(weighted_result, unweighted_result), "Unit weights should equal standard MSE"


def test_weight_scaling_verification():
    output = torch.randn(2, 3)
    target = torch.randn(2, 3)
    weight = torch.tensor([1.0, 2.0, 3.0])
    doubled_weight = torch.tensor([2.0, 4.0, 6.0])
    
    loss1 = weighted_mse_loss(output, target, weight)
    loss2 = weighted_mse_loss(output, target, doubled_weight)
    
    assert torch.allclose(loss2, 2 * loss1), "Doubling weights should double the loss"


def test_perfect_prediction_case():
    output = torch.randn(5, 3)
    target = output.clone()
    weight = torch.randn(3).abs()
    
    result = weighted_mse_loss(output, target, weight)
    
    assert torch.allclose(result, torch.tensor(0.0)), "Perfect prediction should give zero loss"


def test_zero_weights_case():
    output = torch.randn(4, 2)
    target = torch.randn(4, 2)
    weight = torch.zeros(2)
    
    result = weighted_mse_loss(output, target, weight)
    
    assert torch.allclose(result, torch.tensor(0.0)), "Zero weights should give zero loss"

def test_weight_broadcasting_verification():
    output = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target = torch.zeros_like(output)
    weight = torch.tensor([2.0, 0.5])
    
    # (target - output)^2 = [[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]]
    # weight * (target - output)^2 = [[2.0, 2.0], [18.0, 8.0], [50.0, 18.0]]
    # mean of all elements = (2.0 + 2.0 + 18.0 + 8.0 + 50.0 + 18.0) / 6 = 98.0 / 6
    expected = 98.0 / 6
    
    result = weighted_mse_loss(output, target, weight)
    
    assert torch.allclose(result, torch.tensor(expected), atol=1e-4), "Weights should broadcast across batch dimension"


def test_scalar_weight_handling():
    output = torch.randn(2, 3)
    target = torch.randn(2, 3)
    scalar_weight = torch.tensor(2.0)  # Wrong shape
    
    with pytest.raises(RuntimeError):
        weighted_mse_loss(output, target, scalar_weight)


def test_device_consistency():
    output = torch.randn(2, 3)
    target = torch.randn(2, 3)
    weight = torch.randn(3)
    
    if torch.cuda.is_available():
        output_cuda = output.cuda()
        
        with pytest.raises(RuntimeError):
            weighted_mse_loss(output_cuda, target, weight)


def test_requires_grad_preservation():
    output = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 3)
    weight = torch.randn(3)
    
    result = weighted_mse_loss(output, target, weight)
    
    assert result.requires_grad, "Result should require gradients when output does"


def test_scalar_output_verification():
    output = torch.randn(10, 5)
    target = torch.randn(10, 5)
    weight = torch.randn(5)
    
    result = weighted_mse_loss(output, target, weight)
    
    assert result.dim() == 0, "Output should be a scalar tensor"


@pytest.mark.parametrize("output_shape,target_shape,weight_shape,error_type", [
    ((2, 3), (2, 4), (3,), RuntimeError),  # Target shape mismatch
    ((3, 2), (2, 2), (2,), RuntimeError),  # Output shape mismatch  
    ((2, 3), (2, 3), (4,), RuntimeError),  # Weight dimension mismatch
    ((2, 3), (2, 3), (2,), RuntimeError),  # Weight too small
])
def test_input_validation_errors(output_shape, target_shape, weight_shape, error_type):
    output = torch.randn(*output_shape)
    target = torch.randn(*target_shape)
    weight = torch.randn(*weight_shape)
    
    with pytest.raises(error_type):
        weighted_mse_loss(output, target, weight)


@pytest.mark.parametrize("weight_values,expected_sign", [
    ([1.0, 2.0, 3.0], "positive"),     # All positive
    ([0.0, 1.0, 2.0], "non_negative"), # Mixed zero/positive
    ([-1.0, -2.0, -3.0], "negative"),  # All negative
    ([-1.0, 0.0, 2.0], "mixed"),       # Mixed pos/neg/zero
])
def test_weight_value_behaviour(weight_values, expected_sign):
    output = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target = torch.zeros_like(output)
    weight = torch.tensor(weight_values)
    
    result = weighted_mse_loss(output, target, weight)
    
    if expected_sign == "positive":
        assert result > 0, "Positive weights should give positive loss for non-zero errors"
    elif expected_sign == "non_negative":
        assert result >= 0, "Non-negative weights should give non-negative loss"
    elif expected_sign == "negative":
        assert result < 0, "Negative weights should give negative loss for positive errors"
    # Mixed case: no specific assertion about sign


@pytest.mark.parametrize("weight_config,test_type", [
    ([1.0, 2.0, 3.0], "uniform_scaling"),
    ([2.0, 4.0, 6.0], "double_scaling"),
    ([1.0, 0.0, 2.0], "zero_masking"),
])
def test_gradient_behaviour(weight_config, test_type):
    output = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 3)
    weight = torch.tensor(weight_config)
    
    loss = weighted_mse_loss(output, target, weight)
    loss.backward()
    
    assert output.grad is not None, "Gradients should be computed for output"
    
    if test_type == "double_scaling":
        # Compare with half weights
        output2 = output.detach().clone().requires_grad_(True)
        weight2 = torch.tensor([1.0, 2.0, 3.0])
        loss2 = weighted_mse_loss(output2, target, weight2)
        loss2.backward()
        
        assert torch.allclose(output.grad, 2 * output2.grad), "Gradients should scale with weights"
    
    elif test_type == "zero_masking":
        # Zero weights should give zero gradients for that feature
        assert torch.allclose(output.grad[:, 1], torch.zeros_like(output.grad[:, 1])), "Zero weights should zero out gradients for that feature"