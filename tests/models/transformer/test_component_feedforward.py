import torch
import pytest
from src.models.transformer.components import FeedForward

# ================================ Test init ================================

def test_init():
    """Test that FeedForward initializes without crashing"""
    ff = FeedForward(embed_dim=128, hidden_multiplier=4, dropout=0.1)
    assert isinstance(ff, torch.nn.Module)

# ================================ Test forward ================================

@pytest.fixture
def feedforward():
    return FeedForward(embed_dim=128, hidden_multiplier=4, dropout=0.0)

@pytest.fixture 
def sample_input():
    return torch.randn(2, 10, 128)

def test_forward_shape_preservation(feedforward, sample_input):
    """Test core contract: embed_dim -> embed_dim"""
    output = feedforward(sample_input)
    assert output.shape == sample_input.shape  # Should preserve shape

def test_forward_basic_functionality(feedforward, sample_input):
    """Test basic functionality: doesn't crash, produces finite outputs"""
    output = feedforward(sample_input)
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()

def test_forward_training_vs_eval():
    """Test dropout behavior in training vs eval"""
    ff = FeedForward(embed_dim=64, hidden_multiplier=2, dropout=0.3)
    input_tensor = torch.randn(1, 5, 64)
    
    # Eval should be deterministic
    ff.eval()
    torch.manual_seed(42)
    out1 = ff(input_tensor)
    torch.manual_seed(42) 
    out2 = ff(input_tensor)
    assert torch.allclose(out1, out2)
    
    # Training should vary due to dropout
    ff.train()
    torch.manual_seed(42)
    train_out1 = ff(input_tensor)
    torch.manual_seed(123)
    train_out2 = ff(input_tensor)
    assert not torch.allclose(train_out1, train_out2, atol=1e-6)

def test_forward_gradient_flow(feedforward, sample_input):
    """Test that gradients flow through the network"""
    sample_input.requires_grad_(True)
    output = feedforward(sample_input)
    loss = output.sum()
    loss.backward()
    
    assert sample_input.grad is not None
    assert not torch.allclose(sample_input.grad, torch.zeros_like(sample_input.grad))