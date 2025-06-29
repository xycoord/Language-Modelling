from src.lm_models.transformer.components import MultiHeadAttention, ParallelMultiHeadAttention
from src.lm_models.transformer.config import TransformerConfig
import pytest
import torch

@pytest.fixture(params=[
    MultiHeadAttention,
    ParallelMultiHeadAttention
])
def multihead_attention_class(request):
    """Parameterized fixture providing all multihead attention classes."""
    return request.param

@pytest.fixture
def config():
    """Parameterized fixture providing all multihead attention classes."""
    return TransformerConfig(
        embed_dim=128,
        vocab_size=256,
        block_size=128,
        head_size=128,
        num_heads=4,
        dropout=0.1,
        flash=False
    )

# ================================ Test init ================================

def test_init(multihead_attention_class, config):
    """Test that the multihead attention class is initialized correctly"""
    multihead_attention = multihead_attention_class(config)
    assert isinstance(multihead_attention, torch.nn.Module)

# ================================ Test forward ================================

@pytest.fixture
def multihead_attention(multihead_attention_class, config):
    """Parameterized fixture providing all multihead attention classes."""
    return multihead_attention_class(config)

@pytest.fixture
def sample_inputs(config):
    """Sample attention head inputs for testing"""
    return torch.randn(1, config.block_size, config.embed_dim)


def test_forward_shape_transformation(multihead_attention, sample_inputs, config):
    """Test that the multihead attention class transforms the input shape correctly"""
    output = multihead_attention(sample_inputs)
    assert output.shape == (1, config.block_size, config.head_size)


def test_forward_deterministic_given_weights(multihead_attention, sample_inputs):
    """Test that the multihead attention class is deterministic given the same weights"""
    multihead_attention.eval()
    output1 = multihead_attention(sample_inputs)
    output2 = multihead_attention(sample_inputs)
    assert torch.allclose(output1, output2)


def test_forward_different_sequence_lengths(multihead_attention, config):
    """Test that the multihead attention class works with different sequence lengths"""
    for seq_len in [0, 1, 5, 32, 64]:
        input = torch.randn(1, seq_len, config.embed_dim)
        output = multihead_attention(input)
        assert output.shape == (1, seq_len, config.head_size)

def test_forward_sequence_exceeds_block_size(multihead_attention, config):
    """Test that sequence length exceeds block size raises error"""
    input = torch.randn(1, config.block_size + 1, 128)
    with pytest.raises(ValueError, match="Sequence length .* exceeds block size"):
        multihead_attention(input)


def test_forward_causal_behavior_preserved(multihead_attention, sample_inputs):
    """Test that causal masking works through the projections"""
    multihead_attention.eval()
    
    baseline_output = multihead_attention(sample_inputs)
    
    # Modify future positions - this should not affect past outputs
    modified_input = sample_inputs.clone()
    modified_input[:, -1, :] += 1000  # Large change to last token
    modified_output = multihead_attention(modified_input)
    
    # Past positions should be unaffected (causal property preserved)
    assert torch.allclose(
        baseline_output[:, :-1, :], 
        modified_output[:, :-1, :], 
        atol=1e-6
    ), "Causal masking should work through projections"


def test_forward_gradient_flow(multihead_attention, sample_inputs):
    """Test that the multihead attention class has gradient flow"""
    multihead_attention.train()
    sample_inputs.requires_grad = True
    output = multihead_attention(sample_inputs)
    loss = output.sum()
    loss.backward()
    assert sample_inputs.grad is not None
    assert not torch.allclose(sample_inputs.grad, torch.zeros_like(sample_inputs.grad))


def test_forward_numerical_stability(multihead_attention, sample_inputs):
    """Test that the multihead attention class is numerically stable"""
    multihead_attention.eval()
    output = multihead_attention(sample_inputs)
    assert torch.isfinite(output).all(), "Output should be finite"
    assert not torch.isnan(output).any(), "Output should not contain NaN"


def test_forward_training_vs_eval_modes(multihead_attention, sample_inputs):
    """Test that the multihead attention class works in training and eval modes"""
    multihead_attention.eval()
    torch.manual_seed(42)
    eval_out1 = multihead_attention(sample_inputs)
    torch.manual_seed(42)
    eval_out2 = multihead_attention(sample_inputs)
    assert torch.allclose(eval_out1, eval_out2), "Eval mode should be deterministic"
    
    multihead_attention.train()
    torch.manual_seed(42)
    train_out1 = multihead_attention(sample_inputs)
    torch.manual_seed(123)
    train_out2 = multihead_attention(sample_inputs)
    assert not torch.allclose(train_out1, train_out2, atol=1e-6), "Training mode should have dropout variation"