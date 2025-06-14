from models.transformer.components import AttentionHead
import torch
import pytest

@pytest.fixture
def config_dict():
    return {
        "embed_dim": 128,
        "block_size": 128,
        "head_size": 128,
        "dropout": 0.1,
        "flash": False
    }

# ================================ Test init ================================

def test_init_state_variables(config_dict):
    """Test that documented state variables are set correctly"""
    head = AttentionHead(**config_dict)
    assert head.head_size == config_dict["head_size"]
    assert head.block_size == config_dict["block_size"]

# ================================ Test forward ================================

@pytest.fixture
def attention_head(config_dict):
    return AttentionHead(**config_dict)

@pytest.fixture
def sample_inputs(config_dict):
    """Sample attention head inputs for testing"""
    return torch.randn(1, config_dict["block_size"], config_dict["embed_dim"])


def test_forward_shape_transformation(attention_head, config_dict):
    """Test core contract: transforms embed_dim to head_size"""
    input_tensor = torch.randn(2, 10, config_dict["embed_dim"])
    output = attention_head(input_tensor)
    
    assert output.shape == (2, 10, config_dict["head_size"])


def test_forward_deterministic_given_weights(attention_head, sample_inputs):
    """Test that same input produces same output (weights are fixed after init)"""
    attention_head.eval()
    output1 = attention_head(sample_inputs)
    output2 = attention_head(sample_inputs)
    assert torch.allclose(output1, output2), "Should be deterministic for fixed weights"


def test_forward_different_sequence_lengths(attention_head, config_dict):
    """Test that different input sequence lengths work correctly"""
    for seq_len in [0, 1, 5, 32, 64]:  # Including max block_size
        input_tensor = torch.randn(1, seq_len, config_dict["embed_dim"])
        output = attention_head(input_tensor)
        assert output.shape == (1, seq_len, config_dict["head_size"])

def test_forward_sequence_exceeds_block_size(attention_head, config_dict):
    """Test that sequence length exceeds block size raises error"""
    input_tensor = torch.randn(1, config_dict["block_size"] + 1, config_dict["embed_dim"])
    with pytest.raises(ValueError, match="Sequence length .* exceeds block size"):
        attention_head(input_tensor)


def test_forward_causal_behavior_preserved(attention_head, sample_inputs):
    """Test that causal masking works through the projections"""
    attention_head.eval()
    
    baseline_output = attention_head(sample_inputs)
    
    # Modify future positions - this should not affect past outputs
    modified_input = sample_inputs.clone()
    modified_input[:, -1, :] += 1000  # Large change to last token
    
    modified_output = attention_head(modified_input)
    
    # Past positions should be unaffected (causal property preserved)
    assert torch.allclose(
        baseline_output[:, :-1, :], 
        modified_output[:, :-1, :], 
        atol=1e-6
    ), "Causal masking should work through projections"


def test_forward_gradient_flow(attention_head, sample_inputs):
    """Test that gradients flow through all components"""
    sample_inputs.requires_grad = True
    output = attention_head(sample_inputs)
    loss = output.sum()
    loss.backward()
    assert sample_inputs.grad is not None, "Input should have gradients"
    assert not torch.allclose(sample_inputs.grad, torch.zeros_like(sample_inputs.grad)), "Gradients should not be zero"


def test_forward_numerical_stability(attention_head, sample_inputs):
    """Test that outputs are numerically stable"""
    output = attention_head(sample_inputs)
    assert torch.isfinite(output).all(), "Output should be finite"
    assert not torch.isnan(output).any(), "Output should not contain NaN"


def test_forward_training_vs_eval_modes(attention_head, sample_inputs):
    """Test that training/eval modes work correctly"""
    attention_head.eval()
    torch.manual_seed(42)
    out1 = attention_head(sample_inputs)
    torch.manual_seed(42)
    out2 = attention_head(sample_inputs)
    assert torch.allclose(out1, out2), "Eval mode should be deterministic"
    
    attention_head.train()
    torch.manual_seed(42)
    train_out1 = attention_head(sample_inputs)
    torch.manual_seed(123)
    train_out2 = attention_head(sample_inputs)
    assert not torch.allclose(train_out1, train_out2, atol=1e-6), \
        "Training mode should have dropout variation"