import pytest
import torch
import torch.nn.functional as F

from models.transformer.components import Attention

from tests.models.transformer.mock_flash_availability import mock_flash_availability

# ================================ Test init ================================

@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("has_flash", [True, False])
def test_init_state_variables(use_flash, has_flash):
    """Test that the state variables are set correctly"""
    with mock_flash_availability(has_flash):
        attention = Attention(block_size=10, head_size=10, flash=use_flash)
        assert attention.dropout == 0.0
        assert attention.head_size == 10
        assert attention.flash == (use_flash and has_flash)
        assert attention.block_size == 10

def test_warning_emitted_when_flash_unavailable():
        """Test warning is emitted when flash requested but unavailable"""
        with mock_flash_availability(False):
            with pytest.warns(UserWarning, match="Flash Attention is not available"):
                attention = Attention(block_size=10, head_size=10, flash=True)
                assert attention.flash is False

# ================================ Test forward ================================

@pytest.fixture(
    params=[
        pytest.param("non_flash", id="non_flash_implementation"),
        pytest.param("flash", id="flash_implementation", 
                    marks=pytest.mark.skipif(not hasattr(F, 'scaled_dot_product_attention'),
                                           reason="Flash attention not available"))
    ]
)
def attention_instance(request):
    """Fixture that provides Attention instance configured for specific implementation"""
    flash = request.param == "flash"
    attention = Attention(block_size=16, head_size=32, dropout=0.0, flash=flash)
    if flash and not attention.flash:
        pytest.skip("Flash attention not actually available despite hasattr check")
    yield attention


@pytest.fixture
def sample_inputs():
    """Standard test inputs"""
    torch.manual_seed(42)
    q = torch.randn(2, 8, 32)
    k = torch.randn(2, 8, 32)
    v = torch.randn(2, 8, 32)
    return q, k, v


# ================================ Shape and Basic Functionality ================================

def test_output_shape(attention_instance, sample_inputs):
    """Test output has correct shape - runs on both implementations"""
    q, k, v = sample_inputs
    output = attention_instance(q, k, v)
    assert output.shape == v.shape

def test_different_sequence_lengths(attention_instance):
    """Test attention works with various sequence lengths"""
    for seq_len in [0, 1, 3, 8, 16]:  # Within block_size=16
        q = torch.randn(1, seq_len, 32)
        k = torch.randn(1, seq_len, 32)
        v = torch.randn(1, seq_len, 32)
        
        output = attention_instance(q, k, v)
        assert output.shape == (1, seq_len, 32)


def test_sequence_exceeds_block_size(attention_instance):
    """Test that sequences exceeding block_size raise appropriate error"""
    seq_len = attention_instance.block_size + 1
    # Sequence longer than block_size should raise ValueError
    q = torch.randn(1, seq_len, 32)  # seq_len=8 > block_size=4
    k = torch.randn(1, seq_len, 32)
    v = torch.randn(1, seq_len, 32)
    
    with pytest.raises(ValueError, match="Sequence length .* exceeds block size"):
        attention_instance(q, k, v)


def test_mismatched_input_dimensions(attention_instance):
    """Test error handling for invalid input dimensions"""
    
    # Test mismatched sequence lengths between q,k,v
    q = torch.randn(1, 5, 32)
    k = torch.randn(1, 3, 32)  # Different seq_len
    v = torch.randn(1, 5, 32)
    
    with pytest.raises((RuntimeError, ValueError)):
        attention_instance(q, k, v)
    
    # Test mismatched head dimensions
    q = torch.randn(1, 5, 32)
    k = torch.randn(1, 5, 16)  # Different head_size
    v = torch.randn(1, 5, 32)
    
    with pytest.raises((RuntimeError, ValueError)):
        attention_instance(q, k, v)



# ================================ Causal Behavior ================================

def test_causal_masking_no_future_leakage(attention_instance, sample_inputs):
    """Test that attention is causal - no future information leakage"""
    q, k, v = sample_inputs
    attention_instance.eval()  # Disable dropout for deterministic test
    
    # Get baseline output
    baseline_output = attention_instance(q, k, v)
    
    # Modify future tokens (last token)
    q_modified = q.clone()
    k_modified = k.clone()
    v_modified = v.clone()
    
    # Make large changes to the last token
    q_modified[:, -1, :] += 1000
    k_modified[:, -1, :] += 1000
    v_modified[:, -1, :] += 1000
    
    modified_output = attention_instance(q_modified, k_modified, v_modified)
    
    # All tokens except the last should be unchanged (causal property)
    seq_len = q.shape[1]
    if seq_len > 1:  # Only test if there are previous tokens
        assert torch.allclose(
            baseline_output[:, :-1, :], 
            modified_output[:, :-1, :], 
            atol=1e-6
        ), "Causal masking failed - future tokens affected past outputs"


# ================================ Training vs Eval Mode ================================

@pytest.fixture(
    params=[
        pytest.param("non_flash", id="non_flash_dropout"),
        pytest.param("flash", id="flash_dropout", 
                    marks=pytest.mark.skipif(not hasattr(F, 'scaled_dot_product_attention'),
                                           reason="Flash attention not available"))
    ]
)
def attention_instance_with_dropout(request):
    """Fixture that provides Attention instance with dropout for testing training modes"""
    flash = request.param == "flash"
    attention = Attention(block_size=16, head_size=32, dropout=0.3, flash=flash)
    if flash and not attention.flash:
        pytest.skip("Flash attention not actually available despite hasattr check")
    yield attention

def test_training_vs_eval_mode_with_dropout(attention_instance_with_dropout, sample_inputs):
    """Test dropout behavior in both implementations"""
    attention = attention_instance_with_dropout
    q, k, v = sample_inputs

    attention.eval()
    torch.manual_seed(123)
    out1 = attention(q, k, v)
    torch.manual_seed(123)
    out2 = attention(q, k, v)
    assert torch.allclose(out1, out2), "Eval mode should be deterministic"
    
    attention.train()
    torch.manual_seed(123)
    train_out1 = attention(q, k, v)
    torch.manual_seed(456)  # Different seed
    train_out2 = attention(q, k, v)
    assert not torch.allclose(train_out1, train_out2, atol=1e-6), \
        "Training mode should have dropout variation"


# ================================ Gradient Flow ================================

def test_gradient_flow(attention_instance, sample_inputs):
    """Test that gradients flow properly through attention"""
    q, k, v = sample_inputs
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    
    output = attention_instance(q, k, v)
    loss = output.sum()
    loss.backward()
    
    # All inputs should have gradients
    assert q.grad is not None, "Query should have gradients"
    assert k.grad is not None, "Key should have gradients"
    assert v.grad is not None, "Value should have gradients"
    
    # Gradients should be non-zero (attention is not a no-op)
    assert not torch.allclose(q.grad, torch.zeros_like(q.grad)), "Query gradients should be non-zero"
    assert not torch.allclose(k.grad, torch.zeros_like(k.grad)), "Key gradients should be non-zero"
    assert not torch.allclose(v.grad, torch.zeros_like(v.grad)), "Value gradients should be non-zero"

# ================================ Equivalence Testing (when both implementations available) ================================

@pytest.mark.skipif(not hasattr(F, 'scaled_dot_product_attention'),
                   reason="Flash attention not available for equivalence test")
def test_flash_vs_nonflash_equivalence(sample_inputs):
    """Test that flash and non-flash implementations produce equivalent results"""
    q, k, v = sample_inputs
    
    attention_flash = Attention(block_size=16, head_size=32, dropout=0.0, flash=True)
    attention_nonflash = Attention(block_size=16, head_size=32, dropout=0.0, flash=False)
    
    attention_flash.eval()
    attention_nonflash.eval()
    
    output_flash = attention_flash(q, k, v)
    output_nonflash = attention_nonflash(q, k, v)
    
    assert torch.allclose(output_flash, output_nonflash, atol=1e-5, rtol=1e-5), \
        "Flash and non-flash implementations should produce equivalent results"

@pytest.mark.skipif(not hasattr(F, 'scaled_dot_product_attention'),
                   reason="Flash attention not available for equivalence test")
def test_equivalence_gradient_computation(sample_inputs):
    """Test that gradients are equivalent between implementations"""
    q, k, v = sample_inputs
    
    # Create separate tensors for each implementation to avoid interference
    q_flash = q.clone().requires_grad_(True)
    k_flash = k.clone().requires_grad_(True)
    v_flash = v.clone().requires_grad_(True)
    
    q_nonflash = q.clone().requires_grad_(True)
    k_nonflash = k.clone().requires_grad_(True)
    v_nonflash = v.clone().requires_grad_(True)
    
    attention_flash = Attention(block_size=16, head_size=32, dropout=0.0, flash=True)
    attention_nonflash = Attention(block_size=16, head_size=32, dropout=0.0, flash=False)
    attention_flash.eval()
    attention_nonflash.eval()
    
    output_flash = attention_flash(q_flash, k_flash, v_flash)
    output_nonflash = attention_nonflash(q_nonflash, k_nonflash, v_nonflash)
    
    loss_flash = output_flash.sum()
    loss_nonflash = output_nonflash.sum()
    loss_flash.backward()
    loss_nonflash.backward()
    
    assert torch.allclose(q_flash.grad, q_nonflash.grad, atol=1e-5, rtol=1e-5), \
        "Query gradients should be equivalent"
    assert torch.allclose(k_flash.grad, k_nonflash.grad, atol=1e-5, rtol=1e-5), \
        "Key gradients should be equivalent"
    assert torch.allclose(v_flash.grad, v_nonflash.grad, atol=1e-5, rtol=1e-5), \
        "Value gradients should be equivalent"