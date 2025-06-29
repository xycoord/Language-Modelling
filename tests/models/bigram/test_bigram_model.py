import pytest
import torch

from src.lm_models.bigram.model import BigramLanguageModel
from src.lm_models.bigram.config import BigramConfig


@pytest.fixture
def small_config():
    """Fixture providing a small config for testing"""
    return BigramConfig(vocab_size=10)


@pytest.fixture
def standard_config():
    """Fixture providing a standard config for testing"""
    return BigramConfig(vocab_size=100)


@pytest.fixture
def small_model(small_config):
    """Fixture providing a small model instance for testing"""
    return BigramLanguageModel(small_config)


@pytest.fixture
def standard_model(standard_config):
    """Fixture providing a standard model instance for testing"""
    return BigramLanguageModel(standard_config)


# Core model contract tests
def test_forward_returns_correct_vocab_dimension(standard_model, standard_config):
    """Forward should return logits with vocab_size as final dimension"""
    context = torch.randint(0, standard_config.vocab_size, (2, 5))
    logits = standard_model(context)
    
    assert logits.shape[-1] == standard_config.vocab_size


def test_forward_returns_same_sequence_length(standard_model):
    """Forward should return logits with same sequence length as input"""
    for seq_len in [1, 5, 10]:
        context = torch.randint(0, 100, (1, seq_len))
        logits = standard_model(context)
        assert logits.shape[1] == seq_len


def test_forward_with_different_vocab_sizes():
    """Forward should work correctly with different vocab sizes"""
    for vocab_size in [5, 50, 500]:
        config = BigramConfig(vocab_size=vocab_size)
        model = BigramLanguageModel(config)
        context = torch.randint(0, vocab_size, (1, 3))
        
        logits = model(context)
        assert logits.shape[-1] == vocab_size


def test_generate_extends_by_exact_token_count(standard_model, standard_config):
    """Generate should extend sequence by exactly max_new_tokens"""
    context = torch.randint(0, standard_config.vocab_size, (1, 4))
    original_length = context.shape[1]
    
    for new_tokens in [0, 1, 5, 10]:
        result = standard_model.generate(context, new_tokens)
        assert result.shape[1] == original_length + new_tokens


def test_generate_preserves_input_context(standard_model):
    """Generate should not modify the original context portion"""
    context = torch.randint(0, 100, (2, 6))
    result = standard_model.generate(context, max_new_tokens=4)
    
    assert torch.equal(result[:, :6], context), "Result should be the same as the original context"


def test_generate_produces_tokens_within_vocabulary(small_model):
    """Generate should only produce tokens within [0, vocab_size)"""
    vocab_size = 10
    context = torch.randint(0, vocab_size, (1, 3))
    
    result = small_model.generate(context, max_new_tokens=20)
    
    # Check all tokens are valid
    assert (result >= 0).all()
    assert (result < vocab_size).all()


def test_generate_uses_only_last_token_for_prediction(standard_model):
    """Generate should base next token prediction only on the last token (bigram property)"""

    context1 = torch.tensor([[10, 20, 30, 40]])
    context2 = torch.tensor([[99, 88, 77, 40]])
    
    torch.manual_seed(42)
    result1 = standard_model.generate(context1, max_new_tokens=1)
    torch.manual_seed(42)
    result2 = standard_model.generate(context2, max_new_tokens=1)
    
    assert result1[0, -1] == result2[0, -1], "The new token should be the same since last context token is the same"


def test_generate_with_zero_tokens_is_identity(standard_model, standard_config):
    """Generate with max_new_tokens=0 should return input unchanged"""
    context = torch.randint(0, standard_config.vocab_size, (3, 7))
    
    result = standard_model.generate(context, max_new_tokens=0)
    
    assert torch.equal(result, context), "Result should be the same as the original context"


def test_model_works_with_single_token_context(standard_model, standard_config):
    """Model should handle contexts with just one token"""
    context = torch.randint(0, standard_config.vocab_size, (1, 1))
    
    logits = standard_model(context)
    assert logits.shape == (1, 1, standard_config.vocab_size)
    
    result = standard_model.generate(context, max_new_tokens=3)
    assert result.shape == (1, 4)
    assert torch.equal(result[:, :1], context)


def test_model_handles_batch_dimension_correctly(standard_model, standard_config):
    """Model should process each item in batch independently"""
    batch_size = 4
    context = torch.randint(0, standard_config.vocab_size, (batch_size, 5))
    
    logits = standard_model(context)
    assert logits.shape[0] == batch_size
    
    result = standard_model.generate(context, max_new_tokens=3)
    assert result.shape[0] == batch_size
    
    for i in range(batch_size):
        assert torch.equal(result[i, :5], context[i]), "Each batch item should have its original context preserved"


def test_model_with_minimal_vocabulary():
    """Model should work correctly with vocabulary size of 1"""
    config = BigramConfig(vocab_size=1)
    model = BigramLanguageModel(config)
    
    # Only token 0 is valid
    context = torch.zeros((1, 3), dtype=torch.long)
    
    logits = model(context)
    assert logits.shape == (1, 3, 1)
    
    result = model.generate(context, max_new_tokens=5)
    assert result.shape == (1, 8)
    assert (result == 0).all(), "All tokens must be 0 since it's the only valid token"


def test_bigram_property_with_repeated_contexts():
    """Model should produce same logits for contexts ending with same token"""
    config = BigramConfig(vocab_size=50)
    model = BigramLanguageModel(config)
    
    context1 = torch.tensor([[1, 2, 3, 25]])
    context2 = torch.tensor([[10, 15, 7, 25]])
    context3 = torch.tensor([[25]])
    
    logits1 = model(context1)
    logits2 = model(context2)
    logits3 = model(context3)
    
    assert torch.allclose(logits1[0, -1, :], logits2[0, -1, :])
    assert torch.allclose(logits1[0, -1, :], logits3[0, -1, :])


def test_model_config_integration():
    """Model should respect the vocab_size from its config"""
    test_sizes = [5, 25, 100, 1000]
    
    for vocab_size in test_sizes:
        config = BigramConfig(vocab_size=vocab_size)
        model = BigramLanguageModel(config)
        
        context = torch.randint(0, vocab_size, (1, 3))
        logits = model(context)
        
        assert logits.shape[-1] == vocab_size, "Output dimension should match config"

def test_config_rejects_invalid_vocab_size():
    """BigramConfig should reject invalid vocab_size values"""
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        BigramConfig(vocab_size=0)
    
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        BigramConfig(vocab_size=-1)
    
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        BigramConfig(vocab_size=-100)

    config = BigramConfig(vocab_size=10)
    assert config.vocab_size == 10

# ================================ Test device ================================

@pytest.fixture(
    params=[
        pytest.param("cpu", id="cpu_device"),
        pytest.param("cuda", id="cuda_device", 
                    marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                           reason="CUDA not available"))
    ]
)
def device(request):
    """Fixture providing available devices for testing"""
    return torch.device(request.param)

def test_device(standard_model, device):
    standard_model = standard_model.to(device)
    assert standard_model.device.type == device.type