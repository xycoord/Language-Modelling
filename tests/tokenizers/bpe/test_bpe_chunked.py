import pytest
from tokenizers import ChunkedBPETokenizer, OptimizedBPETokenizer
from tokenizers.bpe.utils import GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN
import regex as re

@pytest.fixture(params=[
    ChunkedBPETokenizer,
    OptimizedBPETokenizer
])
def chunked_tokenizer_class(request):
    """Parameterized fixture for tokenizers that support split patterns."""
    return request.param


def test_chunked_tokenizer_accepts_split_pattern(chunked_tokenizer_class):
    """Test that chunked tokenizers accept split_pattern parameter."""
    # Should work with default (no parameters)
    default_tokenizer = chunked_tokenizer_class()
    assert default_tokenizer.vocab_size == 256
    
    # Should work with custom split pattern
    custom_tokenizer = chunked_tokenizer_class(split_pattern=r'\w+|\W+')
    assert custom_tokenizer.vocab_size == 256


def test_custom_split_pattern_round_trip(chunked_tokenizer_class):
    """Test that custom split patterns maintain round-trip consistency."""
    test_text = "Hello, world! How are you?"
    
    tokenizer = chunked_tokenizer_class(split_pattern=r'\w+|\W+')
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    assert decoded == test_text


def test_split_pattern_with_training(chunked_tokenizer_class):
    """Test that custom split patterns work correctly with training."""
    tokenizer = chunked_tokenizer_class(split_pattern=r'\w+|\W+')
    
    # Train with repetitive text
    training_text = "hello world! hello world! hello world! " * 50
    tokenizer.train(training_text, target_vocab_size=300)
    
    # Should still handle round-trip correctly after training
    test_text = "hello world!"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    assert decoded == test_text
    
    # Training should have increased vocab size
    assert tokenizer.vocab_size > 256


def test_predefined_split_patterns(chunked_tokenizer_class):
    """Test that predefined GPT2/GPT4 patterns work correctly."""
    text = "Hello! How's it going?"
    
    gpt2_tokenizer = chunked_tokenizer_class(split_pattern=GPT2_SPLIT_PATTERN)
    gpt4_tokenizer = chunked_tokenizer_class(split_pattern=GPT4_SPLIT_PATTERN)
    
    # Both should handle round-trip correctly
    for tokenizer in [gpt2_tokenizer, gpt4_tokenizer]:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text


def test_invalid_regex_patterns(chunked_tokenizer_class):
    """Test behavior with invalid regex patterns."""
    invalid_patterns = [
        r'[',           # Unclosed character class
        r'(?P<',        # Incomplete named group
        r'*',           # Invalid quantifier
        r'(?',          # Incomplete group
    ]
    
    for pattern in invalid_patterns:
        with pytest.raises((ValueError, re.error)):
            chunked_tokenizer_class(split_pattern=pattern)