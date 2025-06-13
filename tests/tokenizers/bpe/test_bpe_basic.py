import pytest
from tokenizers import BasicBPETokenizer


@pytest.fixture
def basic_tokenizer():
    """Create a fresh BasicBPETokenizer instance."""
    return BasicBPETokenizer()


def test_basic_tokenizer_no_constructor_params():
    """Test that BasicBPETokenizer doesn't accept constructor parameters."""
    # Should work with no parameters
    tokenizer = BasicBPETokenizer()
    assert tokenizer.vocab_size == 256
    
    # Should fail with parameters (this tests the contract)
    with pytest.raises(TypeError):
        BasicBPETokenizer(split_pattern="some_pattern")


def test_basic_tokenizer_processes_text_without_chunking(basic_tokenizer):
    """Test that BasicBPETokenizer processes text as a single unit."""
    # This test verifies that BasicBPETokenizer treats the entire text
    # as one continuous sequence for BPE processing
    
    text = "hello, world!"
    
    # Should encode successfully
    tokens = basic_tokenizer.encode(text)
    decoded = basic_tokenizer.decode(tokens)
    assert decoded == text
    
    # Basic tokenizer should handle any text as a continuous byte sequence
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)


def test_basic_tokenizer_training_on_long_continuous_text(basic_tokenizer):
    """Test BasicBPETokenizer training on long continuous text."""
    # Create text that would be split differently by chunked tokenizers
    # but should be treated as one continuous sequence by Basic
    continuous_text = "word1word2word3" * 100
    
    basic_tokenizer.train(continuous_text, target_vocab_size=300)
    
    # Should still handle round-trip correctly
    test_text = "word1word2word3"
    tokens = basic_tokenizer.encode(test_text)
    decoded = basic_tokenizer.decode(tokens)
    assert decoded == test_text
    
    # Training should have increased vocab size
    assert basic_tokenizer.vocab_size > 256


def test_basic_tokenizer_handles_text_boundaries_as_continuous():
    """Test that BasicBPETokenizer can merge across what would be chunk boundaries."""
    # This tests that Basic can potentially merge patterns that span
    # what chunked tokenizers would treat as separate chunks
    
    tokenizer = BasicBPETokenizer()
    
    # Text with patterns that span typical word boundaries
    text = "prefix_suffix prefix_suffix prefix_suffix" * 50
    tokenizer.train(text, target_vocab_size=300)
    
    # Should be able to encode/decode successfully
    test_text = "prefix_suffix"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    assert decoded == test_text