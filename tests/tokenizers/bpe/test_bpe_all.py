import pytest
from tokenizers import BasicBPETokenizer, ChunkedBPETokenizer, OptimizedBPETokenizer
import unicodedata


@pytest.fixture(params=[
    BasicBPETokenizer,
    ChunkedBPETokenizer,
    OptimizedBPETokenizer
])
def tokenizer_class(request):
    """Parameterized fixture providing all tokenizer classes."""
    return request.param


@pytest.fixture
def fresh_tokenizer(tokenizer_class):
    """Create a fresh tokenizer instance for each test."""
    return tokenizer_class()


@pytest.fixture
def trained_tokenizer(tokenizer_class):
    """Create a tokenizer trained on sample text."""
    tokenizer = tokenizer_class()
    training_text = "hello world! this is a test. hello world again and again."
    tokenizer.train(training_text, target_vocab_size=300)
    return tokenizer


def test_fresh_tokenizer_initializes_correctly(fresh_tokenizer):
    """Test that a fresh tokenizer starts with base vocabulary."""
    # Should start with 256 UTF-8 byte tokens
    assert fresh_tokenizer.vocab_size == 256


def test_round_trip_consistency_simple_text(fresh_tokenizer):
    """Test encode->decode round trip preserves text."""
    test_texts = [
        "hello",
        "world",
        "hello world",
        "test123",
        "",  # empty string
        "a",  # single character
    ]
    
    for text in test_texts:
        tokens = fresh_tokenizer.encode(text)
        decoded = fresh_tokenizer.decode(tokens)
        assert decoded == text, f"Round trip failed for: '{text}'"


def test_round_trip_consistency_utf8_text(fresh_tokenizer):
    """Test round trip with non-ASCII UTF-8 characters."""
    test_texts = [
        "café",
        "naïve", 
        "résumé",
        "🙂",  # emoji
        "你好",  # Chinese
        "Γεια σας",  # Greek
        "العربية",  # Arabic
    ]
    
    for text in test_texts:
        tokens = fresh_tokenizer.encode(text)
        decoded = fresh_tokenizer.decode(tokens)
        assert decoded == text, f"UTF-8 round trip failed for: '{text}'"


def test_round_trip_consistency_special_chars(fresh_tokenizer):
    """Test round trip with special characters and whitespace."""
    test_texts = [
        "  ",  # spaces
        "\n",  # newline
        "\t",  # tab
        "\r\n",  # CRLF
        "line1\nline2",
        "word1  word2",  # multiple spaces
        "!@#$%^&*()",  # punctuation
        "'quotes'",
        '"double quotes"',
    ]
    
    for text in test_texts:
        tokens = fresh_tokenizer.encode(text)
        decoded = fresh_tokenizer.decode(tokens)
        assert decoded == text, f"Special char round trip failed for: '{text}'"


def test_encode_returns_list_of_integers(fresh_tokenizer):
    """Test that encode returns a list of integer tokens."""
    tokens = fresh_tokenizer.encode("hello")
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)
    assert all(token >= 0 for token in tokens)


def test_decode_handles_empty_token_list(fresh_tokenizer):
    """Test that decode handles empty token list."""
    result = fresh_tokenizer.decode([])
    assert result == ""


def test_training_increases_vocab_size(fresh_tokenizer):
    """Test that training increases vocabulary size."""
    initial_size = fresh_tokenizer.vocab_size
    training_text = "hello world hello world hello world"
    target_size = 300
    
    fresh_tokenizer.train(training_text, target_size)
    
    # Vocab should have grown, but might not reach target if insufficient patterns
    assert fresh_tokenizer.vocab_size > initial_size
    assert fresh_tokenizer.vocab_size <= target_size


def test_training_with_target_less_than_base_size_fails(fresh_tokenizer):
    """Test that training with target vocab size < 256 raises assertion error."""
    with pytest.raises(AssertionError):
        fresh_tokenizer.train("hello world", target_vocab_size=255)


def test_training_with_minimal_target_size(fresh_tokenizer):
    """Test training with target size equal to base size."""
    fresh_tokenizer.train("hello world", target_vocab_size=256)
    assert fresh_tokenizer.vocab_size == 256  # Should remain unchanged


def test_training_affects_encoding_efficiency(tokenizer_class):
    """Test that training reduces the number of tokens for trained patterns."""
    # Create two tokenizers - one trained, one not
    untrained = tokenizer_class()
    trained = tokenizer_class()
    
    training_text = "hello world " * 100  # Repetitive pattern
    test_text = "hello world"
    
    # Get baseline encoding length
    untrained_tokens = untrained.encode(test_text)
    
    # Train and re-encode
    trained.train(training_text, target_vocab_size=300)
    trained_tokens = trained.encode(test_text)
    
    # Training should reduce token count for repeated patterns
    assert len(trained_tokens) <= len(untrained_tokens)


def test_round_trip_after_training(trained_tokenizer):
    """Test that round trip still works after training."""
    test_texts = [
        "hello world",
        "this is a test",
        "completely new text not in training",
        "café 🙂",  # UTF-8
    ]
    
    for text in test_texts:
        tokens = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(tokens)
        assert decoded == text, f"Post-training round trip failed for: '{text}'"


def test_training_with_insufficient_data(fresh_tokenizer):
    """Test training behaviour when there's insufficient data for target vocab size."""
    # Short text that can't possibly generate enough merges for large vocab
    short_text = "ab"
    large_target = 1000
    
    fresh_tokenizer.train(short_text, target_vocab_size=large_target)
    
    # Should still work and vocab should be much smaller than target
    assert fresh_tokenizer.vocab_size < large_target
    
    # Round trip should still work
    tokens = fresh_tokenizer.encode(short_text)
    decoded = fresh_tokenizer.decode(tokens)
    assert decoded == short_text


def test_consistent_encoding_after_training(trained_tokenizer):
    """Test that encoding is deterministic after training."""
    text = "hello world test"
    
    # Encode the same text multiple times
    tokens1 = trained_tokenizer.encode(text)
    tokens2 = trained_tokenizer.encode(text)
    tokens3 = trained_tokenizer.encode(text)
    
    # Should get identical results
    assert tokens1 == tokens2 == tokens3


def test_different_texts_produce_different_encodings(fresh_tokenizer):
    """Test that different input texts produce different token sequences."""
    text1 = "hello"
    text2 = "world"
    
    tokens1 = fresh_tokenizer.encode(text1)
    tokens2 = fresh_tokenizer.encode(text2)
    
    assert tokens1 != tokens2


def test_longer_text_produces_more_tokens(fresh_tokenizer):
    """Test that longer text generally produces more tokens."""
    short_text = "hi"
    long_text = "hello world this is a much longer text"
    
    short_tokens = fresh_tokenizer.encode(short_text)
    long_tokens = fresh_tokenizer.encode(long_text)
    
    assert len(long_tokens) > len(short_tokens)


def test_very_long_text(fresh_tokenizer):
    """Test handling of very long text."""
    # Create a long repetitive text
    long_text = "hello world! " * 1000
    
    tokens = fresh_tokenizer.encode(long_text)
    decoded = fresh_tokenizer.decode(tokens)
    
    assert decoded == long_text
    assert len(tokens) > 0


def test_text_with_only_whitespace(fresh_tokenizer):
    """Test handling of whitespace-only text."""
    whitespace_texts = [
        " ",
        "   ",  # multiple spaces
        "\n\n\n",  # multiple newlines
        "\t\t",  # tabs
        " \n \t ",  # mixed whitespace
    ]
    
    for text in whitespace_texts:
        tokens = fresh_tokenizer.encode(text)
        decoded = fresh_tokenizer.decode(tokens)
        assert decoded == text


def test_text_with_only_punctuation(fresh_tokenizer):
    """Test handling of punctuation-only text."""
    punct_texts = [
        "!",
        "!@#$%",
        "...",
        "???",
        "---",
    ]
    
    for text in punct_texts:
        tokens = fresh_tokenizer.encode(text)
        decoded = fresh_tokenizer.decode(tokens)
        assert decoded == text


def test_training_on_empty_string(fresh_tokenizer):
    """Test that training on empty string doesn't break anything."""
    initial_size = fresh_tokenizer.vocab_size
    fresh_tokenizer.train("", target_vocab_size=300)
    
    # Vocab size should remain unchanged
    assert fresh_tokenizer.vocab_size == initial_size
    
    # Should still be able to encode/decode
    tokens = fresh_tokenizer.encode("test")
    decoded = fresh_tokenizer.decode(tokens)
    assert decoded == "test"


def test_training_on_single_character(fresh_tokenizer):
    """Test training on text with only one unique character."""
    single_char_text = "a" * 100
    
    fresh_tokenizer.train(single_char_text, target_vocab_size=300)
    
    # Should still work
    tokens = fresh_tokenizer.encode("aaa")
    decoded = fresh_tokenizer.decode(tokens)
    assert decoded == "aaa"


def test_multiple_training_sessions(fresh_tokenizer):
    """Test that multiple training sessions work correctly."""
    # Note: This tests the behaviour but implementations might vary
    # in whether they support multiple training or reset state
    
    # First training with substantial repetitive text
    first_training_text = "the quick brown fox jumps over the lazy dog. " * 50
    fresh_tokenizer.train(first_training_text, target_vocab_size=300)
    intermediate_size = fresh_tokenizer.vocab_size
    
    # Verify first training actually increased vocab size
    assert intermediate_size > 256, "First training should have created new tokens"
    
    # Second training with different patterns - behaviour may vary by implementation
    # but should not crash
    second_training_text = "pack my box with five dozen liquor jugs. " * 50
    fresh_tokenizer.train(second_training_text, target_vocab_size=350)
    
    # Should still be able to encode/decode both types of text
    test_texts = [
        "the quick brown fox",  # from first training
        "pack my box with five",  # from second training  
        "completely new text not in either training set"
    ]
    
    for test_text in test_texts:
        tokens = fresh_tokenizer.encode(test_text)
        decoded = fresh_tokenizer.decode(tokens)
        assert decoded == test_text, f"Failed round-trip for: '{test_text}'"


def test_decode_with_invalid_tokens(fresh_tokenizer):
    """Test decode behavior with invalid token values."""
    # Tokens that don't exist in vocabulary
    with pytest.raises((KeyError, ValueError, IndexError)):
        fresh_tokenizer.decode([999999])
    
    # Negative tokens
    with pytest.raises((KeyError, ValueError, IndexError)):
        fresh_tokenizer.decode([-1])
    
    # Non-integer tokens should fail
    with pytest.raises((KeyError, ValueError, IndexError)):
        fresh_tokenizer.decode([1.5, 2.7])
    
    with pytest.raises((KeyError, ValueError, IndexError)):
        fresh_tokenizer.decode(["hello"])


def test_unicode_normalization_consistency(fresh_tokenizer):
    """Test that different Unicode representations of same text work consistently."""
    
    # Same text in different Unicode normal forms
    text = "café"
    nfc_text = unicodedata.normalize('NFC', text)   # é as single character
    nfd_text = unicodedata.normalize('NFD', text)   # e + combining accent
    
    # Both should round-trip correctly (though may encode differently)
    nfc_tokens = fresh_tokenizer.encode(nfc_text)
    nfc_decoded = fresh_tokenizer.decode(nfc_tokens)
    assert nfc_decoded == nfc_text
    
    nfd_tokens = fresh_tokenizer.encode(nfd_text)
    nfd_decoded = fresh_tokenizer.decode(nfd_tokens)
    assert nfd_decoded == nfd_text#