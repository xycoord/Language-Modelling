import pytest
from src.tokenizers.bpe.basic import BasicBPETokenizer
from src.tokenizers.bpe.chunked import ChunkedBPETokenizer
from src.tokenizers.bpe.optimized import OptimizedBPETokenizer
import unicodedata
import json
import os
from unittest.mock import patch
from .test_helpers import assert_tokenizers_equivalent, create_test_file_with_content

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


# ================================ Test init ================================

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
        "",
        "a",
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
        "🙂",
        "你好",
        "Γεια σας",
        "العربية",
    ]
    
    for text in test_texts:
        tokens = fresh_tokenizer.encode(text)
        decoded = fresh_tokenizer.decode(tokens)
        assert decoded == text, f"UTF-8 round trip failed for: '{text}'"


def test_round_trip_consistency_special_chars(fresh_tokenizer):
    """Test round trip with special characters and whitespace."""
    test_texts = [
        "  ",
        "\n",
        "\t",
        "\r\n",  # CRLF
        "line1\nline2",
        "word1  word2",
        "!@#$%^&*()",
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
    
    assert fresh_tokenizer.vocab_size > initial_size, "Vocab should have grown"
    assert fresh_tokenizer.vocab_size <= target_size, "Vocab should not exceed target size"


def test_training_with_target_less_than_base_size_fails(fresh_tokenizer):
    """Test that training with target vocab size < 256 raises assertion error."""
    with pytest.raises(AssertionError):
        fresh_tokenizer.train("hello world", target_vocab_size=255)


def test_training_with_minimal_target_size(fresh_tokenizer):
    """Test training with target size equal to base size."""
    fresh_tokenizer.train("hello world", target_vocab_size=256)
    assert fresh_tokenizer.vocab_size == 256, "Vocab should remain unchanged"


def test_training_affects_encoding_efficiency(tokenizer_class):
    """Test that training reduces the number of tokens for trained patterns."""
    untrained = tokenizer_class()
    trained = tokenizer_class()
    
    training_text = "hello world " * 100
    test_text = "hello world"
    
    untrained_tokens = untrained.encode(test_text)
    
    trained.train(training_text, target_vocab_size=300)
    trained_tokens = trained.encode(test_text)
    
    assert len(trained_tokens) <= len(untrained_tokens), "Training should reduce token count for repeated patterns"


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
    short_text = "ab"
    large_target = 1000
    
    fresh_tokenizer.train(short_text, target_vocab_size=large_target)
    
    assert fresh_tokenizer.vocab_size < large_target, "Vocab should be much smaller than target"
    
    tokens = fresh_tokenizer.encode(short_text)
    decoded = fresh_tokenizer.decode(tokens)
    assert decoded == short_text, "Round trip should still work"


def test_consistent_encoding_after_training(trained_tokenizer):
    """Test that encoding is deterministic after training."""
    text = "hello world test"
    
    tokens1 = trained_tokenizer.encode(text)
    tokens2 = trained_tokenizer.encode(text)
    tokens3 = trained_tokenizer.encode(text)
    
    assert tokens1 == tokens2 == tokens3, "Encoding should be deterministic"


def test_different_texts_produce_different_encodings(fresh_tokenizer):
    """Test that different input texts produce different token sequences."""
    text1 = "hello"
    text2 = "world"
    
    tokens1 = fresh_tokenizer.encode(text1)
    tokens2 = fresh_tokenizer.encode(text2)
    
    assert tokens1 != tokens2, "Different texts should produce different token sequences"


def test_longer_text_produces_more_tokens(fresh_tokenizer):
    """Test that longer text generally produces more tokens."""
    short_text = "hi"
    long_text = "hello world this is a much longer text"
    
    short_tokens = fresh_tokenizer.encode(short_text)
    long_tokens = fresh_tokenizer.encode(long_text)
    
    assert len(long_tokens) > len(short_tokens), "Longer text should produce more tokens"


def test_very_long_text(fresh_tokenizer):
    """Test handling of very long text."""
    long_text = "hello world! " * 1000
    
    tokens = fresh_tokenizer.encode(long_text)
    decoded = fresh_tokenizer.decode(tokens)
    
    assert decoded == long_text, "Decoding should match original text"
    assert len(tokens) > 0, "Tokens should not be empty"


def test_text_with_only_whitespace(fresh_tokenizer):
    """Test handling of whitespace-only text."""
    whitespace_texts = [
        " ",
        "   ",
        "\n\n\n",
        "\t\t",
        " \n \t ",
    ]
    
    for text in whitespace_texts:
        tokens = fresh_tokenizer.encode(text)
        decoded = fresh_tokenizer.decode(tokens)
        assert decoded == text, "Whitespace-only text should round-trip correctly"


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
        assert decoded == text, "Punctuation-only text should round-trip correctly"


def test_training_on_empty_string(fresh_tokenizer):
    """Test that training on empty string doesn't break anything."""
    initial_size = fresh_tokenizer.vocab_size
    fresh_tokenizer.train("", target_vocab_size=300)
    
    assert fresh_tokenizer.vocab_size == initial_size, "Vocab size should remain unchanged"
    
    tokens = fresh_tokenizer.encode("test")
    decoded = fresh_tokenizer.decode(tokens)
    assert decoded == "test", "Training on empty string should not break anything"


def test_training_on_single_character(fresh_tokenizer):
    """Test training on text with only one unique character."""
    single_char_text = "a" * 100
    
    fresh_tokenizer.train(single_char_text, target_vocab_size=300)
    
    tokens = fresh_tokenizer.encode("aaa")
    decoded = fresh_tokenizer.decode(tokens)
    assert decoded == "aaa", "Training on single character should not break anything"


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
    with pytest.raises((KeyError, ValueError, IndexError)):
        fresh_tokenizer.decode([999999])
    
    with pytest.raises((KeyError, ValueError, IndexError)):
        fresh_tokenizer.decode([-1])
    
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
    
    nfc_tokens = fresh_tokenizer.encode(nfc_text)
    nfc_decoded = fresh_tokenizer.decode(nfc_tokens)
    assert nfc_decoded == nfc_text, "NFC text should round-trip correctly"
    
    nfd_tokens = fresh_tokenizer.encode(nfd_text)
    nfd_decoded = fresh_tokenizer.decode(nfd_tokens)
    assert nfd_decoded == nfd_text, "NFD text should round-trip correctly"


# ================================ Test save/load ================================

def test_save_creates_file(fresh_tokenizer, temp_tokenizer_file):
    """Test that save creates a file."""
    fresh_tokenizer.save(temp_tokenizer_file)
    assert temp_tokenizer_file.exists()


def test_save_overwrites_existing_file(fresh_tokenizer, temp_tokenizer_file):
    """Test that save completely overwrites existing files."""
    # Create initial file with different content
    with open(temp_tokenizer_file, 'w') as f:
        f.write('{"old": "content"}')
    
    fresh_tokenizer.save(temp_tokenizer_file)
    
    # File should now contain tokenizer data, not old content
    with open(temp_tokenizer_file, 'r') as f:
        data = json.load(f)
    assert "tokenizer_type" in data
    assert "old" not in data


def test_save_file_contains_required_fields(fresh_tokenizer, temp_tokenizer_file):
    """Test that saved file contains all required fields."""
    fresh_tokenizer.save(temp_tokenizer_file)
    
    with open(temp_tokenizer_file, 'r') as f:
        data = json.load(f)
    
    required_fields = ["tokenizer_type", "format_version", "vocab", "merges"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_save_invalid_filepath_type(fresh_tokenizer):
    """Test that save rejects invalid filepath types."""
    with pytest.raises(TypeError):
        fresh_tokenizer.save(123)
    
    with pytest.raises(TypeError):
        fresh_tokenizer.save(None)


# --- Load Method Tests ---

def test_load_creates_independent_instance(trained_tokenizer, temp_tokenizer_file):
    """Test that load creates a new independent tokenizer instance."""
    trained_tokenizer.save(temp_tokenizer_file)
    
    loaded1 = trained_tokenizer.__class__.load(temp_tokenizer_file)
    loaded2 = trained_tokenizer.__class__.load(temp_tokenizer_file)
    
    # Should be different objects but equivalent functionality
    assert loaded1 is not loaded2
    assert_tokenizers_equivalent(loaded1, loaded2)


def test_load_invalid_filepath_type(tokenizer_class):
    """Test that load rejects invalid filepath types."""
    with pytest.raises(TypeError):
        tokenizer_class.load(123)
    
    with pytest.raises(TypeError):
        tokenizer_class.load(None)


def test_load_missing_file(tokenizer_class):
    """Test that load raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        tokenizer_class.load("nonexistent_file.json")


# --- Round-Trip Save/Load Tests ---

def test_save_round_trip_fresh_tokenizer(fresh_tokenizer, temp_tokenizer_file):
    """Test save/load round-trip with untrained tokenizer."""
    fresh_tokenizer.save(temp_tokenizer_file)
    loaded = fresh_tokenizer.__class__.load(temp_tokenizer_file)
    
    assert_tokenizers_equivalent(fresh_tokenizer, loaded)


def test_save_round_trip_save_load_trained_tokenizer(trained_tokenizer, temp_tokenizer_file):
    """Test save/load round-trip with trained tokenizer."""
    trained_tokenizer.save(temp_tokenizer_file)
    loaded = trained_tokenizer.__class__.load(temp_tokenizer_file)
    
    assert_tokenizers_equivalent(trained_tokenizer, loaded)


def test_save_round_trip_preserves_encoding(trained_tokenizer, temp_tokenizer_file):
    """Test that save/load preserves encoding behavior."""
    test_texts = [
        "hello world",
        "café naïve résumé",
        "你好世界",
        "🌟🚀🎉",
        "",
        "  multiple  spaces  ",
        "\n\t\r",
        "!@#$%^&*()",
        "'single' and \"double\" quotes",
    ]
    
    trained_tokenizer.save(temp_tokenizer_file)
    loaded = trained_tokenizer.__class__.load(temp_tokenizer_file)
    
    for text in test_texts:
        original_tokens = trained_tokenizer.encode(text)
        loaded_tokens = loaded.encode(text)
        assert original_tokens == loaded_tokens, f"Encoding differs for: {repr(text)}"


def test_save_round_trip_preserves_decoding(trained_tokenizer, temp_tokenizer_file):
    """Test that save/load preserves decoding behavior."""
    test_text = "hello world this is a test"
    tokens = trained_tokenizer.encode(test_text)
    
    trained_tokenizer.save(temp_tokenizer_file)
    loaded = trained_tokenizer.__class__.load(temp_tokenizer_file)
    
    original_decoded = trained_tokenizer.decode(tokens)
    loaded_decoded = loaded.decode(tokens)
    
    assert original_decoded == loaded_decoded == test_text


# --- Atomicity Tests ---

def test_failed_save_preserves_original_serialization_error(trained_tokenizer, temp_tokenizer_file):
    """Test that serialization failure preserves original file."""
    trained_tokenizer.save(temp_tokenizer_file)
    original_content = temp_tokenizer_file.read_text()
    
    # Create different tokenizer and try to save with mocked failure
    different_tokenizer = trained_tokenizer.__class__()
    different_tokenizer.train("different training text", 350)
    
    with patch('json.dump', side_effect=ValueError("Serialization failed")):
        with pytest.raises(ValueError, match="Serialization failed"):
            different_tokenizer.save(temp_tokenizer_file)
    
    preserved_content = temp_tokenizer_file.read_text()
    assert preserved_content == original_content, "Original file should be preserved"
    
    reloaded = trained_tokenizer.__class__.load(temp_tokenizer_file)
    assert_tokenizers_equivalent(trained_tokenizer, reloaded)


def test_atomicity_os_replace_failure(trained_tokenizer, temp_tokenizer_file):
    """Test atomicity when final rename operation fails (most common real failure)."""
    trained_tokenizer.save(temp_tokenizer_file)
    original_content = temp_tokenizer_file.read_text()
    
    different_tokenizer = trained_tokenizer.__class__()
    
    # os.replace failure is the most realistic I/O failure mode:
    # - Cross-filesystem moves
    # - Permission issues 
    # - Network filesystem problems
    # - Antivirus interference
    with patch('os.replace', side_effect=OSError("Cross-device link")):
        with pytest.raises(OSError):
            different_tokenizer.save(temp_tokenizer_file)
    
    # Contract: original file must be preserved and loadable
    assert temp_tokenizer_file.read_text() == original_content
    reloaded = trained_tokenizer.__class__.load(temp_tokenizer_file)
    assert_tokenizers_equivalent(trained_tokenizer, reloaded)



def test_failed_save_preserves_original_rename_error(trained_tokenizer, temp_tokenizer_file):
    """Test that rename failure preserves original file."""
    trained_tokenizer.save(temp_tokenizer_file)
    original_content = temp_tokenizer_file.read_text()
    
    # Try to overwrite with mocked rename failure
    different_tokenizer = trained_tokenizer.__class__()
    with patch('os.replace', side_effect=OSError("Cross-device link")):
        with pytest.raises(OSError, match="Cross-device link"):
            different_tokenizer.save(temp_tokenizer_file)
    
    preserved_content = temp_tokenizer_file.read_text()
    assert preserved_content == original_content, "Original file should be preserved"
    
    reloaded = trained_tokenizer.__class__.load(temp_tokenizer_file)
    assert_tokenizers_equivalent(trained_tokenizer, reloaded)


# ================================ Error handling tests ================================

def test_load_corrupted_json(tokenizer_class):
    """Test load behavior with corrupted JSON files."""
    corrupted_files = [
        '{"tokenizer_type": "BasicBPE", "vocab":',  # Incomplete JSON
        '{"tokenizer_type": "BasicBPE" "vocab": {}}',  # Missing comma
        '{invalid json content}',  # Invalid syntax
        '',  # Empty file
        'not json at all',  # Not JSON
    ]
    
    for corrupted_content in corrupted_files:
        corrupted_file = create_test_file_with_content(corrupted_content)
        try:
            with pytest.raises((json.JSONDecodeError, ValueError)):
                tokenizer_class.load(corrupted_file)
        finally:
            os.unlink(corrupted_file)


def test_load_missing_required_fields(tokenizer_class):
    """Test load behavior when required fields are missing."""
    incomplete_data_sets = [
        {},  # Completely empty
        {"tokenizer_type": "BasicBPE"},  # Missing vocab, merges
        {"vocab": {}, "merges": {}},  # Missing tokenizer_type
        {"tokenizer_type": "BasicBPE", "vocab": {}},  # Missing merges
        {"tokenizer_type": "BasicBPE", "merges": {}},  # Missing vocab
        {"tokenizer_type": "BasicBPE", "vocab": {}, "merges": {}, "format_version": "1.0"},  # Complete but missing for chunked
    ]
    
    for incomplete_data in incomplete_data_sets:
        incomplete_file = create_test_file_with_content(incomplete_data)
        try:
            with pytest.raises((ValueError, KeyError)):
                tokenizer_class.load(incomplete_file)
        finally:
            os.unlink(incomplete_file)


def test_load_wrong_tokenizer_type(tokenizer_class):
    """Test load behavior when tokenizer type doesn't match."""
    wrong_type_data_sets = [
        {"tokenizer_type": "NonExistentType", "vocab": {}, "merges": {}},
        {"tokenizer_type": "SomeOtherTokenizer", "vocab": {}, "merges": {}},
        {"tokenizer_type": "", "vocab": {}, "merges": {}},  # Empty type
        {"tokenizer_type": 123, "vocab": {}, "merges": {}},  # Non-string type
    ]
    
    for wrong_data in wrong_type_data_sets:
        wrong_file = create_test_file_with_content(wrong_data)
        try:
            with pytest.raises(ValueError):
                tokenizer_class.load(wrong_file)
        finally:
            os.unlink(wrong_file)


def test_load_basic_vs_chunked_type_mismatch():
    """Test that BasicBPE can't load ChunkedBPE files and vice versa."""
    chunked_data = {
        "tokenizer_type": "ChunkedBPE",
        "format_version": "1.0",
        "vocab": {"0": "AA==", "1": "AQ=="},
        "merges": {},
        "split_pattern": r'\w+|\W+'
    }
    chunked_file = create_test_file_with_content(chunked_data)
    
    basic_data = {
        "tokenizer_type": "BasicBPE",
        "format_version": "1.0", 
        "vocab": {"0": "AA==", "1": "AQ=="},
        "merges": {}
    }
    basic_file = create_test_file_with_content(basic_data)
    
    try:
        with pytest.raises(ValueError):
            BasicBPETokenizer.load(chunked_file)
        
        with pytest.raises(ValueError):
            ChunkedBPETokenizer.load(basic_file)
        
        with pytest.raises(ValueError):
            OptimizedBPETokenizer.load(basic_file)
    
    finally:
        os.unlink(chunked_file)
        os.unlink(basic_file)


def test_load_invalid_vocab_data(tokenizer_class):
    """Test load behavior with corrupted vocab data."""
    invalid_vocab_data_sets = [
        {"tokenizer_type": "BasicBPE", "vocab": "not_a_dict", "merges": {}},
        {"tokenizer_type": "BasicBPE", "vocab": [], "merges": {}},  # List instead of dict
        {"tokenizer_type": "BasicBPE", "vocab": None, "merges": {}},
        {"tokenizer_type": "BasicBPE", "vocab": {"key": "invalid_base64"}, "merges": {}},
        {"tokenizer_type": "BasicBPE", "vocab": {"not_int": "AA=="}, "merges": {}},  # Non-integer keys as strings
    ]
    
    for invalid_data in invalid_vocab_data_sets:
        invalid_file = create_test_file_with_content(invalid_data)
        try:
            with pytest.raises((ValueError, TypeError, KeyError)):
                tokenizer_class.load(invalid_file)
        finally:
            os.unlink(invalid_file)


def test_load_invalid_merge_data(tokenizer_class):
    """Test load behavior with corrupted merge data."""
    invalid_merge_data_sets = [
        {"tokenizer_type": "BasicBPE", "vocab": {"0": "AA=="}, "merges": "not_a_dict"},
        {"tokenizer_type": "BasicBPE", "vocab": {"0": "AA=="}, "merges": []},  # List instead of dict
        {"tokenizer_type": "BasicBPE", "vocab": {"0": "AA=="}, "merges": None},
        {"tokenizer_type": "BasicBPE", "vocab": {"0": "AA=="}, "merges": {"invalid_pair": 256}},  # Invalid pair format
        {"tokenizer_type": "BasicBPE", "vocab": {"0": "AA=="}, "merges": {"1,2": "not_int"}},  # Non-integer target
    ]
    
    for invalid_data in invalid_merge_data_sets:
        invalid_file = create_test_file_with_content(invalid_data)
        try:
            with pytest.raises((ValueError, TypeError, KeyError)):
                tokenizer_class.load(invalid_file)
        finally:
            os.unlink(invalid_file)