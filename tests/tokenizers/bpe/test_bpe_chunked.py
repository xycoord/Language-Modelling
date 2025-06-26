import pytest
from src.tokenizers.bpe.chunked import ChunkedBPETokenizer
from tokenizers.bpe.deduplicated import DeduplicatedBPETokenizer
from src.tokenizers.bpe.incremental import IncrementalBPETokenizer
from src.tokenizers.bpe.fast_max import FastMaxBPETokenizer
from src.tokenizers.bpe.parallel import ParallelBPETokenizer
from src.tokenizers.bpe.utils import GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN
import regex as re

import tempfile
import json
import os
from .test_helpers import assert_tokenizers_equivalent, create_test_file_with_content

@pytest.fixture(params=[
    ChunkedBPETokenizer,
    DeduplicatedBPETokenizer,
    IncrementalBPETokenizer,
    FastMaxBPETokenizer,
    ParallelBPETokenizer,
])
def chunked_tokenizer_class(request):
    """Parameterized fixture for tokenizers that support split patterns."""
    return request.param

# ================================ Test __init__ ================================

def test_chunked_tokenizer_accepts_split_pattern(chunked_tokenizer_class):
    """Test that chunked tokenizers accept split_pattern parameter."""
    default_tokenizer = chunked_tokenizer_class()
    assert default_tokenizer.vocab_size == 256, "Tokenizer should init without split_pattern"
    
    custom_tokenizer = chunked_tokenizer_class(split_pattern=r'\w+|\W+')
    assert custom_tokenizer.vocab_size == 256, "Tokenizer should init with custom split_pattern"


def test_custom_split_pattern_round_trip(chunked_tokenizer_class):
    """Test that custom split patterns maintain round-trip consistency."""
    test_text = "Hello, world! How are you?"
    
    tokenizer = chunked_tokenizer_class(split_pattern=r'\w+|\W+')
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    assert decoded == test_text, "Round-trip should work with custom split pattern"


def test_split_pattern_with_training(chunked_tokenizer_class):
    """Test that custom split patterns work correctly with training."""
    tokenizer = chunked_tokenizer_class(split_pattern=r'\w+|\W+')
    
    training_text = "hello world! " * 150
    tokenizer.train(training_text, target_vocab_size=300)
    
    test_text = "hello world!"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    assert decoded == test_text, "Round-trip should work with custom split pattern"
    
    assert tokenizer.vocab_size > 256, "Training should have increased vocab size"


def test_predefined_split_patterns(chunked_tokenizer_class):
    """Test that predefined GPT2/GPT4 patterns work correctly."""
    text = "Hello! How's it going?"
    
    gpt2_tokenizer = chunked_tokenizer_class(split_pattern=GPT2_SPLIT_PATTERN)
    gpt4_tokenizer = chunked_tokenizer_class(split_pattern=GPT4_SPLIT_PATTERN)
    
    for tokenizer in [gpt2_tokenizer, gpt4_tokenizer]:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text, "Round-trip should work with GPT patterns"


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


# ================================ Test save/load ================================

# --- Split Pattern Preservation Tests ---

def test_save_round_trip_preserves_split_pattern(chunked_tokenizer_class, temp_tokenizer_file):
    """Test that save/load preserves split_pattern."""
    custom_pattern = r'\w+|\W+'
    tokenizer = chunked_tokenizer_class(split_pattern=custom_pattern)
    
    tokenizer.save(temp_tokenizer_file)
    loaded = chunked_tokenizer_class.load(temp_tokenizer_file)
    
    assert loaded.split_pattern == custom_pattern, "Split pattern should be preserved"


def test_save_chunked_tokenizer_saves_split_pattern(chunked_tokenizer_class, temp_tokenizer_file):
    """Test that chunked tokenizers save split_pattern field."""
    custom_pattern = r'\w+|\W+'
    tokenizer = chunked_tokenizer_class(split_pattern=custom_pattern)
    
    tokenizer.save(temp_tokenizer_file)
    
    with open(temp_tokenizer_file, 'r') as f:
        data = json.load(f)
    
    assert "split_pattern" in data, "Split pattern should be saved"
    assert data["split_pattern"] == custom_pattern, "Split pattern should be preserved"
    assert data["tokenizer_type"] == "ChunkedBPE", "Tokenizer type should be ChunkedBPE"


def test_save_load_with_gpt_patterns(chunked_tokenizer_class, temp_tokenizer_file):
    """Test save/load with predefined GPT patterns."""
    gpt2_tokenizer = chunked_tokenizer_class(split_pattern=GPT2_SPLIT_PATTERN)
    gpt4_tokenizer = chunked_tokenizer_class(split_pattern=GPT4_SPLIT_PATTERN)
    
    gpt2_tokenizer.save(temp_tokenizer_file)
    loaded_gpt2 = chunked_tokenizer_class.load(temp_tokenizer_file)
    assert loaded_gpt2.split_pattern == GPT2_SPLIT_PATTERN, "GPT2 pattern should be preserved"
    
    gpt4_tokenizer.save(temp_tokenizer_file)
    loaded_gpt4 = chunked_tokenizer_class.load(temp_tokenizer_file)
    assert loaded_gpt4.split_pattern == GPT4_SPLIT_PATTERN, "GPT4 pattern should be preserved"


# --- Cross-Compatibility Tests ---

@pytest.mark.parametrize("save_class,load_class", [
    (ChunkedBPETokenizer, DeduplicatedBPETokenizer),
    (DeduplicatedBPETokenizer, ChunkedBPETokenizer),
])
def test_cross_compatibility_save_load(save_class, load_class, temp_tokenizer_file):
    """Test that ChunkedBPE and OptimizedBPE files are interchangeable."""
    tokenizer = save_class(split_pattern=r'\w+|\W+')
    training_text = "hello world! hello world! " * 50
    tokenizer.train(training_text, target_vocab_size=300)
    
    tokenizer.save(temp_tokenizer_file)
    
    loaded = load_class.load(temp_tokenizer_file)
    
    assert_tokenizers_equivalent(tokenizer, loaded)

# ================================ Test cross-compatibility ================================

@pytest.mark.parametrize("save_class,load_class", [
    (ChunkedBPETokenizer, DeduplicatedBPETokenizer),
    (DeduplicatedBPETokenizer, ChunkedBPETokenizer),
])
def test_cross_compatibility_preserves_encoding(save_class, load_class, temp_tokenizer_file):
    """Test that cross-compatibility preserves encoding behavior."""
    tokenizer = save_class()
    training_text = "Hello! How's it going? " * 50
    tokenizer.train(training_text, target_vocab_size=300)
    
    test_texts = [
        "Hello! How's it going?",
        "word1 word2 word3",
        "café naïve",
        "'s 'll 're 've",  # Contractions
        "123.45",
        "",
    ]
    
    tokenizer.save(temp_tokenizer_file)
    loaded = load_class.load(temp_tokenizer_file)
    
    for text in test_texts:
        original_tokens = tokenizer.encode(text)
        loaded_tokens = loaded.encode(text)
        assert original_tokens == loaded_tokens, f"Cross-compatibility encoding differs for: {repr(text)}"


@pytest.mark.parametrize("save_class,load_class", [
    (ChunkedBPETokenizer, DeduplicatedBPETokenizer),
    (DeduplicatedBPETokenizer, ChunkedBPETokenizer),
])
def test_cross_compatibility_preserves_decoding(save_class, load_class, temp_tokenizer_file):
    """Test that cross-compatibility preserves decoding behavior."""
    tokenizer = save_class()
    training_text = "decode test text " * 50
    tokenizer.train(training_text, target_vocab_size=300)
    
    test_text = "decode test text"
    tokens = tokenizer.encode(test_text)
    
    tokenizer.save(temp_tokenizer_file)
    loaded = load_class.load(temp_tokenizer_file)
    
    original_decoded = tokenizer.decode(tokens)
    loaded_decoded = loaded.decode(tokens)
    
    assert original_decoded == loaded_decoded == test_text, "Cross-compatibility should preserve decoding"


def test_cross_compatibility_with_custom_patterns():
    """Test cross-compatibility works with custom split patterns."""
    custom_pattern = r'[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9]+'
    
    chunked = ChunkedBPETokenizer(split_pattern=custom_pattern)
    training_text = "word123 test456! " * 50
    chunked.train(training_text, target_vocab_size=300)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        chunked.save(temp_path)
        optimized = DeduplicatedBPETokenizer.load(temp_path)
        
        assert_tokenizers_equivalent(chunked, optimized)
        
        test_text = "word123 test456!"
        assert chunked.encode(test_text) == optimized.encode(test_text), "Cross-compatibility should preserve encoding"
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
def test_load_missing_split_pattern(chunked_tokenizer_class):
    """Test that chunked tokenizers require split_pattern field."""
    # ChunkedBPE file missing split_pattern
    missing_pattern_data = {
        "tokenizer_type": "ChunkedBPE",
        "format_version": "1.0",
        "vocab": {"0": "AA=="},
        "merges": {}
        # Missing split_pattern
    }
    
    missing_file = create_test_file_with_content(missing_pattern_data)
    try:
        with pytest.raises((ValueError, KeyError)):
            chunked_tokenizer_class.load(missing_file)
    finally:
        os.unlink(missing_file)


def test_load_invalid_split_pattern(chunked_tokenizer_class):
    """Test load behavior with invalid split patterns."""
    invalid_patterns = [
        r'[',           # Unclosed character class
        r'(?P<',        # Incomplete named group
        r'*',           # Invalid quantifier
        r'(?',          # Incomplete group
        None,           # None instead of string
        123,            # Integer instead of string
    ]
    
    for invalid_pattern in invalid_patterns:
        invalid_data = {
            "tokenizer_type": "ChunkedBPE",
            "format_version": "1.0",
            "vocab": {"0": "AA=="},
            "merges": {},
            "split_pattern": invalid_pattern
        }
        
        invalid_file = create_test_file_with_content(invalid_data)
        try:
            with pytest.raises((ValueError, TypeError)):
                chunked_tokenizer_class.load(invalid_file)
        finally:
            os.unlink(invalid_file)