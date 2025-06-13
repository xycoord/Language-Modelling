import pytest
from tokenizers import CharTokenizer


def test_vocab_creation():
    """Test that vocabulary is created correctly from input text"""
    tokenizer = CharTokenizer("hello")
    
    assert tokenizer.vocab_size == 4  # h, e, l, o
    assert set(tokenizer.chars) == {'h', 'e', 'l', 'o'}
    
    # Verify tokens are assigned 0 to vocab_size-1
    tokens = tokenizer.encode("hello")
    assert all(0 <= t < tokenizer.vocab_size for t in tokens)


def test_vocab_sorted():
    """Test that vocabulary is sorted"""
    tokenizer = CharTokenizer("cab")
    assert tokenizer.chars == ['a', 'b', 'c']


def test_encode_decode_round_trip():
    """Test encoding then decoding preserves text"""
    text = "hello world"
    tokenizer = CharTokenizer(text)
    
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_encode_unknown_character():
    """Test encoding a character not in vocabulary raises ValueError"""
    tokenizer = CharTokenizer("abc")
    
    with pytest.raises(ValueError, match="Characters not in vocabulary: {'d'}"):
        tokenizer.encode("d")
    
    # Multiple unknown characters
    with pytest.raises(ValueError, match="Characters not in vocabulary:"):
        tokenizer.encode("def")


def test_decode_invalid_token():
    """Test decoding invalid tokens raises ValueError"""
    tokenizer = CharTokenizer("abc")
    
    with pytest.raises(ValueError, match="Invalid tokens: \\[3\\]"):
        tokenizer.decode([0, 1, 3])  # 3 is out of range
    
    # Negative token
    with pytest.raises(ValueError, match="Invalid tokens: \\[-1\\]"):
        tokenizer.decode([-1])


def test_empty_text():
    """Test tokenizer with empty initialization text"""
    tokenizer = CharTokenizer("")
    
    assert tokenizer.vocab_size == 0
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
    
    # Any character should be unknown
    with pytest.raises(ValueError, match="Characters not in vocabulary"):
        tokenizer.encode("a")


def test_repeated_characters():
    """Test that repeated characters create single vocabulary entry"""
    tokenizer = CharTokenizer("aaaaabbbbb")
    
    assert tokenizer.vocab_size == 2
    assert set(tokenizer.chars) == {'a', 'b'}