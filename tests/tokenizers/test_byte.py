from src.lm_tokenizers.byte import ByteTokenizer

def test_vocab_size():
    """Test that vocab_size is always 256"""
    tokenizer = ByteTokenizer()
    assert tokenizer.vocab_size == 256


def test_encode_returns_list():
    """Test that encode returns a list, not bytes object"""
    tokenizer = ByteTokenizer()
    result = tokenizer.encode("hello")
    
    assert isinstance(result, list)
    assert all(isinstance(token, int) for token in result)


def test_basic_round_trip():
    """Test encoding then decoding preserves text"""
    tokenizer = ByteTokenizer()
    text = "Hello, world! 123"
    
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_empty_string():
    """Test edge case of empty string"""
    tokenizer = ByteTokenizer()
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""