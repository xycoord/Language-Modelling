import pytest
from src.lm_tokenizers.bpe.basic import BasicBPETokenizer
import json

@pytest.fixture
def basic_tokenizer():
    """Create a fresh BasicBPETokenizer instance."""
    return BasicBPETokenizer()


def test_basic_tokenizer_no_constructor_params():
    """Test that BasicBPETokenizer doesn't accept constructor parameters."""
    tokenizer = BasicBPETokenizer()
    assert tokenizer.vocab_size == 256, "Vocab size should be 256"
    
    with pytest.raises(TypeError):
        BasicBPETokenizer(split_pattern="some_pattern")


def test_basic_tokenizer_processes_text_without_chunking(basic_tokenizer):
    """Test that BasicBPETokenizer processes text as a single unit."""
    text = "hello, world!"
    
    tokens = basic_tokenizer.encode(text)
    decoded = basic_tokenizer.decode(tokens)
    assert decoded == text, "Decoding should match original text"
    
    assert isinstance(tokens, list), "Tokens should be a list"
    assert all(isinstance(token, int) for token in tokens), "Tokens should be integers"


def test_basic_tokenizer_training_on_long_continuous_text(basic_tokenizer):
    """Test BasicBPETokenizer training on long continuous text."""
    continuous_text = "word1word2word3" * 100
    
    basic_tokenizer.train(continuous_text, target_vocab_size=300)
    
    test_text = "word1word2word3"
    tokens = basic_tokenizer.encode(test_text)
    decoded = basic_tokenizer.decode(tokens)
    assert decoded == test_text, "Decoding should match original text"

    assert basic_tokenizer.vocab_size > 256, "Vocab size should be greater than 256"


def test_basic_tokenizer_handles_text_boundaries_as_continuous():
    """Test that BasicBPETokenizer can merge across what would be chunk boundaries."""
    tokenizer = BasicBPETokenizer()
    
    text = "prefix_suffix prefix_suffix prefix_suffix" * 50
    tokenizer.train(text, target_vocab_size=300)
    
    test_text = "prefix_suffix"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    assert decoded == test_text, "Decoding should match original text"

# ================================ Test save/load ================================

def test_basic_tokenizer_save_no_split_pattern(basic_tokenizer, temp_tokenizer_file):
    """Test that BasicBPETokenizer doesn't save split_pattern field."""
    basic_tokenizer.save(temp_tokenizer_file)
    
    with open(temp_tokenizer_file, 'r') as f:
        data = json.load(f)
    
    assert "split_pattern" not in data, "Split pattern should not be saved"
    assert data["tokenizer_type"] == "BasicBPE", "Tokenizer type should be BasicBPE"