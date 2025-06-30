import tempfile
import json
from pathlib import Path

def assert_tokenizers_equivalent(tokenizer1, tokenizer2, test_texts=None):
    """Test if two tokenizers are functionally equivalent."""
    if test_texts is None:
        test_texts = [
            "hello world",
            "café naïve", 
            "你好",
            "🙂",
            "",  # empty string
            "  ",  # spaces
            "\n",  # newline
            "!@#$%",  # punctuation
            "'quotes'",
        ]
    
    assert tokenizer1.vocab_size == tokenizer2.vocab_size
    assert tokenizer1.vocab == tokenizer2.vocab
    assert tokenizer1.merges == tokenizer2.merges
    
    if hasattr(tokenizer1, 'split_pattern'):
        assert tokenizer1.split_pattern == tokenizer2.split_pattern
    
    for text in test_texts:
        tokens1 = tokenizer1.encode(text)
        tokens2 = tokenizer2.encode(text)
        assert tokens1 == tokens2, f"Encoding differs for text: {repr(text)}"
        
        decoded1 = tokenizer1.decode(tokens1)  
        decoded2 = tokenizer2.decode(tokens2)
        assert decoded1 == decoded2, f"Decoding differs for text: {repr(text)}"
        assert decoded1 == text, f"Round-trip failed for text: {repr(text)}"

def create_test_file_with_content(content, suffix='.json'):
    """Create a temporary file with specific content."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        if isinstance(content, str):
            f.write(content)
        else:
            json.dump(content, f)
        temp_path = f.name
    return Path(temp_path)

def train_tokenizer_with_text(tokenizer, text, **kwargs):
    """
    Helper to train tokenizer with appropriate preprocessing.
    
    Uses preprocess() for Basic and Chunked tokenizers,
    preprocess_train() for all others (which includes deduplication).
    """
    from src.lm_tokenizers.bpe.basic import BasicBPETokenizer
    from src.lm_tokenizers.bpe.chunked import ChunkedBPETokenizer
    
    # Check exact class type since all tokenizers inherit from BasicBPETokenizer
    tokenizer_class = type(tokenizer)
    if tokenizer_class in (BasicBPETokenizer, ChunkedBPETokenizer):
        preprocessed_data = tokenizer.preprocess(text)
    else:
        preprocessed_data = tokenizer.preprocess_train(text)
    
    return tokenizer.train(preprocessed_data, **kwargs)