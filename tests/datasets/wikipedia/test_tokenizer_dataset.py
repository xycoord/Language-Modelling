import pytest
import regex
from unittest.mock import patch
from collections import Counter
from datasets import Dataset as HFDataset

from src.lm_datasets.wikipedia.tokenizer_dataset import (
    WikipediaPreprocessDataset,
    chunk_counting_preprocess
)

# ============================================================================
# GENERAL FIXTURES
# ============================================================================

@pytest.fixture
def small_hf_dataset():
    """HF Dataset with ~15 articles for integration testing"""
    article_templates = [
        "Article {i} with some common words like the and a",
        "Another article {i} containing different vocabulary sets", 
        "Test document {i} with repeated words words words",
    ]
    # Create 15 articles total - enough to test multiprocessing properly
    all_articles = []
    for i in range(5):  # 5 iterations x 3 templates = 15 articles
        for template in article_templates:
            all_articles.append({
                "text": template.format(i=i)
            })
    return HFDataset.from_list(all_articles)

@pytest.fixture  
def single_article_dataset():
    """HF Dataset with exactly 1 article"""
    return HFDataset.from_list([{"text": "Single test article with words"}])

@pytest.fixture
def empty_hf_dataset():
    """Empty HF Dataset for edge case testing"""
    return HFDataset.from_list([])

@pytest.fixture
def compiled_regex():
    """Pre-compiled regex pattern for testing"""
    return regex.compile(r'\b\w+\b')

# ============================================================================
# TESTS FOR chunk_counting_preprocess
# ============================================================================

@pytest.mark.parametrize("text,expected", [
    ("", {}),
    ("word word", {"word": 2}),
    ("hello world hello", {"hello": 2, "world": 1}),
    ("the quick brown fox", {"the": 1, "quick": 1, "brown": 1, "fox": 1}),
])
def test_chunk_counting_preprocess_basic_cases(text, expected, compiled_regex):
    """Test basic text processing and counting functionality"""
    result = chunk_counting_preprocess(text, compiled_regex)
    assert dict(result) == expected

def test_chunk_counting_preprocess_empty_text(compiled_regex):
    """Test empty text returns empty Counter"""
    result = chunk_counting_preprocess("", compiled_regex)
    assert len(result) == 0
    assert isinstance(result, Counter)

def test_chunk_counting_preprocess_repeated_words(compiled_regex):
    """Test text with repeated words produces correct counts"""
    text = "test test test example example"
    result = chunk_counting_preprocess(text, compiled_regex)
    assert result["test"] == 3
    assert result["example"] == 2
    assert len(result) == 2

# ============================================================================
# TESTS FOR WikipediaPreprocessDataset.__init__
# ============================================================================

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_init_preprocess_func_none(mock_load_dataset, small_hf_dataset):
    """Test preprocess_func is None after initialization"""
    mock_load_dataset.return_value = small_hf_dataset
    
    dataset = WikipediaPreprocessDataset()
    
    assert dataset.preprocess_func is None

# ============================================================================
# TESTS FOR WikipediaPreprocessDataset.__getitem__
# ============================================================================

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_getitem_raises_when_preprocess_func_none(mock_load_dataset, small_hf_dataset):
    """Test __getitem__ raises ValueError when preprocess_func is None"""
    mock_load_dataset.return_value = small_hf_dataset
    
    dataset = WikipediaPreprocessDataset()
    
    with pytest.raises(ValueError, match="Preprocessing function is not set"):
        dataset[0]

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_getitem_successful_processing(mock_load_dataset, small_hf_dataset):
    """Test __getitem__ works when preprocess_func is set"""
    mock_load_dataset.return_value = small_hf_dataset
    
    dataset = WikipediaPreprocessDataset()
    # Simulate what happens in process() method
    compiled_pattern = regex.compile(r'\b\w+\b')
    dataset.preprocess_func = lambda text: chunk_counting_preprocess(text, compiled_pattern)
    
    result = dataset[0]
    
    assert isinstance(result, Counter)
    assert len(result) > 0  # Should find some words

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_getitem_correct_article_passed(mock_load_dataset):
    """Test __getitem__ passes correct article text to preprocessing"""
    test_articles = [{"text": "specific test content"}]
    mock_dataset = HFDataset.from_list(test_articles)
    mock_load_dataset.return_value = mock_dataset
    
    dataset = WikipediaPreprocessDataset()
    
    # Mock preprocessing function to capture what gets passed
    captured_text = []
    def mock_preprocess(text):
        captured_text.append(text)
        return Counter(["test"])
    
    dataset.preprocess_func = mock_preprocess
    
    dataset[0]
    
    assert captured_text[0] == "specific test content"

# ============================================================================
# TESTS FOR WikipediaPreprocessDataset.process
# ============================================================================

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_process_handles_none_workers(mock_load_dataset, small_hf_dataset):
    """Test process method handles num_workers=None without crashing"""
    mock_load_dataset.return_value = small_hf_dataset
    
    dataset = WikipediaPreprocessDataset()
    
    # The important test: does it work when num_workers=None?
    result = dataset.process(chunk_regex=r'\b\w+\b', num_workers=None)
    
    # Just verify it produces valid results - we don't care about the exact worker count
    assert isinstance(result, list)
    if len(small_hf_dataset) > 0:
        assert len(result) > 0

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_process_sets_preprocess_func(mock_load_dataset, small_hf_dataset):
    """Test process method sets preprocess_func correctly"""
    mock_load_dataset.return_value = small_hf_dataset
    
    dataset = WikipediaPreprocessDataset()
    assert dataset.preprocess_func is None
    
    dataset.process(chunk_regex=r'\b\w+\b', num_workers=2)  # Test with multiprocessing
    
    assert dataset.preprocess_func is not None

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_process_return_structure(mock_load_dataset, small_hf_dataset):
    """Test process returns list of WeightedChunk tuples with correct structure"""
    mock_load_dataset.return_value = small_hf_dataset
    
    dataset = WikipediaPreprocessDataset()
    result = dataset.process(num_workers=2)  # Test with realistic multiprocessing
    
    assert isinstance(result, list)
    for chunk_bytes, count in result:
        assert isinstance(chunk_bytes, list)  # list[int] from UTF-8 encoding
        assert isinstance(count, int)
        assert count > 0
        # Verify it's actually UTF-8 encoded bytes
        assert all(isinstance(b, int) and 0 <= b <= 255 for b in chunk_bytes)

@pytest.mark.parametrize("dataset_fixture", ["empty_hf_dataset", "single_article_dataset"])
@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_process_edge_case_datasets(mock_load_dataset, dataset_fixture, request):
    """Test process handles empty and single-article datasets"""
    test_dataset = request.getfixturevalue(dataset_fixture)
    mock_load_dataset.return_value = test_dataset
    
    dataset = WikipediaPreprocessDataset()
    # Use num_workers=1 for edge cases since there's minimal data
    result = dataset.process(num_workers=1)
    
    assert isinstance(result, list)
    if dataset_fixture == "empty_hf_dataset":
        assert len(result) == 0
    else:  # single_article_dataset
        assert len(result) > 0

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_process_utf8_encoding(mock_load_dataset, single_article_dataset):
    """Test process correctly UTF-8 encodes chunk text"""
    mock_load_dataset.return_value = single_article_dataset
    
    dataset = WikipediaPreprocessDataset()
    # Single worker for single article is appropriate
    result = dataset.process(num_workers=1)
    
    # Find a chunk and verify it can be decoded back to text
    for chunk_bytes, count in result:
        decoded_text = bytes(chunk_bytes).decode('utf-8')
        assert isinstance(decoded_text, str)
        assert len(decoded_text) > 0

# ============================================================================
# TESTS FOR WikipediaPreprocessDataset.collate_counters
# ============================================================================

@pytest.fixture
def sample_counters():
    """Sample Counter objects for collate_counters testing"""
    return [
        Counter({"a": 1, "b": 2}),
        Counter({"b": 1, "c": 3}),
        Counter({"a": 2, "d": 1}),
    ]

def test_collate_counters_empty_list():
    """Test collate_counters with empty list returns empty Counter"""
    result = WikipediaPreprocessDataset.collate_counters([])
    assert isinstance(result, Counter)
    assert len(result) == 0

def test_collate_counters_single_counter():
    """Test collate_counters with single counter returns same counter"""
    single_counter = Counter({"test": 5, "word": 3})
    result = WikipediaPreprocessDataset.collate_counters([single_counter])
    
    assert result == single_counter
    assert result is single_counter  # Should return the same object

def test_collate_counters_multiple_merge(sample_counters):
    """Test collate_counters correctly merges multiple counters"""
    result = WikipediaPreprocessDataset.collate_counters(sample_counters)
    
    expected = Counter({"a": 3, "b": 3, "c": 3, "d": 1})  # a:1+2, b:2+1, c:3, d:1
    assert result == expected

def test_collate_counters_modifies_first_inplace(sample_counters):
    """Test collate_counters modifies first counter in-place"""
    first_counter = sample_counters[0]
    original_id = id(first_counter)
    
    result = WikipediaPreprocessDataset.collate_counters(sample_counters)
    
    assert id(result) == original_id  # Same object
    assert first_counter is result

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_end_to_end_processing_pipeline(mock_load_dataset, small_hf_dataset):
    """Test complete end-to-end processing pipeline"""
    mock_load_dataset.return_value = small_hf_dataset
    
    dataset = WikipediaPreprocessDataset()
    result = dataset.process(chunk_regex=r'\b\w+\b', batch_size=4, num_workers=2)  # Realistic parameters
    
    # Verify complete pipeline
    assert isinstance(result, list)
    assert len(result) > 0
    
    # Verify structure
    for chunk_bytes, count in result:
        assert isinstance(chunk_bytes, list)
        assert isinstance(count, int)
        assert count > 0
        
    # Verify some common words are found
    chunk_texts = [bytes(chunk_bytes).decode('utf-8') for chunk_bytes, _ in result]
    assert any('Article' in text or 'article' in text for text in chunk_texts)

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_deterministic_results(mock_load_dataset, small_hf_dataset):
    """Test results are deterministic with single worker"""
    mock_load_dataset.return_value = small_hf_dataset
    
    dataset1 = WikipediaPreprocessDataset()
    result1 = dataset1.process(num_workers=1, batch_size=4)  # Single worker for determinism
    
    dataset2 = WikipediaPreprocessDataset()
    result2 = dataset2.process(num_workers=1, batch_size=4)  # Single worker for determinism
    
    # Sort results for comparison
    result1_sorted = sorted(result1, key=lambda x: bytes(x[0]))
    result2_sorted = sorted(result2, key=lambda x: bytes(x[0]))
    
    assert result1_sorted == result2_sorted

@patch('src.lm_datasets.wikipedia.tokenizer_dataset.load_dataset')
def test_multiprocessing_cleanup(mock_load_dataset, small_hf_dataset):
    """Test that multiprocessing cleans up properly with production cleanup code"""
    mock_load_dataset.return_value = small_hf_dataset
    
    dataset = WikipediaPreprocessDataset()
    result = dataset.process(num_workers=2, batch_size=4)  # Test with real multiprocessing and cleanup
    
    # Basic validation that it completes and returns expected structure
    assert isinstance(result, list)
    # The key test: this should complete without hanging, proving the cleanup works

# ============================================================================
# TEST UTILITIES
# ============================================================================

def assert_weighted_chunks_structure(chunks):
    """Verify each chunk has correct (list[int], int) structure"""
    assert isinstance(chunks, list)
    for chunk_bytes, count in chunks:
        assert isinstance(chunk_bytes, list)
        assert isinstance(count, int)
        assert count > 0
        assert all(isinstance(b, int) and 0 <= b <= 255 for b in chunk_bytes)

def sum_chunk_counts(chunks):
    """Sum all counts for verification"""
    return sum(count for _, count in chunks)

def create_test_dataset_with_known_words(articles):
    """Create HF dataset with predictable content for deterministic testing"""
    article_dicts = [{"text": article} for article in articles]
    return HFDataset.from_list(article_dicts)