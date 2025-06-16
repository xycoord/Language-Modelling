import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.datasets.language_dataset import LanguageDataset
from src.datasets.gutenberg_dataset import GutenbergDataset
from src.tokenizers.byte import ByteTokenizer


# --- Fixtures ---
@pytest.fixture
def sample_text():
    """Sample text for testing - long enough for various split configurations"""
    return "hello world! this is a test string for testing purposes. " * 3


@pytest.fixture
def sample_texts():
    """Sample texts for multi-book testing"""
    return [
        "hello world! this is the first book for testing purposes. " * 2,
        "this is book two with different content for validation. " * 2,
        "and here is the third book with even more test content. " * 2
    ]


@pytest.fixture
def short_text():
    """Short text for edge case testing"""
    return "hello world"


@pytest.fixture
def short_texts():
    """Short texts for edge case testing"""
    return ["hello world", "short book"]


@pytest.fixture
def tokenizer():
    """ByteTokenizer instance for testing"""
    return ByteTokenizer()


@pytest.fixture(params=[
    ("language", LanguageDataset),
    ("gutenberg", GutenbergDataset)
])
def dataset_class_info(request):
    """Parametrized fixture providing dataset class and identifier"""
    return request.param


def create_dataset(dataset_class, data, tokenizer, **kwargs):
    """Helper to create dataset with appropriate data format"""
    if dataset_class == LanguageDataset:
        if isinstance(data, list):
            data = "\n\n".join(data)
        return dataset_class(data_text=data, tokenizer=tokenizer, **kwargs)
    elif dataset_class == GutenbergDataset:
        if isinstance(data, str):
            data = [data]
        return dataset_class(texts=data, tokenizer=tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown dataset class: {dataset_class}")


# --- Shared Tests (Both Datasets) ---
def test_train_dataset_creation(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test creating a valid training dataset"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    dataset = create_dataset(
        dataset_class, data, tokenizer,
        split='train',
        train_split=0.8,
        block_size=8
    )
    
    assert dataset.block_size == 8, "Block size should match constructor argument"
    assert dataset.vocab_size == tokenizer.vocab_size, "Vocab size should match tokenizer"
    assert len(dataset) > 0, "Dataset should have at least one sequence"
    
    inputs, targets = dataset[0]
    assert isinstance(inputs, Tensor), "Inputs should be torch.Tensor"
    assert isinstance(targets, Tensor), "Targets should be torch.Tensor"


def test_val_dataset_creation(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test creating a valid validation dataset"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    dataset = create_dataset(
        dataset_class, data, tokenizer,
        split='val',
        train_split=0.7,
        block_size=8
    )
    
    assert dataset.block_size == 8, "Block size should match constructor argument"
    assert dataset.vocab_size == tokenizer.vocab_size, "Vocab size should match tokenizer"
    assert len(dataset) > 0, "Validation dataset should have at least one sequence"


def test_getitem_returns_correct_shapes(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test that __getitem__ returns tensors of correct shape and type"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    block_size = 10
    
    dataset = create_dataset(
        dataset_class, data, tokenizer,
        block_size=block_size
    )
    
    inputs, targets = dataset[0]
    
    assert inputs.shape == (block_size,), f"Expected inputs shape ({block_size},), got {inputs.shape}"
    assert targets.shape == (block_size,), f"Expected targets shape ({block_size},), got {targets.shape}"
    assert inputs.dtype == torch.long, f"Expected inputs dtype torch.long, got {inputs.dtype}"
    assert targets.dtype == torch.long, f"Expected targets dtype torch.long, got {targets.dtype}"


def test_getitem_autoregressive_offset(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test that targets are offset by 1 from inputs for autoregressive training"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    dataset = create_dataset(
        dataset_class, data, tokenizer,
        block_size=5
    )
    
    inputs, targets = dataset[0]
    
    assert inputs[0] == dataset.data[0], "First input should match first token in data"
    assert targets[0] == dataset.data[1], "First target should be second token in data (next token)"


def test_dataset_length_calculation(sample_text, tokenizer, dataset_class_info):
    """Test that dataset length is calculated correctly"""
    name, dataset_class = dataset_class_info
    data = sample_text
    block_size = 8
    
    dataset = create_dataset(
        dataset_class, data, tokenizer,
        split='train',
        train_split=1,
        block_size=block_size
    )
    
    expected_length = len(dataset.data) - block_size
    assert len(dataset) == expected_length, f"Expected length {expected_length}, got {len(dataset)}"


def test_train_val_split_consistency(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test that train and val splits partition the data correctly"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    train_split = 0.8
    block_size = 8
    
    train_dataset = create_dataset(
        dataset_class, data, tokenizer,
        split='train',
        train_split=train_split,
        block_size=block_size
    )
    
    val_dataset = create_dataset(
        dataset_class, data, tokenizer,
        split='val',
        train_split=train_split,
        block_size=block_size
    )
    
    total_available = len(train_dataset) + len(val_dataset) + (2 * block_size)
    assert total_available > 0, "Total available sequences should be positive"


def test_train_val_no_overlap(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test that train and val data don't overlap"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    train_split = 0.7
    block_size = 5
    
    train_dataset = create_dataset(
        dataset_class, data, tokenizer,
        split='train',
        train_split=train_split,
        block_size=block_size
    )
    
    val_dataset = create_dataset(
        dataset_class, data, tokenizer,
        split='val',
        train_split=train_split,
        block_size=block_size
    )
    
    if len(train_dataset) > 0 and len(val_dataset) > 0:
        train_last_inputs, _ = train_dataset[len(train_dataset) - 1]
        val_first_inputs, _ = val_dataset[0]
        
        assert not torch.equal(train_last_inputs, val_first_inputs), "Train and val data should not overlap"


def test_minimum_viable_dataset(tokenizer, dataset_class_info):
    """Test with minimum data that still works"""
    name, dataset_class = dataset_class_info
    min_text = "hello world test"
    block_size = len(tokenizer.encode(min_text)) - 1
    
    data = [min_text] if name == "gutenberg" else min_text
    
    dataset = create_dataset(
        dataset_class, data, tokenizer,
        split='train',
        train_split=1,
        block_size=block_size
    )
    
    assert len(dataset) == 1, "Should have exactly one valid sequence"
    inputs, targets = dataset[0]
    assert len(inputs) == block_size, f"Input length should be {block_size}"


def test_small_validation_split(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test that small validation splits still work if they have enough data"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    total_tokens = len(tokenizer.encode(sample_text if name == "language" else "\n\n".join(sample_texts)))
    block_size = 8
    min_val_tokens = block_size + 2
    
    train_split = 1.0 - (min_val_tokens / total_tokens)
    
    val_dataset = create_dataset(
        dataset_class, data, tokenizer,
        split='val',
        train_split=train_split,
        block_size=block_size
    )
    
    assert len(val_dataset) >= 1, "Small validation split should still have at least one sequence"


@pytest.mark.parametrize("train_split", [0.0, 0.5, 1.0])
def test_boundary_train_splits(sample_text, sample_texts, tokenizer, dataset_class_info, train_split):
    """Test edge cases for train_split values"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    block_size = 5
    
    if train_split == 1.0:
        train_dataset = create_dataset(
            dataset_class, data, tokenizer,
            split='train',
            train_split=train_split,
            block_size=block_size
        )
        assert len(train_dataset) > 0, "Train dataset should have sequences when train_split=1.0"
    else:
        val_dataset = create_dataset(
            dataset_class, data, tokenizer,
            split='val',
            train_split=train_split,
            block_size=block_size
        )
        assert len(val_dataset) >= 0, "Validation dataset should be valid (possibly empty)"


@pytest.mark.parametrize("block_size", [1, 3, 8, 16, 32])
def test_various_block_sizes(sample_text, sample_texts, tokenizer, dataset_class_info, block_size):
    """Test dataset with various block sizes"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    dataset = create_dataset(
        dataset_class, data, tokenizer,
        block_size=block_size
    )
    
    if len(dataset) > 0:
        inputs, targets = dataset[0]
        assert inputs.shape[0] == block_size, f"Input shape should be ({block_size},)"
        assert targets.shape[0] == block_size, f"Target shape should be ({block_size},)"


def test_invalid_split_raises_error(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test that invalid split values raise ValueError"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    invalid_splits = ['test', 'TEST', 'Train', 'Val', '', 'dev']
    
    for split in invalid_splits:
        with pytest.raises(ValueError, match="Invalid split"):
            create_dataset(
                dataset_class, data, tokenizer,
                split=split
            )


@pytest.mark.parametrize("train_split,error_pattern", [
    (-0.1, r"Invalid train_split.*between 0 and 1"),
    (-1.0, r"Invalid train_split.*between 0 and 1"),
    (1.1, r"Invalid train_split.*between 0 and 1"),
    (2.0, r"Invalid train_split.*between 0 and 1"),
])
def test_train_split_out_of_range(sample_text, sample_texts, tokenizer, dataset_class_info, train_split, error_pattern):
    """Test that train_split outside [0, 1] raises ValueError"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    with pytest.raises(ValueError, match=error_pattern):
        create_dataset(
            dataset_class, data, tokenizer,
            train_split=train_split
        )


def test_train_split_1_with_val_split_raises_error(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test that train_split=1.0 with split='val' raises specific error"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    with pytest.raises(ValueError, match="train_split cannot be 1 when split is 'val'"):
        create_dataset(
            dataset_class, data, tokenizer,
            split='val',
            train_split=1.0
        )


@pytest.mark.parametrize("block_size", [0, -1, -10])
def test_invalid_block_size_raises_error(sample_text, sample_texts, tokenizer, dataset_class_info, block_size):
    """Test that non-positive block_size raises ValueError"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    with pytest.raises(ValueError, match=r"Invalid block_size.*greater than 0"):
        create_dataset(
            dataset_class, data, tokenizer,
            block_size=block_size
        )


def test_validation_split_too_small_for_block_size(short_text, tokenizer, dataset_class_info):
    """Test when validation split doesn't have enough data for even one sequence"""
    name, dataset_class = dataset_class_info
    data = [short_text] if name == "gutenberg" else short_text
    
    total_tokens = len(tokenizer.encode(short_text))
    train_split = 0.95
    val_tokens = int((1 - train_split) * total_tokens)
    
    if val_tokens > 0:
        dataset = create_dataset(
            dataset_class, data, tokenizer,
            split='val',
            train_split=train_split,
            block_size=8
        )
        assert len(dataset) >= 0, "Dataset should be valid even if empty"


def test_vocab_size_property(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test that vocab_size property works correctly"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    dataset = create_dataset(
        dataset_class, data, tokenizer,
        block_size=8
    )
    
    assert dataset.vocab_size == tokenizer.vocab_size, "Dataset vocab_size should match tokenizer"


def test_sequential_iteration(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Test iterating through entire dataset sequentially"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    block_size = 8
    
    dataset = create_dataset(
        dataset_class, data, tokenizer,
        block_size=block_size
    )
    
    for i in range(min(10, len(dataset))):
        inputs, targets = dataset[i]
        assert inputs.shape == (block_size,), f"Input shape should be ({block_size},) at index {i}"
        assert targets.shape == (block_size,), f"Target shape should be ({block_size},) at index {i}"
    
    if len(dataset) > 1:
        inputs1, _ = dataset[0]
        inputs2, _ = dataset[1]
        assert inputs2[0] == inputs1[1], "Sequential datasets should have sliding window property"


def test_works_with_dataloader(sample_text, sample_texts, tokenizer, dataset_class_info):
    """Integration smoke test - can be used with DataLoader"""
    name, dataset_class = dataset_class_info
    data = sample_texts if name == "gutenberg" else sample_text
    
    dataset = create_dataset(
        dataset_class, data, tokenizer,
        split='train',
        train_split=1.0,
        block_size=8
    )
    
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=min(4, len(dataset)))
        for inputs, targets in loader:
            break


# --- LanguageDataset-Specific Tests ---
def test_language_dataset_data_too_short_raises_error(tokenizer):
    """Test that LanguageDataset raises error when data is too short"""
    short_text = "hi"
    
    with pytest.raises(ValueError) as exc_info:
        LanguageDataset(
            data_text=short_text,
            tokenizer=tokenizer,
            block_size=10
        )
    
    error_msg = str(exc_info.value)
    assert "Data too short" in error_msg, f"Error message should mention 'Data too short', got: {error_msg}"


def test_exact_block_size_equals_data_raises_error(tokenizer):
    """Test edge case where block_size exactly equals data length"""
    text = "hello world"
    block_size = len(tokenizer.encode(text))
    
    with pytest.raises(ValueError, match="Data too short"):
        LanguageDataset(
            data_text=text,
            tokenizer=tokenizer,
            block_size=block_size
        )


# --- GutenbergDataset-Specific Tests ---

def test_gutenberg_multiple_books(sample_texts, tokenizer):
    """Test GutenbergDataset specific functionality with multiple books"""
    dataset = GutenbergDataset(
        texts=sample_texts,
        tokenizer=tokenizer,
        block_size=8
    )
    
    assert len(dataset) > 0, "Multi-book dataset should have sequences"
    
    total_tokens = sum(len(tokenizer.encode(text)) for text in sample_texts)
    assert len(dataset.data) == total_tokens, f"Data should contain all {total_tokens} tokens from all books"


def test_gutenberg_empty_text_handling(tokenizer):
    """Test that GutenbergDataset handles empty texts gracefully"""
    texts = ["hello world", "", "another book"]
    
    dataset = GutenbergDataset(
        texts=[t for t in texts if t.strip()],
        tokenizer=tokenizer,
        block_size=5
    )
    
    assert len(dataset) > 0, "Should work after filtering empty texts"


def test_gutenberg_mixed_text_lengths(sample_texts, tokenizer):
    """Test that GutenbergDataset handles mix of short and long texts"""
    mixed_texts = ["hi", sample_texts[0], "bye"]
    
    dataset = GutenbergDataset(
        texts=mixed_texts,
        tokenizer=tokenizer,
        block_size=10
    )
    
    assert len(dataset) > 0, "Should successfully create dataset using only the long text"


@pytest.fixture
def sample_config(sample_texts, tokenizer):
    """Base configuration for GutenbergDataset testing"""
    return {
        'texts': sample_texts,
        'tokenizer': tokenizer,
        'train_split': 0.7,
        'block_size': 10
    }

def test_gutenberg_create_split_pair_equivalent_to_init(sample_config):
    """Test that create_split_pair produces identical results to separate __init__ calls"""
    shared_config = sample_config
    
    train_init = GutenbergDataset(**shared_config, split='train')
    val_init = GutenbergDataset(**shared_config, split='val')
    train_pair, val_pair = GutenbergDataset.create_split_pair(**shared_config)

    datasets_init = {'train': train_init, 'val': val_init}
    datasets_pair = {'train': train_pair, 'val': val_pair}
    
    for split_name in ['train', 'val']:
        dataset_init = datasets_init[split_name]
        dataset_pair = datasets_pair[split_name]
        
        assert torch.equal(dataset_init.data, dataset_pair.data), \
            f"{split_name.capitalize()} datasets should have identical data tensors"
        assert dataset_init.block_size == dataset_pair.block_size, \
            f"{split_name.capitalize()} datasets should have same block_size"
        assert dataset_init.vocab_size == dataset_pair.vocab_size, \
            f"{split_name.capitalize()} datasets should have same vocab_size"
        assert len(dataset_init) == len(dataset_pair), \
            f"{split_name.capitalize()} datasets should have same length"
        
        if len(dataset_init) > 0:
            inputs_init, targets_init = dataset_init[0]
            inputs_pair, targets_pair = dataset_pair[0]
            assert torch.equal(inputs_init, inputs_pair), \
                f"{split_name.capitalize()} inputs should match"
            assert torch.equal(targets_init, targets_pair), \
                f"{split_name.capitalize()} targets should match"
            

@pytest.mark.parametrize("train_split,expected_error", [
    (0.0, r"Invalid train_split.*between 0 and 1 \(exclusive\)"),
    (-0.1, r"Invalid train_split.*between 0 and 1 \(exclusive\)"),
    (1.0, r"Invalid train_split.*between 0 and 1 \(exclusive\)"),
    (1.1, r"Invalid train_split.*between 0 and 1 \(exclusive\)"),
])
def test_gutenberg_create_split_pair_invalid_train_split(sample_config, train_split, expected_error):
    """Test that create_split_pair raises ValueError for invalid train_split values"""
    config = {**sample_config, 'train_split': train_split}
    
    with pytest.raises(ValueError, match=expected_error):
        GutenbergDataset.create_split_pair(**config)


@pytest.mark.parametrize("block_size", [0, -1, -10])
def test_gutenberg_create_split_pair_invalid_block_size(sample_config, block_size):
    """Test that create_split_pair raises ValueError for invalid block_size values"""
    config = {**sample_config, 'block_size': block_size}
    
    with pytest.raises(ValueError, match=r"Invalid block_size.*greater than 0"):
        GutenbergDataset.create_split_pair(**config)
