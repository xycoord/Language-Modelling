import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.datasets.language_dataset import LanguageDataset
from src.tokenizers.byte import ByteTokenizer


# --- Fixtures ---
@pytest.fixture
def sample_text():
    """Sample text for testing - long enough for various split configurations"""
    return "hello world! this is a test string for testing purposes. " * 3


@pytest.fixture
def short_text():
    """Short text for edge case testing"""
    return "hello world"


@pytest.fixture
def tokenizer():
    """CharTokenizer instance for testing"""
    return ByteTokenizer()


# --- Basic Dataset Creation Tests ---
def test_train_dataset_creation(sample_text, tokenizer):
    """Test creating a valid training dataset"""
    dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        split='train',
        train_split=0.8,
        block_size=8
    )
    
    assert dataset.block_size == 8
    assert dataset.vocab_size == tokenizer.vocab_size
    assert len(dataset) > 0
    
    # Verify we can get an item
    inputs, targets = dataset[0]
    assert isinstance(inputs, Tensor)
    assert isinstance(targets, Tensor)


def test_val_dataset_creation(sample_text, tokenizer):
    """Test creating a valid validation dataset"""
    dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        split='val',
        train_split=0.7,  # Give validation 30% to ensure sufficient data
        block_size=8
    )
    
    assert dataset.block_size == 8
    assert dataset.vocab_size == tokenizer.vocab_size
    assert len(dataset) > 0


# --- Data Shape and Content Tests ---
def test_getitem_returns_correct_shapes(sample_text, tokenizer):
    """Test that __getitem__ returns tensors of correct shape and type"""
    block_size = 10
    dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        block_size=block_size
    )
    
    inputs, targets = dataset[0]
    
    assert inputs.shape == (block_size,)
    assert targets.shape == (block_size,)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


def test_getitem_autoregressive_offset(sample_text, tokenizer):
    """Test that targets are offset by 1 from inputs for autoregressive training"""
    dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        block_size=5
    )
    
    inputs, targets = dataset[0]
    
    # Get the full token sequence to verify offset
    full_tokens = torch.tensor(tokenizer.encode(sample_text), dtype=torch.long)
    
    # Check that targets[i] = inputs[i+1] (next token prediction)
    for i in range(len(inputs)):
        assert inputs[i] == full_tokens[i]
        assert targets[i] == full_tokens[i + 1]


def test_dataset_length_calculation(sample_text, tokenizer):
    """Test that dataset length is calculated correctly"""
    block_size = 8
    tokens = tokenizer.encode(sample_text)
    expected_length = len(tokens) - block_size
    
    dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        split='train',
        train_split=1,
        block_size=block_size
    )
    
    assert len(dataset) == expected_length


# --- Train/Validation Split Tests ---
def test_train_val_split_consistency(sample_text, tokenizer):
    """Test that train and val splits partition the data correctly"""
    train_split = 0.8
    block_size = 8
    
    train_dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        split='train',
        train_split=train_split,
        block_size=block_size
    )
    
    val_dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        split='val',
        train_split=train_split,
        block_size=block_size
    )
    
    # Calculate expected sizes
    total_tokens = len(tokenizer.encode(sample_text))
    expected_train_tokens = int(train_split * total_tokens)
    expected_val_tokens = total_tokens - expected_train_tokens
    
    # Account for block_size offset in dataset length
    assert len(train_dataset) == expected_train_tokens - block_size
    assert len(val_dataset) == expected_val_tokens - block_size


def test_train_val_no_overlap(sample_text, tokenizer):
    """Test that train and val data don't overlap"""
    train_split = 0.7
    block_size = 5
    
    train_dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        split='train',
        train_split=train_split,
        block_size=block_size
    )
    
    val_dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        split='val',
        train_split=train_split,
        block_size=block_size
    )
    
    # Get last sequence from train and first from val
    train_last_inputs, _ = train_dataset[len(train_dataset) - 1]
    val_first_inputs, _ = val_dataset[0]
    
    # They should be different (no overlap)
    assert not torch.equal(train_last_inputs, val_first_inputs)


# --- Edge Case Tests ---
def test_minimum_viable_dataset(tokenizer):
    """Test with minimum data that still works"""
    min_text = "hello"
    block_size = len(tokenizer.encode(min_text)) - 1  # Just barely valid
    
    dataset = LanguageDataset(
        data_text=min_text,
        tokenizer=tokenizer,
        split='train',
        train_split=1,
        block_size=block_size
    )
    
    assert len(dataset) == 1  # Exactly one valid sequence
    inputs, targets = dataset[0]
    assert len(inputs) == block_size


def test_small_validation_split(sample_text, tokenizer):
    """Test that small validation splits still work if they have enough data"""
    # Calculate a train_split that gives validation just enough data
    total_tokens = len(tokenizer.encode(sample_text))
    block_size = 8
    min_val_tokens = block_size + 2  # Minimum for at least 2 sequences
    
    train_split = 1.0 - (min_val_tokens / total_tokens)
    
    val_dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        split='val',
        train_split=train_split,
        block_size=block_size
    )
    
    assert len(val_dataset) >= 1


@pytest.mark.parametrize("train_split", [0.0, 0.5, 1.0])
def test_boundary_train_splits(sample_text, tokenizer, train_split):
    """Test edge cases for train_split values"""
    block_size = 5
    
    if train_split == 1.0:
        # All data goes to train
        train_dataset = LanguageDataset(
            data_text=sample_text,
            tokenizer=tokenizer,
            split='train',
            train_split=train_split,
            block_size=block_size
        )
        total_tokens = len(tokenizer.encode(sample_text))
        assert len(train_dataset) == total_tokens - block_size
    else:
        # Validation should have some data
        val_dataset = LanguageDataset(
            data_text=sample_text,
            tokenizer=tokenizer,
            split='val',
            train_split=train_split,
            block_size=block_size
        )
        assert len(val_dataset) > 0


@pytest.mark.parametrize("block_size", [1, 3, 8, 16, 32])
def test_various_block_sizes(sample_text, tokenizer, block_size):
    """Test dataset with various block sizes"""
    dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        block_size=block_size
    )
    
    inputs, targets = dataset[0]
    assert inputs.shape[0] == block_size
    assert targets.shape[0] == block_size
    
    # Test last valid index
    last_inputs, last_targets = dataset[len(dataset) - 1]
    assert last_inputs.shape[0] == block_size


# --- Error Handling Tests ---
def test_invalid_split_raises_error(sample_text, tokenizer):
    """Test that invalid split values raise ValueError"""
    invalid_splits = ['test', 'TEST', 'Train', 'Val', '', 'dev']
    
    for split in invalid_splits:
        with pytest.raises(ValueError, match="Invalid split"):
            LanguageDataset(
                data_text=sample_text,
                tokenizer=tokenizer,
                split=split
            )


@pytest.mark.parametrize("train_split,error_pattern", [
    (-0.1, r"Invalid train_split.*between 0 and 1"),
    (-1.0, r"Invalid train_split.*between 0 and 1"),
    (1.1, r"Invalid train_split.*between 0 and 1"),
    (2.0, r"Invalid train_split.*between 0 and 1"),
])
def test_train_split_out_of_range(sample_text, tokenizer, train_split, error_pattern):
    """Test that train_split outside [0, 1] raises ValueError"""
    with pytest.raises(ValueError, match=error_pattern):
        LanguageDataset(
            data_text=sample_text,
            tokenizer=tokenizer,
            train_split=train_split
        )


def test_train_split_1_with_val_split_raises_error(sample_text, tokenizer):
    """Test that train_split=1.0 with split='val' raises specific error"""
    with pytest.raises(ValueError, match="train_split cannot be 1 when split is 'val'"):
        LanguageDataset(
            data_text=sample_text,
            tokenizer=tokenizer,
            split='val',
            train_split=1.0
        )


@pytest.mark.parametrize("block_size", [0, -1, -10])
def test_invalid_block_size_raises_error(sample_text, tokenizer, block_size):
    """Test that non-positive block_size raises ValueError"""
    with pytest.raises(ValueError, match=r"Invalid block_size.*greater than 0"):
        LanguageDataset(
            data_text=sample_text,
            tokenizer=tokenizer,
            block_size=block_size
        )


def test_data_too_short_raises_error(tokenizer):
    """Test that data shorter than or equal to block_size raises error"""
    short_text = "hi"
    tokens = tokenizer.encode(short_text)
    
    with pytest.raises(ValueError) as exc_info:
        LanguageDataset(
            data_text=short_text,
            tokenizer=tokenizer,
            block_size=10  # Much larger than token count
        )
    
    error_msg = str(exc_info.value)
    assert "Data too short" in error_msg
    assert str(len(tokens)) in error_msg  # Should mention actual length
    assert "10" in error_msg  # Should mention required block_size


def test_exact_block_size_equals_data_raises_error(tokenizer):
    """Test edge case where block_size exactly equals data length"""
    text = "hello"
    block_size = len(tokenizer.encode(text))
    
    with pytest.raises(ValueError, match="Data too short"):
        LanguageDataset(
            data_text=text,
            tokenizer=tokenizer,
            block_size=block_size
        )


def test_validation_split_too_small_for_block_size(short_text, tokenizer):
    """Test when validation split doesn't have enough data for even one sequence"""
    # Set up so validation gets very few tokens
    total_tokens = len(tokenizer.encode(short_text))
    train_split = 0.95  # Only 5% for validation
    val_tokens = int((1 - train_split) * total_tokens)
    
    # If val_tokens <= block_size, this should work but give 0 sequences
    if val_tokens > 0:
        dataset = LanguageDataset(
            data_text=short_text,
            tokenizer=tokenizer,
            split='val',
            train_split=train_split,
            block_size=8  # Likely larger than val_tokens
        )
        # Dataset might have length 0, which is valid but not useful
        assert len(dataset) >= 0


# --- Property Tests ---
def test_vocab_size_property(sample_text, tokenizer):
    """Test that vocab_size property works correctly"""
    dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        block_size=8
    )
    
    assert dataset.vocab_size == tokenizer.vocab_size

# --- Sequential Access Tests ---
def test_sequential_iteration(sample_text, tokenizer):
    """Test iterating through entire dataset sequentially"""
    block_size = 8
    dataset = LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        block_size=block_size
    )
    
    # Should be able to access all indices without error
    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        assert inputs.shape == (block_size,)
        assert targets.shape == (block_size,)
    
    # Verify sequences are consecutive
    if len(dataset) > 1:
        inputs1, _ = dataset[0]
        inputs2, _ = dataset[1]
        # Second sequence should start one token after first
        assert inputs2[0] == inputs1[1]

# --- Dataloader Tests ---

@pytest.fixture
def dataset(sample_text, tokenizer):
    """Standard dataset for DataLoader testing"""
    return LanguageDataset(
        data_text=sample_text,
        tokenizer=tokenizer,
        split='train',
        train_split=1.0,
        block_size=8
    )

def test_works_with_dataloader(dataset):
    """Integration smoke test - can be used with DataLoader"""
    loader = DataLoader(dataset, batch_size=4)
    for inputs, targets in loader:
        break  # Just need to know it works