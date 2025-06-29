import pytest
from unittest.mock import Mock
from src.lm_datasets.offset_sampler import OffsetSampler, OFFSET_MULTIPLIER

# ========== Fixtures ==========

@pytest.fixture
def mock_dataset():
    """Factory fixture for creating mock datasets with specified parameters."""
    def _make_dataset(data_length: int, block_size: int):
        dataset = Mock()
        dataset.data = Mock()
        dataset.data.__len__ = Mock(return_value=data_length)
        dataset.block_size = block_size
        return dataset
    return _make_dataset


@pytest.fixture
def sampler_factory(mock_dataset):
    """Factory for creating samplers with custom configs."""
    def _make_sampler(data_length=1000, block_size=128, shuffle=True, epoch=0):
        dataset = mock_dataset(data_length, block_size)
        sampler = OffsetSampler(dataset, shuffle=shuffle)
        if epoch != 0:
            sampler.set_epoch(epoch)
        return sampler
    return _make_sampler


@pytest.fixture
def collect_all_indices():
    """Helper to collect all indices from a sampler over multiple epochs."""
    def _collect(sampler, num_epochs):
        all_indices = []
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            indices = list(sampler)
            all_indices.extend(indices)
        return all_indices
    return _collect


# ========== Iterator Contract Tests ==========

def test_implements_iterator_protocol(sampler_factory):
    """Sampler must properly implement the iterator protocol."""
    sampler = sampler_factory()
    
    iterator = iter(sampler)
    assert hasattr(iterator, '__next__')
    
    first_value = next(iterator)
    assert isinstance(first_value, int)
    
    # Eventually exhausts (StopIteration)
    all_values = [first_value] + list(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


def test_multiple_iter_calls_independent(sampler_factory):
    """Each __iter__ call must return a fresh, independent iterator."""
    sampler = sampler_factory(data_length=500, block_size=128)
    
    iter1 = iter(sampler)
    iter2 = iter(sampler)
    
    list(iter1)  # Exhaust first iterator
    
    values = list(iter2)
    assert len(values) == len(sampler)
    
    assert values == list(sampler)


def test_iterator_reusable(sampler_factory):
    """Can iterate multiple times over the same sampler."""
    sampler = sampler_factory()
    first_run = list(sampler)
    second_run = list(sampler)
    assert first_run == second_run


def test_partial_iteration(sampler_factory):
    """Can partially consume iterator without affecting future iterations."""
    sampler = sampler_factory(data_length=1000, block_size=128)
    
    # Partially consume
    iterator = iter(sampler)
    first_few = [next(iterator) for _ in range(3)]
    # Don't finish consuming iterator
    
    # New iteration should still give complete sequence
    full_sequence = list(sampler)
    assert len(full_sequence) == len(sampler)
    assert full_sequence[:3] == first_few


# ========== Core Logic Tests ==========

def test_stride_pattern_and_offset_calculation(sampler_factory):
    """Core logic: indices follow offset + k*stride pattern with correct offset per epoch."""
    sampler = sampler_factory(data_length=1000, block_size=128, shuffle=False)
    
    test_cases = [
        (0, 0),
        (1, OFFSET_MULTIPLIER),     
        (2, (2 * OFFSET_MULTIPLIER) % 128),     
        (4, (4 * OFFSET_MULTIPLIER) % 128),     
        (128, 0),    
    ]
    
    for epoch, expected_offset in test_cases:
        sampler.set_epoch(epoch)
        indices = list(sampler)
        
        if indices:
            assert indices[0] == expected_offset
            
            for idx in indices:
                assert (idx - expected_offset) % 128 == 0


@pytest.mark.parametrize(
    "data_length,block_size,epoch,expected_len,expected_first",
    [
        pytest.param(1000, 128, 0, 7, 0, id="normal_epoch_0"),
        pytest.param(1000, 128, 1, 7, OFFSET_MULTIPLIER, id="normal_epoch_1"),
        pytest.param(129, 128, 0, 1, 0, id="barely_fits"),
        pytest.param(129, 128, 1, 0, None, id="offset_exceeds_data"),
        pytest.param(256, 128, 0, 1, 0, id="exact_fit"), 
        pytest.param(200, 128, 3, 0, None, id="large_offset"),
    ]
)
def test_len_matches_iteration_count(sampler_factory, data_length, block_size, epoch, 
                                   expected_len, expected_first):
    """Critical: __len__ must match actual iteration count for all configurations."""
    sampler = sampler_factory(data_length, block_size, epoch=epoch, shuffle=False)
    
    assert len(sampler) == expected_len
    
    indices = list(sampler)
    assert len(indices) == expected_len
    
    if expected_first is not None:
        assert indices[0] == expected_first


# ========== Shuffling Tests ==========

def test_shuffle_deterministic_and_different(sampler_factory):
    """Shuffle is deterministic per epoch but different across epochs."""
    # Need to use epochs that produce the same offset to test shuffling alone
    # For stride=128: epoch 0 and epoch 128 both have offset=0
    sampler1 = sampler_factory(shuffle=True, epoch=0)
    sampler2 = sampler_factory(shuffle=True, epoch=0)
    sampler3 = sampler_factory(shuffle=True, epoch=128)
    
    order1 = list(sampler1)
    order2 = list(sampler2)
    order3 = list(sampler3)
    
    # Same epoch = same order
    assert order1 == order2
    # Different epoch with same offset = different order due to shuffle
    assert order1 != order3
    # But same set of indices (since same offset)
    assert set(order1) == set(order3)


def test_shuffle_vs_no_shuffle(sampler_factory):
    """Verify shuffle=False gives ascending order, shuffle=True does not."""
    sampler_ordered = sampler_factory(data_length=1000, block_size=32, shuffle=False)
    sampler_shuffled = sampler_factory(data_length=1000, block_size=32, shuffle=True)
    
    ordered_indices = list(sampler_ordered)
    shuffled_indices = list(sampler_shuffled)
    
    assert ordered_indices == sorted(ordered_indices)
    assert shuffled_indices != sorted(shuffled_indices)
    assert set(ordered_indices) == set(shuffled_indices)


# ========== Coverage and Cycling Tests ==========

@pytest.mark.parametrize(
    "block_size",
    [8, 16, 32],
    ids=lambda x: f"block_size_{x}"
)
def test_full_coverage_no_duplicates(sampler_factory, collect_all_indices, block_size):
    """Critical: all positions covered exactly once over full cycle."""
    data_length = 256
    sampler = sampler_factory(data_length=data_length, block_size=block_size)
    
    all_indices = collect_all_indices(sampler, block_size)
    
    # Every valid position should appear exactly once
    expected_positions = set(range(data_length - block_size))
    actual_positions = set(all_indices)
    
    assert actual_positions == expected_positions
    # No duplicates
    assert len(all_indices) == len(expected_positions)


def test_offset_multiplier_error(sampler_factory):
    """Test that an error is raised if the block size is incompatible with the offset multiplier."""
    with pytest.raises(ValueError):
        sampler_factory(block_size=OFFSET_MULTIPLIER)


def test_offset_cycles_through_all_values(sampler_factory):
    """Verify offsets cycle through all values 0 to stride-1."""
    block_size = 32
    sampler = sampler_factory(block_size=block_size)
    
    offsets_seen = set()
    for epoch in range(block_size * 2):  # Go beyond one cycle to ensure wrapping
        sampler.set_epoch(epoch)
        offsets_seen.add(sampler._offset)
        
        # Should have seen all offsets after block_size epochs
        if epoch == block_size - 1:
            assert offsets_seen == set(range(block_size))


def test_no_artificial_boundaries(sampler_factory):
    """Adjacent positions must eventually appear in same block."""
    block_size = 16
    sampler = sampler_factory(data_length=100, block_size=block_size)
    
    # Test that positions 30 and 31 appear together
    target_positions = {30, 31}
    found_together = False
    
    for epoch in range(block_size):
        sampler.set_epoch(epoch)
        for idx in sampler:
            block_positions = set(range(idx, idx + block_size))
            if target_positions.issubset(block_positions):
                found_together = True
                break
        if found_together:
            break
    
    assert found_together, f"Adjacent positions {target_positions} never appeared in same block"


# ========== Edge Cases ==========

def test_offset_exceeds_data_boundary(sampler_factory):
    """When offset pushes beyond available data, sampler should be empty."""
    # Small dataset where offset can exceed available positions
    sampler = sampler_factory(data_length=130, block_size=128)
    # max_start = 130 - 128 = 2
    
    # Epoch 1: offset = OFFSET_MULTIPLIER, which is > 2
    sampler.set_epoch(1)
    assert sampler._offset == OFFSET_MULTIPLIER
    assert sampler._offset > sampler._max_start
    
    # Should yield no indices
    assert len(sampler) == 0
    assert list(sampler) == []


def test_varying_data_sizes(sampler_factory):
    """Test correct behavior across various data size edge cases."""
    test_cases = [
        # (data_length, block_size, description)
        (129, 128, "barely_larger_than_block"),
        (256, 128, "exact_multiple"),
        (255, 128, "one_less_than_multiple"),
        (998, 997, "nearly_entire_dataset"),
    ]
    
    for data_length, block_size, desc in test_cases:
        sampler = sampler_factory(data_length=data_length, block_size=block_size)
        
        indices = list(sampler)
        assert len(indices) == len(sampler), f"Failed for {desc}"
        
        for idx in indices:
            assert 0 <= idx <= data_length - block_size, f"Invalid index {idx} for {desc}"


# ========== Integration Test ==========

def test_dataloader_compatibility(sampler_factory):
    """Verify sampler works correctly with PyTorch DataLoader patterns."""
    sampler = sampler_factory(data_length=1000, block_size=128)
    
    # Simulate what DataLoader does
    total_samples = 0
    batch_size = 4
    
    for epoch in range(3):
        sampler.set_epoch(epoch)
        
        # Simulate batching
        iterator = iter(sampler)
        batch_count = 0
        
        while True:
            batch = []
            try:
                for _ in range(batch_size):
                    batch.append(next(iterator))
            except StopIteration:
                if batch:  # Partial batch
                    batch_count += 1
                break
            batch_count += 1
        
        expected_batches = (len(sampler) + batch_size - 1) // batch_size
        assert batch_count == expected_batches
        
        epoch_samples = sum(1 for _ in sampler)
        assert epoch_samples == len(sampler)