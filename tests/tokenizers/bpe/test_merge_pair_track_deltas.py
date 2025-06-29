import pytest
from collections import defaultdict
from copy import deepcopy

from src.lm_tokenizers.bpe.utils import merge_pair_track_deltas, merge_pair_track_deltas_in_place


# Adapters

def in_place_adapter(token_seq, pair, new_token, copies):
            token_seq_copy = deepcopy(token_seq)
            deltas = merge_pair_track_deltas_in_place(token_seq_copy, pair, new_token, copies)
            return token_seq_copy, deltas

@pytest.fixture(params=["pure", "in_place"])
def merge_function(request):
    """Fixture that provides both implementations with normalised interface"""
    if request.param == "pure":
        return merge_pair_track_deltas
    elif request.param == "in_place":
        return in_place_adapter

# Functional helper functions for validating test results

def calculate_pair_counts(token_seq, copies=1):
    """Calculate frequency of each adjacent pair in sequence"""
    counts = defaultdict(int)
    for i in range(len(token_seq) - 1):
        counts[(token_seq[i], token_seq[i+1])] += copies
    return counts


def subtract_pair_counts(counts_1, counts_2):
    """Calculate counts_1 - counts_2, omitting zeros"""
    counts_diff = defaultdict(int)
    all_pairs = set(counts_1.keys()) | set(counts_2.keys())
    for pair in all_pairs:
        diff = counts_1.get(pair, 0) - counts_2.get(pair, 0)
        if diff != 0:
            counts_diff[pair] = diff
    return counts_diff


def deltas_are_equal(deltas_1, deltas_2) -> bool:
    """Check if two delta dicts are equivalent, ignoring zero values"""
    all_pairs = set(deltas_1.keys()) | set(deltas_2.keys())
    
    for pair in all_pairs:
        val_1 = deltas_1.get(pair, 0)
        val_2 = deltas_2.get(pair, 0)
        # Only compare if at least one is non-zero
        if val_1 != 0 or val_2 != 0:
            if val_1 != val_2:
                return False
    return True


def deltas_are_correct(original_seq, result_seq, deltas, copies) -> bool:
    """Check if deltas correctly represent the pair count changes"""
    # Expected deltas = (after - before) * copies
    expected_deltas = subtract_pair_counts(
        calculate_pair_counts(result_seq, copies),
        calculate_pair_counts(original_seq, copies)
    )
    
    return deltas_are_equal(deltas, expected_deltas)


def merge_count_is_correct(deltas, original_seq, result_seq, copies) -> bool:
    """Check if net pair reduction matches number of merges"""
    positive_sum = sum(v for v in deltas.values() if v > 0)
    negative_sum = sum(abs(v) for v in deltas.values() if v < 0)
    net_reduction = (negative_sum - positive_sum) // abs(copies) if copies != 0 else 0
    merges_performed = len(original_seq) - len(result_seq)
    return net_reduction == merges_performed


def copies_are_valid(deltas, copies) -> bool:
    """Check if all delta values are multiples of copies"""
    if copies == 0:
        return True
    return all(delta % copies == 0 for delta in deltas.values())


@pytest.mark.parametrize("token_seq,pair,new_token,expected_seq,expected_deltas", [
    # Empty sequence
    ([], (1, 2), 9, [], {}),
    # Single token
    ([1], (1, 2), 9, [1], {}),
    # Two tokens not matching pair
    ([1, 3], (1, 2), 9, [1, 3], {}),
    # Pair tokens not in sequence at all
    ([3, 4, 5], (1, 2), 9, [3, 4, 5], {}),
])
def test_edge_cases(token_seq, pair, new_token, expected_seq, expected_deltas, merge_function):
    """Test edge cases: empty, single token, no matches"""
    result_seq, result_deltas = merge_pair_track_deltas(token_seq, pair, new_token, copies=1)
    
    assert result_seq == expected_seq, \
        f"Expected sequence {expected_seq}, got {result_seq}"
    assert deltas_are_equal(result_deltas, expected_deltas), \
        f"Expected deltas {expected_deltas}, got {dict(result_deltas)}"


@pytest.mark.parametrize("token_seq,pair,new_token,expected_seq,expected_deltas", [
    # Merge at start
    ([1, 2, 3], (1, 2), 9, [9, 3], 
     {(1, 2): -1, (2, 3): -1, (9, 3): 1}),
    # Merge in middle
    ([0, 1, 2, 3], (1, 2), 9, [0, 9, 3],
     {(1, 2): -1, (0, 1): -1, (2, 3): -1, (0, 9): 1, (9, 3): 1}),
    # Merge at end
    ([0, 1, 2], (1, 2), 9, [0, 9],
     {(1, 2): -1, (0, 1): -1, (0, 9): 1}),
    # Exactly two tokens (matching pair)
    ([1, 2], (1, 2), 9, [9],
     {(1, 2): -1}),
    # Partial match at end (only first element of pair)
    ([0, 1, 2, 1], (1, 2), 9, [0, 9, 1],
     {(1, 2): -1, (0, 1): -1, (2, 1): -1, (0, 9): 1, (9, 1): 1}),
])
def test_single_merge_positions(token_seq, pair, new_token, expected_seq, expected_deltas, merge_function):
    """Test single merge at different positions in sequence"""
    result_seq, result_deltas = merge_function(token_seq, pair, new_token, copies=1)
    
    assert result_seq == expected_seq, \
        f"Expected sequence {expected_seq}, got {result_seq}"
    assert deltas_are_equal(result_deltas, expected_deltas), \
        f"Expected deltas {expected_deltas}, got {dict(result_deltas)}"
    assert deltas_are_correct(token_seq, result_seq, result_deltas, 1), \
        "Deltas do not match pair count changes"


@pytest.mark.parametrize("token_seq,pair,new_token,expected_seq,expected_deltas", [
    # Non-consecutive merges
    ([1, 2, 3, 1, 2], (1, 2), 9, [9, 3, 9],
     {(1, 2): -2, (2, 3): -1, (3, 1): -1, (9, 3): 1, (3, 9): 1}),
    # Consecutive pairs (no overlap)
    ([1, 2, 3, 2, 3], (2, 3), 9, [1, 9, 9],
     {(2, 3): -2, (1, 2): -1, (3, 2): -1, (1, 9): 1, (9, 9): 1}),
    # Entire sequence is pairs
    ([1, 2, 1, 2, 1, 2], (1, 2), 9, [9, 9, 9],
     {(1, 2): -3, (2, 1): -2, (9, 9): 2}),
    # Alternating pattern
    ([3, 1, 2, 3, 1, 2, 4], (1, 2), 9, [3, 9, 3, 9, 4],
     {(1, 2): -2, (3, 1): -2, (2, 3): -1, (2, 4): -1, 
      (3, 9): 2, (9, 3): 1, (9, 4): 1}),
    # Interleaved matches
    ([1, 2, 5, 1, 2], (1, 2), 9, [9, 5, 9],
     {(1, 2): -2, (2, 5): -1, (5, 1): -1, (9, 5): 1, (5, 9): 1}),
])
def test_multiple_merges(token_seq, pair, new_token, expected_seq, expected_deltas, merge_function):
    """Test multiple merges in various patterns"""
    result_seq, result_deltas = merge_function(token_seq, pair, new_token, copies=1)
    
    assert result_seq == expected_seq, \
        f"Expected sequence {expected_seq}, got {result_seq}"
    assert deltas_are_equal(result_deltas, expected_deltas), \
        f"Expected deltas {expected_deltas}, got {dict(result_deltas)}"
    assert deltas_are_correct(token_seq, result_seq, result_deltas, 1), \
        "Deltas do not match pair count changes"


@pytest.mark.parametrize("token_seq,pair,new_token,expected_seq,expected_deltas", [
    # BBB with pair BB → XB (using 2,2,2)
    ([2, 2, 2], (2, 2), 9, [9, 2],
     {(2, 2): -2, (9, 2): 1}),
    # BBBB with pair BB → XX (using 2,2,2,2)
    ([2, 2, 2, 2], (2, 2), 9, [9, 9],
     {(2, 2): -3, (9, 9): 1}),
    # BBBBB with pair BB → XXB (using 2,2,2,2,2)
    ([2, 2, 2, 2, 2], (2, 2), 9, [9, 9, 2],
     {(2, 2): -4, (9, 9): 1, (9, 2): 1}),
    # BBBBBB with pair BB → XXX (using 2,2,2,2,2,2)
    ([2, 2, 2, 2, 2, 2], (2, 2), 9, [9, 9, 9],
     {(2, 2): -5, (9, 9): 2}),
    # ABABA with pair AB → XXA (using 1,2,1,2,1)
    ([1, 2, 1, 2, 1], (1, 2), 9, [9, 9, 1],
     {(1, 2): -2, (2, 1): -2, (9, 9): 1, (9, 1): 1}),
    # ABABAB with pair AB → XXX (using 1,2,1,2,1,2)
    ([1, 2, 1, 2, 1, 2], (1, 2), 9, [9, 9, 9],
     {(1, 2): -3, (2, 1): -2, (9, 9): 2}),
    # Pattern where new_token creates mergeable pair (1,3,1 with pair 1,3 → 1)
    ([1, 3, 1], (1, 3), 1, [1, 1],
     {(1, 3): -1, (3, 1): -1, (1, 1): 1}),
])
def test_overlapping_patterns(token_seq, pair, new_token, expected_seq, expected_deltas, merge_function):
    """Test overlapping patterns and greedy matching behaviour"""
    result_seq, result_deltas = merge_function(token_seq, pair, new_token, copies=1)
    
    assert result_seq == expected_seq, \
        f"Expected sequence {expected_seq}, got {result_seq}"
    assert deltas_are_equal(result_deltas, expected_deltas), \
        f"Expected deltas {expected_deltas}, got {dict(result_deltas)}"
    assert deltas_are_correct(token_seq, result_seq, result_deltas, 1), \
        "Deltas do not match pair count changes"


@pytest.mark.parametrize("copies,token_seq,pair,new_token", [
    (0, [1, 2, 3], (1, 2), 9),
    (1, [1, 2, 3], (1, 2), 9),
    (5, [1, 2, 3], (1, 2), 9),
    (-2, [1, 2, 3], (1, 2), 9),
    (1000, [1, 2, 3], (1, 2), 9),
])
def test_copies_parameter(copies, token_seq, pair, new_token, merge_function):
    """Test that copies parameter correctly scales all deltas"""
    result_seq, result_deltas = merge_function(token_seq, pair, new_token, copies)
    
    # Expected sequence doesn't change with copies
    assert result_seq == [9, 3], \
        f"Copies parameter shouldn't affect sequence, got {result_seq}"
    
    # All deltas should be multiples of copies
    assert copies_are_valid(result_deltas, copies), \
        f"Not all deltas are multiples of copies={copies}"
    
    # Specific checks for this test case
    if copies != 0:
        assert result_deltas[(1, 2)] == -copies, \
            f"Expected delta for merged pair to be -{copies}"
        assert result_deltas[(9, 3)] == copies, \
            f"Expected delta for new pair to be {copies}"


@pytest.mark.parametrize("token_seq,pair,new_token,merge_position,has_prev,has_next", [
    # Isolated merge (no neighbours)
    ([1, 2], (1, 2), 9, 0, False, False),
    # Merge with preceding token only
    ([0, 1, 2], (1, 2), 9, 1, True, False),
    # Merge with following token only
    ([1, 2, 3], (1, 2), 9, 0, False, True),
    # Merge with both neighbours
    ([0, 1, 2, 3], (1, 2), 9, 1, True, True),
])
def test_delta_tracking_scenarios(token_seq, pair, new_token, merge_position, has_prev, has_next, merge_function):
    """Test delta tracking for different merge contexts"""
    result_seq, result_deltas = merge_function(token_seq, pair, new_token, copies=1)
    
    # Always remove the merged pair
    assert result_deltas[pair] == -1, \
        "Merged pair delta should be -1"
    
    if has_prev:
        prev_token = token_seq[merge_position - 1]
        # Should remove (prev, first)
        assert result_deltas[(prev_token, pair[0])] == -1, \
            "Should remove preceding pair"
        # Should add (prev, new_token)
        assert result_deltas[(prev_token, new_token)] == 1, \
            "Should add new preceding pair"
    
    if has_next:
        next_token = token_seq[merge_position + 2]
        # Should remove (second, next)
        assert result_deltas[(pair[1], next_token)] == -1, \
            "Should remove following pair"
        # Should add (new_token, next)
        assert result_deltas[(new_token, next_token)] == 1, \
            "Should add new following pair"
    
    assert deltas_are_correct(token_seq, result_seq, result_deltas, 1), \
        "Deltas do not match pair count changes"


def test_delta_conservation(merge_function):
    """Verify net pair reduction matches merge count across various cases"""
    test_cases = [
        ([1, 2, 3, 1, 2], (1, 2), 9, 3),
        ([2, 2, 2, 2], (2, 2), 9, 1),
        ([1, 2, 1, 2, 1, 2], (1, 2), 9, 10),
    ]
    
    for token_seq, pair, new_token, copies in test_cases:
        result_seq, result_deltas = merge_function(token_seq, pair, new_token, copies)
        assert merge_count_is_correct(result_deltas, token_seq, result_seq, copies), \
            "Net pair reduction doesn't match merge count"
        assert deltas_are_correct(token_seq, result_seq, result_deltas, copies), \
            "Deltas do not match pair count changes"


def test_immutability(merge_function):
    """Verify original token sequence is not modified"""
    original_seq = [1, 2, 3, 1, 2]
    seq_copy = deepcopy(original_seq)
    
    merge_function(original_seq, (1, 2), 9, copies=1)
    
    assert original_seq == seq_copy, \
        "Original sequence was modified"


def test_new_token_equals_pair_element(merge_function):
    """Test special case where new_token equals one of the pair elements"""
    # new_token equals first element
    result_seq, result_deltas = merge_function(
        [1, 2, 3], (1, 2), 1, copies=1
    )
    assert result_seq == [1, 3], \
        f"Expected [1, 3], got {result_seq}"
    assert deltas_are_correct([1, 2, 3], result_seq, result_deltas, 1), \
        "Deltas do not match pair count changes"
    
    # new_token equals second element  
    result_seq, result_deltas = merge_function(
        [1, 2, 3], (1, 2), 2, copies=1
    )
    assert result_seq == [2, 3], \
        f"Expected [2, 3], got {result_seq}"
    assert deltas_are_correct([1, 2, 3], result_seq, result_deltas, 1), \
        "Deltas do not match pair count changes"


def test_same_new_pair_multiple_sources(merge_function):
    """Verify delta accumulation when same pair is created multiple times"""
    # Single merge operation creates multiple 9-9 pairs
    result_seq, result_deltas = merge_function(
        [9, 1, 2, 9, 1, 2, 9], (1, 2), 9, copies=1
    )
    assert result_seq == [9, 9, 9, 9, 9], \
        f"Expected [9, 9, 9, 9, 9], got {result_seq}"
    
    # Should have created 4 (9,9) pairs total (accumulated)
    assert result_deltas[(9, 9)] == 4, \
        f"Expected (9, 9) delta to be 4, got {result_deltas[(9, 9)]}"
    
    # Verify all expected deltas
    assert result_deltas[(1, 2)] == -2, "Should remove 2 (1,2) pairs"
    assert result_deltas[(9, 1)] == -2, "Should remove 2 (9,1) pairs"
    assert result_deltas[(2, 9)] == -2, "Should remove 2 (2,9) pairs"
    
    assert deltas_are_correct([9, 1, 2, 9, 1, 2, 9], result_seq, result_deltas, 1), \
        "Deltas do not match pair count changes"


def test_single_token_pair(merge_function):
    """Test special case where pair consists of same token repeated"""
    result_seq, result_deltas = merge_function(
        [1, 1, 1], (1, 1), 9, copies=1
    )
    assert result_seq == [9, 1], \
        f"Expected [9, 1], got {result_seq}"
    
    # Should have removed two (1,1) pairs and added one (9,1) pair
    assert result_deltas[(1, 1)] == -2, \
        f"Expected (1, 1) delta to be -2"
    assert result_deltas[(9, 1)] == 1, \
        f"Expected (9, 1) delta to be 1"
    
    assert deltas_are_correct([1, 1, 1], result_seq, result_deltas, 1), \
        "Deltas do not match pair count changes"