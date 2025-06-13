from tokenizers.bpe.utils import count_pairs, merge_pair


# --- count_pairs tests ---
def test_count_pairs_basic():
    """Test basic pair counting"""
    tokens = [1, 2, 3, 2, 3]
    counts = count_pairs(tokens)
    
    assert counts == {
        (1, 2): 1,
        (2, 3): 2,
        (3, 2): 1
    }


def test_count_pairs_empty():
    """Test counting pairs in empty list"""
    assert count_pairs([]) == {}


def test_count_pairs_single_element():
    """Test counting pairs with single element"""
    assert count_pairs([1]) == {}


def test_count_pairs_num_copies():
    """Test num_copies parameter (for weighted counting)"""
    tokens = [1, 2, 3]
    counts = count_pairs(tokens, num_copies=3)
    
    assert counts == {
        (1, 2): 3,
        (2, 3): 3
    }


def test_count_pairs_with_existing_counts():
    """Test accumulating counts across chunks"""
    tokens1 = [1, 2, 3]
    counts = count_pairs(tokens1)
    
    tokens2 = [2, 3, 4]
    counts = count_pairs(tokens2, counts=counts)
    
    assert counts == {
        (1, 2): 1,
        (2, 3): 2,  # Appears in both chunks
        (3, 4): 1
    }


# --- merge_pair tests ---
def test_merge_pair_basic():
    """Test basic pair merging"""
    tokens = [1, 2, 3, 2, 3]
    result = merge_pair(tokens, (2, 3), 99)
    
    assert result == [1, 99, 99]


def test_merge_pair_no_occurrences():
    """Test merging when pair doesn't exist"""
    tokens = [1, 2, 3, 4]
    result = merge_pair(tokens, (5, 6), 99)
    
    assert result == [1, 2, 3, 4]  # Unchanged


def test_merge_pair_overlapping():
    """Test merging with overlapping pairs"""
    tokens = [1, 1, 1, 1]
    result = merge_pair(tokens, (1, 1), 99)
    
    # Should merge non-overlapping pairs only
    assert result == [99, 99]  # Not [99, 1, 1] or [99, 1, 99]


def test_merge_pair_at_boundaries():
    """Test merging pairs at start and end"""
    # Pair at start
    tokens = [1, 2, 3, 4]
    result = merge_pair(tokens, (1, 2), 99)
    assert result == [99, 3, 4]
    
    # Pair at end
    tokens = [1, 2, 3, 4]
    result = merge_pair(tokens, (3, 4), 99)
    assert result == [1, 2, 99]


def test_merge_pair_empty():
    """Test merging in empty list"""
    assert merge_pair([], (1, 2), 99) == []


def test_merge_pair_single_element():
    """Test merging with single element"""
    assert merge_pair([1], (1, 2), 99) == [1]