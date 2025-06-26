from typing import Optional, TypeVar
from collections import defaultdict
from ..base import Token
from .types import TokenPair, WeightedChunk


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def count_pairs(token_seq: list[Token], num_copies: int = 1, 
                counts: Optional[dict[TokenPair, int]] = None
                ) -> dict[TokenPair, int]:
    """Count frequency of adjacent token pairs in sequence.
    Args:
        token_seq: list of tokens
        num_copies: number of copies of the token_seq chunk
        counts: from previous token_seq (optional)
                used when chunking
    Note:
        If counts is provided, it will be updated in place.
    """
    counts = {} if counts is None else counts
    for pair in zip(token_seq, token_seq[1:]):
        counts[pair] = counts.get(pair, 0) + num_copies
    return counts


def count_pairs_chunked(chunks: list[WeightedChunk]) -> dict[TokenPair, int]:
    """Count frequency of adjacent token pairs in a list of token chunks.
    Args:
        chunks: list of token chunks with number of copies
    Returns:
        dict of token pair counts
    """
    counts = defaultdict(int)
    for token_chunk, num_copies in chunks:
        count_pairs(token_chunk, num_copies, counts)
    return counts


def merge_pair(token_seq: list[Token], pair: TokenPair, new_token: Token) -> list[Token]:
    """Replace all occurrences of pair with new_token in sequence."""
    new_token_seq = []
    i = 0
    while i < len(token_seq):
        if token_seq[i] == pair[0] and i < len(token_seq) - 1 and token_seq[i+1] == pair[1]:
            new_token_seq.append(new_token)
            i += 2
        else:
            new_token_seq.append(token_seq[i])
            i += 1
    return new_token_seq

def merge_pair_track_deltas(token_seq: list[Token], pair: TokenPair, new_token: Token, copies: int) -> tuple[list[Token], dict[TokenPair, int]]:
    """Replace all occurrences of pair with new_token in sequence.
    Compute how the merges affect the pair counts (pair_deltas).
    Args:
        token_seq: list of tokens
        pair: token pair to merge
        new_token: new token to replace pair with
        copies: weight of the chunk (token_seq) from deduplication
    Returns:
        new_token_seq: token sequence after merge
        pair_deltas: +copies for new pairs, -copies for removed pairs
    """
    new_token_seq = []
    pair_deltas = defaultdict(int)
    token_seq_len = len(token_seq)
    if token_seq_len < 2:
        return token_seq, pair_deltas

    i = 0
    while i + 1 < token_seq_len:
        if token_seq[i] == pair[0] and token_seq[i+1] == pair[1]:
            # === Update pair deltas ===
            # Each merge effectively operates on the list: 
            # new_token_seq + token_seq[i:]
            # Particularly the 4 tokens:
            # new_token_seq[-1], token_seq[i], token_seq[i+1], token_seq[i+2]
            #       last            first          second           next
            # Remove 3 pairs:
            # (last, first), (first, second), (second, next)
            # Add 2 pairs:
            # (last, new_token), (new_token, next)
            #
            pair_deltas[(token_seq[i], token_seq[i+1])] -= copies        # (first, second)
            if i > 0:
                pair_deltas[(new_token_seq[-1], token_seq[i])] -= copies # (last, first)
                pair_deltas[(new_token_seq[-1], new_token)] += copies    # (last, new_token)
            if i + 2 < token_seq_len:
                pair_deltas[(token_seq[i+1], token_seq[i+2])] -= copies  # (second, next)
                pair_deltas[(new_token, token_seq[i+2])] += copies       # (new_token, next)

            new_token_seq.append(new_token)
            i += 2
        else:
            new_token_seq.append(token_seq[i])
            i += 1

    # If the last pair is not merged, its second token still needs to be added    
    if i < token_seq_len:
        new_token_seq.append(token_seq[i])

    return new_token_seq, pair_deltas

def merge_pair_track_deltas_in_place(token_seq: list[Token], pair: TokenPair, new_token: Token, copies: int) -> dict[TokenPair, int]:
    """
    Merges a pair and tracks deltas by modifying the token_seq list IN-PLACE.
    This is highly memory-efficient as it avoids creating new list objects,
    but the logic is more complex than the original implementation.
    """
    pair_deltas = defaultdict(int)
    
    # We use two pointers: a 'read' pointer to scan the original sequence,
    # and a 'write' pointer to build the new sequence in the same list.
    write_idx = 0
    read_idx = 0

    while read_idx < len(token_seq):
        # Check for a pair match at the current read position
        is_match = (
            read_idx + 1 < len(token_seq) and
            token_seq[read_idx] == pair[0] and
            token_seq[read_idx+1] == pair[1]
        )
        
        if is_match:
            # A pair was found. Calculate deltas BEFORE overwriting the data.
            # 1. The pair being merged is removed.
            pair_deltas[(token_seq[read_idx], token_seq[read_idx+1])] -= copies

            # 2. The pair to the left of the merge site is affected.
            if write_idx > 0:
                pair_deltas[(token_seq[write_idx-1], token_seq[read_idx])] -= copies
                pair_deltas[(token_seq[write_idx-1], new_token)] += copies
            
            # 3. The pair to the right of the merge site is affected.
            if read_idx + 2 < len(token_seq):
                pair_deltas[(token_seq[read_idx+1], token_seq[read_idx+2])] -= copies
                pair_deltas[(new_token, token_seq[read_idx+2])] += copies

            # Perform the in-place write
            token_seq[write_idx] = new_token
            write_idx += 1
            read_idx += 2  # Skip the two tokens we just merged
        else:
            # No match, just copy the token from read to write position.
            # This is only necessary if read and write pointers have diverged.
            if read_idx != write_idx:
                token_seq[write_idx] = token_seq[read_idx]
            write_idx += 1
            read_idx += 1
            
    # After the loop, the new sequence occupies the list up to write_idx.
    # Truncate the list to remove the old, leftover data at the end.
    if write_idx < len(token_seq):
        del token_seq[write_idx:]
    
    return pair_deltas

def apply_deltas(target: dict[TokenPair, int], deltas: dict[TokenPair, int]) -> None:
    """In-place update target with deltas"""
    for pair, delta in deltas.items():
        target[pair] += delta
        if target[pair] == 0:
            del target[pair]


T = TypeVar('T')
def split_list(items: list[T], n: int) -> list[list[T]]:
    """Split a list into n roughly equal parts"""
    chunk_size = len(items) // n
    remainder = len(items) % n
    
    result = []
    start = 0
    
    for i in range(n):
        # First few parts get an extra item if there's a remainder
        end = start + chunk_size + (1 if i < remainder else 0)
        result.append(items[start:end])
        start = end
    
    return result