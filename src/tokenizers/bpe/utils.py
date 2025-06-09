from typing import Optional
from ..base import Token

def count_pairs(token_seq: list[Token], num_copies: int = 1, 
                counts: Optional[dict[tuple[Token, Token], int]] = None
                ) -> dict[tuple[Token, Token], int]:
    """Count frequency of adjacent token pairs in sequence.
    Args:
        token_seq: list of tokens
        num_copies: number of copies of the token_seq chunk
        counts: from previous token_seq (optional)
                used when chunking
    """
    counts = {} if counts is None else counts
    for pair in zip(token_seq, token_seq[1:]):
        counts[pair] = counts.get(pair, 0) + num_copies
    return counts

def merge_pair(token_seq: list[Token], pair: tuple[Token, Token], new_token: Token) -> list[Token]:
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