from ..base import Tokenizer, Token
from .utils import count_pairs, merge_pair

import regex as re


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class ChunkedBPETokenizer(Tokenizer):
    """Byte Pair Encoding tokenizer with regex-based text chunking."""

    def __init__(self, split_pattern: str = GPT4_SPLIT_PATTERN):
        """Initialize tokenizer with UTF-8 byte vocabulary (0-255).
        vocab: token -> bytes[]
        merges: token_pair -> new_token
        """
        self.split_pattern = split_pattern
        self.split_regex = re.compile(split_pattern)
        self.vocab_size = 256
        self.vocab = {token: bytes([token]) for token in range(self.vocab_size)}
        self.merges = {}

    def encode(self, text: str) -> list[Token]:
        """Convert a string to a list of tokens"""
        chunks = self._preprocess_text(text)

        token_seq = []
        for chunk in chunks:
            token_seq.extend(self._encode_chunk(chunk))

        return token_seq

    def _encode_chunk(self, chunk: list[Token]) -> list[Token]:
        """Apply BPE merges to a token sequence"""
        token_seq = list(chunk)
        while len(token_seq) >= 2:
            pair_counts = count_pairs(token_seq)

            pair = min(pair_counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            new_token = self.merges[pair]
            token_seq = merge_pair(token_seq, pair, new_token)

        return token_seq

    def decode(self, tokens: list[Token]) -> str:
        """Convert a list of tokens to a string"""
        byte_sequences = [self.vocab[token] for token in tokens]
        text = self._postprocess_text(byte_sequences)
        return text


    def train(self, text: str, target_vocab_size: int):
        """Learn BPE merges from text to expand vocabulary.
        Merges across chunks are not allowed.
        Modifies the tokenizer in-place.
        
        Args:
            text: Training text as a string
            target_vocab_size: Desired vocabulary size (must be >= 256) 
        """
        assert target_vocab_size >= self.vocab_size
        next_token = self.vocab_size

        print("Preprocessing text...")
        chunks = self._preprocess_text(text)

        merges = self.merges.copy()
        vocab = self.vocab.copy()

        print("Training...")
        while next_token < target_vocab_size:

            pair_counts = {}
            for chunk in chunks:
                pair_counts.update(count_pairs(chunk, counts=pair_counts))
            
            if not pair_counts: 
                # no more pairs to merge, we're done
                break

            most_common_pair = max(pair_counts, key=pair_counts.get)

            # mint a new token
            new_token = next_token

            # update merges
            merges[most_common_pair] = new_token

            # update vocab
            vocab[new_token] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]

            # merge the most common pair
            chunks = [merge_pair(chunk, most_common_pair, new_token) for chunk in chunks]

            next_token += 1

        self.merges = merges
        self.vocab = vocab
        self.vocab_size = len(vocab)
        print("Training complete")

    def _preprocess_text(self, text: str) -> list[list[Token]]:
        """Convert a string to a list of chunks of tokens (UTF-8 bytes)"""
        text_chunks = re.findall(self.split_regex, text)
        chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        return chunks
    
    def _postprocess_text(self, byte_sequences: list[bytes]) -> str:
        """Convert a list of UTF-8 byte sequences to a string"""
        text_bytes = b"".join(byte_sequences)
        text = text_bytes.decode('utf-8', errors='replace')
        return text