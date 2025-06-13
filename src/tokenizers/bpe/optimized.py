from ..base import Tokenizer, Token
from .utils import count_pairs, merge_pair, GPT4_SPLIT_PATTERN

import regex as re
from collections import Counter


class OptimizedBPETokenizer(Tokenizer):
    """Byte Pair Encoding tokenizer with regex-based text chunking.
    
    Optimisations:
    - Regex chunking prevents merges across chunk boundaries
    - Chunk caching during encoding to avoid redundant BPE operations
    - Chunk deduplication during training for efficiency
    """

    def __init__(self, split_pattern: str = GPT4_SPLIT_PATTERN):
        """Initialize tokenizer with UTF-8 byte vocabulary (0-255).
        vocab: token -> bytes[]
        merges: token_pair -> new_token
        """
        super().__init__()
        self.split_pattern = split_pattern
        self.split_regex = re.compile(split_pattern)
        self.vocab_size = 256
        self.vocab = {token: bytes([token]) for token in range(self.vocab_size)}
        self.merges = {}

    def encode(self, text: str) -> list[Token]:
        """Convert a string to a list of tokens with chunk caching"""
        chunks = self._preprocess_text_encode(text)

        token_seq = []
        chunk_cache = {}
        for chunk in chunks:
            text_chunk, byte_chunk = chunk
            if text_chunk in chunk_cache:
                token_seq.extend(chunk_cache[text_chunk])
            else:
                token_chunk = self._encode_chunk(byte_chunk)
                token_seq.extend(token_chunk)
                chunk_cache[text_chunk] = token_chunk

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
        Chunks are deduplicated by their text content for efficiency.
        Modifies the tokenizer in-place.
        
        Args:
            text: Training text as a string
            target_vocab_size: Desired vocabulary size (must be >= 256) 
        """
        assert target_vocab_size >= self.vocab_size
        next_token = self.vocab_size

        print("Preprocessing text...")
        chunks = self._preprocess_text_train(text)

        merges = self.merges.copy()
        vocab = self.vocab.copy()

        print("Training...")
        while next_token < target_vocab_size:

            pair_counts = {}
            for token_chunk, num_copies in chunks:
                pair_counts.update(count_pairs(token_chunk, num_copies, pair_counts))
            
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
            chunks = [(merge_pair(token_chunk, most_common_pair, new_token), num_copies) for token_chunk, num_copies in chunks]

            next_token += 1

        self.merges = merges
        self.vocab = vocab
        self.vocab_size = len(vocab)
        print("Training complete")

    def _preprocess_text_encode(self, text: str) -> list[tuple[str, list[bytes]]]:
        """Convert a string to a list of chunks of tokens (UTF-8 bytes)
        Returns:
            chunks: list of (text_chunk, byte_chunk) tuples
                    text_chunk: string
                    byte_chunk: list of UTF-8 bytes
        """
        text_chunks = re.findall(self.split_regex, text)
        chunks = [(text_chunk, list(text_chunk.encode("utf-8"))) for text_chunk in text_chunks]
        return chunks

    def _preprocess_text_train(self, text: str) -> list[tuple[list[bytes], int]]:
        """Convert a string to a list of chunks of tokens (UTF-8 bytes) with chunk deduplication
        Returns:
            chunks: list of (byte_chunk, num_copies) tuples
                    byte_chunk: list of UTF-8 bytes
                    num_copies: number of copies of the chunk
        """
        text_chunks = re.findall(self.split_regex, text)
        chunk_counts = list(Counter(text_chunks).items())
        chunks = [(list(chunk_text.encode("utf-8")), num_copies) for chunk_text, num_copies in chunk_counts]
        return chunks
    
    def _postprocess_text(self, byte_sequences: list[bytes]) -> str:
        """Convert a list of UTF-8 byte sequences to a string"""
        text_bytes = b"".join(byte_sequences)
        text = text_bytes.decode('utf-8', errors='replace')
        return text