from ..base import Tokenizer, Token
from .utils import count_pairs, merge_pair, GPT4_SPLIT_PATTERN
from ..save_utils import atomic_save_json, safe_load_json
import base64
from pathlib import Path
import regex as re


class ChunkedBPETokenizer(Tokenizer):
    """Byte Pair Encoding tokenizer with regex-based text chunking."""

    def __init__(self, split_pattern: str = GPT4_SPLIT_PATTERN):
        """Initialize tokenizer with UTF-8 byte vocabulary (0-255).
        vocab: token -> bytes[]
        merges: token_pair -> new_token
        """
        super().__init__()
        self.split_pattern = split_pattern
        try:
            self.split_regex = re.compile(split_pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        
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


    def train(self, text: str, target_vocab_size: int, min_merge_count: int = 2):
        """Learn BPE merges from text to expand vocabulary.
        Merges across chunks are not allowed.
        Modifies the tokenizer in-place.
        
        Args:
            text: Training text as a string
            target_vocab_size: Desired vocabulary size (must be >= current vocab size) 
            min_merge_count: Minimum number of occurrences for a pair to be merged
        """
        if target_vocab_size < self.vocab_size:
            raise ValueError("Target vocabulary size must be >= the current vocabulary size")
        
        next_token = self.vocab_size

        print("Preprocessing text...")
        chunks = self._preprocess_text(text)

        merges = self.merges.copy()
        vocab = self.vocab.copy()

        print("Training...")
        while next_token < target_vocab_size:

            pair_counts = {}
            for chunk in chunks:
                count_pairs(chunk, counts=pair_counts)
            
            if not pair_counts: 
                # no more pairs to merge, we're done
                break

            most_common_pair = max(pair_counts, key=lambda x: pair_counts[x])
            if pair_counts[most_common_pair] < min_merge_count:
                # if the pair is not common enough, we're done
                break

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

    def save(self, filepath: str | Path) -> None:
        """Save the trained tokenizer to a file."""
        
        data = {
            "format_version": "1.0",
            "tokenizer_type": "ChunkedBPE",
            "split_pattern": self.split_pattern,
            "vocab_size": self.vocab_size,
            
            # Convert vocab: int -> bytes to string -> base64
            "vocab": {
                str(token): base64.b64encode(byte_seq).decode('ascii')
                for token, byte_seq in self.vocab.items()
            },
            
            # Convert merges: (int, int) -> int to "int,int" -> int
            "merges": {
                f"{pair[0]},{pair[1]}": new_token
                for pair, new_token in self.merges.items()
            }
        }
        
        atomic_save_json(filepath, data)
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'ChunkedBPETokenizer':
        """Load a trained tokenizer from a file."""
        data = safe_load_json(filepath)
        
        if data.get("tokenizer_type") != "ChunkedBPE":
            raise ValueError(f"Expected ChunkedBPE tokenizer, got {data.get('tokenizer_type')}")
        
        tokenizer = cls(split_pattern=data["split_pattern"])
        tokenizer.vocab_size = data["vocab_size"]
        
        # Reconstruct vocab: string -> base64 to int -> bytes
        tokenizer.vocab = {
            int(token_str): base64.b64decode(b64_bytes)
            for token_str, b64_bytes in data["vocab"].items()
        }
        
        # Reconstruct merges: "int,int" -> int to (int, int) -> int
        tokenizer.merges = {
            (int(parts[0]), int(parts[1])): new_token
            for pair_str, new_token in data["merges"].items()
            for parts in [pair_str.split(',')]
        }
        
        return tokenizer