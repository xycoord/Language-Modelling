from ..base import Token
from .types import TokenPair, WeightedChunk
from .deduplicated import DeduplicatedBPETokenizer
from .utils import count_pairs_chunked, merge_pair_track_deltas_in_place, apply_deltas
from collections import defaultdict

class IncrementalBPETokenizer(DeduplicatedBPETokenizer):
    """Byte Pair Encoding tokenizer with regex-based text chunking.
    
    Optimisations:
    - Incremental pair counting to only update counts for pairs that have changed
    - Stores which chunks contain each pair for efficient searching
    Inherited Optimisations:
    - Regex chunking prevents merges across chunk boundaries
    - Chunk caching during encoding to avoid redundant BPE operations
    - Chunk deduplication during training for efficiency
    """
    def train(self, chunks: list[WeightedChunk], target_vocab_size: int, min_merge_count: int = 2):
        """Learn BPE merges from text to expand vocabulary.
        Merges across chunks are not allowed.
        Chunks are deduplicated by their text content for efficiency.
        Modifies the tokenizer in-place.
        
        Args:
            text: Training text as a string
            target_vocab_size: Desired vocabulary size (must be >= current vocab size) 
            min_merge_count: Minimum number of occurrences for a pair to be merged
        """
        if target_vocab_size < self.vocab_size:
            raise ValueError("Target vocabulary size must be >= the current vocabulary size")
        
        next_token = self.vocab_size

        merges = self.merges.copy()
        vocab = self.vocab.copy()

        pair_counts = count_pairs_chunked(chunks)
        worker = IncrementalBPEWorker(chunks)

        print("Training...")
        while next_token < target_vocab_size:

            if not pair_counts: 
                # prevent max() from raising an error
                # we're done
                break

            most_common_pair = max(pair_counts.items(), key=lambda item: item[1])[0]

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
            deltas = worker.merge_pair_incremental(most_common_pair, new_token)
            apply_deltas(pair_counts, deltas)

            next_token += 1

        self.merges = merges
        self.vocab = vocab
        self.vocab_size = len(vocab)
        print("Training complete")


class IncrementalBPEWorker:
    """A single worker that manages its own chunk partition for BPE training
    
    Invariants:
    - For every pair p and chunk index i, p exists in chunks[i], iff i is in _pair_chunk_locations[p]
    - No empty sets exist in _pair_chunk_locations.values()
    - Only chunks containing the target pair are modified during merge operations
    """
    
    def __init__(self, chunks: list[WeightedChunk]):
        self.chunks = chunks
        self._pair_chunk_locations = self._init_pair_chunk_locations(chunks)
    
    @staticmethod
    def _init_pair_chunk_locations(chunks: list[WeightedChunk]) -> dict[TokenPair, set[int]]:
        """Build initial index of chunks containing each pair"""
        pair_chunks = defaultdict(set)
        for i, (chunk, _) in enumerate(chunks):
            for j in range(len(chunk) - 1):
                pair_chunks[(chunk[j], chunk[j+1])].add(i)
        
        return pair_chunks

    def _update_pair_chunk_locations(self, chunk_index: int, chunk: list[Token], deltas: dict[TokenPair, int]) -> None:
        """Maintain chunk location index after merge operations to preserve invariant"""

        for pair, delta in deltas.items():
            if delta > 0:
                # It must be a new pair (contains the new token)
                self._pair_chunk_locations[pair].add(chunk_index)
            elif delta < 0:
                # pair may not have been removed entirely from the chunk 
                # so double-check before removing chunk_index from its set
                is_pair_still_in_chunk = any(p == pair for p in zip(chunk, chunk[1:]))
                if not is_pair_still_in_chunk:
                    self._pair_chunk_locations[pair].discard(chunk_index)
                    if not self._pair_chunk_locations[pair]:
                        del self._pair_chunk_locations[pair]

    def merge_pair_incremental(self, pair_to_merge: TokenPair, new_token: Token) -> dict[TokenPair, int]:
        """Incrementally merge a token pair across only the chunks that contain it.
        
        Instead of scanning all chunks, uses the _pair_chunk_locations to process only 
        chunks containing the pair. O(chunks) -> O(merges), merges << chunks
        Updates chunk representations and maintains the _pair_chunk_locations for 
        future incremental operations.
        
        Returns pair count deltas which accurately reflect net changes in global pair counts.
        """
        total_pair_deltas = defaultdict(int)

        if pair_to_merge not in self._pair_chunk_locations:
            return total_pair_deltas

        # list() to avoid modifying the iterator while iterating
        for chunk_index in list(self._pair_chunk_locations[pair_to_merge]):
            chunk, copies = self.chunks[chunk_index]

            deltas = merge_pair_track_deltas_in_place(chunk, pair_to_merge, new_token, copies)

            apply_deltas(total_pair_deltas, deltas)
            self._update_pair_chunk_locations(chunk_index, chunk, deltas)

        return total_pair_deltas
