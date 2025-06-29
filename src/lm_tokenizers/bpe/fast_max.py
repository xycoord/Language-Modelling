from .types import TokenPair, WeightedChunk
from .deduplicated import DeduplicatedBPETokenizer
from .utils import count_pairs_chunked
from .incremental import IncrementalBPEWorker
from sortedcontainers import SortedDict
from typing import Optional

class FastMaxBPETokenizer(DeduplicatedBPETokenizer):
    """Byte Pair Encoding tokenizer with regex-based text chunking.
    
    Optimisations:
    - Uses a dual data structure to maintain pair counts with O(1) most common pair search
    Inherited Optimisations:
    - Regex chunking prevents merges across chunk boundaries
    - Chunk caching during encoding to avoid redundant BPE operations
    - Chunk deduplication during training for efficiency
    - Incremental pair counting to only update counts for pairs that have changed
    - Stores which chunks contain each pair for efficient searching
    """
    def train(self, text: str, target_vocab_size: int, min_merge_count: int = 2):
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

        print("Preprocessing text...")
        chunks = self._preprocess_text_train(text)

        merges = self.merges.copy()
        vocab = self.vocab.copy()

        pair_counts = PairCountTracker(chunks)
        worker = IncrementalBPEWorker(chunks)

        print("Training...")
        while next_token < target_vocab_size:

            most_common_pair, max_count = pair_counts.get_most_common_pair()
            if most_common_pair is None or max_count < min_merge_count:
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
            pair_counts.apply_deltas(deltas)

            next_token += 1

        self.merges = merges
        self.vocab = vocab
        self.vocab_size = len(vocab)
        print("Training complete")


class PairCountTracker:
    """
    Tracks pair counts using two synchronised data structures for efficient
    max-pair finding and updates.

    SortedDict allows:
    - O(1) max-pair finding
    - O(1) pair count updates per delta [O(deltas)]

    Invariants: 
    - For every pair p: p is in pair_counts with count c iff p is in counts_to_pairs[c]
    - All values in pair_counts are > 0
    - No empty sets exist in counts_to_pairs.values()
    """
    def __init__(self, chunks: list[WeightedChunk]):
        self.pair_counts: dict[TokenPair, int] = count_pairs_chunked(chunks)
        self.counts_to_pairs: SortedDict = self._init_counts_to_pairs(self.pair_counts)
    
    @staticmethod
    def _init_counts_to_pairs(pair_counts: dict[TokenPair, int]) -> SortedDict:
        """Initialises the counts_to_pairs data structure to satisfy the invariants"""
        counts_to_pairs = SortedDict()
        for pair, count in pair_counts.items():
            if count not in counts_to_pairs:
                counts_to_pairs[count] = set()
            counts_to_pairs[count].add(pair)
        return counts_to_pairs
            
    def __len__(self) -> int:
        """Returns the number of pairs in the state"""
        return len(self.pair_counts)
        
    def __bool__(self) -> bool:
        """Returns True if the state is not empty"""
        return bool(self.pair_counts)

    def get_most_common_pair(self) -> tuple[Optional[TokenPair], int]:
        """
        Gets the most common pair. O(1) time vs O(n) for the naive approach
        Note: this doesn't affect the state
        """
        if len(self.counts_to_pairs) == 0:
            return None, 0

        max_count: int = self.counts_to_pairs.peekitem(-1)[0]  # type: ignore
        
        pairs_with_max_count = self.counts_to_pairs[max_count]
        
        # use min() for a deterministic tie-breaker
        # cheap because the set is probably small
        most_common_pair = min(pairs_with_max_count)
        
        return most_common_pair, max_count

    def apply_deltas(self, deltas: dict[TokenPair, int]) -> None:
        """Applies deltas to the state.
        deltas are an accurate reflection of the net change in pair counts.
        Maintains both pair_counts and counts_to_pairs in sync.
        """
        for pair, delta in deltas.items():
            if delta == 0:
                continue
            
            old_count = self.pair_counts.get(pair, 0)
            new_count = old_count + delta

            if new_count < 0:
                raise ValueError(f"New count is negative: {new_count}, violating invariant")

            if old_count > 0: # pair existed previously
                old_set = self.counts_to_pairs[old_count]
                old_set.remove(pair)
                # clean up
                if not old_set: # remove empty set (essential to remove false max counts)
                    del self.counts_to_pairs[old_count]
                if new_count <= 0: # pair no longer exists
                    del self.pair_counts[pair]
            
            if new_count > 0: # pair exists now
                self.pair_counts[pair] = new_count
                if new_count not in self.counts_to_pairs:
                    self.counts_to_pairs[new_count] = set()
                self.counts_to_pairs[new_count].add(pair)
                