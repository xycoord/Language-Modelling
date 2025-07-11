from ..base import Token
from .incremental import IncrementalBPEWorker
from .deduplicated import DeduplicatedBPETokenizer
from .types import WeightedChunk, TokenPair
from .fast_max import PairCountTracker
from .utils import split_list
import os
from multiprocessing import Process, Queue
import traceback
import time
from typing import Optional
from queue import Empty


class ParallelBPETokenizer(DeduplicatedBPETokenizer):
    """Byte Pair Encoding tokenizer with regex-based text chunking.
    
    Optimisations:
    - Parallel training with multiple workers
    Inherited Optimisations:
    - Incremental pair counting to only update counts for pairs that have changed
    - Stores which chunks contain each pair for efficient searching
    - Regex chunking prevents merges across chunk boundaries
    - Chunk caching during encoding to avoid redundant BPE operations
    - Chunk deduplication during training for efficiency
    """
    def train(self, chunks: list[WeightedChunk], target_vocab_size: int, min_merge_count: int = 2, num_workers: Optional[int] = None):
        """Learn BPE merges from text to expand vocabulary.
        
        Args:
            text: Training text to learn merges from
            target_vocab_size: Desired vocabulary size (must be >= current vocab size)
            min_merge_count: Minimum pair frequency to consider for merging (default: 2)
            num_workers: Number of worker processes (default: CPU count - 1)
        
        Raises:
            ValueError: If target_vocab_size < current vocab size or num_workers < 1
            RuntimeError: If a worker process encounters an error
        
        Note:
            Modifies the tokenizer in-place. Falls back to sequential processing if num_workers=1.
        """
        if target_vocab_size < self.vocab_size:
            raise ValueError("Target vocabulary size must be >= the current vocabulary size")
        
        next_token = self.vocab_size

        merges = self.merges.copy()
        vocab = self.vocab.copy()

        print("Training...")
        with AdaptiveWorkerPool(chunks, num_workers) as workers:
            while next_token < target_vocab_size:
                most_common_pair, max_count = workers.get_most_common_pair()

                if most_common_pair is None or max_count < min_merge_count:
                    # if the pair is not common enough, we're done
                    break

                # mint a new token
                new_token = next_token

                # update merges
                merges[most_common_pair] = new_token

                # update vocab
                vocab[new_token] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]

                # merge the most common pair and update pair counts
                workers.merge_pair(most_common_pair, new_token)

                next_token += 1

        self.merges = merges
        self.vocab = vocab
        self.vocab_size = len(vocab)
        print("Training complete")


class SynchronousWorkerPool:
    """Manages parallel worker processes for BPE merge operations.
    
    Distributes text chunks across multiple worker processes and coordinates
    merge operations. Each merge is applied synchronously across all workers
    before proceeding to the next merge.
    
    Args:
        chunks: List of weighted text chunks to distribute across workers
        num_workers: Number of worker processes to spawn
    
    Attributes:
        pair_counts: Global counts of all character pairs across all chunks
    """

    WORKER_TIMEOUT = 200 

    def __init__(self, chunks: list[WeightedChunk], num_workers: Optional[int]):
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)
        if num_workers < 1:
            raise ValueError("Number of workers must be at least 1")
        
        self.num_workers: int = num_workers
        print(f"Starting {self.num_workers} worker processes")
        
        self.chunks: list[WeightedChunk] = chunks
        self.pair_counts: Optional[PairCountTracker] = None
        self.input_queues: Optional[list[Queue]] = None
        self.output_queue: Optional[Queue] = None
        self.processes: Optional[list[Process]] = None

    def _start_processes(self):        
        if self.is_running:
            raise RuntimeError("Worker processes already started")

        self.processes = []
        self.input_queues = [Queue() for _ in range(self.num_workers)]
        self.output_queue = Queue()

        partitions = split_list(self.chunks, self.num_workers)
        for i, chunks in enumerate(partitions):
            p = Process(
                target=self.worker_process,
                args=(chunks, self.input_queues[i], self.output_queue)
            )
            p.start()
            self.processes.append(p)

    def start(self):
        """Start worker processes and initialise global pair counting.
    
        Raises:
            RuntimeError: If worker processes are already running
        """
        self._start_processes()
        self.pair_counts = PairCountTracker(self.chunks)

    def _shutdown_processes(self):
        if not self.is_running:
            return

        for queue in self.input_queues:
            try:
                queue.put(None)
            except:
                # Queue might be closed/broken, continue anyway
                pass

        # Wait for all processes to finish
        for process in self.processes:
            if process.is_alive():
                process.join(timeout=self.WORKER_TIMEOUT)
                if process.is_alive():
                    process.terminate()
                    process.join()

        # Reset state to allow restart
        self.input_queues = None
        self.output_queue = None
        self.processes = None


    def shutdown(self):
        """Send shutdown signal to all workers and wait for them to finish.
        
        This method is idempotent and can be called multiple times safely.
        After shutdown, start() can be called again to restart the pool.
        """
        self._shutdown_processes()
        self.pair_counts = None

    @property
    def is_running(self) -> bool:
        return self.processes is not None

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    def merge_pair(self, pair_to_merge: tuple[Token, Token], new_token: Token) -> None:
        """Apply a BPE merge across all workers and update global pair counts.
            
        Args:
            pair_to_merge: The token pair to merge
            new_token: The new token to replace the pair with
        
        Raises:
            RuntimeError: If any worker encounters an error during the merge
        """
        if not self.is_running:
            raise RuntimeError("Worker processes not started")

        for worker_id in range(self.num_workers):
            self.input_queues[worker_id].put((pair_to_merge, new_token))

        for _ in range(self.num_workers):
            try:
                result = self.output_queue.get(timeout=self.WORKER_TIMEOUT)
            except Empty:
                self.shutdown()
                raise RuntimeError("Worker timeout - process may have crashed")
            
            match result:
                case ('deltas', deltas):
                    self.pair_counts.apply_deltas(deltas)
                case ('error', error_msg, traceback_str):
                    self.shutdown()
                    raise RuntimeError(f"Worker error: {error_msg}\n{traceback_str}")

    def get_most_common_pair(self) -> tuple[Optional[TokenPair], int]:
        if self.pair_counts is None:
            raise RuntimeError("Pair counts not initialised")
        return self.pair_counts.get_most_common_pair()

    @staticmethod    
    def worker_process(chunks: list[WeightedChunk], input_queue: Queue, output_queue: Queue):
        """Start worker process that handles BPE merge operations for a given set of chunks.
    
        Args:
            chunks: Text chunks assigned to this worker
            input_queue: Queue for receiving merge tasks and shutdown signals
            output_queue: Queue for sending back results or error messages
        """
        worker = IncrementalBPEWorker(chunks)

        while True:
            try:
                task = input_queue.get(timeout=SynchronousWorkerPool.WORKER_TIMEOUT)
            except Empty:
                return
            
            match task:
                case None:
                    break
                case 'chunks':
                    output_queue.put(('chunks', chunks))
                case ((int(first), int(second)), int(new_token)):
                    try:
                        deltas = worker.merge_pair_incremental((first, second), new_token)
                        output_queue.put(('deltas', deltas))
                    except Exception as e:
                        output_queue.put(('error', str(e), traceback.format_exc()))
                        return  # Exit worker after error
                case _:
                    output_queue.put(('error', f"Unknown task: {task}", ""))
                    return




class AdaptiveWorkerPool(SynchronousWorkerPool):
    """Manages parallel worker processes with adaptive sequential fallback for BPE merge operations.

    Distributes text chunks across multiple worker processes and coordinates
    merge operations. Starts in parallel mode but can switch to sequential processing
    when merge operations become too fast to benefit from parallelization.
    Each merge is applied synchronously across all workers before proceeding to the next merge.

    Args:
        chunks: List of weighted text chunks to distribute across workers
        num_workers: Number of worker processes to spawn
        threshold: Threshold in seconds for switching to sequential mode
        window_size: Window size for averaging merge times
    
    Attributes:
        pair_counts: Global counts of all character pairs across all chunks
    """

    def __init__(self, chunks, num_workers, threshold=0.0015, window_size=5):
        super().__init__(chunks, num_workers)
        self.parallel_mode = None
        self.worker = None
        self.threshold = threshold
        self.merge_times = [float('inf')]*window_size

    def start(self):
        """Start worker processes in parallel mode and initialise global pair counting.
    
        Raises:
            RuntimeError: If worker processes are already running
        """
        super().start()
        self.parallel_mode = True

    def shutdown(self):
        """Send shutdown signal to all workers and wait for them to finish.
        
        This method is idempotent and can be called multiple times safely.
        After shutdown, start() can be called again to restart the pool.
        """
        if self.parallel_mode:
            self._shutdown_processes()
        self.pair_counts = None
        self.worker = None
        self.parallel_mode = False

    def _go_sequential(self):
        if not self.is_running:
            raise RuntimeError("Worker processes not started")

        for worker_id in range(self.num_workers):
            self.input_queues[worker_id].put('chunks')

        latest_chunks = []
        for _ in range(self.num_workers):
            try:
                result = self.output_queue.get(timeout=self.WORKER_TIMEOUT)
            except Empty:
                self.shutdown()
                raise RuntimeError("Worker timeout - process may have crashed")
            
            match result:
                case ('chunks', chunks):
                    latest_chunks.extend(chunks)
                case ('error', error_msg, traceback_str):
                    self.shutdown()
                    raise RuntimeError(f"Worker error: {error_msg}\n{traceback_str}")

        self.worker = IncrementalBPEWorker(latest_chunks)

        self._shutdown_processes()

        self.parallel_mode = False

    def merge_pair(self, pair_to_merge: tuple[Token, Token], new_token: Token) -> None:
        """Apply a BPE merge across all workers and update global pair counts.
            
        Args:
            pair_to_merge: The token pair to merge
            new_token: The new token to replace the pair with
        
        Raises:
            RuntimeError: If any worker encounters an error during the merge
        """

        if self.worker is not None:
            deltas = self.worker.merge_pair_incremental(pair_to_merge, new_token)
            self.pair_counts.apply_deltas(deltas)
            return

        merge_start_time = time.time()

        super().merge_pair(pair_to_merge, new_token)

        merge_time = time.time() - merge_start_time
        self.merge_times.append(merge_time)
        self.merge_times.pop(0)

        # Switch to sequential when parallel overhead exceeds merge computation time
        if sum(self.merge_times)/len(self.merge_times) < self.threshold:
            print(f"Switching to sequential mode after token: {new_token}")
            self._go_sequential()