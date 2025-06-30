from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset, DataLoader
from typing import Callable
import multiprocessing
from tqdm import tqdm
from lm_tokenizers import WeightedChunk
from collections import Counter
import regex
import functools

def chunk_counting_preprocess(text: str, pattern: regex.Pattern[str]) -> Counter[str]:
    """Extract chunks from text using regex pattern"""
    text_chunks = pattern.findall(text)
    return Counter(text_chunks)

class WikipediaPreprocessDataset(TorchDataset):
    """Dataset for preprocessing Wikipedia articles with parallel processing"""
    
    def __init__(
        self,
        dataset_name: str = "wikipedia",
        dataset_config: str = "20220301.en",
    ):
        print(f"Loading {dataset_name} dataset...")
        self.full_dataset: HFDataset = load_dataset(dataset_name, dataset_config, split='train', streaming=False)
        self.preprocess_func: Callable[[str], Counter[str]] | None = None  # Set when processing
    
    def __len__(self) -> int:
        return len(self.full_dataset)
    
    def __getitem__(self, idx: int) -> Counter[str]:
        if self.preprocess_func is None:
            raise ValueError("Preprocessing function is not set")
        article_text = self.full_dataset[idx]['text']
        processed = self.preprocess_func(article_text)
        return processed
    
    def process(
        self, 
        chunk_regex: str = r'\b\w+\b',
        batch_size: int = 64,
        num_workers: int | None = None
    ) -> list[WeightedChunk]:
        """
        Process all articles for tokenizer training with global deduplication
        
        PARALLEL/SERIAL WORK SPLIT:
        - PARALLEL: regex + chunk counting per article  
        - SERIAL: Counter merging + byte encoding of unique chunks
        
        Args:
            chunk_regex: Regex pattern to split text into chunks
            batch_size: Batch size for parallel processing
            num_workers: Number of worker processes
        
        Returns:
            List of (byte_chunk, global_count) tuples ready for training
        """
        if num_workers is None:
            cpu_count = multiprocessing.cpu_count() or 1
            num_workers = max(cpu_count - 1, 1)
        
        compiled_pattern = regex.compile(chunk_regex)
        self.preprocess_func = functools.partial(chunk_counting_preprocess, pattern=compiled_pattern)
        
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.collate_counters,
            persistent_workers=True
        )
        try:
            print(f"Counting chunks in {len(self)} articles with {num_workers} workers...")
            global_counter = Counter()
            for batch_counter in tqdm(dataloader, desc="Counting chunks in parallel"):
                global_counter.update(batch_counter)
            
            print(f"Found {len(global_counter)} unique chunks")
            
            print("Encoding unique chunks to bytes...")
            weighted_chunks = [
                # implicitly converts to list[int] to be compatible with merge operation in training
                (list(chunk_text.encode("utf-8")), count) 
                for chunk_text, count in global_counter.items()
            ]
            
            return weighted_chunks
        finally:
            if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
                dataloader._iterator._shutdown_workers()

    @staticmethod
    def collate_counters(counters_list: list[Counter[str]]) -> Counter[str]:
        """Collate a list of counters into a single counter
        Note: first counter is used as the base, all subsequent counters are merged into it
        """
        if len(counters_list) == 0:
            return Counter()

        merged_counter = counters_list[0]
        for counter in counters_list[1:]:
            merged_counter.update(counter)
        return merged_counter