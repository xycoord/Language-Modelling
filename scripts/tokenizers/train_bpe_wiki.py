from lm_datasets.wikipedia.tokenizer_dataset import WikipediaPreprocessDataset
from lm_tokenizers import ParallelBPETokenizer, GPT4_SPLIT_PATTERN
import time
import pickle
import gzip

def save_weighted_chunks_pickle(weighted_chunks: list[tuple[list[int], int]], filepath: str, compress: bool = True):
    """Save weighted chunks using pickle (most compatible with your data structure)"""
    open_func = gzip.open if compress else open
    mode = 'wb'
    
    with open_func(filepath, mode) as f:
        pickle.dump(weighted_chunks, f)
    
    print(f"Saved {len(weighted_chunks)} chunks to {filepath}")

def load_weighted_chunks_pickle(filepath: str, compress: bool = True) -> list[tuple[list[int], int]]:
    """Load weighted chunks from pickle file"""
    open_func = gzip.open if compress else open
    mode = 'rb'
    
    with open_func(filepath, mode) as f:
        weighted_chunks = pickle.load(f)
    
    print(f"Loaded {len(weighted_chunks)} chunks from {filepath}")
    return weighted_chunks


tokenizer = ParallelBPETokenizer()

dataset = WikipediaPreprocessDataset()
print(f"Loaded {len(dataset)} articles")
import time
start_time = time.time()
chunks = dataset.process(
    chunk_regex=GPT4_SPLIT_PATTERN,
    batch_size=63, 
    num_workers=8
)  
end_time = time.time()
print(f"Preprocessing took {end_time - start_time} seconds")


path = "./data/wiki_chunks.pkl.gz"
save_weighted_chunks_pickle(chunks, path)

chunks = load_weighted_chunks_pickle(path)

start_time = time.time()
tokenizer.train(chunks, target_vocab_size=257, num_workers=8)
end_time = time.time()
print(f"Training took {end_time - start_time} seconds")
print(f"Vocab size: {tokenizer.vocab_size}")