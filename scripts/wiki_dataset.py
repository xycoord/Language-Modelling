from lm_datasets.wikipedia.tokenizer_dataset import WikipediaPreprocessDataset
from lm_tokenizers import FastMaxBPETokenizer, GPT4_SPLIT_PATTERN
import time

tokenizer = FastMaxBPETokenizer()

dataset = WikipediaPreprocessDataset()
print(f"Loaded {len(dataset)} articles")

chunks = dataset.process(
    chunk_regex=GPT4_SPLIT_PATTERN,
    batch_size=63, 
    num_workers=7
)  

start_time = time.time()
tokenizer.train(chunks, target_vocab_size=1000)
end_time = time.time()
print(f"Training took {end_time - start_time} seconds")