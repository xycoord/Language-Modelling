from pathlib import Path
from lm_datasets.gutenberg_dataset import load_gutenberg_texts
import time

data_path = Path("data/gutenberg_corpus")

if not data_path.exists():
    raise FileNotFoundError(f"Data file {data_path} not found")

texts, filenames = load_gutenberg_texts(data_path)
text = "\n\n".join(texts)

vocab_size = 100000
min_merge_count = 10

from lm_tokenizers import FastMaxBPETokenizer
tokenizer = FastMaxBPETokenizer()
print(f"Tokenizer: {tokenizer.__class__}")

print("Preprocessing text")
start_time = time.time()
chunks = tokenizer.preprocess_train(text)
print(f"Preprocessing text took {(time.time() - start_time):.2f} seconds")

print("Training tokenizer")
start_time = time.time()
tokenizer.train(chunks, target_vocab_size=vocab_size, min_merge_count=min_merge_count)
print(f"Training tokenizer took {(time.time() - start_time):.2f} seconds")

print(f"Vocab size: {tokenizer.vocab_size}")

output_dir = Path("data/tokenizers")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"gutenberg_tokenizer_{vocab_size}.json"

tokenizer.save(output_path)

print(f"Saved tokenizer to {output_path}")