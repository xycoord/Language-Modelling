from lm_tokenizers import FastMaxBPETokenizer
from pathlib import Path
from lm_datasets.gutenberg_dataset import load_gutenberg_texts

data_path = Path("data/gutenberg_corpus")

if not data_path.exists():
    raise FileNotFoundError(f"Data file {data_path} not found")

texts, filenames = load_gutenberg_texts(data_path)
text = "\n\n".join(texts)

vocab_size = 10000
min_merge_count = 10

tokenizer = FastMaxBPETokenizer()
tokenizer.train(text, target_vocab_size=vocab_size, min_merge_count=min_merge_count)

print(tokenizer.vocab_size)

output_dir = Path("data/tokenizers")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"gutenberg_tokenizer_{vocab_size}.json"

tokenizer.save(output_path)

print(f"Saved tokenizer to {output_path}")