from tokenizers import OptimizedBPETokenizer
from pathlib import Path
from datasets.gutenberg_dataset import load_gutenberg_texts

data_path = Path("data/gutenberg_corpus")

if not data_path.exists():
    raise FileNotFoundError(f"Data file {data_path} not found")

texts, filenames = load_gutenberg_texts(data_path)
text = "\n\n".join(texts)

tokenizer = OptimizedBPETokenizer()

tokenizer.train(text, target_vocab_size=512, min_merge_count=10)

print(tokenizer.vocab_size)

output_dir = Path("data/tokenizers")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "gutenberg_tokenizer_4096.json"

tokenizer.save(output_path)

print(f"Saved tokenizer to {output_path}")