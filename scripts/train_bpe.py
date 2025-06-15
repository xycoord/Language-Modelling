from tokenizers import OptimizedBPETokenizer
from pathlib import Path

data_path = Path("data/shakespeare.txt")

if not data_path.exists():
    raise FileNotFoundError(f"Data file {data_path} not found")

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = OptimizedBPETokenizer()

tokenizer.train(text, target_vocab_size=10000, min_merge_count=10)

print(tokenizer.vocab_size)

output_dir = Path("data/tokenizers")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "shakespeare_tokenizer.json"

tokenizer.save(output_path)

print(f"Saved tokenizer to {output_path}")