import matplotlib.pyplot as plt
import time

from tokenizers import BasicBPETokenizer, ChunkedBPETokenizer, DeduplicatedBPETokenizer, Tokenizer

# load data
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

def benchmark_tokenizer_with_vocab_size(tokenizer: Tokenizer, text: str, name: str, vocab_size: int):
    """Modified version that accepts vocab_size as parameter"""
    print(f"Benchmarking {name} tokenizer with vocab_size={vocab_size}")
    
    start_time = time.time()
    tokenizer.train(text, vocab_size)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.4f}s")

    start_time = time.time()
    encoded_text = tokenizer.encode(text)
    end_time = time.time()
    encoding_time = end_time - start_time
    print(f"Encoding time: {encoding_time:.4f}s")

    start_time = time.time()
    decoded_text = tokenizer.decode(encoded_text)
    end_time = time.time()
    decoding_time = end_time - start_time
    print(f"Decoding time: {decoding_time:.4f}s")

    print(f"Round trip equality: {decoded_text == text}")
    print("-----------------------------------")
    return training_time, encoding_time, decoding_time

# Test parameters
vocab_sizes = [512, 768, 1024]
tokenizer_types = [
    (BasicBPETokenizer, "Basic"),
    (ChunkedBPETokenizer, "Chunked"), 
    (DeduplicatedBPETokenizer, "Optimized")
]

# Store results
results = {
    'Basic': {'training': [], 'encoding': [], 'decoding': []},
    'Chunked': {'training': [], 'encoding': [], 'decoding': []},
    'Optimized': {'training': [], 'encoding': [], 'decoding': []}
}

# Run benchmarks
for vocab_size in vocab_sizes:
    print(f"\n=== Testing with vocab_size = {vocab_size} ===")
    for tokenizer_class, name in tokenizer_types:
        tokenizer = tokenizer_class()  # Create new instance for each test
        training_time, encoding_time, decoding_time = benchmark_tokenizer_with_vocab_size(
            tokenizer, text, name, vocab_size
        )
        
        results[name]['training'].append(training_time)
        results[name]['encoding'].append(encoding_time)
        results[name]['decoding'].append(decoding_time)

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['training', 'encoding', 'decoding']
colors = ['blue', 'orange', 'green']

for i, metric in enumerate(metrics):
    ax = axes[i]
    
    for j, (name, color) in enumerate(zip(results.keys(), colors)):
        ax.plot(vocab_sizes, results[name][metric], 
                marker='o', label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Vocabulary Size')
    ax.set_ylabel(f'{metric.capitalize()} Time (seconds)')
    ax.set_title(f'{metric.capitalize()} Time vs Vocabulary Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(vocab_sizes)

plt.tight_layout()
plt.savefig('tokenizer_benchmark_results.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'tokenizer_benchmark_results.png'")

# Print summary table
print("\n=== Summary Results ===")
print(f"{'Tokenizer':<12} {'Vocab Size':<10} {'Training':<10} {'Encoding':<10} {'Decoding':<10}")
print("-" * 60)

for name in results.keys():
    for i, vocab_size in enumerate(vocab_sizes):
        training = results[name]['training'][i]
        encoding = results[name]['encoding'][i]
        decoding = results[name]['decoding'][i]
        print(f"{name:<12} {vocab_size:<10} {training:<10.4f} {encoding:<10.4f} {decoding:<10.4f}")
    print()