from datasets import load_dataset

# Load English Wikipedia (most similar size + excellent quality)
dataset = load_dataset("wikipedia", "20220301.en")
print(f"Dataset size: {len(dataset['train'])}")

# Check capitalization quality
for i in range(3):
    text = dataset['train'][i]['text'][:10]
    print(f"Sample {i}: {text}")
    # Should see proper names, sentence capitals, etc.