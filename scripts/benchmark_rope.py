import torch
import torch.utils.benchmark as benchmark
from lm_models.transformer.rotary_positional_embedding import RotaryPositionalEmbedding, RotaryEmbeddingFactory, apply_rotary_embedding

torch._dynamo.config.recompile_limit = 100

# --- Benchmarking Configuration ---
CONFIGS = [
    # --- Represents a smaller model, akin to Gemma 2B ---
    # Low batch for latency testing, higher for throughput
    (1, 2048, 8, 256),
    (8, 2048, 8, 256),

    # --- Represents a medium ~7B model, akin to Llama 3 8B or Mistral 7B ---
    # Common sequence lengths up to the standard 8K context window
    (1, 4096, 32, 128),
    (4, 4096, 32, 128),
    (1, 8192, 32, 128),
    (2, 8192, 32, 128),

    # --- Represents a larger model or longer context scenario ---
    # Using parameters from a 70B-class model (e.g., Llama 3 70B often has 64 heads)
    (1, 4096, 64, 128),
    (1, 4096, 64, 256),
    (1, 4096, 64, 512),
    (2, 4096, 64, 512),
    (4, 4096, 64, 512),

]
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    """
    Main function to run the benchmark including the factory method.
    """
    print(f"🚀 Starting benchmark on {DEVICE.upper()} with {DTYPE}...\n")

    # (Place this updated loop in your benchmark script)

    for b, s, h, d in CONFIGS:
        shape_str = f"({b}, {s}, {h}, {d})"
        print(f"--- Benchmarking for shape: {shape_str} ---")
        
        # 1. Setup inputs and models (same as before)
        x = torch.randn(b, s, h, d, device=DEVICE, dtype=DTYPE)
        rope_interleaved = RotaryPositionalEmbedding(dim=d, max_seq_len=s*2, interleaved=True).to(DEVICE, DTYPE)
        rope_half_flipped = RotaryPositionalEmbedding(dim=d, max_seq_len=s*2, interleaved=False).to(DEVICE, DTYPE)
        factory = RotaryEmbeddingFactory(dim=d).to(DEVICE, DTYPE)
        applier = apply_rotary_embedding
        input_pos = torch.arange(s, device=DEVICE)
        positional_embedding = factory(input_pos)

        # 2. Pre-compile all necessary functions and modules
        compiled_interleaved = torch.compile(rope_interleaved, mode="max-autotune")
        compiled_half_flipped = torch.compile(rope_half_flipped, mode="max-autotune")
        # --- THIS IS THE KEY FIX ---
        # Compile the function ONCE before timing.
        compiled_apply_rope = torch.compile(apply_rotary_embedding, mode="max-autotune")

        # 3. Define the statements to be benchmarked
        statements = {
            "Interleaved": "rope_interleaved(x)",
            "Half-Flipped": "rope_half_flipped(x)",
            "Factory Application": "applier(x, positional_embedding)",
            "Compiled Interleaved": "compiled_interleaved(x)",
            "Compiled Half-Flipped": "compiled_half_flipped(x)",
            # Now we call the pre-compiled function
            "Compiled Factory App.": "compiled_apply_rope(x, positional_embedding)"
        }
        
        # 4. Create and run the benchmark (same as before)
        measurements = []
        globals_dict = {
            "x": x,
            "rope_interleaved": rope_interleaved,
            "rope_half_flipped": rope_half_flipped,
            "applier": applier,
            "positional_embedding": positional_embedding,
            "compiled_interleaved": compiled_interleaved,
            "compiled_half_flipped": compiled_half_flipped,
            "compiled_apply_rope": compiled_apply_rope # Add the fixed function to globals
        }
        for label, stmt in statements.items():
            measurements.append(
                benchmark.Timer(
                    stmt=stmt,
                    globals=globals_dict,
                    label=label,
                    description=shape_str,
                    sub_label="RoPE Application Cost"
                ).blocked_autorange(min_run_time=1)
            )
            
        comparison = benchmark.Compare(measurements)
        comparison.print()
        print("-" * 80 + "\n")

if __name__ == "__main__":
    main()