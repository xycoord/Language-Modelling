# Language Modelling

##  **Transformer Implementation** [[code](./src/lm_models/transformer/)]
- **KV Cache** [[code](./src/lm_models/transformer/kv_cache.py)]  
Stateless transformer design with external cache management
- **Rotary Positional Embeddings** [[blog](https://loganthomson.com/RoPE/)] [[code](./src/lm_models/transformer/rotary_positional_embedding/)]  
Both interleaved and half-flipped rotations; direct application and factory patterns for different use cases

## **Mech Interp** [[code](./src/mech_interp/)]
- **Toy Models of Superposition**  
Reproduce 5→2→5 experiments, plotting feature directions in the compressed activation space 
- **Sparse Autoencoders**  
ReLU, TopK and BatchTopK implementations. Trained to recover features from toy models

## **Engineering** 
- **BPE Tokeniser** [[blog](https://loganthomson.com/Optimising-BPE/)] [[code](./src/lm_tokenizers/bpe/)]  
6 training optimisations to go from 8 hours to 13s.
- **Test Suite** [[code](./tests/)]  
Comprehensive testing for transformer and tokeniser implementations