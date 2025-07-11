# Tokenisers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This folder accompanies my blog post: [From Hours to Seconds: Optimising BPE Tokeniser Training](https://medium.com/@logan_16888/from-hours-to-seconds-optimising-bpe-tokeniser-training-f4234300d03e).

The folder `bpe` contains each version of the tokeniser:
1. `BasicBPETokenizer`
2. `ChunkedBPETokenizer`
3. `DeduplicatedBPETokenizer`
4. `IncrementalBPETokenizer`
5. `FastMaxBPETokenizer`
6. `ParallelBPETokenizer`

Each inherits what it can from those before it to highlight the differences.

A full test suite can be found in `tests` folder under the root of this repo.
