import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from .config import BigramConfig

class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model that predicts the next token in a sequence.
    """

    def __init__(self, config: BigramConfig):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.vocab_size)

    def forward(self, context: Tensor) -> Tensor:
        logits = self.token_embedding_table(context)
        return logits

    def generate(self, context: Tensor, max_new_tokens: int) -> Tensor:
        """Generate a sequence of tokens from the model"""
        for _ in range(max_new_tokens):
            logits = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, new_token), dim=1)
        return context
    
    @property
    def device(self) -> str:
        return next(self.parameters()).device