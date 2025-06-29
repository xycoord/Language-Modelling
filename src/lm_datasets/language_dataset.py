import torch
from torch import Tensor
from torch.utils.data import Dataset

from lm_tokenizers.base import Tokenizer

class LanguageDataset(Dataset):
    """
    Dataset for autoregressive language model training that tokenizes input text.
    Splits the data into train and val sets, preserving temporal order.
    Creates sliding windows of token sequences for next-token prediction.
    """
    def __init__(self, 
                 data_text: str, 
                 tokenizer: Tokenizer,
                 split: str = 'train', 
                 train_split: float = 1.0, 
                 block_size: int = 8, 
                ):
        self.block_size = block_size
        self.tokenizer = tokenizer

        # Parameter validation
        if split not in ('train', 'val'):
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")
        if train_split < 0 or train_split > 1:
            raise ValueError(f"Invalid train_split: {train_split}. Must be between 0 and 1.")
        if split == 'val' and train_split == 1:
            raise ValueError("train_split cannot be 1 when split is 'val'.")
        if block_size < 1:
            raise ValueError(f"Invalid block_size: {block_size}. Must be greater than 0.")
        if split == 'train' and train_split == 1:
            print("Note: Using all data for training. Consider setting train_split < 1.0 for validation.")
        
        data = torch.tensor(self.tokenizer.encode(data_text), dtype=torch.long)
        if len(data) <= block_size:
            raise ValueError(f"Data too short: {len(data)} tokens, need > {block_size}")

        train_end_idx = int(train_split * len(data))
        if split == 'train':
            self.data = data[:train_end_idx]
        elif split == 'val': # pragma: no branch
            self.data = data[train_end_idx:]

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Returns an (inputs, targets) pair for autoregressive training.
        
        target[t] is the next token after input[:t+1], enabling the model
        to learn next-token prediction at each position simultaneously.
        
        Args:
            index: Starting position in the token sequence
            
        Returns:
            inputs - (block_size,)
            targets - (block_size,)
        """
        inputs = self.data[index : index+self.block_size]
        targets = self.data[index+1 : index+self.block_size+1]
        return inputs, targets

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size