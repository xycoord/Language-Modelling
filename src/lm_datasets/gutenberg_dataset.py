from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset

from lm_tokenizers.base import Tokenizer


def load_gutenberg_texts(corpus_dir="gutenberg_corpus", file_pattern="*.txt") -> tuple[list[str], list[str]]:
    """
    Load all text files from a directory into a list of strings
    
    Args:
        corpus_dir: Directory containing the text files
        file_pattern: Pattern to match files (default: "*.txt")
    
    Returns:
        list: List of strings, one per file
        list: List of corresponding filenames (optional)
    """
    
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory '{corpus_dir}' not found")
    
    text_files = list(corpus_path.glob(file_pattern))
    if not text_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in '{corpus_dir}'")
    
    print(f"Loading {len(text_files)} text files...")
    
    texts = []
    filenames = []
    failed_files = []
    
    for file_path in sorted(text_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                if content:
                    texts.append(content)
                    filenames.append(file_path.name)
                else:
                    print(f"Warning: Empty file {file_path.name}")
                    
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            failed_files.append(file_path.name)
            continue
    
    print(f"Successfully loaded {len(texts)} texts")
    if failed_files:
        print(f"Failed to load {len(failed_files)} files: {failed_files}")
    
    return texts, filenames



class GutenbergDataset(Dataset):
    """
    Dataset for autoregressive language model training that tokenizes input text.
    Splits the data into train and val sets, preserving temporal order.
    Creates sliding windows of token sequences for next-token prediction.
    """
    def __init__(self, 
                 texts: list[str], 
                 tokenizer: Tokenizer,
                 split: str = 'train', 
                 train_split: float = 1.0, 
                 block_size: int = 8, 
                ):
        self.block_size = block_size
        self.tokenizer = tokenizer

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
        
        train_tokens, val_tokens = self._tokenize_and_split(texts, tokenizer, train_split, block_size)
        
        if split == 'train':
            self.data = torch.cat(train_tokens) if train_tokens else torch.tensor([], dtype=torch.long)
        elif split == 'val': # pragma: no branch
            self.data = torch.cat(val_tokens) if val_tokens else torch.tensor([], dtype=torch.long)
            

    @classmethod
    def create_split_pair(cls, 
                              texts: list[str], 
                              tokenizer: Tokenizer,
                              train_split: float = 0.9, 
                              block_size: int = 8
                              ) -> tuple['GutenbergDataset', 'GutenbergDataset']:
        """Create both train and validation datasets with only one tokenization pass."""

        if train_split <= 0 or train_split >= 1:
            raise ValueError(f"Invalid train_split: {train_split}. Must be between 0 and 1 (exclusive).")
        if block_size < 1:
            raise ValueError(f"Invalid block_size: {block_size}. Must be greater than 0.")
        
        train_tokens, val_tokens = cls._tokenize_and_split(texts, tokenizer, train_split, block_size)
        
        train_data = torch.cat(train_tokens) if train_tokens else torch.tensor([], dtype=torch.long)
        val_data = torch.cat(val_tokens) if val_tokens else torch.tensor([], dtype=torch.long)
        
        train_dataset = cls.__new__(cls)
        train_dataset.tokenizer = tokenizer
        train_dataset.block_size = block_size
        train_dataset.data = train_data
        
        val_dataset = cls.__new__(cls)
        val_dataset.tokenizer = tokenizer
        val_dataset.block_size = block_size
        val_dataset.data = val_data
        
        return train_dataset, val_dataset
 

    @staticmethod
    def _tokenize_and_split(texts: list[str], 
                            tokenizer: Tokenizer, 
                            train_split: float, 
                            block_size: int) -> tuple[list[Tensor], list[Tensor]]:
        """
        Helper method that does the expensive tokenization work once and returns
        lists of train and validation token tensors.
        
        Args:
            texts: List of text strings to tokenize
            tokenizer: Tokenizer to use
            train_split: Fraction of data to use for training (0-1)
            block_size: Minimum size for including a text
            
        Returns:
            tuple: (list of train token tensors, list of val token tensors)
        """
        print(f"Tokenizing {len(texts)} texts...")
        
        train_tokens = []
        val_tokens = []
        
        for text in texts:
            tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

            train_end_idx = int(train_split * len(tokens))
            train_tokens.append(tokens[:train_end_idx])
            val_tokens.append(tokens[train_end_idx:])
        
        return train_tokens, val_tokens


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