import torch
from torch.utils.data import Dataset

class LanguageDataset(Dataset):
    """
    A dataset that returns a batch of data for training a language model.
    """
    def __init__(self, data_text, tokenizer, split='train', train_split=0.9, block_size=8, device='cpu'):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.device = device
        # encode data
        data = torch.tensor(self.tokenizer.encode(data_text), dtype=torch.long)
        # split data into train and val
        split_n = int(train_split * len(data))
        self.data = data[:split_n] if split == 'train' else data[split_n:]
        self.data = self.data.to(self.device)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx+self.block_size]
        y = self.data[idx+1 : idx+self.block_size+1]
        return x, y

    def vocab_size(self):
        return self.tokenizer.vocab_size