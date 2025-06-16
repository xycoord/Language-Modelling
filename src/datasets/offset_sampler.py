from typing import Iterator
from torch.utils.data.sampler import Sampler
import torch

OFFSET_MULTIPLIER = 37

class OffsetSampler(Sampler):
    """
    Samples dataset indices with epoch-based offsets to ensure full coverage over multiple epochs.
    Sampling ensures that the dataset is (almost) fully covered once per epoch.
    """

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.stride = dataset.block_size
        if self.stride % OFFSET_MULTIPLIER == 0:
            raise ValueError(f"Stride {self.stride} must not be divisible by offset multiplier {OFFSET_MULTIPLIER}")
        self.set_epoch(0)
        
    def __iter__(self) -> Iterator[int]:
        """Returns an iterator over the dataset indices.
        - The indices are offset by the epoch * OFFSET_MULTIPLIER % self.stride
        - Shuffled if shuffle is True
        - Shuffling uses the epoch as the seed
        """
        indices = list(range(self._offset, self._max_start, self.stride))
        
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            shuffled = torch.randperm(len(indices), generator=g)
            indices = [indices[i] for i in shuffled]
            
        return iter(indices)
    
    def __len__(self) -> int:
        return (self._max_start - self._offset + self.stride - 1) // self.stride
    
    def set_epoch(self, epoch: int) -> None:
        """ Sets the epoch
        Effects:
        - Shuffling order is changed (if shuffle is True)
        - Offset is changed
        """
        self.epoch = epoch
        self._offset = (epoch * OFFSET_MULTIPLIER) % self.stride
        self._max_start = len(self.dataset.data) - self.dataset.block_size