import torch

class SyntheticSparseDataGenerator():
    """
    Generates synthetic sparse data on the fly.
    Based on the data generation from "Toy Models of Superposition".

    Args:
        batch_size
        sparsity: independent sparsity of each feature, this is the probability that the feature is zero
        device: device to create tensors on (default: cpu)

    Each feature is 0 with probability sparsity[i] and uniformly distributed on [0, 1] otherwise.
    The data will match the shape of the sparsity tensor (e.g. (num_models, feature_dim)).
    """
    def __init__(self, batch_size: int, sparsity: torch.Tensor, device: torch.device = torch.device("cpu")):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not torch.all((sparsity >= 0) & (sparsity <= 1)):
            raise ValueError("sparsity must be between 0 and 1")
        
        self.batch_size = batch_size
        self._feature_probability = 1-sparsity.to(device)
        self._feature_probability = self._feature_probability.unsqueeze(0).expand(batch_size, -1, -1)
        self._data_shape = (batch_size, *sparsity.shape)
        self.device = device

    def generate_batch(self):
        raw_vector = torch.rand(self._data_shape, device=self.device)
        active_mask = torch.bernoulli(self._feature_probability)
        feature_vector = raw_vector * active_mask
        return feature_vector


