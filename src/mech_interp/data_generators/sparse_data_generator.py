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
        self._data_shape = (batch_size, *sparsity.shape)
        self._feature_probability = 1-sparsity.to(device)
        self._feature_probability = self._feature_probability.unsqueeze(0).expand(self._data_shape)
        self.device = device

    def generate_batch(self) -> torch.Tensor:
        raw_vector = torch.rand(self._data_shape, device=self.device)
        active_mask = torch.bernoulli(self._feature_probability)
        feature_vector = raw_vector * active_mask
        return feature_vector

    def to(self, device: torch.device):
        self.device = device
        return self

def create_uniform_sparsity(feature_dim: int, sparsity: float) -> torch.Tensor:
    """
    Create uniform sparsity tensor.
    Return shape: (feature_dim,)
    """
    if not (0 <= sparsity <= 1):
        raise ValueError("Sparsity must be between 0 and 1")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be > 0")
    return torch.full((feature_dim,), sparsity)

def create_sparsity_range(min_sparsity: float,
                          max_sparsity: float,
                          num_models: int,
                          feature_dim: int
                          ) -> torch.Tensor:
    """
    Create sparsity tensor for parallel models with log-spaced sparsity between min and max.
    Sparsity varies across models, but is constant across features.
    
    Note: max_sparsity must be < 1 (else would cause division by zero)
    Return shape: (num_models, feature_dim)
    """
    if not (0 <= min_sparsity <= max_sparsity < 1):
        raise ValueError("Sparsity values must satisfy: 0 <= min_sparsity < max_sparsity <= 1")
    if num_models <= 0:
        raise ValueError("num_models must be > 0")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be > 0")

    # Convert to probabilities to leverage exponential spacing in probability space
    # This ensures more intuitive sparsity progression than linear spacing
    min_feature_probability = 1 - max_sparsity
    max_feature_probability = 1 - min_sparsity

    sparsity = 1 - (
        max_feature_probability 
        * torch.logspace(0, -1, num_models, base=max_feature_probability/min_feature_probability)
        )

    # Repeat sparsity across feature dimension: (num_models,) -> (num_models, feature_dim)
    sparsity = sparsity.unsqueeze(1).expand(-1, feature_dim)
    return sparsity