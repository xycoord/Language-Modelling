import torch

def create_uniform_sparsity(feature_dim: int, sparsity: float) -> torch.Tensor:
    """
    Create uniform sparsity tensor.
    Return shape: (feature_dim,)
    """
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
    if not (0 <= min_sparsity < max_sparsity < 1):
        raise ValueError("Sparsity values must satisfy: 0 <= min_sparsity < max_sparsity < 1")
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

def create_importance(feature_dim: int, importance_decay: float) -> torch.Tensor:
    """
    Create feature importance weights with exponential decay across features.
    Return shape: (feature_dim,)
    """
    importance = importance_decay ** torch.arange(feature_dim, dtype=torch.float)
    return importance

def weighted_mse_loss(output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted MSE loss.
    Args:
        output: (batch_size, feature_dim)
        target: (batch_size, feature_dim)  
        weight: (feature_dim,)
    Return shape: scalar
    """
    return torch.mean(weight * (target - output) ** 2)
