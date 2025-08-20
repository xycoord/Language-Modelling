import torch

def importance_decay_by_ratio(feature_dim: int, decay_ratio: float) -> torch.Tensor:
    """
    Create feature importance weights with exponential decay across features.
    
    Args:
        feature_dim: int
        decay_ratio: float
    Returns:
        importance: (feature_dim,) 
            where importance[i] = decay_ratio ** i
          
    """
    importance = decay_ratio ** torch.arange(feature_dim, dtype=torch.float)
    return importance

def importance_decay_by_min(feature_dim: int, min_importance: float) -> torch.Tensor:
    """
    Create feature importance weights with exponential decay across features from 1.0 to min_importance.
    
    Args:
        feature_dim: int
        min_importance: float
    Returns:
        importance:  (feature_dim,)
            = [1.0, ..., min_importance]
    """
    decay_ratio = min_importance ** (1 / (feature_dim - 1))
    return importance_decay_by_ratio(feature_dim, decay_ratio)

