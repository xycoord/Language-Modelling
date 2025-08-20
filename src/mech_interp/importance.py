import torch

def importance_decay_by_ratio(feature_dim: int, decay_ratio: float) -> torch.Tensor:
    """
    Create feature importance weights with exponential decay across features.
    
    Args:
        feature_dim: int >= 1
        decay_ratio: float >= 0
    Returns:
        importance: (feature_dim,) 
            where importance[i] = decay_ratio ** i
          
    """
    if int(feature_dim) != feature_dim:
        raise ValueError("feature_dim must be an integer")
    if feature_dim < 1:
        raise ValueError("feature_dim must be >= 1")
    if decay_ratio < 0:
        raise ValueError("decay_ratio must be >= 0")
    
    importance = decay_ratio ** torch.arange(feature_dim, dtype=torch.float)
    return importance

def importance_decay_by_min(feature_dim: int, min_importance: float) -> torch.Tensor:
    """
    Create feature importance weights with exponential decay across features from 1.0 to min_importance.
    
    Args:
        feature_dim: int >= 1
        min_importance: float >= 0
    Returns:
        importance:  (feature_dim,)
            = [1.0, ..., min_importance]
    """
    if int(feature_dim) != feature_dim:
        raise ValueError("feature_dim must be an integer")
    if feature_dim < 1:
        raise ValueError("feature_dim must be >= 1")
    if min_importance < 0:
        raise ValueError("min_importance must be >= 0")
    
    if feature_dim == 1: # special case prevents division by zero
        return torch.tensor([1.0])
    
    decay_ratio = min_importance ** (1 / (feature_dim - 1))
    return importance_decay_by_ratio(feature_dim, decay_ratio)

