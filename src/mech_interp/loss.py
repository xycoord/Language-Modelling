import torch

def weighted_mse_loss(output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted MSE loss.
    Args:
        output: (batch_size, feature_dim)
        target: (batch_size, feature_dim)  
        weight: (feature_dim,)
    Return shape: scalar
    """
    if output.shape != target.shape:
        raise RuntimeError("output and target must have the same shape")
    if weight.ndim != 1 or weight.shape[0] != output.shape[-1]:
        raise RuntimeError("weight must have shape (feature_dim,)")
    
    return torch.mean(weight * (target - output) ** 2)
