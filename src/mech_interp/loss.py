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
    return torch.mean(weight * (target - output) ** 2)
