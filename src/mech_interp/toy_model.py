import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyModel(nn.Module):
    """
    Single instance toy model used to demonstrate superposition of features in compressed space.

    Uses a tied linear layer to project features to a lower dimension and back.
    The upwards projection is followed by an additional bias and a ReLU activation.
    This allows the model to ignore interference between sparse features in the compressed space.

    Mathematical operations:
        hidden = features @ weights
        out = ReLU(hidden @ weights.T + bias)

    Args:
        feature_dim: the dimension of the input features
        hidden_dim: the dimension of the hidden layer
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(feature_dim, hidden_dim))
        nn.init.xavier_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(feature_dim))
    
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [..., feature_dim] input features
        Returns:
            out: [..., feature_dim] reconstructed features after ReLU
            hidden: [..., hidden_dim] hidden representation
        """
        hidden = features @ self.weight
        out = hidden @ self.weight.T + self.bias
        out = F.relu(out)
        return out, hidden

    def get_feature_directions(self) -> torch.Tensor:
        """Returns: [feature_dim, hidden_dim]"""
        return self.weight.detach()
    
    def get_bias(self) -> torch.Tensor:
        """Returns: [feature_dim]"""
        return self.bias.detach()

class ParallelToyModel(nn.Module):
    """
    Parallel per-instance toy model used to demonstrate superposition of features in compressed space.

    Each instance uses a tied linear layer to project features to a lower dimension and back.
    The upwards projection is followed by an additional untied bias and a ReLU activation.
    This allows the model to ignore interference between sparse features in the compressed space.

    Instances are parallelised for efficiency.

    Per instance:
        hidden = features @ weights
        out = ReLU(hidden @ weights.T + bias)

    Args:
        n_instances: the number of instances
        feature_dim: the dimension of the input features
        hidden_dim: the dimension of the hidden layer
    """
    def __init__(self, 
                n_instances: int, 
                feature_dim: int, 
                hidden_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty((n_instances, feature_dim, hidden_dim)))
        nn.init.xavier_normal_(self.weights)
        self.bias = nn.Parameter(torch.zeros((n_instances, feature_dim)))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [..., n_instances, feature_dim]
        Returns:
            out: [..., n_instances, feature_dim]
        """
        # For each implicit batch element `...` and each instance `i`,
        # features @ `weights[i]`
        hidden = torch.einsum("...if,ifh->...ih", features, self.weights) # [..., n_instances, hidden_dim]
        # hidden @ `weights[i].T`
        out = torch.einsum("...ih,ifh->...if", hidden, self.weights) # [..., n_instances, feature_dim]
        out = out + self.bias
        out = F.relu(out)
        return out, hidden

    def get_feature_directions(self) -> torch.Tensor:
        """
        Returns:
            feature_directions: [n_instances, feature_dim, hidden_dim]
        """
        return self.weights.detach()

    def get_bias(self) -> torch.Tensor:
        """
        Returns:
            bias: [n_instances, feature_dim]
        """
        return self.bias.detach()
