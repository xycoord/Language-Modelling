import torch
import torch.nn as nn
import torch.nn.functional as F

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
        n_features: the dimension of the input features
        n_hidden: the dimension of the hidden layer
    """
    def __init__(self, 
                n_instances, 
                n_features, 
                n_hidden):
        super().__init__()
        self.weights = nn.Parameter(torch.empty((n_instances, n_features, n_hidden)))
        nn.init.xavier_normal_(self.weights)
        self.bias = nn.Parameter(torch.zeros((n_instances, n_features)))

    def forward(self, features):
        """
        Args:
            features: [..., n_instances, n_features]
        Returns:
            out: [..., n_instances, n_features]
        """
        # For each implicit batch element `...` and each instance `i`,
        # features @ `weights[i]`
        hidden = torch.einsum("...if,ifh->...ih", features, self.weights) # [..., n_instances, n_hidden]
        # hidden @ `weights[i].T`
        out = torch.einsum("...ih,ifh->...if", hidden, self.weights) # [..., n_instances, n_features]
        out = out + self.bias
        out = F.relu(out)
        return out

    def get_feature_directions(self) -> torch.Tensor:
        """
        Returns:
            feature_directions: [n_instances, n_features, n_hidden]
        """
        return self.weights.detach().cpu()

