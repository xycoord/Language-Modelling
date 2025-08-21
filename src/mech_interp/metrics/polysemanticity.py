import torch

def compute_polysemanticity(W):
    """
    Summary statistic: How much does each feature interfere with other features?

    This is the formulation from the notebook accompanying the paper.
    https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb

    Args:
        W: Weight matrices for each model (n_instances, feature_dim, hidden_dim)

    Returns:
        polysemanticity: Summary statistic for each model (n_instances, feature_dim)
    """
    n_features = W.shape[1]

    W_norm = W / (1e-5 + torch.linalg.norm(W, 2, dim=-1, keepdim=True)) # (n_instances, n_features, n_hidden)

    # interference[i,f,g] = how much activation would feature g contribute 
    # when we look in the direction of feature f?
    interference = torch.bmm(W_norm, W.transpose(-1, -2)) # (n_instances, n_features, n_features)

    # Set diagonal to 0 to remove self-interference
    mask = torch.eye(n_features, device=W.device).bool()
    interference[:, mask] = 0

    # Summary statistic:
    # How much does each feature interfere with other features?
    polysemanticity = torch.linalg.norm(interference, dim=-1) # (n_instances, n_features)

    return polysemanticity


def compute_polysemanticity_paper(W):
    """
    Original `polysemanticity` formulation.
    Per-feature summary statistic: 
       How much does each feature interfere with other features?

    This is the formulation given in the paper but it is not scale invariant.
    This leads to features with small norms having low polysemanticity which 
    is not what is shown in the paper.
    """
    n_features = W.shape[1]

    interference = torch.bmm(W, W.transpose(-1, -2))  # (n_instances, n_features, n_features)

    # Set diagonal to 0 to remove self-interference
    mask = torch.eye(n_features, device=W.device).bool()
    interference[:, mask] = 0

    # Per-feature summary statistic (sum of squares):
    polysemanticity = torch.sum(interference ** 2, dim=-1) # (n_instances, n_features)

    return polysemanticity

