import torch
import torch.nn.functional as F

def cosine_similarity_matrix(W_sae: torch.Tensor, W_true: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between learnt SAE features and true feature directions.
    
    Args:
        W_sae: (..., n_sae_features, hidden_dim)
        W_true: (..., n_true_features, hidden_dim)
        
    Returns:
        similarity: (..., n_sae_features, n_true_features)
    """
    if W_sae.dim() < 2 or W_true.dim() < 2:
        raise ValueError(f"w_sae and w_true must be at least 2D, got {W_sae.dim()}D and {W_true.dim()}D")
    if W_sae.dim() != W_true.dim():
        raise ValueError(f"Tensors must have same number of dimensions: {W_sae.dim()} vs {W_true.dim()}")
    if W_sae.shape[:-2] != W_true.shape[:-2]:
        raise ValueError(f"Batch dims mismatch: {W_sae.shape[:-2]} vs {W_true.shape[:-2]}")
    if W_sae.shape[-1] != W_true.shape[-1]:
        raise ValueError(f"Hidden dimension mismatch: {W_sae.shape[-1]} vs {W_true.shape[-1]}")
    if W_sae.device != W_true.device:
        raise ValueError(f"Device mismatch: {W_sae.device} vs {W_true.device}")
    
    similarity = F.cosine_similarity(
        W_sae.unsqueeze(-2),   # (..., n_sae, 1, hidden_dim)
        W_true.unsqueeze(-3),  # (..., 1, n_true, hidden_dim)  
        dim=-1
    )
    
    return similarity


def max_feature_similarity(similarity_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the best SAE match for each true feature.
    
    Args:
        similarity_matrix: (..., n_sae_features, n_true_features)
        
    Returns:
        max_similarities: (..., n_true_features) - best similarity for each true feature
    """
    if similarity_matrix.dim() < 2:
        raise ValueError(f"Similarity matrix must be at least 2D, got {similarity_matrix.dim()}D")
    
    # Take max across SAE features (second-to-last dimension)
    max_similarities = similarity_matrix.max(dim=-2)[0]
    
    return max_similarities



def injective_feature_matching(similarity_matrix: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Greedy assignment of SAE features to true features, avoiding double-assignment.

    Finds per-batch best (sae_feature, true_feature) matches iteratively until
    similarity drops below threshold. Matches are not replaced, guaranteeing an
    injective mapping. The threshold is applied per-batch such that the batches
    are treated independently.

    Supports arbitrary batch dimensions (...), which are flattened for processing.

    Args:
        similarity_matrix: (..., n_sae, n_true)
        threshold: [0...1] minimum similarity to count as a match

    Returns:
        similarities: (..., n_true) - similarity for each true feature, 0 if unmatched

    Note:
        The threshold is validated to be in [0, 1], but the similarity values 
        themselves are assumed to be in this range and are not validated.
    """
    dtype = similarity_matrix.dtype

    if similarity_matrix.dim() < 2:
        raise ValueError(f"Similarity matrix must be at least 2D, got {similarity_matrix.dim()}D")
    if threshold < 0 or threshold > 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    n_sae, n_true = similarity_matrix.shape[-2:]
    batch_shape = similarity_matrix.shape[:-2]
    batch_size = torch.prod(torch.tensor(batch_shape)) if batch_shape else 1

    
    # Clone to avoid modifying the original tensor
    similarity_matrix = similarity_matrix.clone()
    flat_sim = similarity_matrix.reshape(batch_size, n_sae * n_true) # view for max
    matrix_sim = flat_sim.view(batch_size, n_sae, n_true) # view for masking
    batch_indices = torch.arange(batch_size, device=similarity_matrix.device)

    # Initialize result
    result = torch.zeros(batch_size, n_true, device=similarity_matrix.device, dtype=dtype)
    active_batches = torch.ones(batch_size, device=similarity_matrix.device, dtype=torch.bool)

    if n_sae == 0 or n_true == 0: # avoid max on empty tensors
        return result.view(*batch_shape, n_true)

    while active_batches.any():
        # Find max similarity for each batch
        max_vals, flat_indices = flat_sim.max(dim=-1)           # (batch_size,), (batch_size,)

        # Convert flat indices back to 2D coordinates  
        sae_indices = flat_indices // n_true                    # (batch_size,)
        true_indices = flat_indices % n_true                    # (batch_size,)

        # Update active batch indices
        active_batches = max_vals >= threshold

        # Record the selected similarity (inactive batches unchanged)
        result[batch_indices, true_indices] = torch.where(
            active_batches,                     # if active batch
            max_vals,                           # use max val
            result[batch_indices, true_indices] # else use existing value
        )

        # Mask out this SAE feature row with -inf 
        matrix_sim[batch_indices, sae_indices, :] = float('-inf')
        # Mask out this true feature column with -inf 
        matrix_sim[batch_indices, :, true_indices] = float('-inf')
        

    return result.view(*batch_shape, n_true)