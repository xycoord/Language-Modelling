import torch

def weighted_average(points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute the weighted average of a set of points.
    """
    # Divide by the weights sum first to avoid numerical instability
    weights = weights / weights.sum()
    return (points * weights.view(-1, 1)).sum(dim=0)

def geometric_median(
    points: torch.Tensor,
    eps: float = 1e-6,
    maxiter: int = 100,
    convergence_tolerance: float = 1e-20,
) -> torch.Tensor:
    """
    Compute the geometric median of a set of points using the Weiszfeld algorithm (equal weights).
    The geometric median is the point that minimises the sum of distances to the points.

    Args:
        points: Tensor of shape (n, d)
        eps: Smallest allowed value of denominator, to avoid divide by zero. Default 1e-6.
        maxiter: Maximum number of Weiszfeld iterations. Default 100
        convergence_tolerance: If objective value does not improve by at least this fraction, terminate. Default 1e-20.
    Returns:
        Geometric median tensor of shape (d,)
    """
    if maxiter < 1:
        raise ValueError("maxiter must be at least 1")
    
    with torch.no_grad():
        objective_value = None
        distance_weights = None

        # Weiszfeld iterations
        for i in range(maxiter):
            # Initialize with mean on first iteration, then use weighted average
            median = weighted_average(points, distance_weights) if i > 0 else points.mean(dim=0)
            distances = torch.linalg.norm(points - median.view(1, -1), dim=1)
            distance_weights = 1.0 / torch.clamp(distances, min=eps)
            new_objective_value = distances.sum()

            if objective_value is not None:
                if abs(new_objective_value - objective_value) <= convergence_tolerance * objective_value:
                    break

            objective_value = new_objective_value

    # Run final iteration outside of no-grad to allow autodiff to track it
    median = weighted_average(points, distance_weights)
    return median