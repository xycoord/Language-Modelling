import matplotlib.pyplot as plt
import torch

def plot_feature_directions(feature_directions: torch.Tensor, feature_probabilities: torch.Tensor, importance: torch.Tensor, save_path: str):
    """Plot feature directions for each toy model.
    The feature directions are plotted as arrows from the origin to the feature direction.

    Args:
        feature_directions: feature directions for each model (n_instances, n_features, n_hidden)
        feature_probabilities: feature probabilities for each model (used to set the titles)
        importance: importance of each feature (used to colour the feature directions)
        save_path: path to save the plot
    """
    # Importance-based colors
    n_features = feature_directions[0].shape[0]
    colors = plt.cm.viridis(importance.cpu().numpy() / importance.max())

    # Use the corrected plotting function
    n_plots = len(feature_directions)
    columns = min(5, n_plots)
    rows = (n_plots + columns - 1) // columns
    
    fig, axes = plt.subplots(rows, columns, figsize=(columns*3, rows*3), squeeze=False)
    axes = axes.flatten()
    axes_limit = 1.5

    for i, W in enumerate(feature_directions):
        ax = axes[i]

        # Set up the plot
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
        
        # Title with sparsity
        sparsity = 1 - feature_probabilities[i].item()
        ax.set_title(f'Sparsity = {sparsity:.3f}', fontsize=12, weight='bold')

        # Plot feature directions
        for feature_idx in range(n_features):
            x, y = W[feature_idx]
            # Plot the vector
            ax.arrow(0, 0, x, y, 
                    head_width=0.05, head_length=0.05, 
                    fc=colors[feature_idx], ec=colors[feature_idx],
                    linewidth=2)
            # Label the feature index
            ax.text(x*1.1, y*1.1, f'{feature_idx}', fontsize=10, ha='center', weight='bold')
        
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature directions plot saved to {save_path}")
    plt.close()