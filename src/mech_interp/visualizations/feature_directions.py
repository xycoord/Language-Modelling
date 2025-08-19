import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import numpy as np

arrow_config = {
    'mode': 'lines+markers',
    'line': {'width': 3},
    'marker': {
        'size': [0, 8],
        'symbol': ['circle', 'triangle-up'],
    },
    'showlegend': False,
    'hoverinfo': 'skip'
}
label_config = {
    'mode': 'text',
    'textfont': {'color': 'black', 'size': 12},
    'showlegend': False,
    'hoverinfo': 'skip'
}
annotation_config = {
        'font': {'size': 14}
    }
layout_config = {
    'showlegend': False,
    'template': "plotly_white",
    'title_text': "Feature Directions",
    'margin': {'l': 40, 'r': 40, 'b': 40, 't': 80},
    'annotations': [annotation_config]
}
axes_config = {
    'range': [-1.2, 1.2],
    'tickmode': 'array',
    'tickvals': [-1, -0.5, 0, 0.5, 1],
    'ticktext': ['-1', '', '0', '', '1'],
    'showgrid': True,
    'zeroline': True, 
    'zerolinewidth': 1,
    'gridcolor': 'lightgrey',
    'zerolinecolor': 'lightgrey',
}

def plot_feature_directions(
    feature_directions: torch.Tensor,
    labels: list[str],
    importance: torch.Tensor,
    show_arrow_labels: bool = False,
    eps: float = 1e-2,
    colorscale: str = 'Viridis',
):
    """Plot feature directions for each toy model using Plotly.

    The feature directions are plotted as arrows from the origin, colored by importance.

    Note: The returned figure must be displayed using `fig.show()` to be visible.

    Args:
        feature_directions: Feature directions for each model (n_instances, feature_dim, 2).
        labels: Labels for each model (n_instances).
        importance: Importance of each feature (feature_dim).
        show_arrow_labels: Whether to show labels for the arrows.
        eps: Small value to ignore very small feature directions.

    Returns:
        fig: Plotly figure object.
    """
    if feature_directions.ndim == 2:  # Shape: (feature_dim, 2)
        feature_directions = feature_directions.unsqueeze(0) # -> (1, feature_dim, 2)
    if len(labels) != feature_directions.shape[0]:
        raise ValueError("Number of labels must match number of feature directions")

    feature_directions = feature_directions.cpu().numpy()
    importance = importance.cpu().numpy()

    n_plots, n_features, _ = feature_directions.shape

    # Set up the Subplot Grid
    cols = min(5, n_plots)
    rows = (n_plots + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols, 
        subplot_titles=labels,
    )

    # Create Importance-Based Color Scale using Plotly colors
    norm_importance = importance / importance.max()
    colors = pc.sample_colorscale(colorscale, norm_importance, colortype='rgb')

    for i, experiment_directions in enumerate(feature_directions):
        row = i // cols + 1
        col = i % cols + 1

        # Ensure the subplots have the same aspect ratio
        x_axis = "x" if i == 0 else f"x{i+1}"
        fig.update_yaxes(scaleanchor=x_axis, scaleratio=1, row=row, col=col)

        for feature_idx in range(n_features):
            x, y = experiment_directions[feature_idx]
            color = colors[feature_idx]
            
            if abs(x) < eps and abs(y) < eps:
                continue

            # Create arrow as a line from origin to endpoint
            arrow_trace = go.Scatter(
                x=[0, x],
                y=[0, y],
                **arrow_config
            )
            angle = np.degrees(np.arctan2(x, y))
            arrow_trace.update(
                line={'color': color}, 
                marker={'color': color,'angle': angle}
                )
            fig.add_trace(arrow_trace, row=row, col=col)
            
            if show_arrow_labels:
                fig.add_trace(
                    go.Scatter(
                        x=[x * 1.15],
                        y=[y * 1.15],
                        text=[f'{feature_idx}'],
                        **label_config
                    ),
                    row=row, col=col
                )

    # Calculate figure dimensions to make subplots approximately square
    subplot_size = 120
    margin_total = 160
    fig_width = cols * subplot_size + margin_total
    fig_height = rows * subplot_size + margin_total

    # --- 5. Apply Global Styling to All Subplots ---
    fig.update_layout(height=fig_height, width=fig_width, **layout_config)
    fig.update_xaxes(**axes_config)
    fig.update_yaxes(**axes_config)

    return fig