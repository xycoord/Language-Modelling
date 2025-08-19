import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


bar_config = {
    'marker': dict(
        cmin=0,
        cmax=1
    ),
    'width': 0.8  # Relative bar width within subplot
}
hovertemplate = 'In: %{x}<br>Out: %{y}<br>Weight: %{z:0.2f}'
heatmap_config = {
    'colorscale': 'Picnic',
    'zmin': -1,
    'zmax': 1,
    'hovertemplate': hovertemplate
}
v_line_config = {
    'line': {
        'width': 0.5, 
        'color': 'black'
    },
    'col': 1,
}
title_config = {
    'text': "Toy Models of Superposition",
    'x': 0.05,
    'font': {'size': 24, 'color': 'black', 'weight': 600, 'family':'Arial'},
}
layout_config = {
    'showlegend': False, 
    'margin': {
        't': 100, 'b': 30,'l': 40, 'r': 20
    },
    'title': title_config,
    'font': {'family':'Georgia'}
}
annotation_config = {
    'font': {'size': 14, 'color': 'black'},
    'showarrow': False,
}

label_config = {
    **annotation_config,
    'xanchor': 'right', 
    'yanchor': 'middle', 
    'textangle': -90, 
    'x':-0.01, 
    'y':0.5
} 
heading_config = {
    **annotation_config,
    'x':0, 
    'y':1.17, 
    'xanchor': 'left',
    'yanchor': 'top',
}
bias_heading_config = {
    **annotation_config,
    'x':0, 
    'y':1.17, 
    'xanchor': 'center',
    'yanchor': 'top',
}

# Sizing logic: Keep heatmaps square, bars proportional to n_features
heatmap_size = 200  # Square size for heatmaps
bar_section_width = 500
bias_section_width = 5 
    
# Calculate relative proportions for subplot layout
total_width = bar_section_width + heatmap_size
bar_proportion = bar_section_width / total_width
heatmap_proportion = heatmap_size / total_width
bias_proportion = bias_section_width / total_width

subplot_config = {
    'shared_xaxes': True,
    'vertical_spacing': 0.02,
    'horizontal_spacing': 0.02,
    'column_widths': [bar_proportion, heatmap_proportion, bias_proportion],
}

def plot_feature_analysis(
    weights: torch.Tensor,
    bias: torch.Tensor,
    labels: list[str]|None = None,
    use_paper_polysemanticity: bool = False
    ) -> go.Figure:
    """
    Plot results of the Toy Models of Superposition.
    - Norm: ∥Wᵢ∥ (bar chart)
    - Polysemanticity: (bar chart colors)
        How much does each feature interfere with other features? 
        (see `compute_polysemanticity`)
    - Interference: WᵀW (heatmap)
    - Bias: b (heatmap)

    Note: The returned figure must be displayed using `fig.show()` to be visible.

    Args:
        weights: Weight matrices for each model (n_instances, feature_dim, hidden_dim)
        bias: Bias vectors for each model (n_instances, feature_dim)
        labels: Labels for each model (n_instances)

    Returns:
        fig: Plotly figure object.
    """
    if weights.ndim == 2:  # Shape: (feature_dim, 2)
        weights = weights.unsqueeze(0) # -> (1, feature_dim, 2)
    if bias.ndim == 1:  # Shape: (feature_dim,)
        bias = bias.unsqueeze(0) # -> (1, feature_dim)

    n_instances, n_features, n_hidden = weights.shape

    if labels is not None and len(labels) != n_instances:
        raise ValueError("Number of labels must match number of instances")


    # Bar X-Axis
    x = torch.arange(n_features)
    # Bar heights
    norms = torch.linalg.norm(weights, 2, dim=-1).cpu()
    # Bar Colors
    if use_paper_polysemanticity:
        polysemanticity = compute_polysemanticity_paper(weights).cpu()
    else:
        polysemanticity = compute_polysemanticity(weights).cpu()

    # Heatmap Colors
    WtW = torch.bmm(weights, weights.transpose(-1, -2)).cpu()
    
    bias = bias.unsqueeze(-1).cpu() # -> (n_instances, feature_dim, 1)

    fig = make_subplots(
        rows=n_instances,
        cols=3,
        specs=[[{"secondary_y": False}]*3] * n_instances,
        **subplot_config,
    )

    for inst in range(n_instances):

        row = inst+1 # row is 1-indexed
        bar_index = '' if inst == 0 else 3*inst+1
        heatmap_index = 3*inst+2
        bias_index = 3*inst+3

        # Add polysemanticity bar chart
        fig.add_trace(
            go.Bar(x=x, 
                    y=norms[inst],
                    **bar_config
            ).update(marker={'color': polysemanticity[inst]}),
            row=row, col=1
        )

        # Add WtW heatmap
        fig.add_trace(
            go.Heatmap(
                z=WtW[inst],
                showscale=False,
                **heatmap_config
            ),
            row=row, col=2
        )
        # Make heatmap square
        fig.update_yaxes(
            scaleanchor=f"x{heatmap_index}",
            scaleratio=1, 
            constrain='domain',
            autorange='reversed',
            row=row, col=2
        )
        fig.update_xaxes(constrain='domain', row=row, col=2)

        # Add bias heatmap
        fig.add_trace(
            go.Heatmap(
                z=bias[inst],
                showscale=False,
                **heatmap_config
            ),
            row=row, col=3
        )
        fig.update_yaxes(
            constrain='domain',
            autorange='reversed',
            row=row, col=3
        )
        fig.update_xaxes(constrain='domain', row=row, col=3)


        # Add headings
        fig.add_annotation(
            text=f"∥Wᵢ∥",
            xref=f"x{bar_index} domain",
            yref=f"y{bar_index} domain",
            **heading_config
        )
        fig.add_annotation(
            text=f"WᵀW",
            xref=f"x{heatmap_index} domain",
            yref=f"y{heatmap_index} domain",
            **heading_config
        )
        fig.add_annotation(
            text=f"b",
            xref=f"x{bias_index} domain",
            yref=f"y{bias_index} domain",
            **bias_heading_config
        )

        if labels is not None:
            # Add label
            fig.add_annotation(
                text=labels[inst],
                xref=f"x{bar_index} domain",  # Bar chart x-axis (column 1)
                yref=f"y{bar_index} domain",  # Bar chart y-axis (column 1)
                **label_config
            )

    # Add vertical line to separate feature types
    fig.add_vline(x=(x[n_hidden-1] + x[n_hidden]) / 2, **v_line_config)
      
    
    # Set figure size based on content
    fig.update_layout(
        width=total_width,
        height=heatmap_size * n_instances,
        **layout_config
    )
    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig