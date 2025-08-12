import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from abc import ABC, abstractmethod

class SparseAutoencoder(ABC, nn.Module):
    """
    Sparse Autoencoder (SAE) for extracting features from activations.

    This implementation is partially based on:
    - https://transformer-circuits.pub/2023/monosemantic-features/index.html
    
    It features:
    - Tied pre and post bias (initialised to initial_bias) 
    - A decoder with weight normalisation (pytorch implementation) (different)
    - Uniform direction initialisation for the decoder (different)
    - Initially tied encoder and decoder weights (see https://arxiv.org/abs/2406.04093)

    Note: To follow the paper, pass initial_bias as the geometric median of the activations.
    """

    def __init__(self, activations_dim: int, feature_dim: int, initial_bias: torch.Tensor):
        super().__init__()
        if initial_bias.shape != (activations_dim,):
            raise ValueError(f"initial_bias must have shape (activations_dim,), but got {initial_bias.shape}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, but got {feature_dim}")
        if activations_dim <= 0:
            raise ValueError(f"activations_dim must be positive, but got {activations_dim}")

        self.pre_post_bias = nn.Parameter(initial_bias)

        unnormalized_decoder = nn.Linear(feature_dim, activations_dim, bias=False)
        with torch.no_grad():
            # Initialise the decoder with gaussian,
            # When normalised, this gives us a uniform distribution over the unit sphere.
            gaussian_directions = torch.randn(activations_dim, feature_dim)
            unnormalized_decoder.weight.data = gaussian_directions

        self.decoder = weight_norm(unnormalized_decoder, dim=0)
        self.encoder = nn.Linear(activations_dim, feature_dim, bias=True)

        # Initialise the encoder with the same weights as the decoder 
        with torch.no_grad():
            self.encoder.weight.data = get_weight_norm_weights(self.decoder).T

    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            activations: [batch_size, activations_dim]
        Returns:
            reconstructed_activations: [batch_size, activations_dim]
            features: [batch_size, feature_dim]
        """
        features = self.encode(activations)
        reconstructed_activations = self.decode(features)
        return reconstructed_activations, features

    @abstractmethod
    def apply_activation(self, raw_features: torch.Tensor) -> torch.Tensor:
        """
        Apply the activation function to the raw features.
        Subclasses should implement the activation function here.
        Args:
            raw_features: [batch_size, feature_dim]
        Returns:
            features: [batch_size, feature_dim]
        """
        pass

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Encode the activations into features.
        The features are then passed through the activation function.
        Args:
            activations: [batch_size, activations_dim]
        Returns:
            features: [batch_size, feature_dim]
        """
        shifted_activations = activations - self.pre_post_bias
        encoder_output = self.encoder(shifted_activations)
        features = self.apply_activation(encoder_output)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode the features into activations.
        Args:
            features: [batch_size, feature_dim]
        Returns:
            reconstructed_activations: [batch_size, activations_dim]
        """
        reconstructed_activations = self.decoder(features) + self.pre_post_bias
        return reconstructed_activations

    @torch.no_grad()
    def get_feature_directions(self) -> torch.Tensor:
        """
        Get the feature directions of the SAE.
        Returns: [feature_dim, activations_dim]
        """
        return get_weight_norm_weights(self.decoder).T


class ReLUSparseAutoencoder(SparseAutoencoder):
    """
    Sparse Autoencoder using ReLU activation between the encoder and decoder.
    This is the same as the original SAE from:
    - https://transformer-circuits.pub/2023/monosemantic-features/index.html
    """
    def apply_activation(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return F.relu(encoder_output)

class TopKSparseAutoencoder(SparseAutoencoder):
    """
    Sparse Autoencoder using per-sample top-k activation.
    Between the encoder and decoder, top k values in each sample may pass through.

    This implementation is partially based on:
    - https://arxiv.org/abs/2406.04093

    Following the code for the paper, we use a ReLU *after* the top-k activation.
    """
    def __init__(self, activations_dim: int, feature_dim: int, initial_bias: torch.Tensor, k: int):
        if k <= 0:
            raise ValueError(f"k must be positive, but got {k}")
        if k > feature_dim:
            raise ValueError(f"k must be less than or equal to feature_dim, but got {k} and {feature_dim}")
        self.k = k
        super().__init__(activations_dim, feature_dim, initial_bias)

    def apply_activation(self, raw_features: torch.Tensor) -> torch.Tensor:
        """
        Apply the top-k per sample activation function to the encoder output.
        The top k values in each sample are unaltered, and the rest are set to 0.
        The output is then passed through a ReLU activation function.
        """
        topk_values, topk_indices = torch.topk(raw_features, k=self.k, dim=-1)
        topk_features = (
            torch.zeros_like(raw_features)
            .scatter_(-1, topk_indices, topk_values)
            )

        features = F.relu(topk_features)
        return features

class BatchTopKSparseAutoencoder(SparseAutoencoder):
    """
    Sparse Autoencoder using batch top-k activation.
    Between the encoder and decoder, top k*batch_size features per batch can pass through.

    This implementation is based on:
    - https://arxiv.org/abs/2412.06410

    Following the code for the paper, we use a ReLU *before* the top-k activation.
    """
    def __init__(self, activations_dim: int, feature_dim: int, initial_bias: torch.Tensor, k: int):
        if k <= 0:
            raise ValueError(f"k must be positive, but got {k}")
        if k > feature_dim:
            raise ValueError(f"k must be less than or equal to feature_dim, but got {k} and {feature_dim}")
        self.k = k
        super().__init__(activations_dim, feature_dim, initial_bias)

    def apply_activation(self, raw_features: torch.Tensor) -> torch.Tensor:
        """
        Apply the top-k per batch activation function to the encoder output.
        First, ReLU the raw features.
        Then, the top k*batch_size values in each batch are unaltered, and the rest are set to 0.
        """
        batch_size = raw_features.shape[0]
        original_shape = raw_features.shape

        positive_features = F.relu(raw_features)
        
        flattened_features = positive_features.flatten()
        topk_values, topk_indices = torch.topk(flattened_features, self.k * batch_size, dim=-1)
        topk_features = (
            torch.zeros_like(flattened_features)
            .scatter(-1, topk_indices, topk_values)
            .reshape(original_shape)
        )
        
        return topk_features

@torch.no_grad()
def get_weight_norm_weights(layer: nn.Linear) -> torch.Tensor:
    """
    Extract the implicit weights of a weight_norm linear layer.
    This is the normalized direction vector of the weight_norm layer.
    """
    unnormalized_weights = layer.parametrizations.weight.original1
    unit_norm_weights = F.normalize(unnormalized_weights, p=2, dim=0)
    return unit_norm_weights
