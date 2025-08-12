import torch
from torch.distributions import Beta, Dirichlet

class SyntheticClusteredDataGenerator:
    """
    Generates feature vectors with controllable latent topic structure using an 
    LDA-inspired approach, loosely modelling feature relationships in language. 
    This creates synthetic data where features are correlated by topic clustering, 
    giving more realistic structure than independent features.

    Key relationship: P(feature active) = Σᵢ P(topicᵢ)×P(feature | topicᵢ)

    Args:
        n_topics: Number of latent topics
        n_features: Number of features per sample
        alpha: Dirichlet concentration parameter for topic proportions
        beta_params: Beta distribution parameters for P(feature | topic). 
            Default (0.1, 2.0) creates sparse topic signatures.
        device: Device for computations
    """
    def __init__(self, 
        n_topics: int,
        n_features: int, 
        alpha: float = 1.0, 
        beta_params: tuple[float, float] = (0.1, 2.0),
        device: torch.device | str | None = None
        ):

        self.n_topics = n_topics
        self.n_features = n_features

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.alpha = torch.ones(n_topics, device=self.device) * alpha
        self.feature_prob_given_topic = Beta(*beta_params).sample((n_topics, n_features)).to(self.device)
        
        self.topic_weights_distribution = Dirichlet(self.alpha)

    def to(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        if device == self.device:
            return self
        
        self.device = device
        self.alpha = self.alpha.to(device)
        self.feature_prob_given_topic = self.feature_prob_given_topic.to(device)
        self.topic_weights_distribution = Dirichlet(self.alpha)
        return self
    
    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate a batch of samples.
        Returns:
            features: (batch_size, n_features) 
                tensor with random values for active features, zeros for inactive features
        """
        # Sample topic mixture for each sample from Dirichlet
        topic_probs = self.topic_weights_distribution.sample((batch_size,))

        # Total law of probability: P(feature) = Σᵢ P(topicᵢ)×P(feature|topicᵢ)
        feature_probs = topic_probs @ self.feature_prob_given_topic
        active_features = torch.bernoulli(feature_probs)
        
        # Generate feature values: random values for active features, zero otherwise
        features = torch.where(
            active_features.bool(),
            torch.rand(batch_size, self.n_features, device=self.device),
            torch.zeros_like(active_features)
        )
        return features
    
    def state_dict(self) -> dict:
        """Return state dictionary for saving."""
        return {
            'n_topics': self.n_topics,
            'n_features': self.n_features,
            'alpha': self.alpha,
            'feature_prob_given_topic': self.feature_prob_given_topic
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state from dictionary."""
        self.n_topics = state_dict['n_topics']
        self.n_features = state_dict['n_features']
        self.alpha = state_dict['alpha'].to(self.device)
        self.feature_prob_given_topic = state_dict['feature_prob_given_topic'].to(self.device)
        self.topic_weights_distribution = Dirichlet(self.alpha)