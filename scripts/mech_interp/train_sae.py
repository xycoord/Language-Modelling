import torch
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

from mech_interp import ToyModel, SyntheticSparseDataGenerator 
from mech_interp.sparse_autoencoder import TopKSparseAutoencoder
from mech_interp.script_utils import create_uniform_sparsity, create_importance
from mech_interp.geometric_median import geometric_median
from mech_interp.plot import plot_feature_directions

script_dir = Path(__file__).resolve().parent
model_path = script_dir / 'toy_model_for_sae.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"

feature_dim = 5
hidden_dim = 2
sparsity = 0.9
importance_decay = 0.9

uniform_sparsity = create_uniform_sparsity(feature_dim, sparsity)
importance = create_importance(feature_dim, importance_decay)

model = ToyModel(feature_dim, hidden_dim)
model.load_state_dict(torch.load(model_path))
model.requires_grad_(False)
model.to(device)

# === Setup SAE ===

# Collect activations for SAE initialisation
num_samples = 10_000
sample_data = SyntheticSparseDataGenerator(
                batch_size=num_samples,
                sparsity=uniform_sparsity,
                device=device
                ).generate_batch()
_, activations = model(sample_data)

# Following the paper, we initialise the SAE bias as the geometric median of the activations.
initial_sae_bias = geometric_median(activations)

sae = TopKSparseAutoencoder(activations_dim=hidden_dim,
                            feature_dim=feature_dim,
                            initial_bias=initial_sae_bias,
                            k=1)
sae.to(device)

# === Train SAE ===

iterations = 20_000
sparsity_penalty = 0.00001
learning_rate = 0.005
batch_size = 2048

data_generator = SyntheticSparseDataGenerator(
                batch_size=batch_size,
                sparsity=uniform_sparsity,
                device=device
                )

optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate, betas=(0.0, 0.999))

model_feature_directions = model.get_feature_directions().cpu()
feature_directions = [model_feature_directions]
labels = ['Target']

progress_bar = tqdm(range(iterations))
for step in progress_bar:
    if step % 5000 == 0:
        feature_directions.append(sae.get_feature_directions().cpu())
        labels.append(f'SAE@{step}')

    optimizer.zero_grad(set_to_none=True)

    # Collect activations
    batch = data_generator.generate_batch()
    _, activations = model(batch)

    reconstruction, features = sae(activations)

    reconstruction_loss = F.mse_loss(activations, reconstruction)
    sparsity_loss = sparsity_penalty * features.abs().sum()
    loss = reconstruction_loss #+ sparsity_loss

    loss.backward()
    optimizer.step()

    progress_bar.set_postfix(loss=f"{loss.item():.3f}", reconstruction_loss=f"{reconstruction_loss.item():.3f}", sparsity_loss=f"{sparsity_loss.item():.3f}")

feature_directions.append(sae.get_feature_directions().cpu())
labels.append(f'SAE@{iterations}')
feature_directions = torch.stack(feature_directions, dim=0)

# === Plot feature directions ===

plot_path = script_dir / 'sae_learnt_feature_directions.png'
plot_feature_directions(feature_directions, labels, importance, save_path=plot_path)