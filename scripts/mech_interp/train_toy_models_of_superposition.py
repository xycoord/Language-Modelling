import torch
from tqdm import tqdm
from pathlib import Path

from mech_interp import plot_feature_directions, ParallelToyModel
from mech_interp.data_generators import SyntheticSparseDataGenerator
from mech_interp.script_utils import create_sparsity_range, create_importance, weighted_mse_loss

# ==== Parameters ====

feature_dim = 5
hidden_dim = 2

importance_decay = 0.9
min_sparsity = 0.0
max_sparsity = 0.95

num_models = 10
batch_size = 1024
num_steps = 10_000
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== Set up ====

sparsity = create_sparsity_range(min_sparsity, max_sparsity, num_models, feature_dim)

importance = create_importance(feature_dim, importance_decay).to(device)

model = ParallelToyModel(num_models, feature_dim, hidden_dim).to(device)

data_generator = SyntheticSparseDataGenerator(batch_size=batch_size, sparsity=sparsity, device=device)

# ==== Training ====

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

progress_bar = tqdm(range(num_steps))

for step in progress_bar:
    optimizer.zero_grad(set_to_none=True)

    batch = data_generator.generate_batch()

    output, _ = model(batch)

    loss = weighted_mse_loss(output, batch, importance)
    loss.backward()
    optimizer.step()

    progress_bar.set_postfix(loss=loss.item())

# ==== Plot Feature Directions ====

feature_directions = model.get_feature_directions().cpu()

script_dir = Path(__file__).resolve().parent
save_path = script_dir / 'toy_model_feature_directions.png'

sparcity_list = [sparsity[i][0].item() for i in range(num_models)]
labels = [f'Sparsity = {sparcity:.3f}' for sparcity in sparcity_list]

plot_feature_directions(feature_directions, labels, importance, save_path=save_path)
print(f"Saved feature directions to {save_path}")