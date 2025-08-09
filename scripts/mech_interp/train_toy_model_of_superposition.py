import torch
from tqdm import tqdm
from pathlib import Path

from mech_interp import SyntheticSparseDataGenerator, plot_feature_directions, ParallelToyModel

# ==== Parameters ====

feature_dim = 5
hidden_dim = 2

importance_decay = 0.9
min_feature_probability = 0.05

num_models = 10
batch_size = 1024
num_steps = 10_000
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== Set up ====

feature_probabilities = torch.logspace(0, -1, num_models, base=1/min_feature_probability)
sparsity = (1 - feature_probabilities)
# Repeat sparsity across feature dimension: (num_models,) -> (num_models, feature_dim)
sparsity = sparsity.unsqueeze(1).expand(-1, feature_dim)

importance = importance_decay ** torch.arange(feature_dim, dtype=torch.float)
importance = importance.to(device)

model = ParallelToyModel(n_instances=num_models, n_features=feature_dim, n_hidden=hidden_dim).to(device)

data_generator = SyntheticSparseDataGenerator(batch_size=batch_size, sparsity=sparsity, device=device)

# ==== Training ====

def weighted_mse_loss(output, target, weight):
    per_feature_loss = weight * (target - output) ** 2
    per_instance_loss = torch.mean(per_feature_loss, dim=(0,2))
    return per_instance_loss.mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

progress_bar = tqdm(range(num_steps))

for step in progress_bar:
    optimizer.zero_grad(set_to_none=True)

    batch = data_generator.generate_batch()

    output = model(batch)

    loss = weighted_mse_loss(output, batch, importance)
    loss.backward()
    optimizer.step()

    progress_bar.set_postfix(loss=loss.item())

# ==== Interpretability ====

feature_directions = model.get_feature_directions()

script_dir = Path(__file__).resolve().parent
save_path = script_dir / 'toy_model_feature_directions.png'
plot_feature_directions(feature_directions, feature_probabilities, importance, save_path=save_path)
print(f"Saved feature directions to {save_path}")