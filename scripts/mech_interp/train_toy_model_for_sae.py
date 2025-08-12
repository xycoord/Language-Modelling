import torch
from tqdm import tqdm
from pathlib import Path

from mech_interp import plot_feature_directions, ToyModel
from mech_interp.data_generators import SyntheticSparseDataGenerator
from mech_interp.script_utils import create_uniform_sparsity, create_importance, weighted_mse_loss

# ==== Parameters ====

feature_dim = 5
hidden_dim = 2

sparsity = 0.9
importance_decay = 0.9

batch_size = 1024
num_steps = 10_000
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== Set up ====

uniform_sparsity = create_uniform_sparsity(feature_dim, sparsity)
importance = create_importance(feature_dim, importance_decay).to(device)

model = ToyModel(feature_dim, hidden_dim).to(device)

data_generator = SyntheticSparseDataGenerator(batch_size=batch_size, sparsity=uniform_sparsity, device=device)

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


# ==== Save and plot ====

script_dir = Path(__file__).resolve().parent
plot_path = script_dir / 'toy_model_for_sae.png'
model_path = script_dir / 'toy_model_for_sae.pth'

torch.save(model.state_dict(), model_path)
print(f"Saved model to {model_path}")

feature_directions = model.get_feature_directions().cpu()
labels = [f'Sparsity = {sparsity:.3f}']
plot_feature_directions(feature_directions, labels, importance, save_path=plot_path)
print(f"Saved feature directions to {plot_path}")