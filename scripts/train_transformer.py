import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.transformer import TransformerLanguageModel, TransformerConfig
from datasets.language_dataset import LanguageDataset
from tokenizers import CharTokenizer, Tokenizer
from utils.mixed_precision import get_autocast_ctx

# hyperparameters
torch.manual_seed(1337)

# model
embed_dim = 384
num_heads = 6
head_size = embed_dim // num_heads
n_layers = 6
dropout = 0.2

# data
train_split = 0.90

# training
batch_size = 64
block_size = 256
epochs = 1
max_train_steps = 5000
eval_interval = 1000
example_interval = 1000
learning_rate = 3e-4
training_data_path = 'data/shakespeare.txt'

compile_model = True

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# precision
# Enable TF32 for faster training (negligible change with bfloat16)
torch.set_float32_matmul_precision('high')
# autocast context manager for mixed precision training
mixed_precision_ctx = get_autocast_ctx(device)

# load data
with open(training_data_path, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharTokenizer(text)
    
train_dataset = LanguageDataset(text, tokenizer, split='train', train_split=train_split, block_size=block_size)
val_dataset = LanguageDataset(text, tokenizer, split='val', train_split=train_split, block_size=block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

transformer_config = TransformerConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    head_size=head_size,
    n_layers=n_layers,
    dropout=dropout
)
model = TransformerLanguageModel(transformer_config).to(device)

if compile_model:
    print("compiling the model...")
    model = torch.compile(model)
    print("model compiled")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def generate_example(model: TransformerLanguageModel, tokenizer: Tokenizer, max_tokens: int) -> str:
    """Generate a single example from the model"""

    model.eval()
    idx = torch.zeros((1,1), dtype=torch.long).to(device)
    with mixed_precision_ctx:
        raw_prediction = model.generate(idx, max_new_tokens=max_tokens-1)[0].tolist()
    model.train()
    return tokenizer.decode(raw_prediction)

def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute the cross entropy loss for a batch of logits and targets
    logits: (B, T, vocab_size)
    targets: (B, T)
    returns: (1)
    """
    batch_size, block_size, vocab_size = logits.shape
    logits = logits.view(batch_size*block_size, vocab_size)
    targets = targets.view(batch_size*block_size)
    return F.cross_entropy(logits, targets)

@torch.no_grad()
def evaluate_model(model: TransformerLanguageModel, val_loader: DataLoader) -> float:
    """Evaluate the model on the validation set"""
    model.eval()
    losses = []
    for batch in val_loader:
        context, targets = batch
        context = context.to(device)
        targets = targets.to(device)
        with mixed_precision_ctx:
            logits = model(context)
            loss = compute_loss(logits, targets)
        losses.append(loss.item())
    model.train()
    return torch.tensor(losses).mean().item()

def train_loop(
        model: TransformerLanguageModel, 
        optimizer: torch.optim.Optimizer, 
        train_loader: DataLoader, 
        val_loader: DataLoader
        ) -> float:
    """Train the model for a given number of epochs"""
    global_step = 0

    val_loss = evaluate_model(model, val_loader)
    print(f'Validation loss: {val_loss}, global_step: {global_step}')

    print(f'Training for {max_train_steps} steps')
    for epoch in range(epochs):

        remaining_steps = max_train_steps - global_step
        steps_this_epoch = min(len(train_loader), remaining_steps)
        if steps_this_epoch <= 0:
            break

        progress_bar = tqdm(total=steps_this_epoch, desc=f'Epoch {epoch}', leave=False)

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            context, targets = batch
            context = context.to(device) # (B, T)
            targets = targets.to(device) # (B, T)

            with mixed_precision_ctx:
                logits = model(context) # (B, T, vocab_size)
                loss = compute_loss(logits, targets)

            loss.backward()
            optimizer.step()

            global_step += 1
            progress_bar.update(1)

            progress_bar.set_postfix(loss=loss.item(), global_step=global_step)

            if global_step % eval_interval == 0 or global_step == max_train_steps:
                val_loss = evaluate_model(model, val_loader)
                print(f'Validation loss: {val_loss}, global_step: {global_step}')

            if global_step % example_interval == 0 or global_step == max_train_steps:
                print("================================================")
                print(generate_example(model, tokenizer, max_tokens=block_size))
                print("================================================")
            
            if global_step >= max_train_steps:
                break

        progress_bar.close()
    
    return loss.item()


loss = train_loop(model, optimizer, train_loader, val_loader)
print(f'Final loss: {loss}')