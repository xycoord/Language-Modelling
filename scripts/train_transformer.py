import torch
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.transformer import TransformerLanguageModel
from datasets.language_dataset import LanguageDataset
from tokenizers import CharTokenizer
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
split_p = 0.90

# training
batch_size = 64
block_size = 256
epochs = 1
max_train_steps = 5000
eval_interval = 1000
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
    
train_dataset = LanguageDataset(text, tokenizer, split='train', train_split=0.9, block_size=block_size)
val_dataset = LanguageDataset(text, tokenizer, split='val', train_split=0.9, block_size=block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

model = TransformerLanguageModel(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    head_size=head_size,
    n_layers=n_layers,
    dropout=dropout
).to(device)

if compile_model:
    print("compiling the model...")
    model = torch.compile(model)
    print("model compiled")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def generate_example(model, tokenizer, max_tokens):
    """Generate a single example from the model"""

    model.eval()
    idx = torch.zeros((1,1), dtype=torch.long).to(device)
    with mixed_precision_ctx:
        raw_prediction = model.generate(idx, max_new_tokens=max_tokens-1)[0].tolist()
    model.train()
    return tokenizer.decode(raw_prediction)

def compute_loss(logits, targets):
    """Compute the loss for a batch of logits and targets"""
    batch_size, block_size, vocab_size = logits.shape
    logits = logits.view(batch_size*block_size, vocab_size)
    targets = targets.view(batch_size*block_size)
    return F.cross_entropy(logits, targets)

@torch.no_grad()
def evaluate_model(model, val_loader):
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

def train_loop(model, optimizer, train_loader, val_loader):
    """Train the model for a given number of epochs"""
    global_step = 0
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)

            context, targets = batch
            context = context.to(device)
            targets = targets.to(device)

            with mixed_precision_ctx:
                logits = model(context)
                loss = compute_loss(logits, targets)

            loss.backward()
            optimizer.step()

            if global_step % 100 == 0 or step == len(progress_bar) - 1:
                progress_bar.set_postfix(loss=loss.item(), global_step=global_step)

            if global_step % eval_interval == 0:
                val_loss = evaluate_model(model, val_loader)
                print(f'validation loss: {val_loss}, global_step: {global_step}')
                print(generate_example(model, tokenizer, max_tokens=block_size))
            
            global_step += 1

            if global_step >= max_train_steps:
                break

        print(generate_example(model, tokenizer, max_tokens=block_size))

    return loss.item()

loss = train_loop(model, optimizer, train_loader, val_loader)
print(f'final loss: {loss}')
print(generate_example(model, tokenizer, max_tokens=block_size))