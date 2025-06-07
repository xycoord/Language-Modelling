import torch
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from bigram_language_model import BigramLanguageModel
from language_dataset import LanguageDataset
from tokenizers import CharTokenizer

# hyperparameters
torch.manual_seed(1337)
batch_size = 32
block_size = 8
split_p = 0.9
epochs = 1
eval_iters = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharTokenizer(text)
    
train_dataset = LanguageDataset(text, tokenizer, split='train', train_split=0.9, block_size=block_size)
val_dataset = LanguageDataset(text, tokenizer, split='val', train_split=0.9, block_size=block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

model = BigramLanguageModel(tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def generate_example(model, tokenizer, max_new_tokens=50):
    """Generate a single example from the model"""

    model.eval()
    idx = torch.zeros((1,1), dtype=torch.long).to(device)
    raw_prediction = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
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
        logits = model(context)
        loss = compute_loss(logits, targets)
        losses.append(loss.item())
    model.train()
    return torch.tensor(losses).mean().item()

def train_loop(model, optimizer, train_loader, val_loader):
    """Train the model for a given number of epochs"""
    global_step = 0
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', mininterval=0.5)

        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)

            context, targets = batch

            context = context.to(device)
            targets = targets.to(device)

            logits = model(context)

            loss = compute_loss(logits, targets)

            loss.backward()
            optimizer.step()

            if global_step % 100 == 0 or step == len(progress_bar) - 1:
                progress_bar.set_postfix(loss=loss.item(), global_step=global_step)

            if global_step % eval_iters == 0:
                val_loss = evaluate_model(model, val_loader)
                print(f'validation loss: {val_loss}, global_step: {global_step}')
            
            global_step += 1

        print(generate_example(model, tokenizer))

    return loss.item()

loss = train_loop(model, optimizer, train_loader, val_loader)
print(f'final loss: {loss}')
print(generate_example(model, tokenizer, max_new_tokens=500))