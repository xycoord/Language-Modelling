import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from pathlib import Path

import wandb

from models.transformer import TransformerLanguageModel
from datasets.language_dataset import LanguageDataset
from tokenizers import Tokenizer, OptimizedBPETokenizer

from utils import Config, ArgsParser, setup_precision, get_autocast_ctx


def generate_example(model: TransformerLanguageModel, tokenizer: Tokenizer, max_tokens: int, config: Config) -> str:
    """Generate a single example from the model"""
    mixed_precision_ctx = get_autocast_ctx(config)
    model.eval()
    idx = torch.zeros((1,1), dtype=torch.long).to(model.device)
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
def evaluate_model(model: TransformerLanguageModel, val_loader: DataLoader, config: Config) -> float:
    """Evaluate the model on the validation set"""
    mixed_precision_ctx = get_autocast_ctx(config)
    model.eval()
    losses = []
    for batch in val_loader:
        context, targets = batch
        context = context.to(model.device)
        targets = targets.to(model.device)
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
        val_loader: DataLoader,
        tokenizer: Tokenizer,
        config: Config
        ) -> float:
    """Train the model for a given number of epochs"""
    global_step = 0

    val_loss = evaluate_model(model, val_loader, config)
    print(f'Validation loss: {val_loss}, global_step: {global_step}')
    wandb.log({"val_loss": val_loss}, step=global_step)

    # autocast context manager for mixed precision training
    mixed_precision_ctx = get_autocast_ctx(config)

    print(f'Training for {config.max_train_steps} steps')
    for epoch in range(config.epochs):

        remaining_steps = config.max_train_steps - global_step
        steps_this_epoch = min(len(train_loader), remaining_steps)
        if steps_this_epoch <= 0:
            break

        progress_bar = tqdm(total=steps_this_epoch, desc=f'Epoch {epoch}', leave=False)

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            context, targets = batch
            context = context.to(model.device) # (B, T)
            targets = targets.to(model.device) # (B, T)

            with mixed_precision_ctx:
                logits = model(context) # (B, T, vocab_size)
                loss = compute_loss(logits, targets)

            loss.backward()
            optimizer.step()

            global_step += 1
            progress_bar.update(1)

            progress_bar.set_postfix(loss=loss.item(), global_step=global_step)
            wandb.log({"loss": loss.item()}, step=global_step)

            if global_step % config.eval_interval == 0 or global_step == config.max_train_steps:
                val_loss = evaluate_model(model, val_loader, config)
                print(f'Validation loss: {val_loss}, global_step: {global_step}')
                wandb.log({"val_loss": val_loss}, step=global_step)

            if global_step % config.example_interval == 0 or global_step == config.max_train_steps:
                print("================================================")
                print(generate_example(model, tokenizer, max_tokens=config.block_size, config=config))
                print("================================================")
            
            if global_step >= config.max_train_steps:
                break

        progress_bar.close()
    
    return loss.item()


def main():
    parser = ArgsParser()
    config_path, overrides = parser.parse_config_args()
    config = Config.from_file(config_path, overrides)

    torch.manual_seed(config.seed)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {config.device}')

    setup_precision(config)

    # load data
    data_path = Path(config.data_dir) / 'shakespeare.txt'
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = OptimizedBPETokenizer.load(config.tokenizer_path)
        
    train_dataset = LanguageDataset(text, tokenizer, split='train', 
                                    train_split=config.train_split, 
                                    block_size=config.block_size)
    val_dataset = LanguageDataset(text, tokenizer, split='val', 
                                  train_split=config.train_split, 
                                  block_size=config.block_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    config.model_config_typed.vocab_size = tokenizer.vocab_size
    print(f'Vocab size: {config.model_config_typed.vocab_size}')
    model = TransformerLanguageModel(config.model_config_typed).to(config.device)

    if config.compile_model:
        print("compiling the model...")
        model = torch.compile(model)
        print("model compiled")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    wandb.init(project="language-modelling", config=config)

    final_loss = train_loop(model, optimizer, train_loader, val_loader, tokenizer, config)
    print(f'Final loss: {final_loss}')

    wandb.finish()


if __name__ == "__main__":
    main()