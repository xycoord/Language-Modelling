import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from lm_models.bigram import BigramLanguageModel, BigramConfig
from lm_datasets.language_dataset import LanguageDataset
from lm_tokenizers import CharTokenizer, Tokenizer
from script_utils import ArgsParser, Config
from pathlib import Path

def main():

    parser = ArgsParser()
    config_path, overrides = parser.parse_config_args()
    print(f'Loading config from {config_path} with overrides {overrides}')
    config = Config.from_file(config_path, overrides)

    torch.manual_seed(config.seed)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {config.device}')

    # load data
    data_path = Path(config.data_dir) / 'shakespeare.txt'
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
        
    train_dataset = LanguageDataset(text, tokenizer, split='train', train_split=config.train_split, block_size=config.block_size)
    val_dataset = LanguageDataset(text, tokenizer, split='val', train_split=config.train_split, block_size=config.block_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    model_config = BigramConfig(vocab_size=tokenizer.vocab_size)
    model = BigramLanguageModel(model_config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    final_loss = train_loop(model, optimizer, train_loader, val_loader, tokenizer, config)
    print(f'Final loss: {final_loss}')
    print(generate_example(model, tokenizer, max_new_tokens=500))

def generate_example(model: BigramLanguageModel, tokenizer: Tokenizer, max_new_tokens: int = 50) -> str:
    """Generate a single example from the model"""

    model.eval()
    idx = torch.zeros((1,1), dtype=torch.long).to(model.device)
    raw_prediction = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    model.train()
    return tokenizer.decode(raw_prediction)

def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute the loss for a batch of logits and targets"""
    batch_size, block_size, vocab_size = logits.shape
    logits = logits.view(batch_size*block_size, vocab_size)
    targets = targets.view(batch_size*block_size)
    return F.cross_entropy(logits, targets)

@torch.no_grad()
def evaluate_model(model: BigramLanguageModel, val_loader: DataLoader) -> float:
    """Evaluate the model on the validation set"""
    model.eval()
    losses = []
    for batch in val_loader:
        context, targets = batch
        context = context.to(model.device)
        targets = targets.to(model.device)
        logits = model(context)
        loss = compute_loss(logits, targets)
        losses.append(loss.item())
    model.train()
    return torch.tensor(losses).mean().item()

def train_loop(
        model: BigramLanguageModel, 
        optimizer: torch.optim.Optimizer, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        tokenizer: Tokenizer,
        config: Config,
        ) -> float:
    """Train the model for a given number of epochs"""
    global_step = 0
    for epoch in range(config.epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', mininterval=0.5)

        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)

            context, targets = batch

            context = context.to(model.device)
            targets = targets.to(model.device)

            logits = model(context)

            loss = compute_loss(logits, targets)

            loss.backward()
            optimizer.step()

            if global_step % 100 == 0 or step == len(progress_bar) - 1:
                progress_bar.set_postfix(loss=loss.item(), global_step=global_step)

            if global_step % config.eval_interval == 0:
                val_loss = evaluate_model(model, val_loader)
                print(f'validation loss: {val_loss}, global_step: {global_step}')
            
            global_step += 1

        print(generate_example(model, tokenizer))

    return loss.item()


if __name__ == "__main__":
    main()