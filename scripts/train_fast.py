"""
Fast training script for RoFormer LLM on MPS
Optimized for speed while maintaining quality
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import RoFormerLLM, RoFormerLLMConfig
from word_tokenizer import WordTokenizer


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_length, add_bos=True, add_eos=True)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return input_ids, target_ids


def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    padded_target_ids = torch.zeros(len(target_ids), max_len, dtype=torch.long)
    
    for i, (inp, tgt) in enumerate(zip(input_ids, target_ids)):
        padded_input_ids[i, :len(inp)] = inp
        padded_target_ids[i, :len(tgt)] = tgt
    
    attention_mask = (padded_input_ids != 0).long()
    return padded_input_ids, padded_target_ids, attention_mask


def train_epoch(model, dataloader, optimizer, criterion, device, pad_token_id, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch_idx, (input_ids, target_ids, attention_mask) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        logits = model(input_ids, attention_mask=attention_mask)
        
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    # MPS device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading training data...")
    with open('data/training_chunks.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # Use subset for faster training
    texts = texts[:1000]  # First 1000 chunks
    print(f"Using {len(texts)} training chunks")
    
    # Tokenizer
    print("\nBuilding tokenizer...")
    tokenizer = WordTokenizer(texts, vocab_size=5000)
    
    # Fast but capable model
    config = RoFormerLLMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=384,      # Smaller but capable
        num_heads=6,
        num_layers=6,
        ff_dim=1536,
        max_position_embeddings=256,
        dropout=0.1,
        pad_token_id=tokenizer.get_pad_token_id()
    )
    
    model = RoFormerLLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        max_position_embeddings=config.max_position_embeddings,
        dropout=config.dropout,
        pad_token_id=config.pad_token_id
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters (~{total_params * 4 / 1024 / 1024:.1f} MB)")
    print(f"Embeddings: {config.embed_dim}, Heads: {config.num_heads}, Layers: {config.num_layers}")
    
    # Dataset
    dataset = TextDataset(texts, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_pad_token_id())
    
    # Training
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Batch size: 32, Learning rate: 5e-4\n")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, config.pad_token_id, epoch)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
    
    # Save
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'tokenizer': tokenizer
    }, 'roformer_shakespeare_fast.pt')
    
    # Test
    print("\nTesting generation...")
    model.eval()
    
    prompts = ["To be", "What's in", "All the", "Shall I"]
    
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, max_length=50, add_bos=False, add_eos=False)
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated = model.generate(prompt_ids, max_new_tokens=20, temperature=0.8, do_sample=True)
        
        generated_text = tokenizer.decode(generated[0].cpu().numpy())
        print(f"Prompt: {prompt} -> {generated_text}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
