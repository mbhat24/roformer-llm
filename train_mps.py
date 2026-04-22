"""
Training script for RoFormer LLM on real Shakespeare dataset with MPS (Mac GPU) acceleration
Building a larger, more capable model
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import RoFormerLLM, RoFormerLLMConfig
from word_tokenizer import WordTokenizer


class TextDataset(Dataset):
    """Text dataset for language modeling"""
    
    def __init__(self, texts, tokenizer, max_length=512):
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
    """Collate function to pad sequences"""
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
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch_idx, (input_ids, target_ids, attention_mask) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)
        
        # Compute loss
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    """Main training function with MPS acceleration"""
    # Check for MPS (Mac GPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using MPS (Mac GPU) acceleration")
    else:
        device = torch.device('cpu')
        print("✗ MPS not available, using CPU")
    
    # Load training data
    print("\nLoading training data...")
    with open('data/training_chunks.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(texts)} training chunks")
    
    # Build word-level tokenizer with larger vocabulary
    print("\nBuilding word-level tokenizer...")
    tokenizer = WordTokenizer(texts, vocab_size=10000)  # Increased vocab
    
    # Optimized model configuration for MPS memory limits
    config = RoFormerLLMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=512,      # Reduced from 768
        num_heads=8,        # Reduced from 12
        num_layers=8,       # Reduced from 12
        ff_dim=2048,        # Reduced from 3072
        max_position_embeddings=512,  # Reduced from 1024
        dropout=0.1,
        pad_token_id=tokenizer.get_pad_token_id()
    )
    
    # Create model
    print("\nCreating ambitious model...")
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print("MODEL SPECIFICATIONS")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    print(f"Embedding dimension: {config.embed_dim}")
    print(f"Attention heads: {config.num_heads}")
    print(f"Transformer layers: {config.num_layers}")
    print(f"Feed-forward dimension: {config.ff_dim}")
    print(f"Max sequence length: {config.max_position_embeddings}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_pad_token_id())
    
    # Training loop
    num_epochs = 10
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: 16")
    print(f"Learning rate: 3e-5")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, config.pad_token_id, epoch)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'config': config.to_dict(),
                'tokenizer': tokenizer
            }, 'roformer_shakespeare_best.pt')
            print(f"✓ Saved best model (loss: {train_loss:.4f})")
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'config': config.to_dict(),
            'tokenizer': tokenizer
        }, f'roformer_shakespeare_epoch_{epoch+1}.pt')
    
    # Save final model
    print("\nSaving final model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'tokenizer': tokenizer
    }, 'roformer_shakespeare_final.pt')
    
    # Test generation
    print("\n" + "="*60)
    print("GENERATION TEST")
    print("="*60)
    model.eval()
    
    prompts = [
        "To be, or not to be",
        "What's in a name",
        "All the world's a stage",
        "Shall I compare thee",
        "The course of true love"
    ]
    
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, max_length=100, add_bos=False, add_eos=False)
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated = model.generate(
                prompt_ids,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(generated[0].cpu().numpy())
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
