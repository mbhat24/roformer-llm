"""
Training script for RoFormer LLM on real Shakespeare dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import RoFormerLLM, RoFormerLLMConfig
from word_tokenizer import WordTokenizer


class TextDataset(Dataset):
    """Text dataset for language modeling"""
    
    def __init__(self, texts, tokenizer, max_length=256):
        """
        Args:
            texts: List of text strings
            tokenizer: WordTokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Encode text
        tokens = self.tokenizer.encode(text, max_length=self.max_length, add_bos=True, add_eos=True)
        
        # For language modeling, input = tokens[:-1], target = tokens[1:]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        return input_ids, target_ids


def collate_fn(batch):
    """Collate function to pad sequences"""
    input_ids, target_ids = zip(*batch)
    
    # Pad sequences
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    padded_target_ids = torch.zeros(len(target_ids), max_len, dtype=torch.long)
    
    for i, (inp, tgt) in enumerate(zip(input_ids, target_ids)):
        padded_input_ids[i, :len(inp)] = inp
        padded_target_ids[i, :len(tgt)] = tgt
    
    # Create attention mask
    attention_mask = (padded_input_ids != 0).long()
    
    return padded_input_ids, padded_target_ids, attention_mask


def train_epoch(model, dataloader, optimizer, criterion, device, pad_token_id):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (input_ids, target_ids, attention_mask) in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)
        
        # Compute loss (ignore padding tokens)
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
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    """Main training function"""
    # Load training data
    print("Loading training data...")
    with open('data/training_chunks.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(texts)} training chunks")
    
    # Build word-level tokenizer
    print("\nBuilding word-level tokenizer...")
    tokenizer = WordTokenizer(texts, vocab_size=5000)
    
    # Larger model configuration
    config = RoFormerLLMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=512,      # Increased from 256
        num_heads=8,        # Same
        num_layers=8,       # Increased from 6
        ff_dim=2048,        # Increased from 1024
        max_position_embeddings=512,
        dropout=0.1,
        pad_token_id=tokenizer.get_pad_token_id()
    )
    
    # Create model
    print("\nCreating larger model...")
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
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    print(f"Device: {device}")
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_pad_token_id())
    
    # Training loop
    num_epochs = 5
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, config.pad_token_id)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
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
    }, 'roformer_shakespeare.pt')
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    
    prompts = [
        "To be, or not to be",
        "What's in a name",
        "All the world's a stage",
        "Shall I compare thee"
    ]
    
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, max_length=50, add_bos=False, add_eos=False)
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated = model.generate(
                prompt_ids,
                max_new_tokens=30,
                temperature=0.8,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(generated[0].cpu().numpy())
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
