"""
Training script for RoFormer LLM
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from model import RoFormerLLM, RoFormerLLMConfig
from tokenizer import SimpleTokenizer


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
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
        tokens = self.tokenizer.encode(text, max_length=self.max_length, add_eos=True)
        
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
    
    # Create attention mask (1 for real tokens, 0 for padding)
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


def evaluate(model, dataloader, criterion, device, pad_token_id):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_ids, target_ids, attention_mask in tqdm(dataloader, desc="Evaluating"):
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
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    """Main training function"""
    # Configuration
    config = RoFormerLLMConfig(
        vocab_size=100,  # Will be set by tokenizer
        embed_dim=256,   # Small model for demo
        num_heads=8,
        num_layers=6,
        ff_dim=1024,
        max_position_embeddings=512,
        dropout=0.1,
        pad_token_id=0
    )
    
    # Sample training data (in practice, use real dataset)
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Transformers have revolutionized natural language processing.",
        "Rotary position embeddings improve transformer performance.",
        "The attention mechanism allows models to focus on relevant parts.",
        "Language models can generate coherent text.",
        "Training large models requires significant computational resources.",
        "RoFormer combines attention with rotary position embeddings.",
        "Position encoding helps models understand word order."
    ] * 100  # Repeat to create more training data
    
    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = SimpleTokenizer(texts)
    config.vocab_size = tokenizer.get_vocab_size()
    config.pad_token_id = tokenizer.get_pad_token_id()
    
    # Create model
    print("Creating model...")
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
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Device: {device}")
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_pad_token_id())
    
    # Training loop
    num_epochs = 10
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, config.pad_token_id)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
    
    # Save model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'tokenizer': tokenizer
    }, 'roformer_llm.pt')
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    prompt = "The attention mechanism"
    prompt_ids = tokenizer.encode(prompt, max_length=50, add_eos=False).unsqueeze(0).to(device)
    
    generated = model.generate(
        prompt_ids,
        max_new_tokens=50,
        temperature=0.8,
        do_sample=True
    )
    
    generated_text = tokenizer.decode(generated[0].cpu().numpy())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text[len(prompt):]}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
