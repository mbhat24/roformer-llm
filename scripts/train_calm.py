"""
CALM: Curiosity-Driven Action-Sensitive Language Models
Training Script
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from calm import CALM, CALMConfig
from calm.data_collection import (
    ActionDataCollector,
    LanguageActionDataset,
    SimpleTokenizer,
    generate_descriptions
)


def create_mock_environment():
    """Create a mock environment for demonstration"""
    class MockEnv:
        def __init__(self, state_dim=10, action_dim=3):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.max_steps = 200
        
        def reset(self):
            return np.random.randn(self.state_dim)
        
        def step(self, action):
            next_state = np.random.randn(self.state_dim)
            reward = np.random.randn()
            done = np.random.random() < 0.01
            info = {}
            return next_state, reward, done, info
    
    return MockEnv()


def collect_action_data(
    num_episodes: int = 100,
    max_steps: int = 200,
    save_dir: str = "data/calm"
):
    """
    Collect action data using curiosity-driven exploration
    
    Args:
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        save_dir: Directory to save data
    """
    print("Creating mock environment...")
    env = create_mock_environment()
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"Collecting {num_episodes} episodes...")
    collector = ActionDataCollector(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        curiosity_weight=0.1,
        save_dir=save_dir
    )
    
    sequences = collector.collect(
        num_episodes=num_episodes,
        max_steps=max_steps,
        save_every=50
    )
    
    stats = collector.get_statistics()
    print("\n=== Collection Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return sequences


def create_dataset(
    action_sequences,
    tokenizer,
    max_seq_len: int = 512
):
    """
    Create language-action dataset
    
    Args:
        action_sequences: List of action sequences
        tokenizer: Tokenizer for language
        max_seq_len: Maximum sequence length
        
    Returns:
        LanguageActionDataset
    """
    print("Generating language descriptions...")
    descriptions = generate_descriptions(action_sequences)
    
    print("Creating dataset...")
    dataset = LanguageActionDataset(
        action_sequences=action_sequences,
        descriptions=descriptions,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    
    return dataset


def train_calm(
    model: CALM,
    dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = 'cpu',
    save_dir: str = "checkpoints/calm"
):
    """
    Train CALM model
    
    Args:
        model: CALM model
        dataloader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining CALM for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    
    history = {
        'total_loss': [],
        'lm_loss': [],
        'align_loss': [],
        'curiosity_loss': [],
        'forward_loss': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_losses = {
            'total_loss': [],
            'lm_loss': [],
            'align_loss': [],
            'curiosity_loss': [],
            'forward_loss': []
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                action_sequence=action_sequence,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Backward pass
            optimizer.zero_grad()
            outputs['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record losses
            for key in epoch_losses.keys():
                if key in outputs:
                    epoch_losses[key].append(outputs[key].item() if isinstance(outputs[key], torch.Tensor) else outputs[key])
            
            # Update progress bar
            loss_components = outputs.get('loss_components', {})
            progress_bar.set_postfix({
                'total': f"{loss_components.get('total', 0):.4f}",
                'lm': f"{loss_components.get('lm', 0):.4f}",
                'align': f"{loss_components.get('align', 0):.4f}",
                'curiosity': f"{loss_components.get('curiosity', 0):.4f}"
            })
        
        # Compute epoch averages
        for key in epoch_losses:
            if epoch_losses[key]:
                avg_loss = np.mean(epoch_losses[key])
                history[key].append(avg_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        for key, losses in history.items():
            if losses:
                print(f"  {key}: {losses[-1]:.4f}")
        
        # Save checkpoint
        checkpoint_path = save_dir / f"calm_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    return model, history


def test_calm(
    model: CALM,
    tokenizer,
    device: str = 'cpu',
    num_samples: int = 5
):
    """
    Test CALM model
    
    Args:
        model: CALM model
        tokenizer: Tokenizer
        device: Device to test on
        num_samples: Number of test samples
    """
    model.eval()
    model = model.to(device)
    
    print("\n=== Testing CALM ===")
    
    # Generate sample text (use prompts that will be in vocab)
    test_prompts = [
        "the agent performed 50 actions with average curiosity 3.15",
        "the agent performed 100 actions with average curiosity 3.15",
        "the agent performed 200 actions with average curiosity 3.15"
    ]
    
    for prompt in test_prompts[:num_samples]:
        # Tokenize prompt
        encoded = tokenizer(prompt, max_length=128, padding='max_length', truncation=True)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Clamp token IDs to vocab_size
        vocab_size = model.vocab_size
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        print(f"\nPrompt: {prompt}")
        print(f"Input IDs: {input_ids[0].cpu().numpy()}")
        
        # Predict action
        with torch.no_grad():
            action = model.predict_action(input_ids, attention_mask)
        print(f"Predicted action: {action[0].cpu().numpy()}")
        print(f"Action shape: {action.shape}")
    
    print("\n=== Test Complete ===")


def main():
    """Main training pipeline"""
    
    # Configuration
    config = CALMConfig(
        vocab_size=50000,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        action_dim=3,
        max_seq_len=128,
        hidden_dim=128,
        latent_dim=64,
        lambda_align=1.0,
        gamma_curiosity=0.1,
        delta_forward=0.5
    )
    
    # Create model
    print("Creating CALM model...")
    model = config.create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Collect action data
    print("\n=== Stage 1: Action Data Collection ===")
    sequences = collect_action_data(
        num_episodes=100,
        max_steps=200,
        save_dir="data/calm"
    )
    
    # Create dataset
    print("\n=== Stage 2: Dataset Creation ===")
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    dataset = create_dataset(
        action_sequences=sequences,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: 8")
    print(f"Number of batches: {len(dataloader)}")
    
    # Train model
    print("\n=== Stage 3: Training ===")
    device = 'cpu'  # Use 'cuda' if available
    model, history = train_calm(
        model=model,
        dataloader=dataloader,
        num_epochs=5,
        learning_rate=1e-4,
        device=device,
        save_dir="checkpoints/calm"
    )
    
    # Test model
    print("\n=== Stage 4: Testing ===")
    test_calm(model, tokenizer, device=device, num_samples=3)
    
    print("\n=== Training Complete ===")
    print("Model saved to checkpoints/calm/")
    print("Action data saved to data/calm/")


if __name__ == "__main__":
    main()
