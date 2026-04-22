"""
Deep training script for RoFormer LLM with professional training loop
Includes learning rate scheduling, checkpointing, logging, and evaluation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm
import numpy as np

from roformer.model.llm import RoFormerLLM, RoFormerLLMConfig
from roformer.data.tokenizer import WordTokenizer
from roformer.data.dataset import TextDataset, collate_fn
from roformer.data.custom_data import CustomDataLoader
from roformer.utils.config import Config
from roformer.utils.logger import get_logger
from roformer.utils.hardware import detect_hardware, get_optimal_device
from roformer.training.checkpoint import CheckpointManager
from roformer.evaluation.benchmarks import LLMBenchmarks


class Trainer:
    """Professional trainer for RoFormer LLM"""
    
    def __init__(self, config: Config):
        """
        Initialize trainer
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger("roformer", config.logging.log_dir)
        self.device = self._setup_device()
        
        # Load data
        self.logger.info("Loading training data...")
        self.texts = self._load_data()
        self.tokenizer = self._build_tokenizer()
        
        # Create datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self._create_datasets()
        
        # Create model
        self.logger.info("Creating model...")
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=config.data.shuffle,
            collate_fn=collate_fn,
            num_workers=config.data.num_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.data.num_workers
        )
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Create loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_pad_token_id())
        
        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.logging.checkpoint_dir,
            max_checkpoints=5
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        self.logger.info("Trainer initialized successfully")
    
    def _setup_device(self):
        """Setup device based on configuration"""
        if self.config.hardware.device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(self.config.hardware.device)
        
        self.logger.info(f"Using device: {device}")
        return device
    
    def _load_data(self):
        """Load training data"""
        if os.path.exists(self.config.data.data_path):
            with open(self.config.data.data_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            # Try to use custom data loader
            self.logger.warning(f"Data file not found: {self.config.data.data_path}")
            self.logger.info("Using built-in Shakespeare dataset...")
            texts = CustomDataLoader.from_file('data/shakespeare.txt')
        
        self.logger.info(f"Loaded {len(texts)} text chunks")
        return texts
    
    def _build_tokenizer(self):
        """Build tokenizer"""
        tokenizer = WordTokenizer(self.texts, vocab_size=self.config.tokenizer.vocab_size)
        self.logger.info(f"Tokenizer built with {tokenizer.get_vocab_size()} words")
        return tokenizer
    
    def _create_datasets(self):
        """Create train/val/test splits"""
        total_size = len(self.texts)
        train_size = int(total_size * self.config.data.train_split)
        val_size = int(total_size * self.config.data.val_split)
        test_size = total_size - train_size - val_size
        
        dataset = TextDataset(
            self.texts,
            self.tokenizer,
            max_length=self.config.data.max_length
        )
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size]
        )
        
        self.logger.info(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
        return train_dataset, val_dataset, test_dataset
    
    def _create_model(self):
        """Create model from configuration"""
        model_config = RoFormerLLMConfig(
            vocab_size=self.config.model.vocab_size,
            embed_dim=self.config.model.embed_dim,
            num_heads=self.config.model.num_heads,
            num_layers=self.config.model.num_layers,
            ff_dim=self.config.model.ff_dim,
            max_position_embeddings=self.config.model.max_position_embeddings,
            dropout=self.config.model.dropout,
            pad_token_id=self.config.model.pad_token_id
        )
        
        model = RoFormerLLM(
            vocab_size=model_config.vocab_size,
            embed_dim=model_config.embed_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            ff_dim=model_config.ff_dim,
            max_position_embeddings=model_config.max_position_embeddings,
            dropout=model_config.dropout,
            pad_token_id=model_config.pad_token_id
        )
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer"""
        if self.config.optimizer.type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer.type}")
        
        self.logger.info(f"Optimizer: {self.config.optimizer.type}")
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        warmup_steps = self.config.training.warmup_steps
        total_steps = len(self.train_loader) * self.config.training.num_epochs
        
        if self.config.scheduler.type == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.scheduler.min_lr
            )
        elif self.config.scheduler.type == "linear":
            scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=total_steps - warmup_steps
            )
        else:
            scheduler = None
        
        self.logger.info(f"Scheduler: {self.config.scheduler.type}")
        return scheduler
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, (input_ids, target_ids, attention_mask) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.logging.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_metrics({
                    'loss': loss.item(),
                    'lr': lr,
                    'epoch': epoch + 1
                }, self.global_step)
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids, attention_mask in tqdm(self.val_loader, desc="Validating"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.config.training.num_epochs}")
        self.logger.info(f"Total steps: {len(self.train_loader) * self.config.training.num_epochs}")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            self.logger.info(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                self.logger.info(f"New best model! Loss: {val_loss:.4f}")
            
            if (epoch + 1) % self.config.logging.save_interval == 0 or is_best:
                self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    self.global_step,
                    val_loss,
                    self.config.to_dict(),
                    self.tokenizer,
                    is_best=is_best
                )
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_loss:.4f}")


def main():
    """Main training function"""
    # Load configuration
    config_path = "configs/default.yaml"
    if os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()
    
    # Detect hardware and adjust config
    hardware_info = detect_hardware()
    if config.hardware.device == "auto":
        config.hardware.device = hardware_info['device']
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()
    
    # Final evaluation
    trainer.logger.info("Running final evaluation...")
    benchmarks = LLMBenchmarks(trainer.model, trainer.tokenizer, trainer.device)
    
    test_texts = [trainer.texts[i] for i in range(min(100, len(trainer.texts)))]
    prompts = ["To be", "What's in", "All the", "Shall I"]
    
    results = benchmarks.run_full_benchmark(test_texts, prompts)
    trainer.logger.info(f"Final perplexity: {results['perplexity']:.2f}")


if __name__ == "__main__":
    main()
