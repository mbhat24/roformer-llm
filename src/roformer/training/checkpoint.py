"""
Checkpoint management for RoFormer LLM training
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class CheckpointManager:
    """Manage model checkpoints during training"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """Load existing checkpoints from directory"""
        if self.checkpoint_dir.exists():
            for ckpt_file in self.checkpoint_dir.glob("*.pt"):
                self.checkpoints.append(ckpt_file)
            self.checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        loss: float,
        config: Dict[str, Any],
        tokenizer: Any,
        is_best: bool = False
    ):
        """
        Save checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current step
            loss: Current loss
            config: Configuration dictionary
            tokenizer: Tokenizer
            is_best: Whether this is the best model so far
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"
        if is_best:
            checkpoint_name = f"best_model_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config,
            'tokenizer': tokenizer,
            'timestamp': timestamp
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            # Also save as best_model.pt for easy loading
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Manage checkpoint count
        self.checkpoints.append(checkpoint_path)
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints if exceeding max"""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop()
            if oldest.name != "best_model.pt":  # Don't delete best model
                oldest.unlink()
    
    def load(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint. If None, loads latest
        
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            if not self.checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = self.checkpoints[0]
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint
    
    def load_best(self) -> Dict[str, Any]:
        """Load best checkpoint"""
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError("Best model checkpoint not found")
        return self.load(str(best_path))
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        return self.checkpoints[0] if self.checkpoints else None
