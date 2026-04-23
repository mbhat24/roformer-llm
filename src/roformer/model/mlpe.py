"""
MLPE: Meta-Learned Position Encoding
A novel meta-learning approach to position encoding
"""

import torch
import torch.nn as nn
import math


class TaskMetadata:
    """
    Metadata for a specific task/domain
    """
    def __init__(
        self,
        domain_id,  # Domain identifier (0: code, 1: literature, 2: dialogue, etc.)
        seq_len_mean,  # Mean sequence length
        seq_len_std,  # Standard deviation of sequence lengths
        vocab_size,  # Vocabulary size
        token_freq_stats,  # Token frequency statistics
        max_seq_len  # Maximum sequence length
    ):
        self.domain_id = domain_id
        self.seq_len_mean = seq_len_mean
        self.seq_len_std = seq_len_std
        self.vocab_size = vocab_size
        self.token_freq_stats = token_freq_stats
        self.max_seq_len = max_seq_len
        
    def to_tensor(self):
        """Convert metadata to tensor for hypernetwork input"""
        return torch.tensor([
            self.domain_id,
            self.seq_len_mean / 1000.0,  # Normalize
            self.seq_len_std / 1000.0,
            self.vocab_size / 50000.0,
            self.max_seq_len / 10000.0
        ], dtype=torch.float32)


class PositionEncodingHypernetwork(nn.Module):
    """
    Hypernetwork that generates position encoding parameters from task metadata
    """
    
    def __init__(
        self,
        metadata_dim=5,
        hidden_dim=128,
        output_dim=64,  # Output RoPE parameters
        num_layers=3
    ):
        """
        Args:
            metadata_dim: Dimension of task metadata
            hidden_dim: Hidden dimension of hypernetwork
            output_dim: Output dimension (RoPE parameters)
            num_layers: Number of layers in hypernetwork
        """
        super().__init__()
        
        self.metadata_dim = metadata_dim
        self.output_dim = output_dim
        
        # Hypernetwork layers
        layers = []
        input_dim = metadata_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, metadata):
        """
        Generate position encoding parameters from task metadata
        
        Args:
            metadata: Task metadata tensor of shape (batch_size, metadata_dim)
            
        Returns:
            RoPE parameters of shape (batch_size, output_dim)
        """
        return self.network(metadata)


class MLPE(nn.Module):
    """
    Meta-Learned Position Encoding
    
    Uses a hypernetwork to generate task-specific position encoding parameters
    from task metadata, enabling zero-shot and few-shot adaptation to new domains.
    """
    
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        hypernetwork_hidden=128,
        hypernetwork_layers=3
    ):
        """
        Args:
            dim: Dimension of the rotary embedding (head_dim)
            max_position_embeddings: Maximum sequence length
            base: Base for the frequency computation
            hypernetwork_hidden: Hidden dimension of hypernetwork
            hypernetwork_layers: Number of layers in hypernetwork
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        
        # Hypernetwork to generate RoPE parameters
        self.hypernetwork = PositionEncodingHypernetwork(
            metadata_dim=5,
            hidden_dim=hypernetwork_hidden,
            output_dim=dim // 2,  # Generate frequency multipliers
            num_layers=hypernetwork_layers
        )
        
        # Base frequencies (standard RoPE)
        self.register_buffer(
            "base_freqs",
            1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False
        )
        
        # Current task metadata (set during forward)
        self.current_metadata = None
        
    def set_task_metadata(self, metadata):
        """Set task metadata for current task"""
        self.current_metadata = metadata
        
    def forward(self, x, positions=None):
        """
        Apply Meta-Learned Position Encoding
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            positions: Optional position indices
            
        Returns:
            Rotated tensor with meta-learned position encoding
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, dtype=torch.float, device=x.device)
        
        # Generate frequency multipliers from task metadata
        if self.current_metadata is not None:
            metadata_tensor = self.current_metadata.to(x.device)
            if metadata_tensor.dim() == 1:
                metadata_tensor = metadata_tensor.unsqueeze(0)
            
            # Generate multipliers
            freq_multipliers = self.hypernetwork(metadata_tensor)  # (1, head_dim // 2)
            freq_multipliers = freq_multipliers.squeeze(0)  # (head_dim // 2)
            
            # Scale multipliers to [0.5, 2.0] range for stability
            freq_multipliers = 0.5 + 1.5 * torch.sigmoid(freq_multipliers)
        else:
            # Default to no scaling if no metadata
            freq_multipliers = torch.ones(self.dim // 2, device=x.device)
        
        # Compute adaptive frequencies
        adaptive_freqs = self.base_freqs * freq_multipliers  # (head_dim // 2)
        
        # Compute position frequencies
        positions_expanded = positions.view(seq_len, 1)  # (seq_len, 1)
        freqs = positions_expanded * adaptive_freqs  # (seq_len, head_dim // 2)
        
        # Cos and sin
        freqs_cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim // 2)
        freqs_sin = freqs.sin().unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim // 2)
        
        # Apply rotation
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_rotated = torch.stack([
            x_even * freqs_cos - x_odd * freqs_sin,
            x_even * freqs_sin + x_odd * freqs_cos
        ], dim=-1)
        
        x_rotated = x_rotated.reshape(x.shape)
        
        return x_rotated


class MAMLMetaLearner:
    """
    MAML-style meta-learner for MLPE
    
    Implements Model-Agnostic Meta-Learning for position encoding
    """
    
    def __init__(
        self,
        model,
        inner_lr=0.01,
        inner_steps=5,
        meta_lr=0.001
    ):
        """
        Args:
            model: The model with MLPE position encoding
            inner_lr: Learning rate for inner loop (task adaptation)
            inner_steps: Number of gradient steps for task adaptation
            meta_lr: Learning rate for outer loop (meta-optimization)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr
        
        # Meta-optimizer (only optimize hypernetwork)
        self.meta_optimizer = torch.optim.Adam(
            model.hypernetwork.parameters(),
            lr=meta_lr
        )
        
    def inner_loop(self, task_data, task_metadata):
        """
        Inner loop: Adapt position encoding to a specific task
        
        Args:
            task_data: Support set data for the task
            task_metadata: Metadata for the task
            
        Returns:
            Adapted model parameters
        """
        # Set task metadata
        self.model.set_task_metadata(task_metadata)
        
        # Clone hypernetwork parameters for adaptation
        adapted_params = {}
        for name, param in self.model.hypernetwork.named_parameters():
            adapted_params[name] = param.clone().detach()
        
        # Inner loop adaptation
        for step in range(self.inner_steps):
            # Compute loss on support set
            loss = self._compute_task_loss(task_data, adapted_params)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=True
            )
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params
    
    def outer_loop(self, task_distribution):
        """
        Outer loop: Optimize meta-learner for fast adaptation
        
        Args:
            task_distribution: List of tasks (each with support and query sets)
            
        Returns:
            Meta-loss across all tasks
        """
        meta_loss = 0.0
        
        for task in task_distribution:
            support_data, query_data, metadata = task
            
            # Inner loop: Adapt to task
            adapted_params = self.inner_loop(support_data, metadata)
            
            # Evaluate on query set with adapted parameters
            query_loss = self._compute_query_loss(query_data, adapted_params)
            meta_loss += query_loss
        
        meta_loss /= len(task_distribution)
        
        # Meta-optimization step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def _compute_task_loss(self, task_data, adapted_params):
        """Compute loss on support set with adapted parameters"""
        # This would be implemented based on the actual task
        # For now, return a placeholder
        return torch.tensor(0.0, requires_grad=True)
    
    def _compute_query_loss(self, query_data, adapted_params):
        """Compute loss on query set with adapted parameters"""
        # This would be implemented based on the actual task
        # For now, return a placeholder
        return torch.tensor(0.0, requires_grad=True)
