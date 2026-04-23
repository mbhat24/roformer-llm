"""
Learned Adaptive Position Encoding (LAPE)
A novel approach to dynamic, context-aware position encoding
"""

import torch
import torch.nn as nn
import math


class LAPE(nn.Module):
    """
    Learned Adaptive Position Encoding
    
    Instead of fixed frequency schedules, LAPE learns to predict
    optimal frequency scaling for each position based on input context.
    
    This allows the model to adapt its position encoding to different
    sequence patterns and content, potentially enabling better extrapolation
    beyond training lengths.
    """
    
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        adapter_dim=64,
        num_layers=2
    ):
        """
        Args:
            dim: Dimension of the rotary embedding (head_dim)
            max_position_embeddings: Maximum sequence length
            base: Base for the frequency computation (default 10000)
            adapter_dim: Hidden dimension for the adapter network
            num_layers: Number of layers in the adapter network
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        
        # Base frequencies (standard RoPE)
        self.register_buffer(
            "base_freqs",
            1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False
        )
        
        # Adapter network to predict frequency multipliers
        # Input: position index + aggregated context
        # Output: frequency multiplier for each frequency band
        self.adapter = nn.Sequential(
            nn.Linear(1 + dim, adapter_dim),  # position + context
            nn.LayerNorm(adapter_dim),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(adapter_dim, adapter_dim),
                    nn.LayerNorm(adapter_dim),
                    nn.ReLU()
                )
                for _ in range(num_layers - 1)
            ],
            nn.Linear(adapter_dim, dim // 2),  # Output: multiplier per frequency band
            nn.Sigmoid()  # Multipliers in [0, 2] range
        )
        
        # Context aggregation network
        self.context_aggregator = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, dim)
        )
    
    def forward(self, x, positions=None):
        """
        Apply Learned Adaptive Position Encoding
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            positions: Optional position indices (default: 0 to seq_len-1)
            
        Returns:
            Rotated tensor with adaptive position encoding
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, dtype=torch.float, device=x.device)
        
        # Aggregate context from the input
        # Average across heads to get per-position context
        context = x.mean(dim=2)  # (batch_size, seq_len, head_dim)
        context = self.context_aggregator(context)  # (batch_size, seq_len, head_dim)
        
        # For each position, predict frequency multipliers
        freq_multipliers = []
        
        for pos_idx in range(seq_len):
            # Create input: [position, aggregated_context]
            pos_tensor = torch.full((batch_size,), positions[pos_idx], device=x.device)
            pos_tensor = pos_tensor.unsqueeze(-1)  # (batch_size, 1)
            
            pos_context = torch.cat([pos_tensor, context[:, pos_idx]], dim=-1)  # (batch_size, 1 + head_dim)
            
            # Predict multipliers
            multipliers = self.adapter(pos_context)  # (batch_size, head_dim // 2)
            
            # Scale multipliers to [0.5, 2.0] range for stability
            multipliers = 0.5 + 1.5 * multipliers
            
            freq_multipliers.append(multipliers)
        
        freq_multipliers = torch.stack(freq_multipliers, dim=1)  # (batch_size, seq_len, head_dim // 2)
        
        # Compute adaptive frequencies
        # base_freqs: (head_dim // 2)
        # freq_multipliers: (batch_size, seq_len, head_dim // 2)
        adaptive_freqs = self.base_freqs.unsqueeze(0).unsqueeze(0) * freq_multipliers  # (batch_size, seq_len, head_dim // 2)
        
        # Compute position frequencies
        positions_expanded = positions.view(1, seq_len, 1)  # (1, seq_len, 1)
        freqs = positions_expanded * adaptive_freqs  # (batch_size, seq_len, head_dim // 2)
        
        # Split into cos and sin
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        
        # Apply rotation
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Broadcast freqs to match batch and head dimensions
        freqs_cos = freqs_cos.unsqueeze(2)  # (batch_size, seq_len, 1, head_dim // 2)
        freqs_sin = freqs_sin.unsqueeze(2)  # (batch_size, seq_len, 1, head_dim // 2)
        
        x_rotated = torch.stack([
            x_even * freqs_cos - x_odd * freqs_sin,
            x_even * freqs_sin + x_odd * freqs_cos
        ], dim=-1)
        
        x_rotated = x_rotated.reshape(x.shape)
        
        return x_rotated


class LAPELite(nn.Module):
    """
    Lightweight version of LAPE with lower computational overhead
    """
    
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        adapter_dim=32
    ):
        """
        Args:
            dim: Dimension of the rotary embedding (head_dim)
            max_position_embeddings: Maximum sequence length
            base: Base for the frequency computation
            adapter_dim: Hidden dimension for the adapter
        """
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Base frequencies
        self.register_buffer(
            "base_freqs",
            1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False
        )
        
        # Simple MLP for frequency scaling
        self.adapter = nn.Sequential(
            nn.Linear(1, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, dim // 2),
            nn.Sigmoid()
        )
    
    def forward(self, x, positions=None):
        """
        Apply lightweight LAPE
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            positions: Optional position indices
            
        Returns:
            Rotated tensor
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, dtype=torch.float, device=x.device)
        
        # Predict multipliers based only on position (no context)
        positions_tensor = positions.view(-1, 1)  # (seq_len, 1)
        multipliers = self.adapter(positions_tensor)  # (seq_len, head_dim // 2)
        multipliers = 0.5 + 1.5 * multipliers  # Scale to [0.5, 2.0]
        
        # Compute frequencies
        adaptive_freqs = self.base_freqs * multipliers  # (seq_len, head_dim // 2)
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
