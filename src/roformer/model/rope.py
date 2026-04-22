"""
Rotary Position Embedding (RoPE) implementation
Based on the RoFormer paper: https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
import math


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    Encodes absolute position with a rotation matrix and incorporates
    explicit relative position dependency in self-attention.
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Args:
            dim: Dimension of the embedding (must be even)
            max_position_embeddings: Maximum sequence length
            base: Base for the frequency computation (default 10000 as in paper)
        """
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cache for position embeddings
        self._build_cache()
    
    def _build_cache(self):
        """Pre-compute and cache the rotation matrix for all positions"""
        # Create position indices
        position = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        
        # Compute the angles: m * theta_i where theta_i = base^(-2i/d)
        freqs = torch.outer(position, self.inv_freq)
        
        # Create the rotation matrix components (cos and sin)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Split into cos and sin for rotation
        cos = emb.cos()
        sin = emb.sin()
        
        # Register as buffer (not a learnable parameter)
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)
    
    def rotate_half(self, x):
        """
        Rotate half the dimensions
        
        For a vector [x1, x2, x3, x4, ...], this transforms it to
        [-x2, x1, -x4, x3, ...]
        
        This is equivalent to multiplying by the rotation matrix.
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x, cos, sin):
        """
        Apply rotary position embedding to the input tensor
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            cos: Cosine values of shape (seq_len, head_dim)
            sin: Sine values of shape (seq_len, head_dim)
        
        Returns:
            Rotated tensor of same shape as x
        """
        # Apply rotation: x * cos + rotate_half(x) * sin
        # This is equivalent to: R * x where R is the rotation matrix
        return (x * cos) + (self.rotate_half(x) * sin)
    
    def forward(self, x, seq_len=None):
        """
        Apply rotary position embedding to input
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
                  or (batch_size, seq_len, dim)
            seq_len: Sequence length (if None, inferred from x)
        
        Returns:
            Rotated tensor of same shape as x
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        assert seq_len <= self.max_position_embeddings, \
            f"Sequence length {seq_len} exceeds maximum {self.max_position_embeddings}"
        
        # Get the cached cos and sin for this sequence length
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Reshape for broadcasting based on input shape
        if x.dim() == 4:
            # x shape: (batch, seq_len, num_heads, head_dim) - before transpose in attention
            # cos/sin shape: (seq_len, head_dim)
            # Need: (1, seq_len, 1, head_dim) for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 3:
            # x shape: (batch, seq_len, dim) - single head or no head dimension
            # cos/sin shape: (seq_len, head_dim)
            # Need: (1, seq_len, head_dim) for broadcasting
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")
        
        # Apply rotation
        rotated = self.apply_rotary_pos_emb(x, cos, sin)
        
        return rotated


def apply_rotary_pos_emb(q, k, rope):
    """
    Apply RoPE to query and key tensors
    
    This is the main function used in attention mechanisms.
    
    Args:
        q: Query tensor (batch_size, seq_len, num_heads, head_dim)
        k: Key tensor (batch_size, seq_len, num_heads, head_dim)
        rope: RotaryPositionEmbedding module
    
    Returns:
        Rotated q and k tensors
    """
    return rope(q), rope(k)
