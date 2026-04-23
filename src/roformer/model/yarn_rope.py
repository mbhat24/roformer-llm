"""
YaRN (Yet another RoPE extension) - Extended Rotary Position Embeddings
Provides better extrapolation beyond training sequence length
"""

import torch
import torch.nn as nn
import math


class YarnRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE extension) for better length extrapolation
    
    YaRN applies a non-linear scaling to position indices to improve
    performance on sequences longer than those seen during training.
    
    Based on: "YaRN: Efficient Context Window Extension of Large Language Models"
    https://arxiv.org/abs/2309.00071
    """
    
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        alpha=1.0,
        beta=1.0,
        mscale=0.5
    ):
        """
        Args:
            dim: Dimension of the rotary embedding (head_dim)
            max_position_embeddings: Maximum sequence length
            base: Base for the frequency computation (default 10000)
            alpha: Alpha scaling factor for YaRN
            beta: Beta scaling factor for YaRN
            mscale: M-scale factor for interpolation
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.mscale = mscale
        
        # Build frequency cache
        self.register_buffer(
            "freqs",
            self._compute_freqs(max_position_embeddings),
            persistent=False
        )
    
    def _compute_freqs(self, seq_len):
        """
        Compute frequencies for YaRN scaling
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Frequencies tensor of shape (seq_len, dim // 2)
        """
        # Compute standard RoPE frequencies
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Apply YaRN scaling to position indices
        # YaRN uses a non-linear scaling: position^alpha * (1 + beta * position / max_len)
        positions = torch.arange(seq_len, dtype=torch.float)
        
        # Apply YaRN scaling
        if self.alpha != 1.0 or self.beta != 0.0:
            scaled_positions = positions ** self.alpha * (1 + self.beta * positions / seq_len)
        else:
            scaled_positions = positions
        
        # Apply m-scale for interpolation
        if self.mscale != 1.0:
            scaled_positions = scaled_positions * self.mscale
        
        # Compute frequencies
        freqs = torch.outer(scaled_positions, theta)
        
        return freqs
    
    def forward(self, x, seq_len=None):
        """
        Apply YaRN rotary embeddings
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            seq_len: Optional sequence length override
            
        Returns:
            Rotated tensor of same shape
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # Recompute frequencies if needed
        if seq_len > self.freqs.size(0):
            freqs = self._compute_freqs(seq_len).to(x.device)
        else:
            freqs = self.freqs[:seq_len].to(x.device)
        
        # Split into cos and sin
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        
        # Reshape freqs to match x shape: (seq_len, dim//2) -> (seq_len, 1, 1, dim//2)
        freqs_cos = freqs_cos.view(seq_len, 1, 1, self.dim // 2)
        freqs_sin = freqs_sin.view(seq_len, 1, 1, self.dim // 2)
        
        # Apply rotation
        # x shape: (batch_size, seq_len, num_heads, head_dim)
        # Split into even and odd dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Rotate
        x_rotated = torch.stack([
            x_even * freqs_cos - x_odd * freqs_sin,
            x_even * freqs_sin + x_odd * freqs_cos
        ], dim=-1)
        
        # Interleave back
        x_rotated = x_rotated.reshape(x.shape)
        
        return x_rotated


def apply_yarn_rotary_pos_emb(q, k, rope):
    """
    Apply YaRN rotary position embeddings to query and key tensors
    
    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        rope: YarnRotaryEmbedding module
        
    Returns:
        Rotated q and k tensors
    """
    q_rotated = rope(q)
    k_rotated = rope(k)
    return q_rotated, k_rotated
