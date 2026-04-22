"""
Multi-Head Attention with Rotary Position Embeddings
"""

import torch
import torch.nn as nn
import math
from rope import RotaryPositionEmbedding


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with RoPE (Rotary Position Embedding)
    
    Implements the attention mechanism from the Transformer paper,
    enhanced with RoFormer's rotary position embeddings.
    """
    
    def __init__(self, embed_dim, num_heads, max_position_embeddings=2048, dropout=0.1):
        """
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            max_position_embeddings: Maximum sequence length for RoPE
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Rotary Position Embedding
        self.rope = RotaryPositionEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass of multi-head attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                  or (batch_size, 1, seq_len, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, embed_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, embed_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply Rotary Position Embedding to Q and K
        q = self.rope(q)  # (batch_size, seq_len, num_heads, head_dim)
        k = self.rope(k)  # (batch_size, seq_len, num_heads, head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # = (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # = (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        output = output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        output = output.contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    """
    
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        """
        Args:
            embed_dim: Input/output dimension
            ff_dim: Hidden dimension (typically 4x embed_dim)
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer Block with RoPE-enhanced Multi-Head Attention
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, max_position_embeddings=2048, dropout=0.1):
        """
        Args:
            embed_dim: Model dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            max_position_embeddings: Maximum sequence length for RoPE
            dropout: Dropout probability
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, max_position_embeddings, dropout
        )
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Pre-norm architecture (like GPT-2/3)
        # Attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x
