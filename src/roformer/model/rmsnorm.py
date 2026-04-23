"""
RMSNorm - Root Mean Square Layer Normalization
More stable and efficient alternative to LayerNorm
Used in LLaMA and other modern transformer architectures
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    RMSNorm is a simpler and more efficient alternative to LayerNorm.
    It normalizes by the root mean square of the activations without
    centering (subtracting mean), which makes it more stable and faster.
    
    Formula: output = input / sqrt(mean(input^2) + eps) * weight
    
    Args:
        dim: Dimension of the input
        eps: Small constant for numerical stability
        elementwise_affine: Whether to use learnable scale parameter
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RMS normalization
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Normalized tensor
        """
        # Compute RMS along the last dimension
        # RMS = sqrt(mean(x^2))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Normalized and scaled tensor
        """
        output = self._norm(x.float()).type_as(x)
        
        if self.weight is not None:
            return output * self.weight
        return output


class RMSNorm1d(nn.Module):
    """
    RMSNorm for 1D tensors (for use in other contexts)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 1D tensors
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Normalized and scaled tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
