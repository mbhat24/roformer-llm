"""
RoFormer-enhanced Language Model
A decoder-only transformer with rotary position embeddings
"""

import torch
import torch.nn as nn
import math
from attention import TransformerBlock


class RoFormerLLM(nn.Module):
    """
    RoFormer-enhanced Language Model (Decoder-only Transformer)
    
    This is a GPT-style autoregressive language model that uses
    RoFormer's rotary position embeddings instead of traditional
    absolute position embeddings.
    """
    
    def __init__(
        self,
        vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        ff_dim=3072,
        max_position_embeddings=2048,
        dropout=0.1,
        pad_token_id=0
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Model embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            ff_dim: Feed-forward hidden dimension (typically 4x embed_dim)
            max_position_embeddings: Maximum sequence length
            dropout: Dropout probability
            pad_token_id: Token ID for padding
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Note: We don't need learned position embeddings with RoPE!
        # The rotary position embeddings are applied in the attention mechanism
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                max_position_embeddings=max_position_embeddings,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
        
        # Tie input and output embeddings (weight tying)
        self.token_embedding.weight = self.lm_head.weight
    
    def create_causal_mask(self, seq_len, device):
        """
        Create causal (autoregressive) mask
        
        Args:
            seq_len: Sequence length
            device: Tensor device
        
        Returns:
            Mask tensor of shape (seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
                           1 for valid tokens, 0 for padding
        
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        x = self.dropout(x)
        
        # Create causal mask for autoregressive generation
        causal_mask = self.create_causal_mask(seq_len, input_ids.device)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len) -> (batch_size, 1, seq_len, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask * causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary logits
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        do_sample=True,
        pad_token_id=None
    ):
        """
        Generate text autoregressively
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling (if None, use full vocabulary)
            do_sample: If False, use greedy decoding
            pad_token_id: Token ID for padding
        
        Returns:
            Generated token IDs
        """
        self.eval()
        
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        batch_size = input_ids.shape[0]
        
        for _ in range(max_new_tokens):
            # Truncate to max_position_embeddings
            if input_ids.shape[1] > self.max_position_embeddings:
                input_ids = input_ids[:, -self.max_position_embeddings:]
            
            # Forward pass
            logits = self(input_ids)
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample or greedy decode
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


class RoFormerLLMConfig:
    """Configuration class for RoFormer LLM"""
    
    def __init__(
        self,
        vocab_size=50257,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        ff_dim=3072,
        max_position_embeddings=2048,
        dropout=0.1,
        pad_token_id=0
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.pad_token_id = pad_token_id
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'max_position_embeddings': self.max_position_embeddings,
            'dropout': self.dropout,
            'pad_token_id': self.pad_token_id
        }
