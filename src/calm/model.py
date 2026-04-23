"""
CALM: Curiosity-Driven Action-Sensitive Language Models
Main Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from .maths import (
    CuriosityModule,
    ForwardModel,
    ActionLanguageAlignment,
    CALMLoss,
    MotorEntropyLoss,
    InformationGain
)


class CALM(nn.Module):
    """
    Curiosity-Driven Action-Sensitive Language Model
    
    Architecture:
    1. Language Encoder: Processes text input
    2. Action Encoder: Processes action sequences
    3. Alignment Module: Aligns language and action representations
    4. Curiosity Module: Computes curiosity-driven exploration signal
    5. Forward Model: Predicts next state from action
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        action_dim: int = 3,
        max_seq_len: int = 512,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        lambda_align: float = 1.0,
        gamma_curiosity: float = 0.1,
        delta_forward: float = 0.5
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        
        # Language encoder (transformer-based)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.language_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Action-language alignment
        self.alignment = ActionLanguageAlignment(
            lang_dim=embed_dim,
            action_dim=embed_dim,
            hidden_dim=hidden_dim
        )
        
        # Curiosity module
        self.curiosity = CuriosityModule(
            state_dim=embed_dim,
            action_dim=action_dim,
            latent_dim=latent_dim
        )
        
        # Forward model
        self.forward_model = ForwardModel(
            state_dim=embed_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        # Information gain
        self.info_gain = InformationGain(
            state_dim=embed_dim,
            hidden_dim=hidden_dim
        )
        
        # Loss function
        self.calm_loss = CALMLoss(
            lambda_align=lambda_align,
            gamma_curiosity=gamma_curiosity,
            delta_forward=delta_forward
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        # Action prediction head
        self.action_head = nn.Linear(embed_dim, action_dim)
        
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text input
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Encoded representation (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        
        # Encode with transformer
        if attention_mask is not None:
            # Convert attention mask to transformer format
            mask = (attention_mask == 0)
            encoded = self.language_encoder(embeddings, src_key_padding_mask=mask)
        else:
            encoded = self.language_encoder(embeddings)
        
        return encoded
    
    def encode_action_sequence(
        self,
        action_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode action sequence
        
        Args:
            action_sequence: Action sequence (batch_size, seq_len, action_dim)
            
        Returns:
            Encoded representation (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = action_sequence.shape
        
        # Encode each action
        action_sequence_flat = action_sequence.view(-1, self.action_dim)
        encoded_flat = self.action_encoder(action_sequence_flat)
        encoded = encoded_flat.view(batch_size, seq_len, self.embed_dim)
        
        return encoded
    
    def forward(
        self,
        input_ids: torch.Tensor,
        action_sequence: Optional[torch.Tensor] = None,
        next_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            action_sequence: Action sequence (batch_size, seq_len, action_dim)
            next_state: Next state representation (batch_size, embed_dim)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target labels for language modeling (batch_size, seq_len)
            
        Returns:
            Dictionary with outputs and losses
        """
        batch_size, seq_len = input_ids.shape
        
        # Encode text
        text_encoded = self.encode_text(input_ids, attention_mask)
        
        # Get pooled representation (mean over sequence)
        text_pooled = text_encoded.mean(dim=1)
        
        # Language modeling logits
        lm_logits = self.lm_head(text_encoded)
        
        outputs = {
            'text_encoded': text_encoded,
            'text_pooled': text_pooled,
            'lm_logits': lm_logits
        }
        
        # Compute language modeling loss
        lm_loss = torch.tensor(0.0, device=input_ids.device)
        if labels is not None:
            lm_loss = F.cross_entropy(
                lm_logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        # Action-related computations
        align_loss = torch.tensor(0.0, device=input_ids.device)
        curiosity_loss = torch.tensor(0.0, device=input_ids.device)
        forward_loss = torch.tensor(0.0, device=input_ids.device)
        
        if action_sequence is not None:
            # Encode action sequence
            action_encoded = self.encode_action_sequence(action_sequence)
            action_pooled = action_encoded.mean(dim=1)
            
            # Compute alignment loss
            align_loss = self.alignment.compute_alignment_loss(text_pooled, action_pooled)
            
            outputs['action_encoded'] = action_encoded
            outputs['action_pooled'] = action_pooled
            
            # Compute curiosity loss
            if next_state is not None:
                # Use last action in sequence
                last_action = action_sequence[:, -1, :]
                curiosity_loss = self.curiosity.compute_curiosity(
                    text_pooled,
                    last_action,
                    next_state
                )
                
                # Compute forward model loss
                predicted_next = self.forward_model(text_pooled, last_action)
                forward_loss = F.mse_loss(predicted_next, next_state)
                
                outputs['predicted_next'] = predicted_next
                outputs['curiosity_score'] = curiosity_loss
        
        # Compute total CALM loss
        total_loss, loss_components = self.calm_loss(lm_loss, align_loss, curiosity_loss, forward_loss)
        
        outputs['total_loss'] = total_loss
        outputs['lm_loss'] = lm_loss
        outputs['align_loss'] = align_loss
        outputs['curiosity_loss'] = curiosity_loss
        outputs['forward_loss'] = forward_loss
        outputs['loss_components'] = loss_components
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        curiosity_threshold: float = 0.5,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate text with curiosity-driven exploration
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            curiosity_threshold: Minimum curiosity for exploration
            attention_mask: Attention mask
            
        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Encode current sequence
                outputs = self.forward(current_ids, attention_mask=attention_mask)
                lm_logits = outputs['lm_logits']
                
                # Get logits for last position
                next_token_logits = lm_logits[:, -1, :] / temperature
                
                # Sample from distribution
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(batch_size, 1, device=device)
                    ], dim=1)
        
        return current_ids
    
    def predict_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict action from text
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask
            
        Returns:
            Predicted action (batch_size, action_dim)
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask=attention_mask)
            text_pooled = outputs['text_pooled']
            action = self.action_head(text_pooled)
            
        return action
    
    def compute_curiosity_score(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute curiosity score for state-action-next-state triplet
        
        Args:
            state: Current state (batch_size, embed_dim)
            action: Action (batch_size, action_dim)
            next_state: Next state (batch_size, embed_dim)
            
        Returns:
            Curiosity score (batch_size,)
        """
        return self.curiosity.compute_curiosity(state, action, next_state)


class CALMConfig:
    """Configuration for CALM model"""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        action_dim: int = 3,
        max_seq_len: int = 512,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        lambda_align: float = 1.0,
        gamma_curiosity: float = 0.1,
        delta_forward: float = 0.5
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lambda_align = lambda_align
        self.gamma_curiosity = gamma_curiosity
        self.delta_forward = delta_forward
    
    def create_model(self) -> CALM:
        """Create CALM model from config"""
        return CALM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            action_dim=self.action_dim,
            max_seq_len=self.max_seq_len,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            lambda_align=self.lambda_align,
            gamma_curiosity=self.gamma_curiosity,
            delta_forward=self.delta_forward
        )
