"""
CALM: Curiosity-Driven Action-Sensitive Language Models
Mathematical Formulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class CuriosityModule(nn.Module):
    """
    Curiosity computation using information gain (KL divergence)
    
    Mathematical formulation:
    Curiosity(s_t, a_t, s_{t+1}) = D_KL[q(z_t|o_t, h_{t-1}) || p(z_t|h_{t-1})]
    
    Where:
    - q(z_t|o_t, h_{t-1}) is the posterior distribution over latent variables
    - p(z_t|h_{t-1}) is the prior distribution over latent variables
    - D_KL is the Kullback-Leibler divergence
    """
    
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Prior network: p(z_t|h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Mean and log_std
        )
        
        # Posterior network: q(z_t|o_t, h_{t-1})
        self.posterior_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Mean and log_std
        )
        
    def reparameterize(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling"""
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def compute_curiosity(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute curiosity as information gain (KL divergence)
        
        Args:
            state: Current state (batch_size, state_dim)
            action: Action taken (batch_size, action_dim)
            next_state: Next state (batch_size, state_dim)
            
        Returns:
            Curiosity score (batch_size,)
        """
        batch_size = state.shape[0]
        
        # Compute prior parameters
        prior_params = self.prior_net(state)
        prior_mean = prior_params[:, :self.latent_dim]
        prior_log_std = prior_params[:, self.latent_dim:]
        
        # Compute posterior parameters
        state_action = torch.cat([state, action], dim=-1)
        posterior_params = self.posterior_net(state_action)
        posterior_mean = posterior_params[:, :self.latent_dim]
        posterior_log_std = posterior_params[:, self.latent_dim:]
        
        # Sample from posterior
        z_sample = self.reparameterize(posterior_mean, posterior_log_std)
        
        # Compute KL divergence
        # KL(q || p) = 0.5 * (log_std_p - log_std_q + (std_q^2 + (mean_q - mean_p)^2) / std_p^2 - 1)
        prior_std = torch.exp(prior_log_std)
        posterior_std = torch.exp(posterior_log_std)
        
        kl_div = 0.5 * (
            prior_log_std - posterior_log_std +
            (posterior_std ** 2 + (posterior_mean - prior_mean) ** 2) / (prior_std ** 2 + 1e-8) - 1
        )
        
        # Sum over latent dimensions
        curiosity = kl_div.sum(dim=-1)
        
        return curiosity
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_curiosity(state, action, next_state)


class ForwardModel(nn.Module):
    """
    Forward model: Predict next state from current state and action
    
    Mathematical formulation:
    s_{t+1} = f(s_t, a_t) + ε
    
    Where f is the forward model and ε is noise
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state"""
        state_action = torch.cat([state, action], dim=-1)
        predicted_next_state = self.network(state_action)
        return predicted_next_state


class ActionLanguageAlignment(nn.Module):
    """
    Action-Language Alignment Module
    
    Mathematical formulation:
    L_align = ||f_lang(text) - f_action(action_sequence)||^2
    
    Where:
    - f_lang: Language encoder
    - f_action: Action encoder
    """
    
    def __init__(
        self,
        lang_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.lang_dim = lang_dim
        self.action_dim = action_dim
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Alignment projection
        self.align_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def encode_language(self, lang_repr: torch.Tensor) -> torch.Tensor:
        """Encode language representation"""
        return self.lang_encoder(lang_repr)
    
    def encode_action(self, action_repr: torch.Tensor) -> torch.Tensor:
        """Encode action representation"""
        return self.action_encoder(action_repr)
    
    def compute_alignment_loss(
        self,
        lang_repr: torch.Tensor,
        action_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment loss between language and action representations
        
        Args:
            lang_repr: Language representation (batch_size, lang_dim)
            action_repr: Action representation (batch_size, action_dim)
            
        Returns:
            Alignment loss (scalar)
        """
        lang_encoded = self.encode_language(lang_repr)
        action_encoded = self.encode_action(action_repr)
        
        # Project to common space
        lang_projected = self.align_proj(lang_encoded)
        
        # Compute MSE loss
        alignment_loss = F.mse_loss(lang_projected, action_encoded)
        
        return alignment_loss
    
    def forward(
        self,
        lang_repr: torch.Tensor,
        action_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        lang_encoded = self.encode_language(lang_repr)
        action_encoded = self.encode_action(action_repr)
        return lang_encoded, action_encoded


class CALMLoss(nn.Module):
    """
    Complete CALM loss function
    
    Mathematical formulation:
    L_total = L_lm + λ * L_align + γ * L_curiosity + δ * L_forward
    
    Where:
    - L_lm: Standard language modeling loss
    - L_align: Action-language alignment loss
    - L_curiosity: Curiosity-driven generation loss
    - L_forward: Forward model prediction loss
    - λ, γ, δ: Hyperparameters
    """
    
    def __init__(
        self,
        lambda_align: float = 1.0,
        gamma_curiosity: float = 0.1,
        delta_forward: float = 0.5
    ):
        super().__init__()
        self.lambda_align = lambda_align
        self.gamma_curiosity = gamma_curiosity
        self.delta_forward = delta_forward
        
    def forward(
        self,
        lm_loss: torch.Tensor,
        align_loss: torch.Tensor,
        curiosity_loss: torch.Tensor,
        forward_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total CALM loss
        
        Args:
            lm_loss: Language modeling loss
            align_loss: Action-language alignment loss
            curiosity_loss: Curiosity loss
            forward_loss: Forward model loss
            
        Returns:
            Total loss and loss components dictionary
        """
        total_loss = (
            lm_loss +
            self.lambda_align * align_loss +
            self.gamma_curiosity * curiosity_loss +
            self.delta_forward * forward_loss
        )
        
        loss_components = {
            'total': total_loss.item(),
            'lm': lm_loss.item(),
            'align': align_loss.item(),
            'curiosity': curiosity_loss.item(),
            'forward': forward_loss.item()
        }
        
        return total_loss, loss_components


class MotorEntropyLoss(nn.Module):
    """
    Motor entropy loss for encouraging diverse actions
    
    Mathematical formulation:
    L_entropy = -H(π(a_t|h_{t-1}))
    
    Where H is the entropy of the policy distribution
    """
    
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, action_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute motor entropy loss
        
        Args:
            action_logits: Policy logits (batch_size, action_dim)
            
        Returns:
            Entropy loss (scalar)
        """
        action_probs = F.softmax(action_logits, dim=-1)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        entropy_loss = -self.alpha * entropy.mean()
        
        return entropy_loss


class InformationGain(nn.Module):
    """
    Information gain computation for curiosity
    
    Mathematical formulation:
    IG(s_t, a_t) = H(s_t) - H(s_t|a_t)
    
    Where H is entropy
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        
        # State entropy estimator
        self.entropy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def estimate_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate entropy of state"""
        return self.entropy_net(state).squeeze(-1)
    
    def compute_information_gain(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute information gain from action
        
        Args:
            state: Current state (batch_size, state_dim)
            action: Action taken (batch_size, action_dim)
            
        Returns:
            Information gain (batch_size,)
        """
        state_entropy = self.estimate_entropy(state)
        
        # In a full implementation, we would compute conditional entropy
        # For now, use a simple approximation
        action_effect = action.norm(dim=-1)
        information_gain = state_entropy * action_effect
        
        return information_gain
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.compute_information_gain(state, action)
