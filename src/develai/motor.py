"""
DevelAI Motor Control Networks
Neural networks for learning motor patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class MotorCortex(nn.Module):
    """
    Motor cortex: Generates motor commands from sensory input
    Mimics the brain's motor planning and execution
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 3,  # Number of arm segments
        num_layers: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Motor planning layers
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Motor memory (short-term motor memory)
        self.motor_memory = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, observation: torch.Tensor, prev_actions: torch.Tensor = None):
        """
        Generate motor commands from observation
        
        Args:
            observation: Current state observation
            prev_actions: Previous motor actions (for temporal consistency)
            
        Returns:
            Motor commands (torques)
        """
        # Generate base motor command
        motor_command = self.network(observation)
        
        # Apply motor memory if previous actions provided
        if prev_actions is not None:
            # Reshape for LSTM
            if prev_actions.dim() == 2:
                prev_actions = prev_actions.unsqueeze(1)
            
            lstm_out, _ = self.motor_memory(prev_actions)
            memory_bias = lstm_out[:, -1, :]
            
            # Combine with base command
            motor_command = motor_command + 0.1 * memory_bias[:, :self.output_dim]
        
        # Clamp to realistic torque limits
        motor_command = torch.tanh(motor_command) * 10.0
        
        return motor_command


class Cerebellum(nn.Module):
    """
    Cerebellum: Motor learning and refinement
    Learns to correct motor errors through supervised learning
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Predictive model: predict outcome of action
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # Predict next state
        )
        
        # Error correction network
        self.corrector = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # Current + predicted
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Correction to action
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Predict next state and compute action correction
        
        Args:
            state: Current state
            action: Proposed action
            
        Returns:
            predicted_next_state, action_correction
        """
        # Predict outcome
        state_action = torch.cat([state, action], dim=-1)
        predicted_next_state = self.predictor(state_action)
        
        # Compute correction (will be used when actual next state is known)
        return predicted_next_state
        
    def compute_correction(self, state: torch.Tensor, predicted_next: torch.Tensor, actual_next: torch.Tensor):
        """
        Compute action correction based on prediction error
        
        Args:
            state: Current state
            predicted_next: Predicted next state
            actual_next: Actual next state
            
        Returns:
            Action correction
        """
        error = actual_next - predicted_next
        state_error = torch.cat([state, error], dim=-1)
        correction = self.corrector(state_error)
        
        return correction


class BasalGanglia(nn.Module):
    """
    Basal Ganglia: Action selection and reinforcement learning
    Implements dopamine-like reward prediction
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Value function (critic)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Policy network (actor)
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor):
        """
        Get value estimate and action probabilities
        
        Args:
            state: Current state
            
        Returns:
            value, action_logits
        """
        value = self.value_network(state)
        action_logits = self.policy_network(state)
        
        return value.squeeze(-1), action_logits
        
    def select_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Select action using policy
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            action, log_prob
        """
        _, action_logits = self.forward(state)
        
        if deterministic:
            action = torch.tanh(action_logits)
            log_prob = torch.zeros_like(action[:, 0])
        else:
            # Sample from Gaussian policy
            mean = action_logits
            std = torch.ones_like(mean) * 0.5
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action, log_prob


class MotorLearningSystem:
    """
    Complete motor learning system
    Combines motor cortex, cerebellum, and basal ganglia
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Motor components
        self.motor_cortex = MotorCortex(state_dim, hidden_dim, action_dim)
        self.cerebellum = Cerebellum(state_dim, action_dim, hidden_dim // 2)
        self.basal_ganglia = BasalGanglia(state_dim, action_dim, hidden_dim // 2)
        
        # Optimizers
        self.motor_optimizer = torch.optim.Adam(self.motor_cortex.parameters(), lr=1e-4)
        self.cerebellum_optimizer = torch.optim.Adam(self.cerebellum.parameters(), lr=1e-4)
        self.basal_optimizer = torch.optim.Adam(self.basal_ganglia.parameters(), lr=1e-4)
        
        # Training mode
        self.training_mode = True
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Select action using motor system
        
        Args:
            state: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            action, info dict
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            # Get action from motor cortex
            motor_action = self.motor_cortex(state_tensor)
            
            # Get action from basal ganglia (policy)
            bg_action, log_prob = self.basal_ganglia.select_action(state_tensor, deterministic)
            
            # Combine actions (motor cortex provides base, BG provides exploration)
            if self.training_mode and not deterministic:
                action = 0.7 * motor_action + 0.3 * bg_action
            else:
                action = motor_action
                
        return action.squeeze(0).numpy(), {
            'motor_action': motor_action.squeeze(0).numpy(),
            'bg_action': bg_action.squeeze(0).numpy(),
            'log_prob': log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob
        }
        
    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Update motor system based on experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        if not self.training_mode:
            return
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])
        
        # Update cerebellum (predictive learning)
        predicted_next = self.cerebellum(state_tensor, action_tensor)
        cerebellum_loss = F.mse_loss(predicted_next, next_state_tensor)
        
        self.cerebellum_optimizer.zero_grad()
        cerebellum_loss.backward()
        self.cerebellum_optimizer.step()
        
        # Update basal ganglia (reinforcement learning)
        value, _ = self.basal_ganglia(state_tensor)
        next_value, _ = self.basal_ganglia(next_state_tensor)
        
        # TD learning
        td_target = reward_tensor + 0.99 * next_value * (1 - float(done))
        td_error = td_target - value
        
        value_loss = F.mse_loss(value, td_target.detach())
        
        self.basal_optimizer.zero_grad()
        value_loss.backward()
        self.basal_optimizer.step()
        
        # Update motor cortex (supervised by basal ganglia)
        motor_action = self.motor_cortex(state_tensor)
        _, bg_action = self.basal_ganglia(state_tensor)
        
        # Motor cortex learns to mimic successful basal ganglia actions
        motor_loss = F.mse_loss(motor_action, bg_action.detach())
        
        self.motor_optimizer.zero_grad()
        motor_loss.backward()
        self.motor_optimizer.step()
        
        return {
            'cerebellum_loss': cerebellum_loss.item(),
            'value_loss': value_loss.item(),
            'motor_loss': motor_loss.item(),
            'td_error': td_error.item()
        }
        
    def eval(self):
        """Set to evaluation mode"""
        self.training_mode = False
        self.motor_cortex.eval()
        self.cerebellum.eval()
        self.basal_ganglia.eval()
        
    def train(self):
        """Set to training mode"""
        self.training_mode = True
        self.motor_cortex.train()
        self.cerebellum.train()
        self.basal_ganglia.train()
