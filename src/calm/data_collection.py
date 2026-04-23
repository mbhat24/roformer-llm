"""
CALM: Curiosity-Driven Action-Sensitive Language Models
Action Data Collection Pipeline
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle
from tqdm import tqdm


class ActionSequence:
    """Container for a single action sequence"""
    
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        curiosity_scores: np.ndarray,
        description: str = ""
    ):
        self.states = states  # (seq_len, state_dim)
        self.actions = actions  # (seq_len, action_dim)
        self.rewards = rewards  # (seq_len,)
        self.curiosity_scores = curiosity_scores  # (seq_len,)
        self.description = description
    
    def __len__(self):
        return len(self.states)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'curiosity_scores': self.curiosity_scores,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ActionSequence':
        """Create from dictionary"""
        return cls(
            states=data['states'],
            actions=data['actions'],
            rewards=data['rewards'],
            curiosity_scores=data['curiosity_scores'],
            description=data['description']
        )


class CuriosityAgent:
    """
    Curiosity-driven agent for action data collection
    
    Uses intrinsic curiosity reward to explore environment
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        curiosity_weight: float = 0.1,
        exploration_noise: float = 0.1
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.curiosity_weight = curiosity_weight
        self.exploration_noise = exploration_noise
        
        # Simple policy network (could be replaced with more sophisticated)
        import torch.nn as nn
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        # Forward model for curiosity computation
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        
        # Experience buffer
        self.experience_buffer = []
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using policy with exploration
        
        Args:
            state: Current state (state_dim,)
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action (action_dim,)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.policy(state_tensor).squeeze(0).numpy()
        
        # Add exploration noise
        if not deterministic:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
        
        return action
    
    def compute_curiosity(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """
        Compute curiosity as prediction error
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Curiosity score
        """
        state_action = np.concatenate([state, action])
        state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        with torch.no_grad():
            predicted_next = self.forward_model(state_action_tensor)
            prediction_error = torch.norm(predicted_next - next_state_tensor, dim=-1)
        
        return prediction_error.item()
    
    def collect_episode(
        self,
        env,
        max_steps: int = 200
    ) -> ActionSequence:
        """
        Collect a single episode of action data
        
        Args:
            env: Environment with step() method
            max_steps: Maximum steps per episode
            
        Returns:
            ActionSequence with collected data
        """
        states = []
        actions = []
        rewards = []
        curiosity_scores = []
        
        state = env.reset()
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state, deterministic=False)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Compute curiosity
            curiosity = self.compute_curiosity(state, action, next_state)
            
            # Store data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            curiosity_scores.append(curiosity)
            
            state = next_state
            
            if done:
                break
        
        return ActionSequence(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            curiosity_scores=np.array(curiosity_scores)
        )


class ActionDataCollector:
    """
    Collects action data using curiosity-driven exploration
    """
    
    def __init__(
        self,
        env,
        state_dim: int,
        action_dim: int,
        curiosity_weight: float = 0.1,
        save_dir: str = "data/calm"
    ):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.curiosity_weight = curiosity_weight
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent = CuriosityAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            curiosity_weight=curiosity_weight
        )
        
        self.sequences: List[ActionSequence] = []
        
    def collect(
        self,
        num_episodes: int = 1000,
        max_steps: int = 200,
        save_every: int = 100
    ) -> List[ActionSequence]:
        """
        Collect action data across multiple episodes
        
        Args:
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            save_every: Save checkpoint every N episodes
            
        Returns:
            List of collected action sequences
        """
        print(f"Collecting {num_episodes} episodes of action data...")
        
        for episode in tqdm(range(num_episodes), desc="Collecting"):
            sequence = self.agent.collect_episode(self.env, max_steps)
            self.sequences.append(sequence)
            
            # Save checkpoint
            if (episode + 1) % save_every == 0:
                self.save(f"checkpoint_ep{episode + 1}.pkl")
                print(f"Saved checkpoint at episode {episode + 1}")
        
        # Final save
        self.save("final.pkl")
        
        return self.sequences
    
    def save(self, filename: str):
        """Save collected sequences to file"""
        filepath = self.save_dir / filename
        
        data = [seq.to_dict() for seq in self.sequences]
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(self.sequences)} sequences to {filepath}")
    
    def load(self, filename: str) -> List[ActionSequence]:
        """Load sequences from file"""
        filepath = self.save_dir / filename
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.sequences = [ActionSequence.from_dict(d) for d in data]
        
        print(f"Loaded {len(self.sequences)} sequences from {filepath}")
        
        return self.sequences
    
    def get_statistics(self) -> Dict:
        """Get statistics about collected data"""
        if not self.sequences:
            return {}
        
        total_steps = sum(len(seq) for seq in self.sequences)
        avg_curiosity = np.mean([seq.curiosity_scores.mean() for seq in self.sequences])
        avg_reward = np.mean([seq.rewards.mean() for seq in self.sequences])
        
        return {
            'num_sequences': len(self.sequences),
            'total_steps': total_steps,
            'avg_steps_per_sequence': total_steps / len(self.sequences),
            'avg_curiosity': avg_curiosity,
            'avg_reward': avg_reward
        }


class LanguageActionDataset(torch.utils.data.Dataset):
    """
    Dataset for language-action alignment training
    
    Pairs language descriptions with action sequences
    """
    
    def __init__(
        self,
        action_sequences: List[ActionSequence],
        descriptions: List[str],
        tokenizer,
        max_seq_len: int = 512
    ):
        self.action_sequences = action_sequences
        self.descriptions = descriptions
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        assert len(action_sequences) == len(descriptions), \
            "Number of sequences and descriptions must match"
    
    def __len__(self):
        return len(self.action_sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training example
        
        Returns:
            Dictionary with input_ids, action_sequence, labels
        """
        sequence = self.action_sequences[idx]
        description = self.descriptions[idx]
        
        # Tokenize description
        encoded = self.tokenizer(
            description,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Pad or truncate action sequence to fixed length
        action_seq = sequence.actions
        seq_len = action_seq.shape[0]
        action_dim = action_seq.shape[1]
        
        if seq_len > self.max_seq_len:
            action_seq = action_seq[:self.max_seq_len]
        else:
            padding = np.zeros((self.max_seq_len - seq_len, action_dim))
            action_seq = np.vstack([action_seq, padding])
        
        # Pad or truncate curiosity scores
        curiosity_scores = sequence.curiosity_scores
        if len(curiosity_scores) > self.max_seq_len:
            curiosity_scores = curiosity_scores[:self.max_seq_len]
        else:
            curiosity_scores = np.pad(curiosity_scores, (0, self.max_seq_len - len(curiosity_scores)))
        
        # Create labels (shift input_ids for next token prediction)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last token
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'action_sequence': torch.FloatTensor(action_seq),
            'labels': labels,
            'curiosity_scores': torch.FloatTensor(curiosity_scores)
        }


class SimpleTokenizer:
    """Simple tokenizer for demonstration (replace with proper tokenizer)"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 0
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        words = text.lower().split()
        ids = []
        for word in words:
            if word not in self.word_to_id:
                if self.next_id < self.vocab_size:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
                else:
                    # Use unknown token if vocab is full
                    word = '<unk>'
            # Ensure ID is within vocab bounds
            word_id = self.word_to_id.get(word, 0)
            if word_id >= self.vocab_size:
                word_id = 0  # Use padding token as fallback
            ids.append(word_id)
        return ids
    
    def __call__(self, text: str, **kwargs):
        """Tokenizer interface"""
        ids = self.encode(text)
        max_length = kwargs.get('max_length', 512)
        padding = kwargs.get('padding', 'max_length')
        truncation = kwargs.get('truncation', True)
        
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        
        if padding == 'max_length':
            ids = ids + [0] * (max_length - len(ids))
        
        return {
            'input_ids': torch.tensor(ids).unsqueeze(0),
            'attention_mask': torch.ones(len(ids)).unsqueeze(0)
        }


def generate_descriptions(
    action_sequences: List[ActionSequence],
    template: str = "The agent performed {} actions with average curiosity {}"
) -> List[str]:
    """
    Generate language descriptions for action sequences
    
    Args:
        action_sequences: List of action sequences
        template: Template for descriptions
        
    Returns:
        List of descriptions
    """
    descriptions = []
    
    for seq in action_sequences:
        avg_curiosity = seq.curiosity_scores.mean()
        num_actions = len(seq)
        
        description = template.format(num_actions, f"{avg_curiosity:.2f}")
        descriptions.append(description)
    
    return descriptions
