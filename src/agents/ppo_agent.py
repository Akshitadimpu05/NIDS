"""
Proximal Policy Optimization (PPO) Agent for NIDS traffic control.
Lightweight implementation suitable for fog computing deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import deque
import pickle

logger = logging.getLogger(__name__)


class PPONetwork(nn.Module):
    """Neural network for PPO agent with shared feature extraction."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Initialize PPO network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
        """
        super(PPONetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action_probabilities, state_value)
        """
        features = self.shared_net(state)
        action_probs = self.policy_head(features)
        state_value = self.value_head(features)
        
        return action_probs, state_value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get action from current policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, log_prob, state_value)
        """
        action_probs, state_value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)))
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, state_value


class PPOBuffer:
    """Experience buffer for PPO training."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize buffer with given capacity."""
        self.capacity = capacity
        self.clear()
    
    def clear(self):
        """Clear the buffer."""
        self.states = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.log_probs = deque(maxlen=self.capacity)
        self.values = deque(maxlen=self.capacity)
        self.dones = deque(maxlen=self.capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float,
            log_prob: torch.Tensor, value: torch.Tensor, done: bool):
        """Add experience to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.dones.append(done)
    
    def get_batch(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Get all experiences as tensors."""
        return {
            'states': torch.FloatTensor(list(self.states)).to(device),
            'actions': torch.LongTensor(list(self.actions)).to(device),
            'rewards': torch.FloatTensor(list(self.rewards)).to(device),
            'log_probs': torch.stack(list(self.log_probs)).to(device),
            'values': torch.stack(list(self.values)).squeeze().to(device),
            'dones': torch.BoolTensor(list(self.dones)).to(device)
        }
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization Agent for NIDS traffic control.
    Optimized for lightweight deployment on fog nodes.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [128, 64],
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 batch_size: int = 64,
                 buffer_capacity: int = 10000,
                 device: str = 'cpu'):
        """
        Initialize PPO Agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs
            batch_size: Batch size for training
            buffer_capacity: Experience buffer capacity
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Initialize network and optimizer
        self.network = PPONetwork(state_dim, action_dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = PPOBuffer(buffer_capacity)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'explained_variance': []
        }
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, Dict[str, Any]]:
        """
        Get action from current policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, info_dict)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)
        
        info = {
            'log_prob': log_prob,
            'value': value,
            'action_probs': None
        }
        
        # Get action probabilities for analysis
        with torch.no_grad():
            action_probs, _ = self.network(state_tensor)
            info['action_probs'] = action_probs.cpu().numpy().flatten()
        
        return action, info
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, info: Dict[str, Any]):
        """Store experience in buffer."""
        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            log_prob=info['log_prob'],
            value=info['value'],
            done=done
        )
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, next_value: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Reward tensor
            values: Value tensor
            dones: Done flags tensor
            next_value: Value of next state (for last step)
            
        Returns:
            Tuple of (advantages, returns)
        """
        if next_value is None:
            next_value = torch.zeros(1, device=self.device)
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """
        Update the agent using PPO algorithm.
        
        Returns:
            Dictionary of training statistics
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Get batch data
        batch = self.buffer.get_batch(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            batch['rewards'], batch['values'], batch['dones']
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store old log probs for KL divergence
        old_log_probs = batch['log_probs']
        
        # PPO update loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_div = 0
        
        for epoch in range(self.ppo_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(len(batch['states']))
            
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Mini-batch data
                states_mb = batch['states'][batch_indices]
                actions_mb = batch['actions'][batch_indices]
                old_log_probs_mb = old_log_probs[batch_indices]
                advantages_mb = advantages[batch_indices]
                returns_mb = returns[batch_indices]
                
                # Forward pass
                action_probs, values = self.network(states_mb)
                
                # Compute new log probabilities
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()
                
                # Compute ratio and surrogate losses
                ratio = torch.exp(new_log_probs - old_log_probs_mb)
                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_mb
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), returns_mb)
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_coef * value_loss - 
                             self.entropy_coef * entropy)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                
                # Compute KL divergence
                with torch.no_grad():
                    kl_div = (old_log_probs_mb - new_log_probs).mean()
                    total_kl_div += kl_div.item()
        
        # Compute explained variance
        explained_var = 1 - torch.var(returns - batch['values']) / torch.var(returns)
        
        # Update statistics
        num_updates = self.ppo_epochs * (len(batch['states']) // self.batch_size)
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'total_loss': (total_policy_loss + total_value_loss - total_entropy_loss) / num_updates,
            'kl_divergence': total_kl_div / num_updates,
            'explained_variance': explained_var.item()
        }
        
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        # Clear buffer
        self.buffer.clear()
        
        return stats
    
    def save_model(self, filepath: str):
        """Save agent model and training statistics."""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef
            }
        }
        torch.save(checkpoint, filepath)
        logger.info(f"PPO agent saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load agent model and training statistics."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        logger.info(f"PPO agent loaded from {filepath}")
        return checkpoint.get('config', {})
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information for fog deployment."""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        # Estimate memory usage (in MB)
        param_size = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': param_size,
            'is_lightweight': param_size < 5  # Consider < 5MB as lightweight for RL
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            'training_stats': self.training_stats.copy(),
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'average_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'total_episodes': len(self.episode_rewards)
        }
