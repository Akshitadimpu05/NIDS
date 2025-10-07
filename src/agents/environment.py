"""
NIDS Environment for Reinforcement Learning.
Defines the state space, action space, and reward function for traffic decisions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
import logging
from enum import IntEnum

logger = logging.getLogger(__name__)


class TrafficAction(IntEnum):
    """Traffic control actions."""
    ALLOW = 0
    BLOCK = 1
    THROTTLE = 2


class NIDSEnvironment(gym.Env):
    """
    NIDS Environment for RL-based traffic control decisions.
    
    State: Combined features from Autoencoder and CapsNet
    Actions: Allow, Block, or Throttle traffic
    Rewards: Based on security effectiveness and network performance
    """
    
    def __init__(self,
                 state_dim: int = 12,
                 max_episode_steps: int = 1000,
                 security_weight: float = 0.7,
                 performance_weight: float = 0.3):
        """
        Initialize NIDS Environment.
        
        Args:
            state_dim: Dimension of state space (AE + CapsNet features)
            max_episode_steps: Maximum steps per episode
            security_weight: Weight for security in reward calculation
            performance_weight: Weight for performance in reward calculation
        """
        super(NIDSEnvironment, self).__init__()
        
        self.state_dim = state_dim
        self.max_episode_steps = max_episode_steps
        self.security_weight = security_weight
        self.performance_weight = performance_weight
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(TrafficAction))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.current_state = None
        self.traffic_buffer = []
        self.performance_metrics = {
            'throughput': 1.0,
            'latency': 0.0,
            'blocked_attacks': 0,
            'false_positives': 0,
            'total_flows': 0
        }
        
        # Reward tracking
        self.episode_rewards = []
        self.cumulative_reward = 0.0
        
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.traffic_buffer = []
        self.cumulative_reward = 0.0
        
        # Reset performance metrics
        self.performance_metrics = {
            'throughput': 1.0,
            'latency': 0.0,
            'blocked_attacks': 0,
            'false_positives': 0,
            'total_flows': 0
        }
        
        info = {
            'episode': len(self.episode_rewards),
            'performance_metrics': self.performance_metrics.copy()
        }
        
        return self.current_state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=Allow, 1=Block, 2=Throttle)
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Process action and compute reward
        reward = self._compute_reward(action)
        self.cumulative_reward += reward
        
        # Update performance metrics based on action
        self._update_performance_metrics(action)
        
        # Generate next state (in real deployment, this comes from traffic analysis)
        next_state = self._generate_next_state()
        self.current_state = next_state
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            'action_taken': TrafficAction(action).name,
            'step_reward': reward,
            'cumulative_reward': self.cumulative_reward,
            'performance_metrics': self.performance_metrics.copy(),
            'security_score': self._compute_security_score(),
            'performance_score': self._compute_performance_score()
        }
        
        if terminated or truncated:
            self.episode_rewards.append(self.cumulative_reward)
            info['episode_reward'] = self.cumulative_reward
            info['episode_length'] = self.current_step
        
        return next_state, reward, terminated, truncated, info
    
    def _compute_reward(self, action: int) -> float:
        """
        Compute reward based on action and current state.
        
        Args:
            action: Action taken
            
        Returns:
            Reward value
        """
        # Extract features from current state
        # Assume first part is AE reconstruction error, rest is CapsNet features
        ae_error = self.current_state[0] if len(self.current_state) > 0 else 0.0
        caps_features = self.current_state[1:] if len(self.current_state) > 1 else np.array([0.0])
        
        # Determine if traffic is likely malicious
        is_anomaly = ae_error > 0.1  # Threshold for anomaly detection
        attack_probability = np.mean(np.abs(caps_features))  # Simple heuristic
        
        # Base reward components
        security_reward = 0.0
        performance_reward = 0.0
        
        if action == TrafficAction.ALLOW:
            if is_anomaly or attack_probability > 0.5:
                # Allowed malicious traffic - negative security reward
                security_reward = -1.0
                self.performance_metrics['false_positives'] += 0  # No false positive
            else:
                # Allowed benign traffic - positive rewards
                security_reward = 0.1
                performance_reward = 1.0  # Good for throughput
                
        elif action == TrafficAction.BLOCK:
            if is_anomaly or attack_probability > 0.5:
                # Blocked malicious traffic - positive security reward
                security_reward = 1.0
                self.performance_metrics['blocked_attacks'] += 1
            else:
                # Blocked benign traffic - negative performance reward
                security_reward = -0.1
                performance_reward = -0.5  # False positive penalty
                self.performance_metrics['false_positives'] += 1
                
        elif action == TrafficAction.THROTTLE:
            if is_anomaly or attack_probability > 0.5:
                # Throttled malicious traffic - moderate security reward
                security_reward = 0.5
                performance_reward = 0.2  # Some throughput maintained
            else:
                # Throttled benign traffic - slight performance penalty
                security_reward = 0.0
                performance_reward = -0.2
        
        # Combine rewards with weights
        total_reward = (self.security_weight * security_reward + 
                       self.performance_weight * performance_reward)
        
        # Add bonus for maintaining good performance metrics
        if self.performance_metrics['total_flows'] > 0:
            false_positive_rate = (self.performance_metrics['false_positives'] / 
                                 self.performance_metrics['total_flows'])
            if false_positive_rate < 0.05:  # Less than 5% false positives
                total_reward += 0.1
        
        return total_reward
    
    def _update_performance_metrics(self, action: int):
        """Update performance metrics based on action."""
        self.performance_metrics['total_flows'] += 1
        
        if action == TrafficAction.ALLOW:
            self.performance_metrics['throughput'] = min(1.0, 
                self.performance_metrics['throughput'] + 0.01)
            self.performance_metrics['latency'] = max(0.0,
                self.performance_metrics['latency'] - 0.001)
                
        elif action == TrafficAction.BLOCK:
            self.performance_metrics['throughput'] = max(0.0,
                self.performance_metrics['throughput'] - 0.02)
            self.performance_metrics['latency'] += 0.001
            
        elif action == TrafficAction.THROTTLE:
            self.performance_metrics['throughput'] = max(0.0,
                self.performance_metrics['throughput'] - 0.005)
            self.performance_metrics['latency'] += 0.0005
    
    def _generate_next_state(self) -> np.ndarray:
        """Generate next state (simulated for training)."""
        # In real deployment, this would come from AE + CapsNet analysis
        # For simulation, generate realistic traffic patterns
        
        # Simulate AE reconstruction error (0 = normal, >0.1 = anomaly)
        if np.random.random() < 0.1:  # 10% anomaly rate
            ae_error = np.random.uniform(0.1, 1.0)
        else:
            ae_error = np.random.uniform(0.0, 0.05)
        
        # Simulate CapsNet features (normalized between -1 and 1)
        caps_features = np.random.normal(0, 0.3, self.state_dim - 1)
        caps_features = np.clip(caps_features, -1, 1)
        
        # Add some correlation with AE error for realism
        if ae_error > 0.1:
            caps_features += np.random.normal(0, 0.2, len(caps_features))
            caps_features = np.clip(caps_features, -1, 1)
        
        next_state = np.concatenate([[ae_error], caps_features])
        return next_state.astype(np.float32)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate early."""
        # Terminate if performance degrades too much
        if (self.performance_metrics['throughput'] < 0.1 or 
            self.performance_metrics['latency'] > 1.0):
            return True
        
        # Terminate if too many false positives
        if self.performance_metrics['total_flows'] > 100:
            false_positive_rate = (self.performance_metrics['false_positives'] / 
                                 self.performance_metrics['total_flows'])
            if false_positive_rate > 0.3:  # More than 30% false positives
                return True
        
        return False
    
    def _compute_security_score(self) -> float:
        """Compute current security effectiveness score."""
        if self.performance_metrics['total_flows'] == 0:
            return 1.0
        
        total_attacks = self.performance_metrics['blocked_attacks']
        false_positives = self.performance_metrics['false_positives']
        
        # Simple security score based on attack detection vs false positives
        if total_attacks + false_positives == 0:
            return 1.0
        
        security_score = total_attacks / (total_attacks + false_positives)
        return security_score
    
    def _compute_performance_score(self) -> float:
        """Compute current network performance score."""
        throughput_score = self.performance_metrics['throughput']
        latency_score = max(0, 1.0 - self.performance_metrics['latency'])
        
        return (throughput_score + latency_score) / 2.0
    
    def set_state(self, state: np.ndarray):
        """Set current state (for real-time deployment)."""
        self.current_state = state.astype(np.float32)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            'episode_rewards': self.episode_rewards.copy(),
            'current_performance': self.performance_metrics.copy(),
            'security_score': self._compute_security_score(),
            'performance_score': self._compute_performance_score(),
            'average_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'total_episodes': len(self.episode_rewards)
        }
