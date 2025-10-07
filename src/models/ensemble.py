"""
Hybrid NIDS Model combining Autoencoder, Capsule Network, and RL Agent.
This is the main model used by fog nodes for traffic analysis and decision making.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Any, Optional
import logging

from .autoencoder import TrafficAutoencoder
from .capsnet import CapsuleNetwork
from ..agents.ppo_agent import PPOAgent
from ..agents.environment import NIDSEnvironment, TrafficAction

logger = logging.getLogger(__name__)


class HybridNIDSModel:
    """
    Hybrid NIDS Model combining deep learning and reinforcement learning.
    
    Architecture:
    1. Autoencoder: Detects anomalies in traffic patterns
    2. CapsNet: Extracts hierarchical features for classification
    3. RL Agent: Makes final traffic control decisions
    """
    
    def __init__(self,
                 input_dim: int = 78,
                 ae_config: Dict[str, Any] = None,
                 capsnet_config: Dict[str, Any] = None,
                 rl_config: Dict[str, Any] = None,
                 device: str = 'cpu'):
        """
        Initialize Hybrid NIDS Model.
        
        Args:
            input_dim: Input feature dimension
            ae_config: Autoencoder configuration
            capsnet_config: CapsNet configuration
            rl_config: RL agent configuration
            device: Device to run on
        """
        self.input_dim = input_dim
        self.device = device
        
        # Default configurations
        self.ae_config = ae_config or {
            'hidden_dims': [64, 32, 16, 8],
            'latent_dim': 4,
            'dropout_rate': 0.2
        }
        
        self.capsnet_config = capsnet_config or {
            'primary_caps_dim': 8,
            'primary_caps_num': 32,
            'digit_caps_dim': 16,
            'digit_caps_num': 10,
            'routing_iterations': 3
        }
        
        self.rl_config = rl_config or {
            'state_dim': 12,  # AE features + CapsNet features
            'action_dim': 3,  # Allow, Block, Throttle
            'hidden_dims': [128, 64],
            'learning_rate': 3e-4
        }
        
        # Initialize models
        self._initialize_models()
        
        # Model states
        self.is_trained = {
            'autoencoder': False,
            'capsnet': False,
            'rl_agent': False
        }
        
        # Performance metrics
        self.metrics = {
            'total_flows': 0,
            'anomalies_detected': 0,
            'actions_taken': {'ALLOW': 0, 'BLOCK': 0, 'THROTTLE': 0},
            'false_positives': 0,
            'true_positives': 0,
            'processing_time': []
        }
        
    def _initialize_models(self):
        """Initialize all component models."""
        # Autoencoder
        self.autoencoder = TrafficAutoencoder(
            input_dim=self.input_dim,
            **self.ae_config
        ).to(self.device)
        
        # Capsule Network
        self.capsnet = CapsuleNetwork(
            input_dim=self.input_dim,
            **self.capsnet_config
        ).to(self.device)
        
        # RL Environment and Agent
        self.environment = NIDSEnvironment(
            state_dim=self.rl_config['state_dim']
        )
        
        self.rl_agent = PPOAgent(
            state_dim=self.rl_config['state_dim'],
            action_dim=self.rl_config['action_dim'],
            hidden_dims=self.rl_config['hidden_dims'],
            learning_rate=self.rl_config['learning_rate'],
            device=self.device
        )
        
        logger.info("Hybrid NIDS Model initialized successfully")
    
    def extract_features(self, traffic_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from traffic data using AE and CapsNet.
        
        Args:
            traffic_data: Input traffic features
            
        Returns:
            Dictionary containing extracted features
        """
        # Convert to tensor
        if isinstance(traffic_data, np.ndarray):
            x = torch.FloatTensor(traffic_data).to(self.device)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        else:
            x = traffic_data
        
        features = {}
        
        # Autoencoder features
        with torch.no_grad():
            self.autoencoder.eval()
            reconstruction, latent = self.autoencoder(x)
            
            # Compute reconstruction error
            recon_error = torch.mean((x - reconstruction) ** 2, dim=1)
            
            features['ae_reconstruction_error'] = recon_error.cpu().numpy()
            features['ae_latent'] = latent.cpu().numpy()
            features['ae_reconstruction'] = reconstruction.cpu().numpy()
        
        # CapsNet features
        with torch.no_grad():
            self.capsnet.eval()
            class_predictions, digit_caps, _ = self.capsnet(x)
            
            features['capsnet_predictions'] = class_predictions.cpu().numpy()
            features['capsnet_features'] = digit_caps.view(x.size(0), -1).cpu().numpy()
        
        return features
    
    def create_rl_state(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create RL state from extracted features.
        
        Args:
            features: Features from AE and CapsNet
            
        Returns:
            RL state vector
        """
        # Combine features for RL state
        ae_error = features['ae_reconstruction_error']
        capsnet_pred = features['capsnet_predictions']
        
        # Take first few CapsNet features to fit state dimension
        capsnet_features = features['capsnet_features']
        if capsnet_features.shape[1] > self.rl_config['state_dim'] - 1:
            capsnet_features = capsnet_features[:, :self.rl_config['state_dim'] - 1]
        
        # Pad if necessary
        if capsnet_features.shape[1] < self.rl_config['state_dim'] - 1:
            padding = np.zeros((capsnet_features.shape[0], 
                              self.rl_config['state_dim'] - 1 - capsnet_features.shape[1]))
            capsnet_features = np.concatenate([capsnet_features, padding], axis=1)
        
        # Combine AE error with CapsNet features
        rl_state = np.concatenate([ae_error.reshape(-1, 1), capsnet_features], axis=1)
        
        return rl_state
    
    def predict(self, traffic_data: np.ndarray, 
                deterministic: bool = False) -> Dict[str, Any]:
        """
        Make prediction on traffic data.
        
        Args:
            traffic_data: Input traffic features
            deterministic: Whether to use deterministic policy
            
        Returns:
            Prediction results including action and analysis
        """
        import time
        start_time = time.time()
        
        # Extract features
        features = self.extract_features(traffic_data)
        
        # Create RL state
        rl_state = self.create_rl_state(features)
        
        # Get action from RL agent
        if len(rl_state.shape) > 1:
            rl_state = rl_state[0]  # Take first sample if batch
        
        action, action_info = self.rl_agent.get_action(rl_state, deterministic)
        
        # Process results
        processing_time = time.time() - start_time
        self.metrics['processing_time'].append(processing_time)
        self.metrics['total_flows'] += 1
        
        # Update action counts
        action_name = TrafficAction(action).name
        self.metrics['actions_taken'][action_name] += 1
        
        # Determine if anomaly detected
        is_anomaly = features['ae_reconstruction_error'][0] > 0.1  # Threshold
        if is_anomaly:
            self.metrics['anomalies_detected'] += 1
        
        result = {
            'action': action,
            'action_name': action_name,
            'confidence': np.max(action_info['action_probs']),
            'action_probabilities': action_info['action_probs'],
            'is_anomaly': is_anomaly,
            'anomaly_score': features['ae_reconstruction_error'][0],
            'capsnet_predictions': features['capsnet_predictions'][0],
            'processing_time': processing_time,
            'features': features,
            'rl_state': rl_state
        }
        
        return result
    
    def train_autoencoder(self, train_loader, val_loader, epochs: int = 100):
        """Train the autoencoder component."""
        from .autoencoder import TrafficAETrainer
        
        trainer = TrafficAETrainer(
            model=self.autoencoder,
            learning_rate=0.001,
            device=self.device
        )
        
        logger.info("Training Autoencoder...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_component('autoencoder', f'data/models/best_autoencoder.pth')
            
            if epoch % 10 == 0:
                logger.info(f"AE Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        self.is_trained['autoencoder'] = True
        logger.info("Autoencoder training completed")
    
    def train_capsnet(self, train_loader, val_loader, epochs: int = 50):
        """Train the CapsNet component."""
        from .capsnet import CapsNetTrainer
        
        trainer = CapsNetTrainer(
            model=self.capsnet,
            learning_rate=0.001,
            device=self.device
        )
        
        logger.info("Training CapsNet...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_losses = trainer.train_epoch(train_loader)
            val_losses = trainer.validate(val_loader)
            
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                self.save_component('capsnet', f'data/models/best_capsnet.pth')
            
            if epoch % 10 == 0:
                logger.info(f"CapsNet Epoch {epoch}: Train Loss = {train_losses['total_loss']:.4f}, "
                          f"Val Loss = {val_losses['total_loss']:.4f}")
        
        self.is_trained['capsnet'] = True
        logger.info("CapsNet training completed")
    
    def train_rl_agent(self, num_episodes: int = 1000):
        """Train the RL agent using the environment."""
        logger.info("Training RL Agent...")
        
        for episode in range(num_episodes):
            state, _ = self.environment.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Get action
                action, action_info = self.rl_agent.get_action(state)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = self.environment.step(action)
                
                # Store experience
                self.rl_agent.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=terminated or truncated,
                    info=action_info
                )
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Update agent
            if len(self.rl_agent.buffer) >= self.rl_agent.batch_size:
                training_stats = self.rl_agent.update()
            
            # Track episode statistics
            self.rl_agent.episode_rewards.append(episode_reward)
            self.rl_agent.episode_lengths.append(episode_length)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.rl_agent.episode_rewards[-100:])
                logger.info(f"RL Episode {episode}: Avg Reward = {avg_reward:.4f}")
        
        self.is_trained['rl_agent'] = True
        self.save_component('rl_agent', 'data/models/best_rl_agent.pth')
        logger.info("RL Agent training completed")
    
    def save_component(self, component: str, filepath: str):
        """Save individual component."""
        if component == 'autoencoder':
            torch.save(self.autoencoder.state_dict(), filepath)
        elif component == 'capsnet':
            torch.save(self.capsnet.state_dict(), filepath)
        elif component == 'rl_agent':
            self.rl_agent.save_model(filepath)
        
        logger.info(f"Saved {component} to {filepath}")
    
    def load_component(self, component: str, filepath: str):
        """Load individual component."""
        if component == 'autoencoder':
            self.autoencoder.load_state_dict(torch.load(filepath, map_location=self.device))
            self.is_trained['autoencoder'] = True
        elif component == 'capsnet':
            self.capsnet.load_state_dict(torch.load(filepath, map_location=self.device))
            self.is_trained['capsnet'] = True
        elif component == 'rl_agent':
            self.rl_agent.load_model(filepath)
            self.is_trained['rl_agent'] = True
        
        logger.info(f"Loaded {component} from {filepath}")
    
    def save_model(self, filepath: str):
        """Save complete hybrid model."""
        checkpoint = {
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'capsnet_state_dict': self.capsnet.state_dict(),
            'rl_agent_checkpoint': {
                'network_state_dict': self.rl_agent.network.state_dict(),
                'optimizer_state_dict': self.rl_agent.optimizer.state_dict()
            },
            'config': {
                'input_dim': self.input_dim,
                'ae_config': self.ae_config,
                'capsnet_config': self.capsnet_config,
                'rl_config': self.rl_config
            },
            'is_trained': self.is_trained,
            'metrics': self.metrics
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Hybrid NIDS model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load complete hybrid model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.capsnet.load_state_dict(checkpoint['capsnet_state_dict'])
        self.rl_agent.network.load_state_dict(checkpoint['rl_agent_checkpoint']['network_state_dict'])
        self.rl_agent.optimizer.load_state_dict(checkpoint['rl_agent_checkpoint']['optimizer_state_dict'])
        
        self.is_trained = checkpoint.get('is_trained', self.is_trained)
        self.metrics = checkpoint.get('metrics', self.metrics)
        
        logger.info(f"Hybrid NIDS model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        ae_info = self.autoencoder.get_model_size()
        capsnet_info = self.capsnet.get_model_size()
        rl_info = self.rl_agent.get_model_size()
        
        total_size = ae_info['model_size_mb'] + capsnet_info['model_size_mb'] + rl_info['model_size_mb']
        total_params = ae_info['total_parameters'] + capsnet_info['total_parameters'] + rl_info['total_parameters']
        
        return {
            'components': {
                'autoencoder': ae_info,
                'capsnet': capsnet_info,
                'rl_agent': rl_info
            },
            'total_model_size_mb': total_size,
            'total_parameters': total_params,
            'is_lightweight': total_size < 50,  # Consider < 50MB as lightweight for full system
            'training_status': self.is_trained,
            'performance_metrics': self.metrics
        }
