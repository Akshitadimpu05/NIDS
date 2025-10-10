#!/usr/bin/env python3
"""
Train Reinforcement Learning agent using the joint-trained NIDS model.
"""

import os
import sys
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Setup paths
sys.path.append('/home/tejasri/nids/NIDS')
sys.path.append('/home/tejasri/nids/NIDS/src')

from joint_training import JointNIDSModel
from src.agents.ppo_agent import PPOAgent
from src.utils.data_preprocessing import DataPreprocessor

class NIDSRLEnvironment:
    """RL Environment for NIDS decision making."""
    
    def __init__(self, joint_model, data_loader, device='cpu'):
        self.joint_model = joint_model
        self.data_loader = data_loader
        self.device = device
        self.data_iter = iter(data_loader)
        self.current_batch = None
        self.batch_idx = 0
        
        # Action space: 0=ALLOW, 1=BLOCK, 2=THROTTLE
        self.action_space = 3
        
        # State space: joint model features
        self.state_dim = self._get_state_dim()
        
    def _get_state_dim(self):
        """Get the dimension of the state space."""
        # Get a sample to determine state dimensions
        try:
            batch_x, batch_y = next(iter(self.data_loader))
            with torch.no_grad():
                outputs = self.joint_model(batch_x[:1])
                # Combine autoencoder latent + capsnet predictions
                state_dim = outputs['ae_encoded'].shape[1] + outputs['capsnet_pred'].shape[1]
            return state_dim
        except:
            return 8  # Default fallback
    
    def reset(self):
        """Reset environment and return initial state."""
        try:
            self.current_batch = next(self.data_iter)
            self.batch_idx = 0
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            self.current_batch = next(self.data_iter)
            self.batch_idx = 0
        
        return self._get_current_state()
    
    def _get_current_state(self):
        """Get current state from joint model."""
        if self.current_batch is None:
            return np.zeros(self.state_dim)
        
        batch_x, batch_y = self.current_batch
        sample_x = batch_x[self.batch_idx:self.batch_idx+1]
        
        with torch.no_grad():
            outputs = self.joint_model(sample_x)
            # Combine features: autoencoder latent + capsnet predictions
            ae_features = outputs['ae_encoded'].cpu().numpy().flatten()
            capsnet_features = outputs['capsnet_pred'].cpu().numpy().flatten()
            state = np.concatenate([ae_features, capsnet_features])
        
        return state
    
    def step(self, action):
        """Take action and return next state, reward, done."""
        if self.current_batch is None:
            return self.reset(), 0, True
        
        batch_x, batch_y = self.current_batch
        true_label = batch_y[self.batch_idx].item()
        
        # Calculate reward based on action and true label
        reward = self._calculate_reward(action, true_label)
        
        # Move to next sample
        self.batch_idx += 1
        done = self.batch_idx >= len(batch_x)
        
        if done:
            next_state = self.reset()
        else:
            next_state = self._get_current_state()
        
        return next_state, reward, done
    
    def _calculate_reward(self, action, true_label):
        """Calculate reward based on action and true label."""
        # Reward structure:
        # - Correct classification: +1
        # - Wrong classification: -1
        # - Conservative actions (BLOCK) for suspicious traffic: +0.5
        # - Risky actions (ALLOW) for malicious traffic: -2
        
        if true_label == 0:  # Non-Tor (normal traffic)
            if action == 0:  # ALLOW
                return 1.0
            elif action == 2:  # THROTTLE
                return 0.5
            else:  # BLOCK
                return -0.5
        else:  # Tor, VPN, NonVPN (potentially suspicious)
            if action == 1:  # BLOCK
                return 1.0
            elif action == 2:  # THROTTLE
                return 0.7
            else:  # ALLOW
                return -1.0

def train_rl_agent():
    """Train RL agent using joint model features."""
    print("🎯 RL Training with Joint NIDS Model")
    print("=" * 60)
    
    # Load configuration
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if joint model exists
    joint_model_path = 'data/models/joint_nids_model.pth'
    if not os.path.exists(joint_model_path):
        print("❌ Joint model not found! Please run joint_training.py first.")
        return False
    
    # Load and preprocess data
    print("📊 Loading data...")
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor('data/models/joint_preprocessor.pkl')
    
    features_df, labels_df = preprocessor.load_cic_darknet2020('data/Darknet.CSV')
    features = preprocessor.preprocess_features(features_df)
    labels = preprocessor.preprocess_labels(labels_df)
    
    # Create train split for RL
    splits = preprocessor.create_train_test_split(features, labels, test_size=0.2, validation_size=0.1)
    X_train = torch.FloatTensor(splits['X_train'])
    y_train = torch.LongTensor(splits['y_train'])
    
    # Create data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print(f"✅ Training data: {len(X_train)} samples")
    
    # Load joint model
    print("🤖 Loading joint model...")
    input_dim = features.shape[1]
    joint_model = JointNIDSModel(input_dim, config['autoencoder'], config['capsnet'])
    joint_model.load_state_dict(torch.load(joint_model_path, map_location='cpu'))
    joint_model.eval()  # Set to evaluation mode
    
    print("✅ Joint model loaded successfully")
    
    # Create RL environment
    print("🌍 Creating RL environment...")
    env = NIDSRLEnvironment(joint_model, train_loader)
    
    # Initialize RL agent
    print("🤖 Initializing RL agent...")
    rl_agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space,
        **{k: v for k, v in config['rl_agent'].items() if k in ['lr', 'gamma', 'eps_clip', 'k_epochs']}
    )
    
    print(f"✅ RL Agent initialized - State dim: {env.state_dim}, Action dim: {env.action_space}")
    
    # Training loop
    print("\n🚀 Starting RL training...")
    num_episodes = 1000
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from agent
            action, action_info = rl_agent.get_action(state, deterministic=False)
            
            # Take step in environment
            next_state, reward, done = env.step(action)
            
            # Store experience with required info
            rl_agent.store_experience(state, action, reward, next_state, done, action_info)
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        # Update agent
        if (episode + 1) % 10 == 0:  # Update every 10 episodes
            rl_agent.update()
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1:4d}/{num_episodes} | Avg Reward: {avg_reward:.4f}")
    
    # Save trained RL agent
    rl_model_path = 'data/models/joint_rl_agent.pth'
    rl_agent.save_model(rl_model_path)
    
    print(f"\n🎉 RL Training completed!")
    print(f"📁 RL Agent saved: {rl_model_path}")
    print(f"📊 Final average reward: {np.mean(episode_rewards[-100:]):.4f}")
    
    return True

if __name__ == "__main__":
    success = train_rl_agent()
    if success:
        print("\n✅ RL training completed successfully!")
        print("🚀 Ready for deployment with joint model + RL agent!")
    else:
        print("\n❌ RL training failed")
        sys.exit(1)
