#!/usr/bin/env python3
"""
Docker Model Adapter for Joint-Trained NIDS Model
Converts joint model format to HybridNIDSModel format for Docker deployment
"""

import os
import sys
import torch
import yaml
import logging
from pathlib import Path

# Setup paths
sys.path.append('/home/tejasri/nids/NIDS')
sys.path.append('/home/tejasri/nids/NIDS/src')

from joint_training import JointNIDSModel
from src.agents.ppo_agent import PPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DockerModelAdapter:
    """Adapter to convert joint model to Docker-compatible format."""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self):
        """Load model configuration."""
        with open('config/model_config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def convert_joint_model_to_docker_format(self):
        """Convert joint model to format expected by Docker fog nodes."""
        logger.info("🔄 Converting joint model to Docker-compatible format...")
        
        # Load joint model
        logger.info("Loading joint NIDS model...")
        joint_model = JointNIDSModel(54, self.config['autoencoder'], self.config['capsnet'])
        joint_model.load_state_dict(torch.load('data/models/joint_nids_model.pth', map_location='cpu'))
        joint_model.eval()
        
        # Load RL agent
        logger.info("Loading RL agent...")
        rl_agent = PPOAgent(
            state_dim=8,  # 4D autoencoder + 4D capsnet
            action_dim=3,
            **{k: v for k, v in self.config['rl_agent'].items() if k in ['lr', 'gamma', 'eps_clip', 'k_epochs']}
        )
        rl_agent.load_model('data/models/joint_rl_agent.pth')
        
        # Create Docker-compatible checkpoint
        docker_checkpoint = {
            'autoencoder_state_dict': joint_model.autoencoder.state_dict(),
            'capsnet_state_dict': joint_model.capsnet.state_dict(),
            'rl_agent_checkpoint': {
                'network_state_dict': rl_agent.network.state_dict(),
                'optimizer_state_dict': rl_agent.optimizer.state_dict()
            },
            'is_trained': True,
            'metrics': {
                'accuracy': 0.9483,
                'joint_training': True,
                'model_type': 'joint_nids'
            },
            'config': self.config
        }
        
        # Save Docker-compatible model
        docker_model_path = 'data/models/docker_hybrid_model.pth'
        torch.save(docker_checkpoint, docker_model_path)
        logger.info(f"✅ Docker-compatible model saved to {docker_model_path}")
        
        return docker_model_path
    
    def create_docker_model_config(self):
        """Create model configuration for Docker deployment."""
        docker_config = {
            'model': {
                'type': 'joint_nids',
                'input_dim': 54,
                'model_path': '/app/data/models/docker_hybrid_model.pth',
                'preprocessor_path': '/app/data/models/joint_preprocessor.pkl'
            },
            'autoencoder': self.config['autoencoder'],
            'capsnet': self.config['capsnet'],
            'rl_agent': self.config['rl_agent'],
            'fog_node': {
                'processing_threads': 4,
                'batch_size': 32,
                'max_queue_size': 10000,
                'enable_mitigation': True
            }
        }
        
        # Save Docker configuration
        docker_config_path = 'config/docker_config.yaml'
        with open(docker_config_path, 'w') as f:
            yaml.dump(docker_config, f, default_flow_style=False)
        
        logger.info(f"✅ Docker configuration saved to {docker_config_path}")
        return docker_config_path
    
    def update_fog_node_for_joint_model(self):
        """Update fog node to use joint model."""
        fog_node_patch = '''
# Add this to src/fog_node/fog_node.py after line 99

    def _load_joint_model(self):
        """Load joint-trained NIDS model."""
        try:
            from joint_training import JointNIDSModel
            
            # Load model configuration
            config_path = '/app/config/docker_config.yaml'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # Fallback configuration
                config = {
                    'autoencoder': {'hidden_dims': [32, 16], 'latent_dim': 4, 'dropout_rate': 0.2},
                    'capsnet': {'primary_caps_dim': 8, 'primary_caps_num': 32, 'digit_caps_dim': 16, 'digit_caps_num': 4, 'routing_iterations': 3}
                }
            
            # Initialize joint model
            self.joint_model = JointNIDSModel(54, config['autoencoder'], config['capsnet'])
            
            # Load trained weights
            model_path = '/app/data/models/docker_hybrid_model.pth'
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Load autoencoder and capsnet from joint model format
                joint_state = {
                    'autoencoder.encoder.0.weight': checkpoint['autoencoder_state_dict']['encoder.0.weight'],
                    'autoencoder.encoder.0.bias': checkpoint['autoencoder_state_dict']['encoder.0.bias'],
                    'autoencoder.encoder.2.weight': checkpoint['autoencoder_state_dict']['encoder.2.weight'],
                    'autoencoder.encoder.2.bias': checkpoint['autoencoder_state_dict']['encoder.2.bias'],
                    'autoencoder.encoder.4.weight': checkpoint['autoencoder_state_dict']['encoder.4.weight'],
                    'autoencoder.encoder.4.bias': checkpoint['autoencoder_state_dict']['encoder.4.bias'],
                    'autoencoder.decoder.0.weight': checkpoint['autoencoder_state_dict']['decoder.0.weight'],
                    'autoencoder.decoder.0.bias': checkpoint['autoencoder_state_dict']['decoder.0.bias'],
                    'autoencoder.decoder.2.weight': checkpoint['autoencoder_state_dict']['decoder.2.weight'],
                    'autoencoder.decoder.2.bias': checkpoint['autoencoder_state_dict']['decoder.2.bias'],
                    'autoencoder.decoder.4.weight': checkpoint['autoencoder_state_dict']['decoder.4.weight'],
                    'autoencoder.decoder.4.bias': checkpoint['autoencoder_state_dict']['decoder.4.bias'],
                }
                
                # Add CapsNet weights
                for key, value in checkpoint['capsnet_state_dict'].items():
                    joint_state[f'capsnet.{key}'] = value
                
                self.joint_model.load_state_dict(joint_state)
                self.joint_model.eval()
                
                logger.info("✅ Joint NIDS model loaded successfully")
                return True
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load joint model: {e}")
            return False
'''
        
        logger.info("📝 Fog node patch for joint model:")
        print(fog_node_patch)
        
        return fog_node_patch

def main():
    """Main function to prepare Docker deployment."""
    print("🐳 Docker Model Adapter for Joint-Trained NIDS")
    print("=" * 60)
    
    adapter = DockerModelAdapter()
    
    # Convert model format
    docker_model_path = adapter.convert_joint_model_to_docker_format()
    
    # Create Docker configuration
    docker_config_path = adapter.create_docker_model_config()
    
    # Show fog node update instructions
    adapter.update_fog_node_for_joint_model()
    
    print("\n✅ Docker preparation completed!")
    print(f"📁 Model: {docker_model_path}")
    print(f"⚙️  Config: {docker_config_path}")
    
    return True

if __name__ == "__main__":
    main()
