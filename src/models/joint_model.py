"""
Joint NIDS Model combining Autoencoder and CapsNet.
Extracted from joint_training.py for modular use.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

try:
    from .autoencoder import TrafficAutoencoder
    from .capsnet import CapsuleNetwork, CapsNetLoss
except ImportError:
    from autoencoder import TrafficAutoencoder
    from capsnet import CapsuleNetwork, CapsNetLoss


class JointNIDSModel(nn.Module):
    """Joint Autoencoder + CapsNet model for encrypted traffic classification."""
    
    def __init__(self, input_dim, ae_config, capsnet_config):
        super(JointNIDSModel, self).__init__()
        
        # Initialize autoencoder
        self.autoencoder = TrafficAutoencoder(
            input_dim=input_dim,
            **{k: v for k, v in ae_config.items() if k in ['hidden_dims', 'latent_dim', 'dropout_rate']}
        )
        
        # Initialize CapsNet
        self.capsnet = CapsuleNetwork(
            input_dim=input_dim,
            **{k: v for k, v in capsnet_config.items() if k in ['primary_caps_dim', 'primary_caps_num', 'digit_caps_dim', 'digit_caps_num', 'routing_iterations']}
        )
        
        # Loss functions
        self.ae_criterion = nn.MSELoss()
        self.capsnet_criterion = CapsNetLoss()
        
        # Loss weights
        self.ae_weight = 0.3
        self.capsnet_weight = 0.7
        
    def forward(self, x, targets=None):
        """Joint forward pass."""
        # Autoencoder forward - returns (reconstruction, latent)
        ae_decoded, ae_encoded = self.autoencoder(x)
        
        # CapsNet forward (use encoded features as additional input)
        capsnet_pred, capsnet_caps, capsnet_recon = self.capsnet(x, targets)
        
        return {
            'ae_encoded': ae_encoded,
            'ae_decoded': ae_decoded,
            'capsnet_pred': capsnet_pred,
            'capsnet_caps': capsnet_caps,
            'capsnet_recon': capsnet_recon
        }
    
    def compute_loss(self, outputs, targets, inputs):
        """Compute joint loss."""
        # Autoencoder reconstruction loss
        ae_loss = self.ae_criterion(outputs['ae_decoded'], inputs)
        
        # CapsNet loss
        capsnet_loss = self.capsnet_criterion(
            outputs['capsnet_pred'], 
            outputs['capsnet_caps'], 
            targets, 
            outputs['capsnet_recon'], 
            inputs
        )
        
        # Combined loss
        total_loss = self.ae_weight * ae_loss + self.capsnet_weight * capsnet_loss
        
        return {
            'total_loss': total_loss,
            'ae_loss': ae_loss,
            'capsnet_loss': capsnet_loss
        }
    
    def predict(self, x):
        """Make predictions without computing loss."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            # Use CapsNet predictions as final output
            predictions = torch.argmax(outputs['capsnet_pred'], dim=1)
            return predictions
    
    def get_anomaly_score(self, x):
        """Get anomaly score from autoencoder reconstruction error."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            reconstruction_error = torch.mean((x - outputs['ae_decoded']) ** 2, dim=1)
            return reconstruction_error
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'ae_weight': self.ae_weight,
            'capsnet_weight': self.capsnet_weight
        }, path)
    
    def load_model(self, path: str, device='cpu'):
        """Load model state."""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.ae_weight = checkpoint.get('ae_weight', 0.3)
        self.capsnet_weight = checkpoint.get('capsnet_weight', 0.7)
        return self
