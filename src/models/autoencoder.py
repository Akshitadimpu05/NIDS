"""
Autoencoder for learning normal encrypted traffic behavior patterns.
Designed for lightweight deployment on fog nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TrafficAutoencoder(nn.Module):
    """
    Lightweight Autoencoder for learning normal traffic patterns.
    Uses reconstruction error as anomaly score.
    """
    
    def __init__(self, 
                 input_dim: int = 78,
                 hidden_dims: list = [64, 32, 16, 8],
                 latent_dim: int = 4,
                 dropout_rate: float = 0.2):
        """
        Initialize the Traffic Autoencoder.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Bottleneck dimension
            dropout_rate: Dropout rate for regularization
        """
        super(TrafficAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Bottleneck layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstruction, latent_representation)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def compute_reconstruction_error(self, x: torch.Tensor, 
                                   reduction: str = 'mean') -> torch.Tensor:
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            x: Input tensor
            reduction: How to reduce the error ('mean', 'sum', 'none')
            
        Returns:
            Reconstruction error tensor
        """
        reconstruction, _ = self.forward(x)
        error = F.mse_loss(reconstruction, x, reduction=reduction)
        return error
    
    def detect_anomalies(self, x: torch.Tensor, 
                        threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies based on reconstruction error.
        
        Args:
            x: Input tensor
            threshold: Anomaly detection threshold
            
        Returns:
            Tuple of (anomaly_scores, is_anomaly)
        """
        self.eval()
        with torch.no_grad():
            reconstruction_errors = self.compute_reconstruction_error(x, reduction='none')
            # Compute per-sample error
            anomaly_scores = torch.mean(reconstruction_errors, dim=1)
            is_anomaly = anomaly_scores > threshold
            
        return anomaly_scores, is_anomaly
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information for fog deployment."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage (in MB)
        param_size = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': param_size,
            'is_lightweight': param_size < 10  # Consider < 10MB as lightweight
        }


class TrafficAETrainer:
    """Trainer class for the Traffic Autoencoder."""
    
    def __init__(self, model: TrafficAutoencoder, 
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: TrafficAutoencoder instance
            learning_rate: Learning rate for optimization
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                x = batch_data[0].to(self.device)
            else:
                x = batch_data.to(self.device)
            
            self.optimizer.zero_grad()
            reconstruction, _ = self.model(x)
            loss = F.mse_loss(reconstruction, x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, dataloader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0].to(self.device)
                else:
                    x = batch_data.to(self.device)
                
                reconstruction, _ = self.model(x)
                loss = F.mse_loss(reconstruction, x)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        self.scheduler.step(avg_loss)
        return avg_loss
    
    def save_model(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'latent_dim': self.model.latent_dim,
                'dropout_rate': self.model.dropout_rate
            }
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {filepath}")
        return checkpoint.get('model_config', {})
