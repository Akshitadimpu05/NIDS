"""
Capsule Network for capturing hierarchical feature relationships 
in network traffic flow metadata.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def squash(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Squashing function for capsule networks.
    
    Args:
        tensor: Input tensor
        dim: Dimension to squash along
        
    Returns:
        Squashed tensor
    """
    squared_norm = torch.sum(tensor ** 2, dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = tensor / torch.sqrt(squared_norm + 1e-8)
    return scale * unit_vector


class PrimaryCapsule(nn.Module):
    """Primary Capsule Layer for initial feature extraction."""
    
    def __init__(self, input_dim: int, caps_dim: int, num_caps: int):
        """
        Initialize Primary Capsule layer.
        
        Args:
            input_dim: Input feature dimension
            caps_dim: Dimension of each capsule
            num_caps: Number of capsules
        """
        super(PrimaryCapsule, self).__init__()
        self.caps_dim = caps_dim
        self.num_caps = num_caps
        
        # Linear transformation to create capsules
        self.capsules = nn.Linear(input_dim, caps_dim * num_caps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through primary capsules.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Capsule outputs of shape (batch_size, num_caps, caps_dim)
        """
        batch_size = x.size(0)
        
        # Transform input
        caps_output = self.capsules(x)
        
        # Reshape to capsule format
        caps_output = caps_output.view(batch_size, self.num_caps, self.caps_dim)
        
        # Apply squashing
        return squash(caps_output, dim=-1)


class DigitCapsule(nn.Module):
    """Digit Capsule Layer for final classification."""
    
    def __init__(self, input_caps_dim: int, input_caps_num: int,
                 output_caps_dim: int, output_caps_num: int,
                 routing_iterations: int = 3):
        """
        Initialize Digit Capsule layer.
        
        Args:
            input_caps_dim: Input capsule dimension
            input_caps_num: Number of input capsules
            output_caps_dim: Output capsule dimension
            output_caps_num: Number of output capsules
            routing_iterations: Number of routing iterations
        """
        super(DigitCapsule, self).__init__()
        self.input_caps_dim = input_caps_dim
        self.input_caps_num = input_caps_num
        self.output_caps_dim = output_caps_dim
        self.output_caps_num = output_caps_num
        self.routing_iterations = routing_iterations
        
        # Weight matrix for transformation
        self.W = nn.Parameter(torch.randn(
            input_caps_num, output_caps_num, output_caps_dim, input_caps_dim
        ))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic routing.
        
        Args:
            x: Input capsules of shape (batch_size, input_caps_num, input_caps_dim)
            
        Returns:
            Output capsules of shape (batch_size, output_caps_num, output_caps_dim)
        """
        batch_size = x.size(0)
        
        # Expand input for matrix multiplication
        x_expanded = x.unsqueeze(2).unsqueeze(4)  # (batch, input_caps, 1, input_dim, 1)
        W_expanded = self.W.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        
        # Compute prediction vectors
        u_hat = torch.matmul(W_expanded, x_expanded).squeeze(-1)  # (batch, input_caps, output_caps, output_dim)
        
        # Initialize routing logits
        b = torch.zeros(batch_size, self.input_caps_num, self.output_caps_num, 
                       device=x.device, dtype=x.dtype)
        
        # Dynamic routing
        for iteration in range(self.routing_iterations):
            # Compute coupling coefficients
            c = F.softmax(b, dim=-1)  # (batch, input_caps, output_caps)
            
            # Compute weighted sum
            s = torch.sum(c.unsqueeze(-1) * u_hat, dim=1)  # (batch, output_caps, output_dim)
            
            # Apply squashing
            v = squash(s, dim=-1)
            
            # Update routing logits (except for last iteration)
            if iteration < self.routing_iterations - 1:
                # Compute agreement
                agreement = torch.sum(u_hat * v.unsqueeze(1), dim=-1)  # (batch, input_caps, output_caps)
                b = b + agreement
        
        return v


class CapsuleNetwork(nn.Module):
    """
    Capsule Network for hierarchical feature learning in network traffic.
    """
    
    def __init__(self,
                 input_dim: int = 54,
                 primary_caps_dim: int = 8,
                 primary_caps_num: int = 32,
                 digit_caps_dim: int = 16,
                 digit_caps_num: int = 10,
                 routing_iterations: int = 3):
        """
        Initialize Capsule Network.
        
        Args:
            input_dim: Input feature dimension
            primary_caps_dim: Primary capsule dimension
            primary_caps_num: Number of primary capsules
            digit_caps_dim: Digit capsule dimension
            digit_caps_num: Number of digit capsules (classes)
            routing_iterations: Number of routing iterations
        """
        super(CapsuleNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.primary_caps_dim = primary_caps_dim
        self.primary_caps_num = primary_caps_num
        self.digit_caps_dim = digit_caps_dim
        self.digit_caps_num = digit_caps_num
        self.routing_iterations = routing_iterations
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Primary capsules
        self.primary_caps = PrimaryCapsule(
            input_dim=128,
            caps_dim=primary_caps_dim,
            num_caps=primary_caps_num
        )
        
        # Digit capsules
        self.digit_caps = DigitCapsule(
            input_caps_dim=primary_caps_dim,
            input_caps_num=primary_caps_num,
            output_caps_dim=digit_caps_dim,
            output_caps_num=digit_caps_num,
            routing_iterations=routing_iterations
        )
        
        # Reconstruction network (for regularization)
        self.reconstruction_net = nn.Sequential(
            nn.Linear(digit_caps_dim * digit_caps_num, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, 
                target: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through CapsNet.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            target: Target labels for reconstruction masking
            
        Returns:
            Tuple of (class_predictions, digit_caps, reconstruction)
        """
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Primary capsules
        primary_caps = self.primary_caps(features)
        
        # Digit capsules
        digit_caps = self.digit_caps(primary_caps)
        
        # Compute class predictions (capsule lengths)
        class_predictions = torch.sqrt(torch.sum(digit_caps ** 2, dim=-1) + 1e-8)
        
        # Reconstruction
        if target is not None:
            # Convert target to one-hot mask for masking digit capsules
            mask = torch.zeros_like(class_predictions)
            mask.scatter_(1, target.unsqueeze(1), 1.0)
            masked_caps = digit_caps * mask.unsqueeze(-1)
            reconstruction_input = masked_caps.view(x.size(0), -1)
        else:
            # Use predicted class for masking
            _, max_indices = class_predictions.max(dim=1)
            mask = torch.zeros_like(class_predictions)
            mask.scatter_(1, max_indices.unsqueeze(1), 1.0)
            masked_caps = digit_caps * mask.unsqueeze(-1)
            reconstruction_input = masked_caps.view(x.size(0), -1)
        
        reconstruction = self.reconstruction_net(reconstruction_input)
        
        return class_predictions, digit_caps, reconstruction
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract hierarchical features for RL agent.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor for RL state representation
        """
        self.eval()
        with torch.no_grad():
            class_predictions, digit_caps, _ = self.forward(x)
            
            # Combine class predictions and capsule features
            caps_features = digit_caps.view(x.size(0), -1)  # Flatten capsules
            features = torch.cat([class_predictions, caps_features], dim=1)
            
        return features
    
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
            'is_lightweight': param_size < 20  # Consider < 20MB as lightweight for CapsNet
        }


class CapsNetLoss(nn.Module):
    """Custom loss function for Capsule Network."""
    
    def __init__(self, margin_loss_weight: float = 1.0,
                 reconstruction_loss_weight: float = 0.0005,
                 m_plus: float = 0.9,
                 m_minus: float = 0.1,
                 lambda_val: float = 0.5):
        """
        Initialize CapsNet loss.
        
        Args:
            margin_loss_weight: Weight for margin loss
            reconstruction_loss_weight: Weight for reconstruction loss
            m_plus: Margin for positive class
            m_minus: Margin for negative class
            lambda_val: Weight for negative class loss
        """
        super(CapsNetLoss, self).__init__()
        self.margin_loss_weight = margin_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val
        
    def forward(self, class_predictions: torch.Tensor,
                target: torch.Tensor,
                reconstruction: torch.Tensor,
                input_data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CapsNet loss.
        
        Args:
            class_predictions: Predicted class probabilities
            target: Target labels (one-hot)
            reconstruction: Reconstructed input
            input_data: Original input data
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        batch_size = class_predictions.size(0)
        
        # Margin loss
        left = F.relu(self.m_plus - class_predictions) ** 2
        right = F.relu(class_predictions - self.m_minus) ** 2
        
        margin_loss = target * left + self.lambda_val * (1 - target) * right
        margin_loss = margin_loss.sum(dim=1).mean()
        
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstruction, input_data)
        
        # Total loss
        total_loss = (self.margin_loss_weight * margin_loss + 
                     self.reconstruction_loss_weight * reconstruction_loss)
        
        loss_components = {
            'margin_loss': margin_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components


class CapsNetTrainer:
    """Trainer class for Capsule Network."""
    
    def __init__(self, model: CapsuleNetwork,
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: CapsuleNetwork instance
            learning_rate: Learning rate for optimization
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.criterion = CapsNetLoss()
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {'margin_loss': 0.0, 'reconstruction_loss': 0.0, 'total_loss': 0.0}
        num_batches = 0
        
        for batch_data in dataloader:
            x, y = batch_data[0].to(self.device), batch_data[1].to(self.device)
            
            self.optimizer.zero_grad()
            class_predictions, _, reconstruction = self.model(x, y)
            
            # Convert labels to one-hot encoding for loss calculation
            num_classes = class_predictions.size(1)
            y_one_hot = torch.zeros(y.size(0), num_classes, device=self.device)
            y_one_hot.scatter_(1, y.unsqueeze(1), 1.0)
            
            loss, loss_components = self.criterion(class_predictions, y_one_hot, reconstruction, x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for key in total_losses:
                total_losses[key] += loss_components[key]
            num_batches += 1
        
        # Average losses
        avg_losses = {key: val / num_batches for key, val in total_losses.items()}
        self.train_losses.append(avg_losses)
        return avg_losses
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_losses = {'margin_loss': 0.0, 'reconstruction_loss': 0.0, 'total_loss': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                x, y = batch_data[0].to(self.device), batch_data[1].to(self.device)
                
                class_predictions, _, reconstruction = self.model(x, y)
                
                # Convert labels to one-hot encoding for loss calculation
                num_classes = class_predictions.size(1)
                y_one_hot = torch.zeros(y.size(0), num_classes, device=self.device)
                y_one_hot.scatter_(1, y.unsqueeze(1), 1.0)
                
                loss, loss_components = self.criterion(class_predictions, y_one_hot, reconstruction, x)
                
                for key in total_losses:
                    total_losses[key] += loss_components[key]
                num_batches += 1
        
        # Average losses
        avg_losses = {key: val / num_batches for key, val in total_losses.items()}
        self.val_losses.append(avg_losses)
        self.scheduler.step(avg_losses['total_loss'])
        return avg_losses
