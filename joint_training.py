#!/usr/bin/env python3
"""
Joint Training for NIDS: Autoencoder + CapsNet
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

# Setup paths
sys.path.append('/home/tejasri/nids/NIDS')
sys.path.append('/home/tejasri/nids/NIDS/src')

from src.models.autoencoder import TrafficAutoencoder
from src.models.capsnet import CapsuleNetwork, CapsNetLoss
from src.utils.data_preprocessing import DataPreprocessor

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
    
    def compute_loss(self, outputs, inputs, targets):
        """Compute joint loss."""
        # Autoencoder reconstruction loss
        ae_loss = self.ae_criterion(outputs['ae_decoded'], inputs)
        
        # Convert targets to one-hot for CapsNet loss
        num_classes = outputs['capsnet_pred'].size(1)
        targets_onehot = torch.zeros(targets.size(0), num_classes, device=inputs.device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # CapsNet loss
        capsnet_loss, _ = self.capsnet_criterion(
            outputs['capsnet_pred'], 
            targets_onehot, 
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

def train_joint_model():
    """Train the joint NIDS model."""
    print("🎯 Joint NIDS Training - Autoencoder + CapsNet")
    print("=" * 60)
    
    # Load configuration
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    features_df, labels_df = preprocessor.load_cic_darknet2020('data/Darknet.CSV')
    
    # Apply preprocessing
    features = preprocessor.preprocess_features(features_df)
    labels = preprocessor.preprocess_labels(labels_df)
    
    print(f"✅ Data shape: {features.shape}, Labels: {labels.shape}")
    print(f"✅ Classes: {np.unique(labels)}")
    
    # Create train/val/test splits
    splits = preprocessor.create_train_test_split(features, labels, test_size=0.2, validation_size=0.1)
    
    # Convert to tensors
    X_train = torch.FloatTensor(splits['X_train'])
    y_train = torch.LongTensor(splits['y_train'])
    X_val = torch.FloatTensor(splits['X_val'])
    y_val = torch.LongTensor(splits['y_val'])
    X_test = torch.FloatTensor(splits['X_test'])
    y_test = torch.LongTensor(splits['y_test'])
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize model
    print("\n🤖 Initializing joint model...")
    input_dim = features.shape[1]
    model = JointNIDSModel(input_dim, config['autoencoder'], config['capsnet'])
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    print("\n🚀 Starting joint training...")
    num_epochs = 30  # Reduced from 50
    best_val_acc = 0.0
    patience = 7  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x, batch_y)
            
            # Compute loss
            loss_dict = model.compute_loss(outputs, batch_x, batch_y)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_losses.append(loss.item())
            
            # Accuracy
            _, predicted = torch.max(outputs['capsnet_pred'], 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x, batch_y)
                loss_dict = model.compute_loss(outputs, batch_x, batch_y)
                val_losses.append(loss_dict['total_loss'].item())
                
                # Accuracy
                _, predicted = torch.max(outputs['capsnet_pred'], 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'data/models/joint_nids_model.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"   No improvement for {patience_counter} epochs")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            print(f"   Best validation accuracy: {best_val_acc:.4f}")
            break
    
    # Test evaluation
    print(f"\n🎯 Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model and test
    model.load_state_dict(torch.load('data/models/joint_nids_model.pth'))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x, batch_y)
            _, predicted = torch.max(outputs['capsnet_pred'], 1)
            
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    test_acc = test_correct / test_total
    
    print(f"\n🎉 FINAL TEST RESULTS:")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Detailed classification report
    class_names = ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
    report = classification_report(all_targets, all_predictions, target_names=class_names)
    print("\n📊 Classification Report:")
    print(report)
    
    # Save preprocessor and model info
    preprocessor.save_preprocessor('data/models/joint_preprocessor.pkl')
    
    print(f"\n✅ Joint training completed!")
    print(f"📁 Model saved: data/models/joint_nids_model.pth")
    print(f"📁 Preprocessor saved: data/models/joint_preprocessor.pkl")

if __name__ == "__main__":
    train_joint_model()
