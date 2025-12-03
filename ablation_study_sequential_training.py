#!/usr/bin/env python3
"""
Ablation Study: Sequential Training (AE + CapsNet)
Trains autoencoder first, then CapsNet separately (non-joint training).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
import json
import time
from datetime import datetime

# Add project paths
sys.path.append('/home/teja/Projects/NIDS')
sys.path.append('/home/teja/Projects/NIDS/src')

from src.models.autoencoder import TrafficAutoencoder
from src.models.capsnet import CapsuleNetwork, CapsNetLoss
from src.utils.data_preprocessing import DataPreprocessor

class SequentialNIDSModel(nn.Module):
    """Sequential training model: AE first, then CapsNet."""
    
    def __init__(self, input_dim=54, ae_config=None, capsnet_config=None):
        super(SequentialNIDSModel, self).__init__()
        
        # Initialize autoencoder
        self.autoencoder = TrafficAutoencoder(
            input_dim=input_dim,
            hidden_dims=ae_config.get('hidden_dims', [64, 32, 16, 8]),
            latent_dim=ae_config.get('latent_dim', 4),
            dropout_rate=ae_config.get('dropout_rate', 0.2)
        )
        
        # Initialize CapsNet
        self.capsnet = CapsuleNetwork(
            input_dim=input_dim,
            primary_caps_dim=capsnet_config.get('primary_caps_dim', 8),
            primary_caps_num=capsnet_config.get('primary_caps_num', 32),
            digit_caps_dim=capsnet_config.get('digit_caps_dim', 16),
            digit_caps_num=capsnet_config.get('digit_caps_num', 4),
            routing_iterations=capsnet_config.get('routing_iterations', 3)
        )
        
        # Loss functions
        self.ae_criterion = nn.MSELoss()
        self.capsnet_criterion = CapsNetLoss()
        
    def forward(self, x, targets=None):
        """Forward pass through both models."""
        # Autoencoder forward
        ae_decoded, ae_encoded = self.autoencoder(x)
        
        # CapsNet forward
        capsnet_pred, capsnet_caps, capsnet_recon = self.capsnet(x, targets)
        
        return {
            'ae_encoded': ae_encoded,
            'ae_decoded': ae_decoded,
            'capsnet_pred': capsnet_pred,
            'capsnet_caps': capsnet_caps,
            'capsnet_recon': capsnet_recon
        }

def train_sequential_ablation():
    """Train Sequential (AE + CapsNet) model for ablation study."""
    
    print("🎯 Ablation Study: Sequential Training (AE + CapsNet)")
    print("=" * 60)
    
    # Create results directory
    results_dir = "results/ablation_study"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Load CIC-Darknet2020 dataset
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
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Model configurations
    ae_config = {
        'hidden_dims': [64, 32, 16, 8],
        'latent_dim': 4,
        'dropout_rate': 0.2
    }
    
    capsnet_config = {
        'primary_caps_dim': 8,
        'primary_caps_num': 32,
        'digit_caps_dim': 16,
        'digit_caps_num': len(np.unique(labels)),
        'routing_iterations': 3
    }
    
    # Initialize model
    print("🤖 Initializing Sequential model...")
    model = SequentialNIDSModel(
        input_dim=X_train.shape[1],
        ae_config=ae_config,
        capsnet_config=capsnet_config
    )
    
    # Training history
    train_history = {
        'phase': [],
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    total_start_time = time.time()
    
    # ===== PHASE 1: Train Autoencoder Only =====
    print("\n" + "="*50)
    print("🔄 PHASE 1: Training Autoencoder Only")
    print("="*50)
    
    # Freeze CapsNet parameters
    for param in model.capsnet.parameters():
        param.requires_grad = False
    
    # Only optimize autoencoder parameters
    ae_optimizer = optim.Adam(model.autoencoder.parameters(), lr=0.001)
    ae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(ae_optimizer, mode='min', patience=5, factor=0.5)
    
    ae_epochs = 25
    best_ae_loss = float('inf')
    ae_patience = 8
    ae_patience_counter = 0
    
    ae_start_time = time.time()
    
    for epoch in range(ae_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            ae_optimizer.zero_grad()
            
            # Only forward through autoencoder
            ae_decoded, ae_encoded = model.autoencoder(batch_x)
            loss = model.ae_criterion(ae_decoded, batch_x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.autoencoder.parameters(), max_norm=1.0)
            ae_optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                ae_decoded, ae_encoded = model.autoencoder(batch_x)
                loss = model.ae_criterion(ae_decoded, batch_x)
                val_losses.append(loss.item())
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Update history
        train_history['phase'].append('AE')
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(0.0)  # No accuracy for AE phase
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(0.0)
        
        # Learning rate scheduling
        ae_scheduler.step(val_loss)
        
        # Print progress
        print(f"AE Epoch {epoch+1:2d}/{ae_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model and early stopping
        if val_loss < best_ae_loss:
            best_ae_loss = val_loss
            ae_patience_counter = 0
            torch.save(model.autoencoder.state_dict(), f'{results_dir}/best_ae_sequential.pth')
            print(f"✅ New best AE model saved! Val Loss: {val_loss:.4f}")
        else:
            ae_patience_counter += 1
            
        # Early stopping
        if ae_patience_counter >= ae_patience:
            print(f"🛑 AE early stopping triggered after {epoch+1} epochs")
            break
    
    ae_training_time = time.time() - ae_start_time
    print(f"\n🎯 Autoencoder training completed in {ae_training_time:.2f} seconds")
    
    # Load best autoencoder
    model.autoencoder.load_state_dict(torch.load(f'{results_dir}/best_ae_sequential.pth'))
    
    # ===== PHASE 2: Train CapsNet Only =====
    print("\n" + "="*50)
    print("🔄 PHASE 2: Training CapsNet Only")
    print("="*50)
    
    # Freeze autoencoder parameters
    for param in model.autoencoder.parameters():
        param.requires_grad = False
    
    # Unfreeze CapsNet parameters
    for param in model.capsnet.parameters():
        param.requires_grad = True
    
    # Only optimize CapsNet parameters
    caps_optimizer = optim.Adam(model.capsnet.parameters(), lr=0.001)
    caps_scheduler = optim.lr_scheduler.ReduceLROnPlateau(caps_optimizer, mode='min', patience=5, factor=0.5)
    
    caps_epochs = 25
    best_caps_acc = 0.0
    caps_patience = 8
    caps_patience_counter = 0
    
    caps_start_time = time.time()
    
    for epoch in range(caps_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            caps_optimizer.zero_grad()
            
            # Convert labels to one-hot encoding
            num_classes = len(np.unique(labels))
            y_onehot = torch.zeros(batch_y.size(0), num_classes)
            y_onehot.scatter_(1, batch_y.unsqueeze(1), 1.0)
            
            # Forward through CapsNet only
            capsnet_pred, capsnet_caps, capsnet_recon = model.capsnet(batch_x, y_onehot)
            loss, _ = model.capsnet_criterion(capsnet_pred, y_onehot, capsnet_recon, batch_x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.capsnet.parameters(), max_norm=1.0)
            caps_optimizer.step()
            
            train_losses.append(loss.item())
            
            # Accuracy
            _, predicted = torch.max(capsnet_pred, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                num_classes = len(np.unique(labels))
                y_onehot = torch.zeros(batch_y.size(0), num_classes)
                y_onehot.scatter_(1, batch_y.unsqueeze(1), 1.0)
                
                capsnet_pred, capsnet_caps, capsnet_recon = model.capsnet(batch_x, y_onehot)
                loss, _ = model.capsnet_criterion(capsnet_pred, y_onehot, capsnet_recon, batch_x)
                val_losses.append(loss.item())
                
                # Accuracy
                _, predicted = torch.max(capsnet_pred, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Update history
        train_history['phase'].append('CapsNet')
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        caps_scheduler.step(val_loss)
        
        # Print progress
        print(f"CapsNet Epoch {epoch+1:2d}/{caps_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model and early stopping
        if val_acc > best_caps_acc:
            best_caps_acc = val_acc
            caps_patience_counter = 0
            torch.save(model.state_dict(), f'{results_dir}/best_sequential_ablation.pth')
            print(f"✅ New best Sequential model saved! Val Acc: {val_acc:.4f}")
        else:
            caps_patience_counter += 1
            
        # Early stopping
        if caps_patience_counter >= caps_patience:
            print(f"🛑 CapsNet early stopping triggered after {epoch+1} epochs")
            break
    
    caps_training_time = time.time() - caps_start_time
    total_training_time = time.time() - total_start_time
    
    print(f"\n🎯 CapsNet training completed in {caps_training_time:.2f} seconds")
    print(f"🎯 Total sequential training time: {total_training_time:.2f} seconds")
    print(f"🎯 Best validation accuracy: {best_caps_acc:.4f}")
    
    # Load best model for testing
    model.load_state_dict(torch.load(f'{results_dir}/best_sequential_ablation.pth'))
    model.eval()
    
    # Test evaluation
    print("\n📊 Evaluating on test set...")
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs['capsnet_pred'], 1)
            
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(test_targets, test_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average='weighted')
    
    # ROC AUC (multi-class)
    try:
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(test_targets)
        if y_test_bin.shape[1] == 1:  # Binary case
            roc_auc = roc_auc_score(test_targets, test_predictions)
        else:  # Multi-class case
            roc_auc = roc_auc_score(y_test_bin, 
                                   LabelBinarizer().fit_transform(test_predictions), 
                                   multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    # Detailed classification report
    class_names = ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
    detailed_report = classification_report(test_targets, test_predictions, 
                                          target_names=class_names, output_dict=True)
    
    # Compile results
    results = {
        'experiment': 'Sequential Training (AE + CapsNet) Ablation',
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'architecture': 'Sequential: Autoencoder -> CapsNet',
            'input_dim': X_train.shape[1],
            'ae_config': ae_config,
            'capsnet_config': capsnet_config
        },
        'training_config': {
            'ae_epochs': epoch + 1 if 'epoch' in locals() else ae_epochs,
            'capsnet_epochs': epoch + 1,
            'total_epochs': len([x for x in train_history['phase'] if x == 'AE']) + len([x for x in train_history['phase'] if x == 'CapsNet']),
            'batch_size': 256,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau'
        },
        'dataset_info': {
            'total_samples': len(features),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_features': X_train.shape[1],
            'num_classes': len(np.unique(labels))
        },
        'performance_metrics': {
            'test_accuracy': float(test_acc),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1),
            'roc_auc_score': float(roc_auc),
            'best_val_accuracy': float(best_caps_acc)
        },
        'per_class_metrics': detailed_report,
        'training_history': train_history,
        'training_time_seconds': {
            'autoencoder': ae_training_time,
            'capsnet': caps_training_time,
            'total': total_training_time
        }
    }
    
    # Save results
    results_file = f"{results_dir}/sequential_ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history as CSV
    history_df = pd.DataFrame(train_history)
    history_df.to_csv(f"{results_dir}/sequential_ablation_history.csv", index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor(f"{results_dir}/sequential_ablation_preprocessor.pkl")
    
    # Print final results
    print("\n" + "="*60)
    print("🎉 SEQUENTIAL TRAINING ABLATION RESULTS")
    print("="*60)
    print(f"📊 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"📈 Test Precision: {precision:.4f}")
    print(f"📈 Test Recall: {recall:.4f}")
    print(f"📈 Test F1-Score: {f1:.4f}")
    print(f"📈 ROC AUC: {roc_auc:.4f}")
    print(f"⏱️  AE Training Time: {ae_training_time:.2f} seconds")
    print(f"⏱️  CapsNet Training Time: {caps_training_time:.2f} seconds")
    print(f"⏱️  Total Training Time: {total_training_time:.2f} seconds")
    print(f"📁 Results saved to: {results_file}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = train_sequential_ablation()
    print("\n✅ Sequential training ablation study completed!")
