#!/usr/bin/env python3
"""
Ablation Study: Autoencoder-Only Training and Evaluation
Trains only the autoencoder component for anomaly detection and classification.
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
from src.utils.data_preprocessing import DataPreprocessor

class AutoencoderClassifier(nn.Module):
    """Autoencoder with classification head for ablation study."""
    
    def __init__(self, input_dim=54, hidden_dims=[64, 32, 16, 8], latent_dim=4, num_classes=4):
        super(AutoencoderClassifier, self).__init__()
        
        # Use the existing autoencoder
        self.autoencoder = TrafficAutoencoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout_rate=0.2
        )
        
        # Add classification head on top of latent representation
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Get reconstruction and latent representation
        reconstruction, latent = self.autoencoder(x)
        
        # Get classification from latent features
        classification = self.classifier(latent)
        
        return {
            'reconstruction': reconstruction,
            'latent': latent,
            'classification': classification
        }
    
    def get_anomaly_score(self, x):
        """Get anomaly score based on reconstruction error."""
        with torch.no_grad():
            outputs = self.forward(x)
            reconstruction_error = torch.mean((x - outputs['reconstruction']) ** 2, dim=1)
            return reconstruction_error

def train_autoencoder_ablation():
    """Train autoencoder-only model for ablation study."""
    
    print("🎯 Ablation Study: Autoencoder-Only Training")
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
    
    # Initialize model
    print("🤖 Initializing Autoencoder model...")
    model = AutoencoderClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32, 16, 8],
        latent_dim=4,
        num_classes=len(np.unique(labels))
    )
    
    # Loss functions and optimizer
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training parameters
    num_epochs = 50
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    # Training history
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'reconstruction_loss': [],
        'classification_loss': []
    }
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        recon_losses = []
        class_losses = []
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            
            # Compute losses
            recon_loss = reconstruction_criterion(outputs['reconstruction'], batch_x)
            class_loss = classification_criterion(outputs['classification'], batch_y)
            
            # Combined loss (weighted)
            total_loss = 0.3 * recon_loss + 0.7 * class_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_losses.append(total_loss.item())
            recon_losses.append(recon_loss.item())
            class_losses.append(class_loss.item())
            
            # Accuracy
            _, predicted = torch.max(outputs['classification'], 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                
                recon_loss = reconstruction_criterion(outputs['reconstruction'], batch_x)
                class_loss = classification_criterion(outputs['classification'], batch_y)
                total_loss = 0.3 * recon_loss + 0.7 * class_loss
                
                val_losses.append(total_loss.item())
                
                # Accuracy
                _, predicted = torch.max(outputs['classification'], 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        avg_recon_loss = np.mean(recon_losses)
        avg_class_loss = np.mean(class_losses)
        
        # Update history
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        train_history['reconstruction_loss'].append(avg_recon_loss)
        train_history['classification_loss'].append(avg_class_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Recon: {avg_recon_loss:.4f} | Class: {avg_class_loss:.4f}")
        
        # Save best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'{results_dir}/best_autoencoder_ablation.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"\n🎯 Training completed in {training_time:.2f} seconds")
    print(f"🎯 Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model for testing
    model.load_state_dict(torch.load(f'{results_dir}/best_autoencoder_ablation.pth'))
    model.eval()
    
    # Test evaluation
    print("\n📊 Evaluating on test set...")
    test_predictions = []
    test_targets = []
    test_anomaly_scores = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            
            # Classification predictions
            _, predicted = torch.max(outputs['classification'], 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
            
            # Anomaly scores
            anomaly_scores = model.get_anomaly_score(batch_x)
            test_anomaly_scores.extend(anomaly_scores.cpu().numpy())
    
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
        'experiment': 'Autoencoder-Only Ablation',
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'architecture': 'Autoencoder + Classification Head',
            'input_dim': X_train.shape[1],
            'hidden_dims': [64, 32, 16, 8],
            'latent_dim': 4,
            'num_classes': len(np.unique(labels)),
            'reconstruction_weight': 0.3,
            'classification_weight': 0.7
        },
        'training_config': {
            'epochs': epoch + 1,
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
            'best_val_accuracy': float(best_val_acc)
        },
        'per_class_metrics': detailed_report,
        'training_history': train_history,
        'training_time_seconds': training_time
    }
    
    # Save results
    results_file = f"{results_dir}/autoencoder_ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history as CSV
    history_df = pd.DataFrame(train_history)
    history_df.to_csv(f"{results_dir}/autoencoder_ablation_history.csv", index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor(f"{results_dir}/autoencoder_ablation_preprocessor.pkl")
    
    # Print final results
    print("\n" + "="*60)
    print("🎉 AUTOENCODER ABLATION RESULTS")
    print("="*60)
    print(f"📊 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"📈 Test Precision: {precision:.4f}")
    print(f"📈 Test Recall: {recall:.4f}")
    print(f"📈 Test F1-Score: {f1:.4f}")
    print(f"📈 ROC AUC: {roc_auc:.4f}")
    print(f"⏱️  Training Time: {training_time:.2f} seconds")
    print(f"📁 Results saved to: {results_file}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = train_autoencoder_ablation()
    print("\n✅ Autoencoder ablation study completed!")
