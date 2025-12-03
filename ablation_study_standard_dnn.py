#!/usr/bin/env python3
"""
Ablation Study: Standard DNN Baseline Training and Evaluation
Trains a standard deep neural network as baseline for comparison.
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

from src.utils.data_preprocessing import DataPreprocessor

class StandardDNN(nn.Module):
    """Standard Deep Neural Network for baseline comparison."""
    
    def __init__(self, input_dim=54, hidden_dims=[128, 64, 32], num_classes=4, dropout_rate=0.2):
        super(StandardDNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

def train_standard_dnn_ablation():
    """Train Standard DNN baseline model for ablation study."""
    
    print("🎯 Ablation Study: Standard DNN Baseline Training")
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
    
    # Initialize Standard DNN model
    print("🤖 Initializing Standard DNN model...")
    model = StandardDNN(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        num_classes=len(np.unique(labels)),
        dropout_rate=0.2
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
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
        'val_acc': []
    }
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_losses.append(loss.item())
            
            # Accuracy
            _, predicted = torch.max(outputs, 1)
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
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
                
                # Accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Update history
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        
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
            torch.save(model.state_dict(), f'{results_dir}/best_standard_dnn_ablation.pth')
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
    model.load_state_dict(torch.load(f'{results_dir}/best_standard_dnn_ablation.pth'))
    model.eval()
    
    # Test evaluation
    print("\n📊 Evaluating on test set...")
    test_predictions = []
    test_targets = []
    test_probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
            test_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(test_targets, test_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average='weighted')
    
    # ROC AUC (multi-class)
    try:
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(test_targets)
        if y_test_bin.shape[1] == 1:  # Binary case
            roc_auc = roc_auc_score(test_targets, [prob[1] for prob in test_probabilities])
        else:  # Multi-class case
            roc_auc = roc_auc_score(y_test_bin, test_probabilities, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    # Detailed classification report
    class_names = ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
    detailed_report = classification_report(test_targets, test_predictions, 
                                          target_names=class_names, output_dict=True)
    
    # Compile results
    results = {
        'experiment': 'Standard DNN Baseline Ablation',
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'architecture': 'Standard Deep Neural Network',
            'input_dim': X_train.shape[1],
            'hidden_dims': [128, 64, 32],
            'num_classes': len(np.unique(labels)),
            'dropout_rate': 0.2
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
    results_file = f"{results_dir}/standard_dnn_ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history as CSV
    history_df = pd.DataFrame(train_history)
    history_df.to_csv(f"{results_dir}/standard_dnn_ablation_history.csv", index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor(f"{results_dir}/standard_dnn_ablation_preprocessor.pkl")
    
    # Print final results
    print("\n" + "="*60)
    print("🎉 STANDARD DNN ABLATION RESULTS")
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
    results = train_standard_dnn_ablation()
    print("\n✅ Standard DNN ablation study completed!")
