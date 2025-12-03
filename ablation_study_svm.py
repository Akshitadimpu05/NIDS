#!/usr/bin/env python3
"""
Ablation Study: SVM Baseline
Trains Support Vector Machine classifier on CIC-Darknet2020 for fair comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import GridSearchCV
import json
import time
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project paths
sys.path.append('/home/teja/Projects/NIDS')
sys.path.append('/home/teja/Projects/NIDS/src')

from src.utils.data_preprocessing import DataPreprocessor

def train_svm_ablation():
    """Train SVM baseline model for ablation study."""
    
    print("🎯 Ablation Study: SVM Baseline")
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
    
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_val = splits['X_val']
    y_val = splits['y_val']
    X_test = splits['X_test']
    y_test = splits['y_test']
    
    print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Additional scaling for SVM (important for SVM performance)
    print("🔄 Applying additional standardization for SVM...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning with GridSearchCV
    print("🔍 Performing hyperparameter tuning...")
    
    # SVM parameter grid (smaller for computational efficiency)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'poly', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
    
    # Use a smaller parameter grid for faster execution
    param_grid_small = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.01]
    }
    
    start_time = time.time()
    
    # Initialize SVM
    svm = SVC(
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        probability=True  # Enable probability estimates for ROC AUC
    )
    
    # Perform grid search on validation set
    print("🔍 Running GridSearchCV (this may take several minutes)...")
    grid_search = GridSearchCV(
        svm, 
        param_grid_small, 
        cv=3,  # 3-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Combine train and validation for hyperparameter tuning
    X_train_val_scaled = np.vstack([X_train_scaled, X_val_scaled])
    y_train_val = np.hstack([y_train, y_val])
    
    grid_search.fit(X_train_val_scaled, y_train_val)
    
    tuning_time = time.time() - start_time
    print(f"✅ Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"🏆 Best parameters: {grid_search.best_params_}")
    print(f"🏆 Best CV score: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    print("🚀 Training final SVM model...")
    best_svm = grid_search.best_estimator_
    
    # Retrain on full training set
    training_start = time.time()
    best_svm.fit(X_train_scaled, y_train)
    training_time = time.time() - training_start
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Validation evaluation
    print("📊 Evaluating on validation set...")
    val_predictions = best_svm.predict(X_val_scaled)
    val_probabilities = best_svm.predict_proba(X_val_scaled)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    print(f"✅ Validation Accuracy: {val_accuracy:.4f}")
    
    # Test evaluation
    print("📊 Evaluating on test set...")
    test_predictions = best_svm.predict(X_test_scaled)
    test_probabilities = best_svm.predict_proba(X_test_scaled)
    
    # Calculate comprehensive metrics
    test_acc = accuracy_score(y_test, test_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_predictions, average='weighted')
    
    # ROC AUC (multi-class)
    try:
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        if y_test_bin.shape[1] == 1:  # Binary case
            roc_auc = roc_auc_score(y_test, test_probabilities[:, 1])
        else:  # Multi-class case
            roc_auc = roc_auc_score(y_test_bin, test_probabilities, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"Warning: Could not compute ROC AUC: {e}")
        roc_auc = 0.0
    
    # Detailed classification report
    class_names = ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
    detailed_report = classification_report(y_test, test_predictions, 
                                          target_names=class_names, output_dict=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('SVM - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Support vectors analysis
    n_support_vectors = best_svm.n_support_
    support_vector_ratio = n_support_vectors.sum() / len(X_train_scaled)
    
    total_time = time.time() - start_time
    
    # Compile results
    results = {
        'experiment': 'SVM Baseline Ablation',
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'algorithm': 'Support Vector Machine',
            'best_parameters': grid_search.best_params_,
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(labels)),
            'class_weight': 'balanced',
            'probability': True,
            'random_state': 42
        },
        'training_config': {
            'hyperparameter_tuning': True,
            'cv_folds': 3,
            'grid_search_params': param_grid_small,
            'best_cv_score': float(grid_search.best_score_),
            'feature_scaling': 'StandardScaler'
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
            'best_val_accuracy': float(val_accuracy)
        },
        'per_class_metrics': detailed_report,
        'model_analysis': {
            'n_support_vectors': n_support_vectors.tolist(),
            'total_support_vectors': int(n_support_vectors.sum()),
            'support_vector_ratio': float(support_vector_ratio),
            'kernel': grid_search.best_params_['kernel'],
            'C_parameter': grid_search.best_params_['C'],
            'gamma_parameter': grid_search.best_params_['gamma']
        },
        'training_time_seconds': {
            'hyperparameter_tuning': tuning_time,
            'final_training': training_time,
            'total': total_time
        }
    }
    
    # Save results
    results_file = f"{results_dir}/svm_ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save trained model and scaler
    model_file = f"{results_dir}/best_svm_model.pkl"
    scaler_file = f"{results_dir}/svm_scaler.pkl"
    
    with open(model_file, 'wb') as f:
        pickle.dump(best_svm, f)
    
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save preprocessor
    preprocessor.save_preprocessor(f"{results_dir}/svm_ablation_preprocessor.pkl")
    
    # Save predictions for further analysis
    predictions_df = pd.DataFrame({
        'true_labels': y_test,
        'predicted_labels': test_predictions,
        'prediction_probabilities': [prob.tolist() for prob in test_probabilities]
    })
    predictions_df.to_csv(f"{results_dir}/svm_predictions.csv", index=False)
    
    # Print final results
    print("\n" + "="*60)
    print("🎉 SVM ABLATION RESULTS")
    print("="*60)
    print(f"📊 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"📈 Test Precision: {precision:.4f}")
    print(f"📈 Test Recall: {recall:.4f}")
    print(f"📈 Test F1-Score: {f1:.4f}")
    print(f"📈 ROC AUC: {roc_auc:.4f}")
    print(f"🏆 Best CV Score: {grid_search.best_score_:.4f}")
    print(f"⏱️  Hyperparameter Tuning: {tuning_time:.2f} seconds")
    print(f"⏱️  Final Training: {training_time:.2f} seconds")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"📁 Results saved to: {results_file}")
    print(f"🤖 Model saved to: {model_file}")
    print(f"📏 Scaler saved to: {scaler_file}")
    print("="*60)
    
    # Print best parameters
    print("\n🏆 Best Hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    # Print support vector analysis
    print(f"\n🔍 Support Vector Analysis:")
    print(f"   Total Support Vectors: {n_support_vectors.sum():,}")
    print(f"   Support Vector Ratio: {support_vector_ratio:.4f}")
    print(f"   Support Vectors per Class: {n_support_vectors}")
    
    return results

if __name__ == "__main__":
    results = train_svm_ablation()
    print("\n✅ SVM ablation study completed!")
