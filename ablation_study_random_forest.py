#!/usr/bin/env python3
"""
Ablation Study: Random Forest Baseline
Trains Random Forest classifier on CIC-Darknet2020 for fair comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
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

def train_random_forest_ablation():
    """Train Random Forest baseline model for ablation study."""
    
    print("🎯 Ablation Study: Random Forest Baseline")
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
    
    # Hyperparameter tuning with GridSearchCV
    print("🔍 Performing hyperparameter tuning...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Use a smaller parameter grid for faster execution
    param_grid_small = {
        'n_estimators': [100, 200],
        'max_depth': [20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    start_time = time.time()
    
    # Initialize Random Forest
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Perform grid search on validation set
    print("🔍 Running GridSearchCV (this may take several minutes)...")
    grid_search = GridSearchCV(
        rf, 
        param_grid_small, 
        cv=3,  # 3-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Combine train and validation for hyperparameter tuning
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])
    
    grid_search.fit(X_train_val, y_train_val)
    
    tuning_time = time.time() - start_time
    print(f"✅ Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"🏆 Best parameters: {grid_search.best_params_}")
    print(f"🏆 Best CV score: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    print("🚀 Training final Random Forest model...")
    best_rf = grid_search.best_estimator_
    
    # Retrain on full training set
    training_start = time.time()
    best_rf.fit(X_train, y_train)
    training_time = time.time() - training_start
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Validation evaluation
    print("📊 Evaluating on validation set...")
    val_predictions = best_rf.predict(X_val)
    val_probabilities = best_rf.predict_proba(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    print(f"✅ Validation Accuracy: {val_accuracy:.4f}")
    
    # Test evaluation
    print("📊 Evaluating on test set...")
    test_predictions = best_rf.predict(X_test)
    test_probabilities = best_rf.predict_proba(X_test)
    
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
    
    # Feature importance analysis
    feature_importance = best_rf.feature_importances_
    feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
    
    # Get top 20 most important features
    top_features_idx = np.argsort(feature_importance)[-20:]
    top_features_importance = feature_importance[top_features_idx]
    top_features_names = [feature_names[i] for i in top_features_idx]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features_importance)), top_features_importance)
    plt.yticks(range(len(top_features_importance)), top_features_names)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances - Random Forest')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Random Forest - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/random_forest_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    total_time = time.time() - start_time
    
    # Compile results
    results = {
        'experiment': 'Random Forest Baseline Ablation',
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'algorithm': 'Random Forest',
            'best_parameters': grid_search.best_params_,
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(labels)),
            'class_weight': 'balanced',
            'random_state': 42
        },
        'training_config': {
            'hyperparameter_tuning': True,
            'cv_folds': 3,
            'grid_search_params': param_grid_small,
            'best_cv_score': float(grid_search.best_score_)
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
        'feature_importance': {
            'top_20_features': {
                'indices': top_features_idx.tolist(),
                'names': top_features_names,
                'importance_scores': top_features_importance.tolist()
            },
            'all_importance_scores': feature_importance.tolist()
        },
        'training_time_seconds': {
            'hyperparameter_tuning': tuning_time,
            'final_training': training_time,
            'total': total_time
        }
    }
    
    # Save results
    results_file = f"{results_dir}/random_forest_ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save trained model
    model_file = f"{results_dir}/best_random_forest_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(best_rf, f)
    
    # Save preprocessor
    preprocessor.save_preprocessor(f"{results_dir}/random_forest_ablation_preprocessor.pkl")
    
    # Save predictions for further analysis
    predictions_df = pd.DataFrame({
        'true_labels': y_test,
        'predicted_labels': test_predictions,
        'prediction_probabilities': [prob.tolist() for prob in test_probabilities]
    })
    predictions_df.to_csv(f"{results_dir}/random_forest_predictions.csv", index=False)
    
    # Print final results
    print("\n" + "="*60)
    print("🎉 RANDOM FOREST ABLATION RESULTS")
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
    print("="*60)
    
    # Print best parameters
    print("\n🏆 Best Hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    # Print top features
    print(f"\n🔍 Top 10 Most Important Features:")
    for i, (idx, importance) in enumerate(zip(top_features_idx[-10:], top_features_importance[-10:])):
        print(f"   {i+1:2d}. Feature {idx:2d}: {importance:.4f}")
    
    return results

if __name__ == "__main__":
    results = train_random_forest_ablation()
    print("\n✅ Random Forest ablation study completed!")
