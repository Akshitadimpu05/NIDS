#!/usr/bin/env python3
"""
Ablation Study: Traditional ML Baselines
Trains multiple traditional ML algorithms on CIC-Darknet2020 for comprehensive comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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

def train_traditional_ml_ablation():
    """Train multiple traditional ML baseline models for ablation study."""
    
    print("🎯 Ablation Study: Traditional ML Baselines")
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
    
    # Prepare scaled data for algorithms that need it
    print("🔄 Preparing scaled data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Define algorithms and their parameter grids
    algorithms = {
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.01],
                'max_depth': [3, 5, 7]
            },
            'use_scaled': False
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            },
            'use_scaled': False
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'use_scaled': True
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'use_scaled': True
        }
    }
    
    class_names = ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
    all_results = {}
    
    total_start_time = time.time()
    
    # Train each algorithm
    for algo_name, algo_config in algorithms.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training {algo_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Choose appropriate data
        if algo_config['use_scaled']:
            X_train_use = X_train_scaled
            X_val_use = X_val_scaled
            X_test_use = X_test_scaled
            print("   Using scaled features")
        else:
            X_train_use = X_train
            X_val_use = X_val
            X_test_use = X_test
            print("   Using original features")
        
        # Hyperparameter tuning
        print(f"🔍 Performing hyperparameter tuning for {algo_name}...")
        
        grid_search = GridSearchCV(
            algo_config['model'],
            algo_config['params'],
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Combine train and validation for hyperparameter tuning
        X_train_val = np.vstack([X_train_use, X_val_use])
        y_train_val = np.hstack([y_train, y_val])
        
        grid_search.fit(X_train_val, y_train_val)
        
        tuning_time = time.time() - start_time
        print(f"✅ Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        print(f"🏆 Best parameters: {grid_search.best_params_}")
        print(f"🏆 Best CV score: {grid_search.best_score_:.4f}")
        
        # Train final model
        print(f"🚀 Training final {algo_name} model...")
        best_model = grid_search.best_estimator_
        
        training_start = time.time()
        best_model.fit(X_train_use, y_train)
        training_time = time.time() - training_start
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Validation evaluation
        val_predictions = best_model.predict(X_val_use)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"✅ Validation Accuracy: {val_accuracy:.4f}")
        
        # Test evaluation
        print(f"📊 Evaluating {algo_name} on test set...")
        test_predictions = best_model.predict(X_test_use)
        
        # Get probabilities if available
        try:
            test_probabilities = best_model.predict_proba(X_test_use)
        except AttributeError:
            test_probabilities = None
            print(f"   Note: {algo_name} doesn't support probability prediction")
        
        # Calculate metrics
        test_acc = accuracy_score(y_test, test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_predictions, average='weighted')
        
        # ROC AUC (if probabilities available)
        roc_auc = 0.0
        if test_probabilities is not None:
            try:
                lb = LabelBinarizer()
                y_test_bin = lb.fit_transform(y_test)
                if y_test_bin.shape[1] == 1:  # Binary case
                    roc_auc = roc_auc_score(y_test, test_probabilities[:, 1])
                else:  # Multi-class case
                    roc_auc = roc_auc_score(y_test_bin, test_probabilities, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"Warning: Could not compute ROC AUC for {algo_name}: {e}")
                roc_auc = 0.0
        
        # Detailed classification report
        detailed_report = classification_report(y_test, test_predictions, 
                                              target_names=class_names, output_dict=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, test_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{algo_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/{algo_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        total_time = time.time() - start_time
        
        # Store results
        algo_results = {
            'experiment': f'{algo_name} Baseline Ablation',
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'algorithm': algo_name,
                'best_parameters': grid_search.best_params_,
                'n_features': X_train.shape[1],
                'n_classes': len(np.unique(labels)),
                'use_scaled_features': algo_config['use_scaled']
            },
            'training_config': {
                'hyperparameter_tuning': True,
                'cv_folds': 3,
                'grid_search_params': algo_config['params'],
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
            'training_time_seconds': {
                'hyperparameter_tuning': tuning_time,
                'final_training': training_time,
                'total': total_time
            }
        }
        
        all_results[algo_name] = algo_results
        
        # Save individual results
        results_file = f"{results_dir}/{algo_name.lower().replace(' ', '_')}_ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(algo_results, f, indent=2)
        
        # Save model
        model_file = f"{results_dir}/best_{algo_name.lower().replace(' ', '_')}_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Print results
        print(f"\n🎉 {algo_name.upper()} RESULTS:")
        print(f"📊 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"📈 Test F1-Score: {f1:.4f}")
        print(f"📈 ROC AUC: {roc_auc:.4f}")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"📁 Results saved to: {results_file}")
    
    total_time = time.time() - total_start_time
    
    # Generate comparison summary
    print(f"\n{'='*60}")
    print("📊 TRADITIONAL ML COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    comparison_data = []
    for algo_name, results in all_results.items():
        metrics = results['performance_metrics']
        comparison_data.append({
            'Algorithm': algo_name,
            'Test Accuracy': metrics['test_accuracy'],
            'Test F1-Score': metrics['test_f1_score'],
            'ROC AUC': metrics['roc_auc_score'],
            'Training Time (s)': results['training_time_seconds']['total']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
    
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Save comparison
    comparison_df.to_csv(f"{results_dir}/traditional_ml_comparison.csv", index=False)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Traditional ML Algorithms Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Test Accuracy', 'Test F1-Score', 'ROC AUC', 'Training Time (s)']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        bars = ax.bar(comparison_df['Algorithm'], comparison_df[metric], alpha=0.8)
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(comparison_df[metric]) * 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/traditional_ml_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comprehensive results
    comprehensive_results = {
        'experiment': 'Traditional ML Baselines Comprehensive Ablation',
        'timestamp': datetime.now().isoformat(),
        'total_algorithms': len(algorithms),
        'total_time_seconds': total_time,
        'best_algorithm': {
            'name': comparison_df.iloc[0]['Algorithm'],
            'accuracy': comparison_df.iloc[0]['Test Accuracy'],
            'f1_score': comparison_df.iloc[0]['Test F1-Score']
        },
        'algorithm_results': all_results,
        'comparison_summary': comparison_df.to_dict('records')
    }
    
    comprehensive_file = f"{results_dir}/traditional_ml_comprehensive_results.json"
    with open(comprehensive_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Save scaler
    scaler_file = f"{results_dir}/traditional_ml_scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save preprocessor
    preprocessor.save_preprocessor(f"{results_dir}/traditional_ml_preprocessor.pkl")
    
    print(f"\n🏆 Best Algorithm: {comprehensive_results['best_algorithm']['name']}")
    print(f"🏆 Best Accuracy: {comprehensive_results['best_algorithm']['accuracy']:.4f}")
    print(f"⏱️  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"📁 Comprehensive results: {comprehensive_file}")
    print(f"📊 Comparison chart: {results_dir}/traditional_ml_comparison.png")
    
    return comprehensive_results

if __name__ == "__main__":
    results = train_traditional_ml_ablation()
    print("\n✅ Traditional ML ablation study completed!")
