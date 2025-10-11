#!/usr/bin/env python3
"""
Comprehensive Research Evaluation for Joint NIDS Model
Generates detailed metrics, ROC curves, and confusion matrices for research paper.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from datetime import datetime
import yaml

# Setup paths
sys.path.append('/home/tejasri/nids/NIDS')
sys.path.append('/home/tejasri/nids/NIDS/src')

from joint_training import JointNIDSModel
from src.utils.data_preprocessing import DataPreprocessor

def create_research_evaluation():
    """Generate comprehensive evaluation for research paper."""
    
    print("🎯 Research Paper Evaluation - Joint NIDS Model")
    print("=" * 70)
    
    # Create results directory
    results_dir = "results/research_paper"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load configuration
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and preprocess data
    print("📊 Loading test data...")
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor('data/models/joint_preprocessor.pkl')
    
    features_df, labels_df = preprocessor.load_cic_darknet2020('data/Darknet.CSV')
    features = preprocessor.preprocess_features(features_df)
    labels = preprocessor.preprocess_labels(labels_df)
    
    # Create test split
    splits = preprocessor.create_train_test_split(features, labels, test_size=0.2, validation_size=0.1)
    X_test = splits['X_test']
    y_test = splits['y_test']
    
    print(f"✅ Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Load joint model
    print("🤖 Loading joint model...")
    joint_model = JointNIDSModel(X_test.shape[1], config['autoencoder'], config['capsnet'])
    joint_model.load_state_dict(torch.load('data/models/joint_nids_model.pth', map_location='cpu'))
    joint_model.eval()
    
    # Get predictions
    print("🔍 Generating predictions...")
    X_test_tensor = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        outputs = joint_model(X_test_tensor)
        y_pred_proba = torch.softmax(outputs['capsnet_pred'], dim=1).numpy()
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Class names
    class_names = ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
    
    # 1. Detailed Classification Report
    print("📋 Generating classification report...")
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Save detailed metrics to CSV
    metrics_data = []
    
    # Overall metrics
    for metric in ['accuracy']:
        metrics_data.append({
            'Metric': 'Overall Accuracy',
            'Value': report[metric],
            'Class': 'Overall'
        })
    
    # Per-class metrics
    for class_name in class_names:
        for metric in ['precision', 'recall', 'f1-score', 'support']:
            metrics_data.append({
                'Metric': metric.replace('-', '_').title(),
                'Value': report[class_name][metric],
                'Class': class_name
            })
    
    # Macro and weighted averages
    for avg_type in ['macro avg', 'weighted avg']:
        for metric in ['precision', 'recall', 'f1-score']:
            metrics_data.append({
                'Metric': f"{avg_type.replace(' ', '_').title()}_{metric.replace('-', '_').title()}",
                'Value': report[avg_type][metric],
                'Class': avg_type.title()
            })
    
    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv_path = f"{results_dir}/joint_nids_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✅ Metrics saved: {metrics_csv_path}")
    
    # 2. Confusion Matrix
    print("🎯 Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Joint NIDS Model\n(Autoencoder + CapsNet)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = f"{results_dir}/confusion_matrix_{timestamp}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    # 3. ROC Curves (Multi-class)
    print("📈 Generating ROC curves...")
    
    # Binarize labels for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    n_classes = y_test_bin.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot ROC curve for each class
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Joint NIDS Model\n(Multi-class Classification)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = f"{results_dir}/roc_curves_{timestamp}.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curves saved: {roc_path}")
    
    # 4. Precision-Recall Curves
    print("📊 Generating Precision-Recall curves...")
    
    plt.figure(figsize=(12, 8))
    
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
        avg_precision = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
        
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{class_names[i]} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Joint NIDS Model', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pr_path = f"{results_dir}/precision_recall_curves_{timestamp}.png"
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Precision-Recall curves saved: {pr_path}")
    
    # 5. Performance Comparison Table
    print("📋 Creating performance comparison...")
    
    # Calculate additional metrics
    overall_accuracy = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    
    comparison_data = {
        'Model': ['Separate Training (Baseline)', 'Joint Training (Proposed)'],
        'Accuracy': [0.2643, overall_accuracy],
        'Macro F1-Score': [0.1815, macro_f1],
        'Weighted F1-Score': [0.3001, weighted_f1],
        'Training Approach': ['Sequential', 'Joint'],
        'Feature Learning': ['Independent', 'Shared'],
        'Improvement': ['Baseline', f'+{((overall_accuracy - 0.2643) / 0.2643 * 100):.1f}%']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = f"{results_dir}/performance_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✅ Performance comparison saved: {comparison_path}")
    
    # 6. Research Summary Report
    print("📄 Generating research summary...")
    
    summary_report = f"""
# Joint NIDS Model - Research Evaluation Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Architecture
- **Approach**: Joint Training (Autoencoder + CapsNet)
- **Input Features**: 54 (after preprocessing)
- **Classes**: 4 (Non-Tor, NonVPN, Tor, VPN)
- **Dataset**: CIC-Darknet2020 (141,530 samples)

## Performance Metrics
- **Test Accuracy**: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)
- **Macro F1-Score**: {macro_f1:.4f}
- **Weighted F1-Score**: {weighted_f1:.4f}
- **Micro-average AUC**: {roc_auc["micro"]:.4f}

## Per-Class Performance
"""
    
    for class_name in class_names:
        class_metrics = report[class_name]
        summary_report += f"""
### {class_name}
- Precision: {class_metrics['precision']:.4f}
- Recall: {class_metrics['recall']:.4f}
- F1-Score: {class_metrics['f1-score']:.4f}
- Support: {int(class_metrics['support'])}
"""
    
    summary_report += f"""
## Improvement over Baseline
- **Accuracy Improvement**: +{((overall_accuracy - 0.2643) / 0.2643 * 100):.1f}%
- **Approach**: Joint training vs. separate training
- **Key Innovation**: Shared feature learning between autoencoder and CapsNet

## Files Generated
- Metrics: {metrics_csv_path}
- Confusion Matrix: {cm_path}
- ROC Curves: {roc_path}
- Precision-Recall Curves: {pr_path}
- Performance Comparison: {comparison_path}
"""
    
    summary_path = f"{results_dir}/research_summary_{timestamp}.md"
    with open(summary_path, 'w') as f:
        f.write(summary_report)
    
    print(f"✅ Research summary saved: {summary_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("🎉 RESEARCH EVALUATION COMPLETED!")
    print("="*70)
    print(f"📊 Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"📈 Improvement: +{((overall_accuracy - 0.2643) / 0.2643 * 100):.1f}% over baseline")
    print(f"📁 Results directory: {results_dir}")
    print("="*70)
    
    return {
        'accuracy': overall_accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'roc_auc_micro': roc_auc["micro"],
        'results_dir': results_dir
    }

if __name__ == "__main__":
    results = create_research_evaluation()
    print(f"\n✅ All research materials generated successfully!")
    print(f"🚀 Ready for research paper submission!")
