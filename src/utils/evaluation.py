"""
Comprehensive evaluation utilities for NIDS model performance assessment.
Includes metrics calculation, visualization, and reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Any
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class NIDSEvaluator:
    """Comprehensive evaluator for NIDS model performance."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names or ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
        self.results = {}
        
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Computing classification metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, 
                                           target_names=self.class_names,
                                           output_dict=True, zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        # ROC curves and AUC (if probabilities available)
        if y_pred_proba is not None:
            roc_results = self._compute_roc_metrics(y_true, y_pred_proba)
            results.update(roc_results)
        
        self.results['classification'] = results
        logger.info("Classification evaluation completed")
        
        return results
    
    def _compute_roc_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Compute ROC curves and AUC metrics."""
        n_classes = len(self.class_names)
        
        # Binarize labels for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        return {
            'roc_fpr': fpr,
            'roc_tpr': tpr,
            'roc_auc': roc_auc
        }
    
    def evaluate_anomaly_detection(self, y_true_anomaly: np.ndarray, 
                                 anomaly_scores: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate anomaly detection performance.
        
        Args:
            y_true_anomaly: True anomaly labels (0=normal, 1=anomaly)
            anomaly_scores: Anomaly scores from autoencoder
            
        Returns:
            Anomaly detection metrics
        """
        logger.info("Computing anomaly detection metrics...")
        
        # ROC curve for anomaly detection
        fpr, tpr, thresholds = roc_curve(y_true_anomaly, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true_anomaly, anomaly_scores)
        avg_precision = average_precision_score(y_true_anomaly, anomaly_scores)
        
        results = {
            'anomaly_roc_fpr': fpr,
            'anomaly_roc_tpr': tpr,
            'anomaly_roc_auc': roc_auc,
            'anomaly_precision': precision,
            'anomaly_recall': recall,
            'anomaly_avg_precision': avg_precision,
            'anomaly_thresholds': thresholds,
            'anomaly_pr_thresholds': pr_thresholds
        }
        
        self.results['anomaly_detection'] = results
        logger.info("Anomaly detection evaluation completed")
        
        return results
    
    def save_metrics_csv(self, filepath: str):
        """Save evaluation metrics to CSV file."""
        logger.info(f"Saving metrics to {filepath}")
        
        metrics_data = []
        
        if 'classification' in self.results:
            cls_results = self.results['classification']
            
            # Overall metrics
            metrics_data.append({
                'Metric': 'Accuracy',
                'Value': cls_results['accuracy'],
                'Type': 'Overall'
            })
            metrics_data.append({
                'Metric': 'Precision (Weighted)',
                'Value': cls_results['precision_weighted'],
                'Type': 'Overall'
            })
            metrics_data.append({
                'Metric': 'Recall (Weighted)',
                'Value': cls_results['recall_weighted'],
                'Type': 'Overall'
            })
            metrics_data.append({
                'Metric': 'F1-Score (Weighted)',
                'Value': cls_results['f1_weighted'],
                'Type': 'Overall'
            })
            
            # Per-class metrics
            for i, class_name in enumerate(self.class_names):
                if i < len(cls_results['precision_per_class']):
                    metrics_data.append({
                        'Metric': f'Precision_{class_name}',
                        'Value': cls_results['precision_per_class'][i],
                        'Type': 'Per-Class'
                    })
                    metrics_data.append({
                        'Metric': f'Recall_{class_name}',
                        'Value': cls_results['recall_per_class'][i],
                        'Type': 'Per-Class'
                    })
                    metrics_data.append({
                        'Metric': f'F1-Score_{class_name}',
                        'Value': cls_results['f1_per_class'][i],
                        'Type': 'Per-Class'
                    })
            
            # ROC AUC if available
            if 'roc_auc' in cls_results:
                for i, class_name in enumerate(self.class_names):
                    if i in cls_results['roc_auc']:
                        metrics_data.append({
                            'Metric': f'ROC_AUC_{class_name}',
                            'Value': cls_results['roc_auc'][i],
                            'Type': 'ROC-AUC'
                        })
                
                metrics_data.append({
                    'Metric': 'ROC_AUC_Micro',
                    'Value': cls_results['roc_auc']['micro'],
                    'Type': 'ROC-AUC'
                })
                metrics_data.append({
                    'Metric': 'ROC_AUC_Macro',
                    'Value': cls_results['roc_auc']['macro'],
                    'Type': 'ROC-AUC'
                })
        
        # Anomaly detection metrics
        if 'anomaly_detection' in self.results:
            anom_results = self.results['anomaly_detection']
            metrics_data.append({
                'Metric': 'Anomaly_ROC_AUC',
                'Value': anom_results['anomaly_roc_auc'],
                'Type': 'Anomaly Detection'
            })
            metrics_data.append({
                'Metric': 'Anomaly_Avg_Precision',
                'Value': anom_results['anomaly_avg_precision'],
                'Type': 'Anomaly Detection'
            })
        
        # Save to CSV
        df = pd.DataFrame(metrics_data)
        df.to_csv(filepath, index=False)
        logger.info(f"Metrics saved to {filepath}")
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot and save confusion matrix."""
        if 'classification' not in self.results:
            logger.warning("No classification results available for confusion matrix")
            return
        
        cm = self.results['classification']['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - NIDS Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, save_path: str = None):
        """Plot and save ROC curves."""
        if 'classification' not in self.results or 'roc_auc' not in self.results['classification']:
            logger.warning("No ROC data available for plotting")
            return
        
        cls_results = self.results['classification']
        fpr = cls_results['roc_fpr']
        tpr = cls_results['roc_tpr']
        roc_auc = cls_results['roc_auc']
        
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curve for each class
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green']
        for i, color in zip(range(len(self.class_names)), colors):
            if i in fpr:
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        # Plot micro and macro averages
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
                label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')
        plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=4,
                label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - NIDS Multi-Class Classification')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_dir: str):
        """Generate comprehensive evaluation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics CSV
        metrics_path = os.path.join(save_dir, f"nids_metrics_{timestamp}.csv")
        self.save_metrics_csv(metrics_path)
        
        # Save confusion matrix
        cm_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
        self.plot_confusion_matrix(cm_path)
        
        # Save ROC curves
        roc_path = os.path.join(save_dir, f"roc_curves_{timestamp}.png")
        self.plot_roc_curves(roc_path)
        
        logger.info(f"Comprehensive evaluation report saved to {save_dir}")
        
        return {
            'metrics_csv': metrics_path,
            'confusion_matrix': cm_path,
            'roc_curves': roc_path
        }
