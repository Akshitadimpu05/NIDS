#!/usr/bin/env python3
"""
Training script for NIDS hybrid models.
Trains Autoencoder, CapsNet, and RL agent components.
"""

import os
import sys
import logging
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add project root and src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

from src.models.ensemble import HybridNIDSModel
from src.utils.data_preprocessing import DataPreprocessor
from src.agents.environment import NIDSEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log')
    ]
)

logger = logging.getLogger(__name__)


class NIDSTrainer:
    """
    Comprehensive trainer for NIDS hybrid model components.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to training configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.preprocessor = DataPreprocessor(
            normalization_method=self.config['data']['normalization'],
            handle_missing=self.config['data']['handle_missing']
        )
        
        # Model will be initialized after data loading to get actual feature count
        self.model = None
        
        # Training data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        logger.info("NIDS Trainer initialized")
    
    def _load_config(self) -> dict:
        """Load training configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def load_data(self, data_path: str):
        """
        Load and preprocess training data.
        
        Args:
            data_path: Path to dataset directory
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load CIC-Darknet2020 dataset
        features_df, labels_df = self.preprocessor.load_cic_darknet2020(data_path)
        
        # Preprocess features and labels
        features = self.preprocessor.preprocess_features(features_df, fit=True)
        labels = self.preprocessor.preprocess_labels(labels_df, fit=True)
        
        # Create train/validation/test splits
        splits = self.preprocessor.create_train_test_split(
            features, labels,
            test_size=self.config['training']['validation_split'],
            validation_size=0.1,
            random_state=42
        )
        
        # Convert to PyTorch datasets
        self.train_data = self._create_dataloader(
            splits['X_train'], splits['y_train'],
            batch_size=self.config['autoencoder']['batch_size'],
            shuffle=True
        )
        
        self.val_data = self._create_dataloader(
            splits['X_val'], splits['y_val'],
            batch_size=self.config['autoencoder']['batch_size'],
            shuffle=False
        )
        
        self.test_data = self._create_dataloader(
            splits['X_test'], splits['y_test'],
            batch_size=self.config['autoencoder']['batch_size'],
            shuffle=False
        )
        
        logger.info(f"Data loaded - Train: {len(splits['X_train'])}, "
                   f"Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
        
        # Initialize model with actual feature count
        self.model = HybridNIDSModel(
            input_dim=features.shape[1],
            ae_config=self.config['autoencoder'],
            capsnet_config=self.config['capsnet'],
            rl_config=self.config['rl_agent'],
            device=self.config['training']['device']
        )
        
        # Save preprocessor
        os.makedirs('data/models', exist_ok=True)
        self.preprocessor.save_preprocessor('data/models/preprocessor.pkl')
    
    def _create_dataloader(self, features: np.ndarray, labels: np.ndarray,
                          batch_size: int, shuffle: bool) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        features_tensor = torch.FloatTensor(features)
        labels_tensor = torch.LongTensor(labels)
        
        dataset = TensorDataset(features_tensor, labels_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=self.config['training']['num_workers'])
    
    def train_autoencoder(self):
        """Train the autoencoder component."""
        # Check if autoencoder is already trained
        autoencoder_path = 'data/models/best_autoencoder.pth'
        if os.path.exists(autoencoder_path):
            logger.info(f"Found existing autoencoder at {autoencoder_path}")
            try:
                self.model.load_component('autoencoder', autoencoder_path)
                self.model.is_trained['autoencoder'] = True
                logger.info("✅ Loaded existing autoencoder, skipping training")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing autoencoder: {e}")
                logger.info("Proceeding with fresh autoencoder training...")
        
        logger.info("Starting autoencoder training...")
        
        # Create anomaly detection data (normal traffic only for AE training)
        normal_data = self._create_normal_traffic_loader()
        normal_val_data = self._create_normal_validation_loader()
        
        # Train autoencoder
        self.model.train_autoencoder(
            train_loader=normal_data,
            val_loader=normal_val_data,
            epochs=self.config['autoencoder']['epochs']
        )
        
        logger.info("Autoencoder training completed")
    
    def train_capsnet(self):
        """Train the CapsNet component."""
        logger.info("Starting CapsNet training...")
        
        # Train CapsNet with full labeled data
        self.model.train_capsnet(
            train_loader=self.train_data,
            val_loader=self.val_data,
            epochs=self.config['capsnet']['epochs']
        )
        
        logger.info("CapsNet training completed")
    
    def train_rl_agent(self):
        """Train the RL agent component."""
        logger.info("Starting RL agent training...")
        
        # Train RL agent using environment simulation
        self.model.train_rl_agent(
            num_episodes=1000  # Can be configured
        )
        
        logger.info("RL agent training completed")
    
    def _create_normal_traffic_loader(self) -> DataLoader:
        """Create DataLoader with only normal traffic for autoencoder training."""
        # Extract normal traffic samples (assuming label 0 is normal)
        normal_samples = []
        
        for batch_features, batch_labels in self.train_data:
            normal_mask = batch_labels == 0
            if normal_mask.any():
                normal_samples.append(batch_features[normal_mask])
        
        if normal_samples:
            normal_features = torch.cat(normal_samples, dim=0)
            normal_labels = torch.zeros(len(normal_features), dtype=torch.long)
            
            dataset = TensorDataset(normal_features, normal_labels)
            return DataLoader(dataset, 
                            batch_size=self.config['autoencoder']['batch_size'],
                            shuffle=True,
                            num_workers=self.config['training']['num_workers'])
        else:
            # If no normal samples found, use all data
            logger.warning("No normal samples found, using all data for AE training")
            return self.train_data
    
    def _create_normal_validation_loader(self) -> DataLoader:
        """Create validation DataLoader with only normal traffic."""
        normal_samples = []
        
        for batch_features, batch_labels in self.val_data:
            normal_mask = batch_labels == 0
            if normal_mask.any():
                normal_samples.append(batch_features[normal_mask])
        
        if normal_samples:
            normal_features = torch.cat(normal_samples, dim=0)
            normal_labels = torch.zeros(len(normal_features), dtype=torch.long)
            
            dataset = TensorDataset(normal_features, normal_labels)
            return DataLoader(dataset,
                            batch_size=self.config['autoencoder']['batch_size'],
                            shuffle=False,
                            num_workers=self.config['training']['num_workers'])
        else:
            return self.val_data
    
    def evaluate_model(self):
        """Comprehensive model evaluation with detailed metrics."""
        try:
            from src.utils.evaluation import NIDSEvaluator
        except ImportError:
            # Fallback for different import contexts
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from src.utils.evaluation import NIDSEvaluator
        
        logger.info("Starting comprehensive model evaluation...")
        
        if self.test_data is None:
            logger.warning("No test data available for evaluation")
            return {}
        
        # Initialize evaluator
        evaluator = NIDSEvaluator(class_names=['Non-Tor', 'NonVPN', 'Tor', 'VPN'])
        
        # Collect predictions and ground truth
        y_true = []
        y_pred = []
        y_pred_proba = []
        anomaly_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_features, batch_labels in self.test_data:
                batch_features = batch_features.to(self.model.device)
                
                # Get model predictions
                predictions = self.model.predict(batch_features.cpu().numpy())
                
                # Collect results
                y_true.extend(batch_labels.cpu().numpy())
                y_pred.extend([pred['action'] for pred in predictions])
                
                # Get probabilities if available
                if 'action_probabilities' in predictions[0]:
                    y_pred_proba.extend([pred['action_probabilities'] for pred in predictions])
                
                # Get anomaly scores
                if 'anomaly_score' in predictions[0]:
                    anomaly_scores.extend([pred['anomaly_score'] for pred in predictions])
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Classification evaluation
        if len(y_pred_proba) > 0:
            y_pred_proba = np.array(y_pred_proba)
            cls_results = evaluator.evaluate_classification(y_true, y_pred, y_pred_proba)
        else:
            cls_results = evaluator.evaluate_classification(y_true, y_pred)
        
        # Anomaly detection evaluation (if we have anomaly scores)
        if len(anomaly_scores) > 0:
            # Convert labels to binary (assume Non-Tor is normal, others are anomalies)
            y_true_anomaly = (y_true != 0).astype(int)  # Non-Tor=0, others=1
            anomaly_scores = np.array(anomaly_scores)
            anom_results = evaluator.evaluate_anomaly_detection(y_true_anomaly, anomaly_scores)
        
        # Generate comprehensive report
        results_dir = "results/evaluation"
        report_files = evaluator.generate_report(results_dir)
        
        # Print summary metrics
        logger.info("=== EVALUATION RESULTS ===")
        logger.info(f"Accuracy: {cls_results['accuracy']:.4f}")
        logger.info(f"Precision (Weighted): {cls_results['precision_weighted']:.4f}")
        logger.info(f"Recall (Weighted): {cls_results['recall_weighted']:.4f}")
        logger.info(f"F1-Score (Weighted): {cls_results['f1_weighted']:.4f}")
        
        if 'roc_auc' in cls_results:
            logger.info(f"ROC AUC (Micro): {cls_results['roc_auc']['micro']:.4f}")
            logger.info(f"ROC AUC (Macro): {cls_results['roc_auc']['macro']:.4f}")
        
        if len(anomaly_scores) > 0:
            logger.info(f"Anomaly Detection AUC: {anom_results['anomaly_roc_auc']:.4f}")
        
        logger.info("=== REPORT FILES ===")
        for report_type, filepath in report_files.items():
            logger.info(f"{report_type}: {filepath}")
        
        return {
            'classification': cls_results,
            'anomaly_detection': anom_results if len(anomaly_scores) > 0 else None,
            'report_files': report_files
        }
    
    def save_model(self, model_path: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def train_complete_pipeline(self, data_path: str, save_path: str = None):
        """Train the complete NIDS pipeline."""
        logger.info("Starting complete NIDS training pipeline...")
        
        # Load and preprocess data
        self.load_data(data_path)
        
        # Train components sequentially
        self.train_autoencoder()
        self.train_capsnet()
        self.train_rl_agent()
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        # Save model
        if save_path:
            self.save_model(save_path)
        else:
            self.save_model('data/models/hybrid_nids_model.pth')
        
        logger.info("Complete training pipeline finished!")
        logger.info(f"Final metrics: {metrics}")
        
        return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train NIDS Hybrid Model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/Darknet.CSV',
                       help='Path to training data directory')
    parser.add_argument('--output', type=str, default='data/models/hybrid_nids_model.pth',
                       help='Output path for trained model')
    parser.add_argument('--component', type=str, choices=['autoencoder', 'capsnet', 'rl', 'all'],
                       default='all', help='Component to train')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting NIDS model training...")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data path: {args.data}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Component: {args.component}")
    
    try:
        # Initialize trainer
        trainer = NIDSTrainer(args.config)
        
        if args.component == 'all':
            # Train complete pipeline
            metrics = trainer.train_complete_pipeline(args.data, args.output)
            logger.info(f"Training completed successfully! Final metrics: {metrics}")
        else:
            # Load data first
            trainer.load_data(args.data)
            
            # Train specific component
            if args.component == 'autoencoder':
                trainer.train_autoencoder()
            elif args.component == 'capsnet':
                trainer.train_capsnet()
            elif args.component == 'rl':
                trainer.train_rl_agent()
            
            # Save model
            trainer.save_model(args.output)
            logger.info(f"Component {args.component} training completed!")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
