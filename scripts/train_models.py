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

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ensemble import HybridNIDSModel
from utils.data_preprocessing import DataPreprocessor
from agents.environment import NIDSEnvironment

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
        
        self.model = HybridNIDSModel(
            input_dim=len(self.config['data']['features']),
            ae_config=self.config['autoencoder'],
            capsnet_config=self.config['capsnet'],
            rl_config=self.config['rl_agent'],
            device=self.config['training']['device']
        )
        
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
        """Evaluate the complete hybrid model."""
        logger.info("Evaluating hybrid model...")
        
        # Evaluation metrics
        total_samples = 0
        correct_predictions = 0
        anomaly_detection_tp = 0
        anomaly_detection_fp = 0
        anomaly_detection_tn = 0
        anomaly_detection_fn = 0
        
        self.model.autoencoder.eval()
        self.model.capsnet.eval()
        
        with torch.no_grad():
            for batch_features, batch_labels in self.test_data:
                batch_size = len(batch_features)
                total_samples += batch_size
                
                # Test each sample
                for i in range(batch_size):
                    sample_features = batch_features[i:i+1]
                    true_label = batch_labels[i].item()
                    
                    # Get model prediction
                    result = self.model.predict(sample_features.numpy())
                    
                    # Evaluate anomaly detection
                    is_anomaly = result['is_anomaly']
                    true_anomaly = true_label != 0  # Assuming 0 is normal
                    
                    if is_anomaly and true_anomaly:
                        anomaly_detection_tp += 1
                    elif is_anomaly and not true_anomaly:
                        anomaly_detection_fp += 1
                    elif not is_anomaly and not true_anomaly:
                        anomaly_detection_tn += 1
                    else:
                        anomaly_detection_fn += 1
                    
                    # Evaluate RL decision (simplified)
                    action = result['action']
                    if (true_anomaly and action in [1, 2]) or (not true_anomaly and action == 0):
                        correct_predictions += 1
        
        # Calculate metrics
        accuracy = correct_predictions / total_samples
        
        precision = anomaly_detection_tp / (anomaly_detection_tp + anomaly_detection_fp) if (anomaly_detection_tp + anomaly_detection_fp) > 0 else 0
        recall = anomaly_detection_tp / (anomaly_detection_tp + anomaly_detection_fn) if (anomaly_detection_tp + anomaly_detection_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Overall accuracy: {accuracy:.4f}")
        logger.info(f"  Anomaly detection precision: {precision:.4f}")
        logger.info(f"  Anomaly detection recall: {recall:.4f}")
        logger.info(f"  Anomaly detection F1-score: {f1_score:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_samples': total_samples
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
    parser.add_argument('--data', type=str, required=True,
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
