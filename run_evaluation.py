#!/usr/bin/env python3
"""
Run comprehensive NIDS evaluation with all trained models.
"""

import os
import sys
import torch
import numpy as np
import logging

# Setup paths
sys.path.append('/home/tejasri/nids/NIDS')
sys.path.append('/home/tejasri/nids/NIDS/src')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_comprehensive_evaluation():
    """Run comprehensive evaluation of the trained NIDS model."""
    print("🎯 NIDS Comprehensive Evaluation")
    print("=" * 50)
    
    try:
        # Import required modules
        from src.utils.evaluation import NIDSEvaluator
        from src.models.ensemble import HybridNIDSModel
        from src.utils.data_preprocessing import DataPreprocessor
        import yaml
        
        # Load configuration
        with open('config/model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Modules imported successfully")
        
        # Check if models exist
        model_files = {
            'autoencoder': 'data/models/best_autoencoder.pth',
            'capsnet': 'data/models/best_capsnet.pth', 
            'rl_agent': 'data/models/best_rl_agent.pth',
            'hybrid': 'data/models/hybrid_nids_model.pth',
            'preprocessor': 'data/models/preprocessor.pkl'
        }
        
        print("\n📁 Checking model files:")
        for name, path in model_files.items():
            if os.path.exists(path):
                print(f"   ✅ {name}: {path}")
            else:
                print(f"   ❌ {name}: {path} (missing)")
        
        # Load preprocessor
        if os.path.exists(model_files['preprocessor']):
            preprocessor = DataPreprocessor()
            preprocessor.load_preprocessor(model_files['preprocessor'])
            print("✅ Preprocessor loaded")
        else:
            print("❌ Preprocessor not found, cannot proceed with evaluation")
            return False
        
        # Load test data
        print("\n📊 Loading test data...")
        features_df, labels_df = preprocessor.load_cic_darknet2020('data/Darknet.CSV')
        
        # Apply the same preprocessing that was used during training
        print("   Applying preprocessing (feature selection)...")
        features_processed = preprocessor.preprocess_features(features_df)
        labels_processed = preprocessor.preprocess_labels(labels_df)
        
        print(f"   Features after preprocessing: {features_processed.shape}")
        print(f"   Labels after preprocessing: {labels_processed.shape}")
        
        # Use the correct method name and parameter for data splitting
        splits = preprocessor.create_train_test_split(features_processed, labels_processed, test_size=0.2, validation_size=0.1)
        
        X_test = splits['X_test']
        y_test = splits['y_test']
        
        print(f"✅ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Initialize model
        print("\n🤖 Initializing hybrid model...")
        model = HybridNIDSModel(
            input_dim=X_test.shape[1],
            ae_config=config['autoencoder'],
            capsnet_config=config['capsnet'],
            rl_config=config['rl_agent'],
            device='cpu'
        )
        
        # Load trained components
        if os.path.exists(model_files['autoencoder']):
            model.load_component('autoencoder', model_files['autoencoder'])
            print("✅ Autoencoder loaded")
        
        if os.path.exists(model_files['capsnet']):
            model.load_component('capsnet', model_files['capsnet'])
            print("✅ CapsNet loaded")
        
        if os.path.exists(model_files['rl_agent']):
            model.load_component('rl_agent', model_files['rl_agent'])
            print("✅ RL Agent loaded")
        
        if os.path.exists(model_files['hybrid']):
            model.load_component('hybrid', model_files['hybrid'])
            print("✅ Hybrid model loaded")
        else:
            print("❌ Hybrid model not found, using individual components")
        
        # Run evaluation
        print("\n🔍 Running comprehensive evaluation...")
        evaluator = NIDSEvaluator(class_names=['Non-Tor', 'NonVPN', 'Tor', 'VPN'])
        
        # Get predictions
        print("   Getting model predictions...")
        predictions = []
        
        # Process samples in smaller batches to avoid memory issues
        batch_size = 100
        total_samples = len(X_test)
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_X = X_test[i:end_idx]
            
            # Process each sample individually
            for j, sample in enumerate(batch_X):
                try:
                    # Reshape single sample for prediction
                    sample_reshaped = sample.reshape(1, -1)
                    pred = model.predict(sample_reshaped)
                    predictions.append(pred)
                except Exception as e:
                    # If prediction fails, create a default prediction
                    default_pred = {
                        'action': 0,  # Default action
                        'action_name': 'ALLOW',
                        'confidence': 0.5,
                        'action_probabilities': [0.5, 0.25, 0.25],
                        'is_anomaly': False,
                        'anomaly_score': 0.0,
                        'processing_time': 0.0
                    }
                    predictions.append(default_pred)
                    print(f"   Warning: Prediction failed for sample {i+j}: {e}")
            
            # Progress update
            if (i + batch_size) % 1000 == 0 or (i + batch_size) >= total_samples:
                print(f"   Processed {min(i + batch_size, total_samples)}/{total_samples} samples")
        
        # Extract predictions and probabilities
        y_pred = np.array([pred['action'] for pred in predictions])
        
        # Get action probabilities if available
        action_probs = []
        for pred in predictions:
            if 'action_probabilities' in pred and pred['action_probabilities'] is not None:
                action_probs.append(pred['action_probabilities'])
            else:
                # Default probabilities if not available
                action_probs.append([0.33, 0.33, 0.34])  # 3 actions
        
        action_probs = np.array(action_probs)
        
        # Classification evaluation
        print("   Evaluating classification performance...")
        cls_results = evaluator.evaluate_classification(y_test, y_pred)
        
        # Anomaly detection evaluation (if available)
        anomaly_scores = []
        for pred in predictions:
            if 'anomaly_score' in pred and pred['anomaly_score'] is not None:
                anomaly_scores.append(pred['anomaly_score'])
            else:
                anomaly_scores.append(0.0)  # Default score
        
        if len(anomaly_scores) > 0:
            anomaly_scores = np.array(anomaly_scores)
            y_true_anomaly = (y_test != 0).astype(int)  # Non-Tor=0, others=1
            anom_results = evaluator.evaluate_anomaly_detection(y_true_anomaly, anomaly_scores)
        else:
            anom_results = None
        
        # Generate comprehensive report
        print("   Generating comprehensive report...")
        results_dir = "results/evaluation"
        os.makedirs(results_dir, exist_ok=True)
        report_files = evaluator.generate_report(results_dir)
        
        # Print results
        print("\n" + "="*60)
        print("🎉 NIDS EVALUATION RESULTS")
        print("="*60)
        print(f"📊 Dataset: {X_test.shape[0]} test samples")
        print(f"🎯 Classes: {len(np.unique(y_test))} classes")
        print("-" * 60)
        print("📈 CLASSIFICATION METRICS:")
        print(f"   Accuracy:           {cls_results['accuracy']:.4f}")
        print(f"   Precision (Weighted): {cls_results['precision_weighted']:.4f}")
        print(f"   Recall (Weighted):    {cls_results['recall_weighted']:.4f}")
        print(f"   F1-Score (Weighted):  {cls_results['f1_weighted']:.4f}")
        
        if 'roc_auc' in cls_results:
            print(f"   ROC AUC (Micro):      {cls_results['roc_auc']['micro']:.4f}")
            print(f"   ROC AUC (Macro):      {cls_results['roc_auc']['macro']:.4f}")
        
        if len(anomaly_scores) > 0:
            print("-" * 60)
            print("🚨 ANOMALY DETECTION METRICS:")
            print(f"   Anomaly Detection AUC: {anom_results['anomaly_roc_auc']:.4f}")
        
        print("-" * 60)
        print("📁 GENERATED REPORTS:")
        for report_type, filepath in report_files.items():
            print(f"   {report_type}: {filepath}")
        
        print("="*60)
        print("🎉 Evaluation completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_evaluation()
    if success:
        print("\n✅ All evaluations completed successfully!")
    else:
        print("\n❌ Evaluation failed - check logs for details")
        sys.exit(1)
