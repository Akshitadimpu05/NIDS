#!/usr/bin/env python3
"""
Test script to verify model dimensions and data shapes.
"""

import sys
import os
sys.path.append('/home/tejasri/nids/NIDS')
sys.path.append('/home/tejasri/nids/NIDS/src')

import torch
import numpy as np
from src.models.autoencoder import TrafficAutoencoder
from src.models.capsnet import CapsuleNetwork
from src.models.ensemble import HybridNIDSModel

def test_model_dimensions():
    print("Testing Model Dimensions...")
    print("=" * 50)
    
    # Test data with 54 features
    batch_size = 256
    input_dim = 54
    test_data = torch.randn(batch_size, input_dim)
    
    print(f"Test data shape: {test_data.shape}")
    
    # Test Autoencoder
    print("\n1. Testing Autoencoder...")
    try:
        autoencoder = TrafficAutoencoder(input_dim=54)
        print(f"Autoencoder input_dim: {autoencoder.input_dim}")
        print(f"Autoencoder encoder: {autoencoder.encoder}")
        
        # Test forward pass
        reconstruction, latent = autoencoder(test_data)
        print(f"✅ Autoencoder works! Input: {test_data.shape}, Output: {reconstruction.shape}")
        
    except Exception as e:
        print(f"❌ Autoencoder failed: {e}")
    
    # Test CapsuleNetwork
    print("\n2. Testing CapsuleNetwork...")
    try:
        capsnet = CapsuleNetwork(input_dim=54)
        print(f"CapsNet input_dim: {capsnet.input_dim}")
        print(f"CapsNet feature_extractor: {capsnet.feature_extractor}")
        
        # Test forward pass
        output = capsnet(test_data)
        print(f"✅ CapsNet works! Input: {test_data.shape}, Output: {output.shape}")
        
    except Exception as e:
        print(f"❌ CapsNet failed: {e}")
    
    # Test Ensemble Model
    print("\n3. Testing Ensemble Model...")
    try:
        config = {
            'autoencoder': {
                'input_dim': 54,
                'hidden_dims': [64, 32, 16, 8],
                'latent_dim': 4,
                'dropout_rate': 0.2
            },
            'capsnet': {
                'input_dim': 54,
                'primary_caps_dim': 8,
                'primary_caps_num': 32,
                'digit_caps_dim': 16,
                'digit_caps_num': 10,
                'routing_iterations': 3
            },
            'rl': {
                'state_dim': 54,
                'action_dim': 3,
                'hidden_dims': [128, 64]
            }
        }
        
        ensemble = HybridNIDSModel(
            input_dim=54,
            ae_config=config['autoencoder'],
            capsnet_config=config['capsnet'],
            rl_config=config['rl']
        )
        
        print(f"Ensemble input_dim: {ensemble.input_dim}")
        print(f"Ensemble autoencoder input_dim: {ensemble.autoencoder.input_dim}")
        print(f"Ensemble capsnet input_dim: {ensemble.capsnet.input_dim}")
        
        # Test prediction
        predictions = ensemble.predict(test_data.numpy())
        print(f"✅ Ensemble works! Predictions: {len(predictions)}")
        
    except Exception as e:
        print(f"❌ Ensemble failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_dimensions()
