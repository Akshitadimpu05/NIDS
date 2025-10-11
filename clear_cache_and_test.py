#!/usr/bin/env python3
"""
Clear Python cache and test model dimensions.
"""

import os
import shutil
import sys

def clear_python_cache():
    """Clear all Python cache files."""
    print("Clearing Python cache files...")
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk('/home/tejasri/nids/NIDS'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_dir = os.path.join(root, dir_name)
                print(f"Removing {cache_dir}")
                shutil.rmtree(cache_dir, ignore_errors=True)
    
    # Remove .pyc files
    for root, dirs, files in os.walk('/home/tejasri/nids/NIDS'):
        for file_name in files:
            if file_name.endswith('.pyc'):
                pyc_file = os.path.join(root, file_name)
                print(f"Removing {pyc_file}")
                os.remove(pyc_file)
    
    print("Cache cleared!")

def test_model_dimensions():
    """Test model dimensions after clearing cache."""
    print("\n" + "="*50)
    print("Testing Model Dimensions (After Cache Clear)")
    print("="*50)
    
    # Add paths
    sys.path.insert(0, '/home/tejasri/nids/NIDS')
    sys.path.insert(0, '/home/tejasri/nids/NIDS/src')
    
    try:
        import torch
        import numpy as np
        
        # Import models (fresh, no cache)
        from src.models.autoencoder import TrafficAutoencoder
        
        print(f"PyTorch version: {torch.__version__}")
        
        # Test data
        batch_size = 256
        input_dim = 54
        test_data = torch.randn(batch_size, input_dim)
        print(f"Test data shape: {test_data.shape}")
        
        # Test Autoencoder
        print("\nTesting Autoencoder...")
        autoencoder = TrafficAutoencoder(input_dim=54)
        print(f"Autoencoder input_dim: {autoencoder.input_dim}")
        
        # Check first layer
        first_layer = autoencoder.encoder[0]
        print(f"First layer: {first_layer}")
        print(f"First layer weight shape: {first_layer.weight.shape}")
        
        # Test forward pass
        try:
            reconstruction, latent = autoencoder(test_data)
            print(f"✅ SUCCESS! Input: {test_data.shape}, Output: {reconstruction.shape}")
        except Exception as e:
            print(f"❌ FAILED: {e}")
            
    except Exception as e:
        print(f"❌ Import or test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clear_python_cache()
    test_model_dimensions()
