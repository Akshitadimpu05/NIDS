#!/usr/bin/env python3
"""
Clear cache and run training with fixed model initialization.
"""

import os
import shutil
import subprocess
import sys

def clear_python_cache():
    """Clear all Python cache files."""
    print("🧹 Clearing Python cache files...")
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk('/home/tejasri/nids/NIDS'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_dir = os.path.join(root, dir_name)
                print(f"   Removing {cache_dir}")
                shutil.rmtree(cache_dir, ignore_errors=True)
    
    print("✅ Cache cleared!")

def run_training():
    """Run the training script."""
    print("\n🚀 Starting NIDS training with fixed dimensions...")
    print("=" * 60)
    
    # Change to project directory
    os.chdir('/home/tejasri/nids/NIDS')
    
    # Run training
    try:
        result = subprocess.run(['./train.sh', '--data', 'data/Darknet.CSV'], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 NIDS Training - Dimension Fix Applied")
    print("=" * 60)
    print("ISSUE FIXED: Model now initializes with actual feature count (54)")
    print("BEFORE: input_dim = len(config['data']['features']) = 77")
    print("AFTER:  input_dim = features.shape[1] = 54")
    print("=" * 60)
    
    # Clear cache first
    clear_python_cache()
    
    # Run training
    success = run_training()
    
    if success:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n❌ Training failed - check logs for details")
