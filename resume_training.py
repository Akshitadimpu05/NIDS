#!/usr/bin/env python3
"""
Resume NIDS training with fixes applied.
"""

import os
import subprocess
import sys

def main():
    print("🚀 NIDS Training - Resume with Fixes")
    print("=" * 50)
    print("✅ FIXED: Autoencoder dimension mismatch (77→54)")
    print("✅ FIXED: CapsNet class count (10→4 classes)")
    print("✅ FIXED: CapsNet masking logic for target labels")
    print("✅ ADDED: Model resumption (skip trained components)")
    print("=" * 50)
    
    # Check if autoencoder exists
    ae_path = 'data/models/best_autoencoder.pth'
    if os.path.exists(ae_path):
        print(f"📁 Found existing autoencoder: {ae_path}")
        print("   → Will skip autoencoder training")
    else:
        print("   → No existing autoencoder found, will train from scratch")
    
    print("\n🎯 Starting training...")
    
    # Change to project directory
    os.chdir('/home/tejasri/nids/NIDS')
    
    # Run training
    try:
        result = subprocess.run(['./train.sh', '--data', 'data/Darknet.CSV'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 Training completed successfully!")
            print("✅ Autoencoder: Trained/Loaded")
            print("✅ CapsNet: Should now train successfully")
            print("✅ RL Agent: Will train after CapsNet")
            print("✅ Evaluation: Complete metrics and visualizations")
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    main()
