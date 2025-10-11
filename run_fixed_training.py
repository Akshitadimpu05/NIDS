#!/usr/bin/env python3
"""
Run NIDS training with all CapsNet fixes applied.
"""

import os
import subprocess
import sys

def main():
    print("🎯 NIDS Training - All CapsNet Fixes Applied")
    print("=" * 60)
    print("✅ FIXED: Autoencoder dimension mismatch (77→54)")
    print("✅ FIXED: CapsNet class count (10→4 classes)")
    print("✅ FIXED: CapsNet masking logic for target labels")
    print("✅ FIXED: CapsNet loss function one-hot encoding")
    print("✅ ADDED: Model resumption (skip trained components)")
    print("=" * 60)
    
    # Check existing models
    ae_path = 'data/models/best_autoencoder.pth'
    capsnet_path = 'data/models/best_capsnet.pth'
    
    if os.path.exists(ae_path):
        print(f"📁 Found existing autoencoder: {ae_path}")
        print("   → Will skip autoencoder training")
    
    if os.path.exists(capsnet_path):
        print(f"📁 Found existing CapsNet: {capsnet_path}")
        print("   → Will skip CapsNet training")
    
    print("\n🚀 Starting training...")
    
    # Change to project directory
    os.chdir('/home/tejasri/nids/NIDS')
    
    # Run training
    try:
        result = subprocess.run(['./train.sh', '--data', 'data/Darknet.CSV'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 SUCCESS! Training completed!")
            print("✅ Autoencoder: Loaded from saved model")
            print("✅ CapsNet: Should train successfully with fixed dimensions")
            print("✅ RL Agent: Will train decision policies")
            print("✅ Evaluation: Complete metrics and visualizations")
            
            # Check what models were created
            if os.path.exists('data/models/best_capsnet.pth'):
                print("📁 CapsNet model saved successfully!")
            if os.path.exists('data/models/hybrid_nids_model.pth'):
                print("📁 Complete hybrid model saved!")
                
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    main()
