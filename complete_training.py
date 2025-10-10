#!/usr/bin/env python3
"""
Complete NIDS training with proper evaluation and results generation.
"""

import os
import sys
import subprocess
import logging

# Setup paths
sys.path.append('/home/tejasri/nids/NIDS')
sys.path.append('/home/tejasri/nids/NIDS/src')

def main():
    print("🎯 NIDS Training - Final Completion")
    print("=" * 60)
    
    # Check training status
    model_files = {
        'autoencoder': 'data/models/best_autoencoder.pth',
        'capsnet': 'data/models/best_capsnet.pth', 
        'rl_agent': 'data/models/best_rl_agent.pth',
        'preprocessor': 'data/models/preprocessor.pkl'
    }
    
    print("📁 Training Status:")
    all_trained = True
    for name, path in model_files.items():
        if os.path.exists(path):
            print(f"   ✅ {name}: Trained and saved")
        else:
            print(f"   ❌ {name}: Missing")
            all_trained = False
    
    if all_trained:
        print("\n🎉 All models are already trained!")
        print("   → Skipping training, proceeding to evaluation")
    else:
        print("\n⚠️  Some models are missing. Running training first...")
        try:
            result = subprocess.run(['./train.sh', '--data', 'data/Darknet.CSV'], 
                                  capture_output=False, text=True, cwd='/home/tejasri/nids/NIDS')
            if result.returncode != 0:
                print("❌ Training failed")
                return False
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    # Save final hybrid model if it doesn't exist
    hybrid_path = 'data/models/hybrid_nids_model.pth'
    if not os.path.exists(hybrid_path) and all_trained:
        print("\n💾 Creating final hybrid model...")
        try:
            # Run a quick script to save the hybrid model
            result = subprocess.run(['python', '-c', '''
import sys
sys.path.append("/home/tejasri/nids/NIDS")
sys.path.append("/home/tejasri/nids/NIDS/src")
from src.models.ensemble import HybridNIDSModel
import yaml
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)
model = HybridNIDSModel(input_dim=54, ae_config=config["autoencoder"], 
                       capsnet_config=config["capsnet"], rl_config=config["rl_agent"])
model.load_component("autoencoder", "data/models/best_autoencoder.pth")
model.load_component("capsnet", "data/models/best_capsnet.pth") 
model.load_component("rl_agent", "data/models/best_rl_agent.pth")
model.save_model("data/models/hybrid_nids_model.pth")
print("✅ Hybrid model saved")
'''], capture_output=True, text=True, cwd='/home/tejasri/nids/NIDS')
            
            if result.returncode == 0:
                print("✅ Hybrid model created successfully")
            else:
                print(f"⚠️  Could not create hybrid model: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Could not create hybrid model: {e}")
    
    # Run evaluation
    print("\n🔍 Running comprehensive evaluation...")
    try:
        result = subprocess.run(['python', 'run_evaluation.py'], 
                              capture_output=False, text=True, cwd='/home/tejasri/nids/NIDS')
        
        if result.returncode == 0:
            print("\n🎉 TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("✅ Autoencoder: Trained for anomaly detection")
            print("✅ CapsNet: Trained for traffic classification") 
            print("✅ RL Agent: Trained for dynamic decision making")
            print("✅ Evaluation: Comprehensive metrics generated")
            print("=" * 60)
            
            # Show results directory
            results_dir = "results/evaluation"
            if os.path.exists(results_dir):
                print(f"📊 Results saved to: {results_dir}")
                print("📁 Generated files:")
                for file in os.listdir(results_dir):
                    print(f"   - {file}")
            
            return True
        else:
            print("❌ Evaluation failed")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 NIDS system is fully trained and evaluated!")
        print("🚀 Ready for deployment!")
    else:
        print("\n❌ Training/evaluation incomplete")
        sys.exit(1)
