#!/usr/bin/env python3
"""
GPU Compatibility Checker for NIDS-RL Training
Automatically detects GPU compatibility and updates config accordingly.
"""

import torch
import yaml
import sys
from pathlib import Path

def check_gpu_compatibility():
    """Check if GPU is available and compatible."""
    print("🔍 Checking GPU compatibility...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False, "CUDA not available"
    
    try:
        # Try to create a simple tensor on GPU
        test_tensor = torch.tensor([1.0]).cuda()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"✅ GPU detected: {gpu_name}")
        print(f"✅ GPU memory: {gpu_memory:.1f} GB")
        
        # Test basic operations
        result = test_tensor * 2
        result.cpu()  # Move back to CPU
        
        print("✅ GPU compatibility test passed")
        return True, f"Compatible GPU: {gpu_name}"
        
    except Exception as e:
        print(f"❌ GPU compatibility test failed: {e}")
        return False, str(e)

def update_config_device(config_path, device):
    """Update the device setting in config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['training']['device'] = device
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        print(f"✅ Updated config to use device: {device}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def main():
    """Main function to check compatibility and update config."""
    print("🔧 NIDS-RL GPU Compatibility Checker")
    print("=" * 50)
    
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    
    # Check GPU compatibility
    gpu_compatible, message = check_gpu_compatibility()
    
    if gpu_compatible:
        device = "cuda"
        print(f"\n🎉 Using GPU for training: {message}")
    else:
        device = "cpu"
        print(f"\n⚠️  Falling back to CPU: {message}")
        print("Note: Training will be slower but should work without issues")
    
    # Update config
    if update_config_device(config_path, device):
        print(f"\n✅ Configuration updated successfully!")
        print(f"Device set to: {device}")
        
        print(f"\n🚀 Ready to train! Run:")
        print("   ./train.sh --data data/Darknet.CSV")
        
        return True
    else:
        print(f"\n❌ Failed to update configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
