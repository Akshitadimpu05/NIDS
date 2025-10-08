#!/usr/bin/env python3
"""
Training launcher script for NIDS-RL.
This script properly sets up the Python path and runs the training.
"""

import os
import sys
from pathlib import Path

# Add the project root and src directories to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"

# Insert at the beginning to ensure our modules are found first
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        project_root / "logs",
        project_root / "models",
        project_root / "data",
        project_root / "results"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"✓ Directory ensured: {directory}")

def check_dependencies():
    """Check if required packages are available."""
    required_packages = ['torch', 'numpy', 'yaml', 'gymnasium']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True

# Now import and run the training script
if __name__ == "__main__":
    print("🚀 NIDS-RL Training Launcher")
    print("=" * 50)
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🔧 Setting up Python path...")
    print(f"✓ Project root: {project_root}")
    print(f"✓ Source path: {src_path}")
    
    try:
        # Import the main function from the training script
        print("\n📥 Importing training modules...")
        from scripts.train_models import main
        print("✓ Training modules imported successfully")
        
        # Run the training
        print("\n🎯 Starting training...")
        print("=" * 50)
        main()
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all files are in the correct locations")
        print("2. Check that all required packages are installed")
        print("3. Verify that the src directory contains all necessary modules")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)
