#!/usr/bin/env python3
"""
Script to fix import issues in the NIDS-RL project.
This script modifies the necessary files to ensure proper imports.
"""

import os
import sys
from pathlib import Path

def fix_ensemble_imports():
    """Fix the imports in ensemble.py to use absolute imports."""
    ensemble_file = Path(__file__).parent / "src" / "models" / "ensemble.py"
    
    if not ensemble_file.exists():
        print(f"❌ File not found: {ensemble_file}")
        return False
    
    # Read the current content
    with open(ensemble_file, 'r') as f:
        content = f.read()
    
    # Replace relative imports with absolute imports
    old_imports = [
        "from ..agents.ppo_agent import PPOAgent",
        "from ..agents.environment import NIDSEnvironment, TrafficAction"
    ]
    
    new_imports = """
# Use absolute imports to avoid relative import issues
import sys
from pathlib import Path

# Add src to path if not already there
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agents.ppo_agent import PPOAgent
from agents.environment import NIDSEnvironment, TrafficAction"""
    
    # Check if we need to replace the imports
    if any(old_import in content for old_import in old_imports):
        # Find the position after the existing imports
        lines = content.split('\n')
        new_lines = []
        import_section_done = False
        
        for line in lines:
            if any(old_import in line for old_import in old_imports):
                if not import_section_done:
                    new_lines.append(new_imports)
                    import_section_done = True
                # Skip the old import lines
                continue
            else:
                new_lines.append(line)
        
        # Write the modified content back
        with open(ensemble_file, 'w') as f:
            f.write('\n'.join(new_lines))
        
        print(f"✓ Fixed imports in {ensemble_file}")
        return True
    else:
        print(f"✓ Imports already fixed in {ensemble_file}")
        return True

def create_init_files():
    """Ensure all necessary __init__.py files exist."""
    init_files = [
        Path(__file__).parent / "src" / "__init__.py",
    ]
    
    for init_file in init_files:
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            with open(init_file, 'w') as f:
                f.write('"""NIDS-RL Package"""\n')
            print(f"✓ Created {init_file}")
        else:
            print(f"✓ {init_file} already exists")

def create_directories():
    """Create necessary directories."""
    directories = [
        Path(__file__).parent / "logs",
        Path(__file__).parent / "models",
        Path(__file__).parent / "data",
        Path(__file__).parent / "results"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"✓ Directory ensured: {directory}")

def main():
    print("🔧 NIDS-RL Import Fixer")
    print("=" * 50)
    
    print("\n📁 Creating necessary directories...")
    create_directories()
    
    print("\n📄 Creating __init__.py files...")
    create_init_files()
    
    print("\n🔄 Fixing import statements...")
    if fix_ensemble_imports():
        print("✅ All imports fixed successfully!")
    else:
        print("❌ Some imports could not be fixed")
        return False
    
    print("\n🎉 Setup complete! You can now run:")
    print("   python run_training.py")
    print("   or")
    print("   python scripts/train_models.py --config config/training_config.yaml")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
