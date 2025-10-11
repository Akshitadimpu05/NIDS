#!/usr/bin/env python3
"""
Install Flask for the web dashboard
"""

import subprocess
import sys

def install_flask():
    """Install Flask and required web dependencies."""
    print("📦 Installing Flask for web dashboard...")
    
    packages = ['flask']
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All web dependencies installed!")
    return True

if __name__ == "__main__":
    install_flask()
