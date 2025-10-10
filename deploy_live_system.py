#!/usr/bin/env python3
"""
Deployment Script for Live NIDS Detection System
Sets up and runs the complete live detection and mitigation system
"""

import os
import sys
import subprocess
import argparse
import time
import threading
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 'matplotlib', 
        'seaborn', 'flask', 'yaml', 'logging'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies satisfied")
    return True

def check_models():
    """Check if trained models exist."""
    print("\n🤖 Checking trained models...")
    
    required_models = [
        'data/models/joint_nids_model.pth',
        'data/models/joint_rl_agent.pth',
        'data/models/joint_preprocessor.pkl'
    ]
    
    missing_models = []
    
    for model_path in required_models:
        if os.path.exists(model_path):
            print(f"✅ {model_path}")
        else:
            missing_models.append(model_path)
            print(f"❌ {model_path}")
    
    if missing_models:
        print(f"\n⚠️  Missing models: {', '.join(missing_models)}")
        print("Run training first:")
        print("  python joint_training.py")
        print("  python train_rl_after_joint.py")
        return False
    
    print("✅ All models available")
    return True

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        'logs',
        'results/live_detection',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}")

def run_command_line_demo():
    """Run command-line live detection demo."""
    print("\n🚀 Starting Command-Line Live Detection Demo")
    print("=" * 60)
    
    try:
        # Import here to avoid issues if models aren't ready
        from live_detection_system import LiveDetectionSystem
        
        # Create detection system
        system = LiveDetectionSystem(detection_rate=5)  # 5 flows per second for demo
        
        print("📊 Demo Configuration:")
        print(f"   Detection Rate: {system.detection_rate} flows/second")
        print(f"   Duration: 30 seconds")
        print("   Press Ctrl+C to stop early")
        
        # Start detection for 30 seconds
        system.start_detection(duration=30)
        
        # Print final statistics
        print("\n" + "="*60)
        print("📊 DEMO COMPLETED - FINAL STATISTICS")
        print("="*60)
        system.print_statistics()
        
        # Save results
        results_file = system.save_results()
        print(f"\n💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True

def run_web_dashboard():
    """Run web dashboard."""
    print("\n🌐 Starting Web Dashboard...")
    print("=" * 60)
    
    try:
        # Install Flask if not available
        try:
            import flask
        except ImportError:
            print("Installing Flask...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask'], check=True)
        
        # Run web dashboard
        print("🚀 Starting web server...")
        print("📊 Dashboard will be available at: http://localhost:5000")
        print("🛡️ Use the web interface to start/stop detection")
        print("   Press Ctrl+C to stop the web server")
        
        # Run the dashboard
        subprocess.run([sys.executable, 'web_dashboard.py'])
        
    except KeyboardInterrupt:
        print("\n🛑 Web dashboard stopped")
    except Exception as e:
        print(f"❌ Web dashboard failed: {e}")
        return False
    
    return True

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy Live NIDS Detection System')
    parser.add_argument('--mode', choices=['demo', 'web', 'both'], default='demo',
                       help='Deployment mode (default: demo)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency and model checks')
    
    args = parser.parse_args()
    
    print("🛡️ Live NIDS Detection System - Deployment")
    print("=" * 60)
    
    # Perform checks unless skipped
    if not args.skip_checks:
        if not check_dependencies():
            print("\n❌ Dependency check failed")
            return False
        
        if not check_models():
            print("\n❌ Model check failed")
            return False
    
    # Setup directories
    setup_directories()
    
    # Run based on mode
    if args.mode == 'demo':
        print("\n🎯 Running Command-Line Demo Mode")
        return run_command_line_demo()
        
    elif args.mode == 'web':
        print("\n🌐 Running Web Dashboard Mode")
        return run_web_dashboard()
        
    elif args.mode == 'both':
        print("\n🎯 Running Both Demo and Web Dashboard")
        
        # Run demo first
        print("\n1️⃣ Running Demo...")
        if not run_command_line_demo():
            return False
        
        print("\n" + "="*60)
        input("Press Enter to continue to web dashboard...")
        
        # Then run web dashboard
        print("\n2️⃣ Starting Web Dashboard...")
        return run_web_dashboard()

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Deployment completed successfully!")
        else:
            print("\n❌ Deployment failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted by user")
    except Exception as e:
        print(f"\n💥 Deployment error: {e}")
        sys.exit(1)
