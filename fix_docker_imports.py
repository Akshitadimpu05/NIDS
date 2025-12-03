#!/usr/bin/env python3
"""
Fix Docker Import Issues
Updates relative imports to work inside Docker containers
"""

import os
import re
from pathlib import Path

def fix_file_imports(file_path, fixes):
    """Fix imports in a specific file."""
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        for old_import, new_import in fixes.items():
            content = content.replace(old_import, new_import)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Fixed imports in {file_path}")
            return True
        else:
            print(f"ℹ️ No changes needed in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def fix_orchestrator_imports():
    """Fix imports in orchestrator files."""
    print("🎛️ Fixing Orchestrator Imports...")
    
    # Fix model_aggregator.py
    fixes = {
        'from ..models.ensemble import HybridNIDSModel': '''# Import with fallback for Docker
try:
    from models.ensemble import HybridNIDSModel
except ImportError:
    try:
        from src.models.ensemble import HybridNIDSModel
    except ImportError:
        from ..models.ensemble import HybridNIDSModel''',
        
        'from ..utils.communication import OrchestratorClient': '''try:
    from utils.communication import OrchestratorClient
except ImportError:
    try:
        from src.utils.communication import OrchestratorClient
    except ImportError:
        from ..utils.communication import OrchestratorClient''',
        
        'from ..utils.metrics import MetricsCollector': '''try:
    from utils.metrics import MetricsCollector
except ImportError:
    try:
        from src.utils.metrics import MetricsCollector
    except ImportError:
        from ..utils.metrics import MetricsCollector'''
    }
    
    fix_file_imports('src/orchestrator/model_aggregator.py', fixes)
    
    # Fix orchestrator.py
    orchestrator_fixes = {
        'from .model_aggregator import ModelAggregator': '''try:
    from orchestrator.model_aggregator import ModelAggregator
except ImportError:
    try:
        from src.orchestrator.model_aggregator import ModelAggregator
    except ImportError:
        from .model_aggregator import ModelAggregator''',
        
        'from ..utils.communication import OrchestratorClient': '''try:
    from utils.communication import OrchestratorClient
except ImportError:
    try:
        from src.utils.communication import OrchestratorClient
    except ImportError:
        from ..utils.communication import OrchestratorClient'''
    }
    
    fix_file_imports('src/orchestrator/orchestrator.py', orchestrator_fixes)

def fix_fog_node_imports():
    """Fix imports in fog node files."""
    print("🌫️ Fixing Fog Node Imports...")
    
    # Fix fog_node.py
    fixes = {
        'from ..models.ensemble import HybridNIDSModel': '''# Import with fallback for Docker
try:
    from models.ensemble import HybridNIDSModel
except ImportError:
    try:
        from src.models.ensemble import HybridNIDSModel
    except ImportError:
        from ..models.ensemble import HybridNIDSModel''',
        
        'from .traffic_capture import TrafficCapture': '''try:
    from fog_node.traffic_capture import TrafficCapture
except ImportError:
    try:
        from src.fog_node.traffic_capture import TrafficCapture
    except ImportError:
        from .traffic_capture import TrafficCapture''',
        
        'from .mitigation import TrafficMitigation': '''try:
    from fog_node.mitigation import TrafficMitigation
except ImportError:
    try:
        from src.fog_node.mitigation import TrafficMitigation
    except ImportError:
        from .mitigation import TrafficMitigation''',
        
        'from ..utils.communication import OrchestratorClient': '''try:
    from utils.communication import OrchestratorClient
except ImportError:
    try:
        from src.utils.communication import OrchestratorClient
    except ImportError:
        from ..utils.communication import OrchestratorClient''',
        
        'from ..utils.metrics import MetricsCollector': '''try:
    from utils.metrics import MetricsCollector
except ImportError:
    try:
        from src.utils.metrics import MetricsCollector
    except ImportError:
        from ..utils.metrics import MetricsCollector'''
    }
    
    fix_file_imports('src/fog_node/fog_node.py', fixes)

def fix_ensemble_imports():
    """Fix imports in ensemble.py for Docker compatibility."""
    print("🤖 Fixing Ensemble Model Imports...")
    
    fixes = {
        'from .autoencoder import TrafficAutoencoder': '''try:
    from models.autoencoder import TrafficAutoencoder
except ImportError:
    try:
        from src.models.autoencoder import TrafficAutoencoder
    except ImportError:
        from .autoencoder import TrafficAutoencoder''',
        
        'from .capsnet import CapsuleNetwork': '''try:
    from models.capsnet import CapsuleNetwork
except ImportError:
    try:
        from src.models.capsnet import CapsuleNetwork
    except ImportError:
        from .capsnet import CapsuleNetwork'''
    }
    
    fix_file_imports('src/models/ensemble.py', fixes)

def fix_run_scripts():
    """Fix the run scripts to set proper Python path."""
    print("📜 Fixing Run Scripts...")
    
    # Fix run_orchestrator.py
    orchestrator_script = '''#!/usr/bin/env python3
"""
Script to run the Central Orchestrator.
"""

import os
import sys
import logging
import asyncio
import uvicorn
from pathlib import Path

# Add paths for Docker compatibility
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import with fallback
try:
    from orchestrator.api_server import create_api_server
except ImportError:
    try:
        from src.orchestrator.api_server import create_api_server
    except ImportError:
        print("❌ Could not import orchestrator.api_server")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/orchestrator.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to run the orchestrator."""
    # Get configuration from environment variables
    host = os.getenv('ORCHESTRATOR_HOST', '0.0.0.0')
    port = int(os.getenv('ORCHESTRATOR_PORT', '8000'))
    log_level = os.getenv('LOG_LEVEL', 'info').lower()
    
    logger.info(f"Starting NIDS Central Orchestrator on {host}:{port}")
    
    # Create FastAPI app
    app = create_api_server()
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
'''
    
    with open('scripts/run_orchestrator.py', 'w') as f:
        f.write(orchestrator_script)
    print("✅ Fixed run_orchestrator.py")
    
    # Fix run_fog_node.py
    fog_node_script = '''#!/usr/bin/env python3
"""
Script to run a NIDS Fog Node.
"""

import os
import sys
import logging
import asyncio
import signal
from pathlib import Path

# Add paths for Docker compatibility
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import with fallback
try:
    from fog_node.fog_node import FogNode, FogNodeConfig
except ImportError:
    try:
        from src.fog_node.fog_node import FogNode, FogNodeConfig
    except ImportError:
        print("❌ Could not import fog_node.fog_node")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/fog_node.log')
    ]
)

logger = logging.getLogger(__name__)


class FogNodeRunner:
    """Runner for fog node with graceful shutdown."""
    
    def __init__(self):
        self.fog_node = None
        self.running = False
    
    async def start(self):
        """Start the fog node."""
        # Get configuration from environment variables
        node_id = os.getenv('NODE_ID', 'fog-node-1')
        orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8000')
        capture_interface = os.getenv('CAPTURE_INTERFACE', 'eth0')
        enable_mitigation = os.getenv('ENABLE_MITIGATION', 'true').lower() == 'true'
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Create fog node configuration
        config = FogNodeConfig(
            node_id=node_id,
            orchestrator_url=orchestrator_url,
            capture_interface=capture_interface,
            enable_mitigation=enable_mitigation,
            log_level=log_level
        )
        
        logger.info(f"Starting fog node {node_id}")
        
        # Create and initialize fog node
        self.fog_node = FogNode(config)
        await self.fog_node.initialize()
        
        # Start fog node
        self.running = True
        await self.fog_node.start()
        
        # Keep running
        while self.running:
            await asyncio.sleep(1)
    
    def stop(self):
        """Stop the fog node."""
        logger.info("Stopping fog node...")
        self.running = False
        if self.fog_node:
            asyncio.create_task(self.fog_node.stop())


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}")
    runner.stop()


def main():
    """Main function."""
    global runner
    runner = FogNodeRunner()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run fog node
    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        logger.info("Fog node stopped by user")
    except Exception as e:
        logger.error(f"Fog node error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    
    with open('scripts/run_fog_node.py', 'w') as f:
        f.write(fog_node_script)
    print("✅ Fixed run_fog_node.py")

def main():
    """Main function to fix all Docker import issues."""
    print("🔧 FIXING DOCKER IMPORT ISSUES")
    print("=" * 50)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Fix imports in different modules
    fix_ensemble_imports()
    fix_orchestrator_imports()
    fix_fog_node_imports()
    fix_run_scripts()
    
    print("\n✅ All Docker import fixes applied!")
    print("\n🚀 Now rebuild and restart Docker services:")
    print("  docker-compose -f docker/docker-compose.yml down")
    print("  docker-compose -f docker/docker-compose.yml build")
    print("  docker-compose -f docker/docker-compose.yml up -d")
    print("\n📊 Then check status:")
    print("  python check_distributed_system.py status")

if __name__ == "__main__":
    main()
