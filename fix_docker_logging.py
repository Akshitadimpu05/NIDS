#!/usr/bin/env python3
"""
Fix Docker Logging Issues
Remove file logging to avoid permission errors in containers
"""

import os

def fix_orchestrator_logging():
    """Fix orchestrator logging to use console only."""
    print("🎛️ Fixing orchestrator logging...")
    
    script_content = '''#!/usr/bin/env python3
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

# Configure logging (console only to avoid permission issues)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to run the orchestrator."""
    # Get configuration from environment variables
    host = os.getenv('ORCHESTRATOR_HOST', '0.0.0.0')
    port = int(os.getenv('ORCHESTRATOR_PORT', '8000'))
    log_level = os.getenv('LOG_LEVEL', 'info').lower()
    
    logger.info(f"🎛️ Starting NIDS Central Orchestrator on {host}:{port}")
    
    try:
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
    except Exception as e:
        logger.error(f"❌ Failed to start orchestrator: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    
    with open('scripts/run_orchestrator.py', 'w') as f:
        f.write(script_content)
    print("✅ Fixed orchestrator logging")

def fix_fog_node_logging():
    """Fix fog node logging to use console only."""
    print("🌫️ Fixing fog node logging...")
    
    script_content = '''#!/usr/bin/env python3
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

# Configure logging (console only to avoid permission issues)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
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
        orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://orchestrator:8000')
        capture_interface = os.getenv('CAPTURE_INTERFACE', 'eth0')
        enable_mitigation = os.getenv('ENABLE_MITIGATION', 'true').lower() == 'true'
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        logger.info(f"🌫️ Starting fog node {node_id}")
        
        try:
            # Create fog node configuration
            config = FogNodeConfig(
                node_id=node_id,
                orchestrator_url=orchestrator_url,
                capture_interface=capture_interface,
                enable_mitigation=enable_mitigation,
                log_level=log_level
            )
            
            # Create and initialize fog node
            self.fog_node = FogNode(config)
            await self.fog_node.initialize()
            
            # Start fog node
            self.running = True
            await self.fog_node.start()
            
            # Keep running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Failed to start fog node {node_id}: {e}")
            sys.exit(1)
    
    def stop(self):
        """Stop the fog node."""
        logger.info("🛑 Stopping fog node...")
        self.running = False
        if self.fog_node:
            asyncio.create_task(self.fog_node.stop())


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"📡 Received signal {signum}")
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
        logger.info("🛑 Fog node stopped by user")
    except Exception as e:
        logger.error(f"❌ Fog node error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    
    with open('scripts/run_fog_node.py', 'w') as f:
        f.write(script_content)
    print("✅ Fixed fog node logging")

def main():
    """Fix all Docker logging issues."""
    print("🔧 FIXING DOCKER LOGGING ISSUES")
    print("=" * 50)
    
    fix_orchestrator_logging()
    fix_fog_node_logging()
    
    print("\n✅ All logging fixes applied!")
    print("\n🚀 Now restart Docker services:")
    print("  docker-compose -f docker/docker-compose.yml down")
    print("  docker-compose -f docker/docker-compose.yml up -d")
    print("\n⏱️ Wait 30 seconds for services to start, then check:")
    print("  python check_distributed_system.py status")
    print("\n🌐 Access Grafana dashboard:")
    print("  http://localhost:3000 (admin/admin)")

if __name__ == "__main__":
    main()
