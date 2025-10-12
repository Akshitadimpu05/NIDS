#!/usr/bin/env python3
"""
Script to run a NIDS Fog Node.
Portable version that works in any environment.
"""

import os
import sys
import logging
import asyncio
import signal
from pathlib import Path

# Add multiple paths - works in Docker and local
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

# Debug: Print paths
print(f"🔍 Python paths:")
for p in sys.path[:5]:
    print(f"  - {p}")

# Try importing with detailed error messages
try:
    from fog_node.fog_node import FogNode, FogNodeConfig
    print("✅ Successfully imported fog_node.fog_node")
except ImportError as e:
    print(f"❌ Failed to import fog_node.fog_node: {e}")
    print(f"📁 Looking for: {project_root / 'src' / 'fog_node' / 'fog_node.py'}")
    print(f"📁 File exists: {(project_root / 'src' / 'fog_node' / 'fog_node.py').exists()}")
    
    # Try alternative import
    try:
        from src.fog_node.fog_node import FogNode, FogNodeConfig
        print("✅ Successfully imported via src.fog_node.fog_node")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
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


async def run_fog_node():
    """Run the fog node with proper async handling."""
    # Get configuration from environment variables
    node_id = os.getenv('NODE_ID', 'fog-node-1')
    orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://orchestrator:8000')
    capture_interface = os.getenv('CAPTURE_INTERFACE', 'eth0')
    enable_mitigation = os.getenv('ENABLE_MITIGATION', 'false').lower() == 'true'
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    logger.info(f"🚀 Starting fog node {node_id}")
    logger.info(f"📡 Orchestrator URL: {orchestrator_url}")
    logger.info(f"⚙️ Configuration: interface={capture_interface}, mitigation={enable_mitigation}")
    
    try:
        # Create fog node configuration
        config = FogNodeConfig(
            node_id=node_id,
            orchestrator_url=orchestrator_url,
            capture_interface=capture_interface,
            enable_mitigation=enable_mitigation,
            log_level=log_level
        )
        
        # Create fog node
        fog_node = FogNode(config)
        
        # Initialize fog node
        logger.info(f"⚙️ Initializing fog node {node_id}...")
        await fog_node.initialize()
        
        # Start fog node
        logger.info(f"▶️ Starting fog node {node_id}...")
        await fog_node.start()
        
        logger.info(f"✅ Fog node {node_id} is running")
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info(f"🛑 Fog node {node_id} received shutdown signal")
            
    except Exception as e:
        logger.error(f"💥 Error in fog node {node_id}: {e}", exc_info=True)
        raise


def main():
    """Main function with proper event loop handling."""
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info("🛑 Received shutdown signal")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create new event loop to avoid "Event loop is closed" errors
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        logger.info("🔄 Starting event loop...")
        
        # Run fog node
        loop.run_until_complete(run_fog_node())
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down fog node (KeyboardInterrupt)...")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up event loop
        try:
            loop.close()
            logger.info("✅ Event loop closed cleanly")
        except Exception as e:
            logger.warning(f"⚠️ Error closing event loop: {e}")


if __name__ == "__main__":
    main()
