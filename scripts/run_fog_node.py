#!/usr/bin/env python3
"""
Script to run a NIDS Fog Node.
Fixed asyncio event loop handling.
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
        print(" Could not import fog_node.fog_node")
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
    enable_mitigation = os.getenv('ENABLE_MITIGATION', 'true').lower() == 'true'
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    logger.info(f" Starting fog node {node_id}")
    logger.info(f" Orchestrator URL: {orchestrator_url}")
    logger.info(f" Configuration: interface={capture_interface}, mitigation={enable_mitigation}")
    
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
        logger.info(f" Initializing fog node {node_id}...")
        await fog_node.initialize()
        
        # Start fog node
        logger.info(f" Starting fog node {node_id}...")
        await fog_node.start()
        
        logger.info(f" Fog node {node_id} is running")
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info(f" Fog node {node_id} received stop signal")
        
        # Stop fog node
        logger.info(f" Stopping fog node {node_id}...")
        await fog_node.stop()
        logger.info(f" Fog node {node_id} stopped gracefully")
        
    except Exception as e:
        logger.error(f" Failed to run fog node {node_id}: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main function with proper event loop handling."""
    # Set up signal handlers for graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    main_task = None
    
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger.info(f" Received signal {signum}, shutting down...")
        if main_task and not main_task.done():
            main_task.cancel()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run fog node
        main_task = loop.create_task(run_fog_node())
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.info(" Fog node stopped by user")
    except Exception as e:
        logger.error(f" Fog node error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up
        try:
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # Wait for all tasks to complete
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            logger.error(f" Error during cleanup: {e}")
        finally:
            loop.close()
            logger.info(" Event loop closed")


if __name__ == "__main__":
    main()
