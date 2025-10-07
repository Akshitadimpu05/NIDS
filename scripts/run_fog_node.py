#!/usr/bin/env python3
"""
Script to run a NIDS Fog Node.
"""

import os
import sys
import logging
import asyncio
import signal
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fog_node.fog_node import FogNode, FogNodeConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/fog_node.log')
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
        
        logger.info(f"Starting fog node: {node_id}")
        logger.info(f"Orchestrator URL: {orchestrator_url}")
        logger.info(f"Capture interface: {capture_interface}")
        logger.info(f"Mitigation enabled: {enable_mitigation}")
        
        # Create and initialize fog node
        self.fog_node = FogNode(config)
        
        try:
            await self.fog_node.initialize()
            self.fog_node.start()
            self.running = True
            
            logger.info(f"Fog node {node_id} started successfully")
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Fog node failed: {e}")
            raise
        finally:
            if self.fog_node:
                self.fog_node.stop()
    
    def stop(self):
        """Stop the fog node."""
        logger.info("Stopping fog node...")
        self.running = False
        if self.fog_node:
            self.fog_node.stop()


# Global runner instance
runner = FogNodeRunner()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    runner.stop()


async def main():
    """Main function to run the fog node."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await runner.start()
    except KeyboardInterrupt:
        logger.info("Fog node stopped by user")
    except Exception as e:
        logger.error(f"Fog node failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
