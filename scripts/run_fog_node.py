#!/usr/bin/env python3
"""
Script to run a NIDS Fog Node.
Fixed asyncio event loop handling and import issues.
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

# Configure logging first (console only to avoid permission issues)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import with comprehensive fallback
def import_fog_node():
    """Import fog node with multiple fallback paths."""
    import_attempts = [
        "src.fog_node.fog_node",
        "fog_node.fog_node",
        "NIDS.src.fog_node.fog_node"
    ]
    
    for attempt in import_attempts:
        try:
            module = __import__(attempt, fromlist=['FogNode', 'FogNodeConfig'])
            FogNode = getattr(module, 'FogNode')
            FogNodeConfig = getattr(module, 'FogNodeConfig')
            return FogNode, FogNodeConfig
        except ImportError as e:
            logger.debug(f"Import attempt failed for {attempt}: {e}")
            continue
    
    # Final attempt with direct path manipulation
    try:
        fog_node_path = Path(__file__).parent.parent / "src" / "fog_node"
        if fog_node_path.exists():
            sys.path.insert(0, str(fog_node_path.parent))
            from fog_node.fog_node import FogNode, FogNodeConfig
            return FogNode, FogNodeConfig
    except ImportError as e:
        logger.error(f"Final import attempt failed: {e}")
    
    logger.error(" Could not import fog_node.fog_node from any path")
    logger.error(f"Python path: {sys.path}")
    logger.error(f"Current working directory: {os.getcwd()}")
    
    # List available files for debugging
    src_path = Path("/app/src")
    if src_path.exists():
        logger.error(f"Contents of /app/src: {list(src_path.iterdir())}")
        fog_node_path = src_path / "fog_node"
        if fog_node_path.exists():
            logger.error(f"Contents of /app/src/fog_node: {list(fog_node_path.iterdir())}")
    
    return None, None


async def run_fog_node():
    """Run the fog node with proper async handling."""
    logger.info(" Starting fog node...")
    
    # Import fog node classes
    FogNode, FogNodeConfig = import_fog_node()
    if FogNode is None or FogNodeConfig is None:
        logger.error(" Failed to import fog node classes")
        return False
    
    # Get configuration from environment variables
    node_id = os.getenv('NODE_ID', 'fog-node-1')
    orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://orchestrator:8000')
    capture_interface = os.getenv('CAPTURE_INTERFACE', 'eth0')
    enable_mitigation = os.getenv('ENABLE_MITIGATION', 'true').lower() == 'true'
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    logger.info(f" Initializing fog node {node_id}")
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
        return False
        
    return True


def main():
    """Main function with proper event loop handling."""
    try:
        # Check if there's already an event loop running
        try:
            loop = asyncio.get_running_loop()
            logger.warning("Event loop already running, creating new task")
            # If we're in an existing loop, create a task
            task = loop.create_task(run_fog_node())
            return task
        except RuntimeError:
            # No event loop running, create one
            pass
        
        # Create new event loop and run
        if sys.platform == 'win32':
            # Windows-specific event loop policy
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Run the fog node
        success = asyncio.run(run_fog_node())
        
        if success:
            logger.info(" Fog node completed successfully")
            sys.exit(0)
        else:
            logger.error(" Fog node failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info(" Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f" Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
