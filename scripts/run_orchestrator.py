#!/usr/bin/env python3
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
def import_orchestrator():
    """Import orchestrator with multiple fallback paths."""
    import_attempts = [
        "src.orchestrator.api_server",
        "orchestrator.api_server", 
        "NIDS.src.orchestrator.api_server"
    ]
    
    for attempt in import_attempts:
        try:
            module = __import__(attempt, fromlist=['create_api_server'])
            return getattr(module, 'create_api_server')
        except ImportError as e:
            logger.debug(f"Import attempt failed for {attempt}: {e}")
            continue
    
    # Final attempt with direct path manipulation
    try:
        orchestrator_path = Path(__file__).parent.parent / "src" / "orchestrator"
        if orchestrator_path.exists():
            sys.path.insert(0, str(orchestrator_path.parent))
            from orchestrator.api_server import create_api_server
            return create_api_server
    except ImportError as e:
        logger.error(f"Final import attempt failed: {e}")
    
    logger.error("❌ Could not import orchestrator.api_server from any path")
    logger.error(f"Python path: {sys.path}")
    logger.error(f"Current working directory: {os.getcwd()}")
    
    # List available files for debugging
    src_path = Path("/app/src")
    if src_path.exists():
        logger.error(f"Contents of /app/src: {list(src_path.iterdir())}")
        orchestrator_path = src_path / "orchestrator"
        if orchestrator_path.exists():
            logger.error(f"Contents of /app/src/orchestrator: {list(orchestrator_path.iterdir())}")
    
    return None


def main():
    """Main function to run the orchestrator."""
    logger.info("🎛️ Starting NIDS Central Orchestrator...")
    
    # Import the API server
    create_api_server = import_orchestrator()
    if create_api_server is None:
        logger.error("❌ Failed to import orchestrator API server")
        sys.exit(1)
    
    # Get configuration from environment variables
    host = os.getenv('ORCHESTRATOR_HOST', '0.0.0.0')
    port = int(os.getenv('ORCHESTRATOR_PORT', '8000'))
    log_level = os.getenv('LOG_LEVEL', 'info').lower()
    
    logger.info(f"🎛️ Starting NIDS Central Orchestrator on {host}:{port}")
    
    try:
        # Create and run the API server
        app = create_api_server()
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
