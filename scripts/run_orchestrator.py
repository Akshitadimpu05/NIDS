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

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.api_server import create_api_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/orchestrator.log')
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
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user")
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        sys.exit(1)
