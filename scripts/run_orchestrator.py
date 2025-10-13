#!/usr/bin/env python3
"""
Script to run the Central Orchestrator.
Portable version that works in any environment.
"""

import os
import sys
import logging
import asyncio
import uvicorn
from pathlib import Path

# Add multiple paths to ensure imports work in both Docker and local
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, "/app")  # Docker fallback
sys.path.insert(0, "/app/src")  # Docker fallback

# Debug: Print paths to verify
print(f"🔍 Python paths:")
for p in sys.path[:5]:
    print(f"  - {p}")

# Try importing with detailed error messages
try:
    from orchestrator.api_server import create_api_server
    print("✅ Successfully imported orchestrator.api_server")
except ImportError as e:
    print(f"❌ Failed to import orchestrator.api_server: {e}")
    print(f"📁 Looking for: {project_root / 'src' / 'orchestrator' / 'api_server.py'}")
    print(f"📁 File exists: {(project_root / 'src' / 'orchestrator' / 'api_server.py').exists()}")
    
    # Try alternative import
    try:
        from src.orchestrator.api_server import create_api_server
        print("✅ Successfully imported via src.orchestrator.api_server")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        sys.exit(1)

# Create logs directory if it doesn't exist
log_dir = Path('logs')
try:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'orchestrator.log'
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(str(log_file))
    ]
except (PermissionError, OSError) as e:
    print(f"⚠️ Could not create log file: {e}")
    print("   Using console logging only")
    handlers = [logging.StreamHandler()]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger(__name__)


def main():
    """Main function to run the orchestrator."""
    # Get configuration from environment variables
    host = os.getenv('ORCHESTRATOR_HOST', '0.0.0.0')
    port = int(os.getenv('ORCHESTRATOR_PORT', '8000'))
    log_level = os.getenv('LOG_LEVEL', 'info').lower()
    
    logger.info(f"🚀 Starting NIDS Central Orchestrator on {host}:{port}")
    
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
