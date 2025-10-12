"""
NIDS Central Orchestrator Package
Contains the central orchestrator for coordinating fog nodes and model updates.
"""

"""
NIDS Orchestrator Module
Central coordinator for distributed fog nodes
"""

from .orchestrator import CentralOrchestrator
from .model_aggregator import ModelAggregator

# Import key components for easier access
try:
    from .api_server import create_api_server
except ImportError:
    pass  # Will be imported when needed

__all__ = ['CentralOrchestrator', 'ModelAggregator', 'create_api_server']
