"""
NIDS Fog Node Package
Contains fog node implementation for edge deployment.
"""

"""
NIDS Fog Node Module
Lightweight edge processing nodes for distributed NIDS
"""

from .fog_node import FogNode
from .traffic_capture import TrafficCapture
from .mitigation import TrafficMitigation

# Import key components
try:
    from .fog_node import FogNode, FogNodeConfig
except ImportError:
    pass  # Will be imported when needed

__all__ = ['FogNode', 'TrafficCapture', 'TrafficMitigation', 'FogNodeConfig']
