"""
Central Orchestrator for NIDS fog computing architecture.
Manages fog nodes, aggregates models, and coordinates global policies.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import threading

from .model_aggregator import ModelAggregator

# Use absolute import instead of relative import
try:
    from models.ensemble import HybridNIDSModel
except ImportError:
    from src.models.ensemble import HybridNIDSModel

logger = logging.getLogger(__name__)


@dataclass
class FogNodeInfo:
    """Information about a registered fog node."""
    node_id: str
    capabilities: Dict[str, Any]
    last_heartbeat: float
    registration_time: float
    status: str = "active"
    metrics: Dict[str, Any] = None
    model_version: str = "1.0.0"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class SecurityEvent:
    """Security event reported by fog nodes."""
    event_id: str
    node_id: str
    timestamp: float
    event_type: str
    severity: str
    details: Dict[str, Any]
    resolved: bool = False


class CentralOrchestrator:
    """
    Central Orchestrator for distributed NIDS architecture.
    
    Responsibilities:
    - Manage fog node registration and health monitoring
    - Aggregate models from fog nodes using federated learning
    - Distribute updated global models to fog nodes
    - Coordinate global security policies
    - Collect and analyze system-wide metrics
    - Manage threat intelligence and security events
    """
    
    def __init__(self,
                 model_aggregation_interval: int = 3600,  # 1 hour
                 node_timeout: int = 300,  # 5 minutes
                 max_nodes: int = 1000,
                 enable_federated_learning: bool = True):
        """
        Initialize Central Orchestrator.
        
        Args:
            model_aggregation_interval: Interval for model aggregation (seconds)
            node_timeout: Timeout for considering nodes offline (seconds)
            max_nodes: Maximum number of fog nodes to manage
            enable_federated_learning: Enable federated learning aggregation
        """
        self.model_aggregation_interval = model_aggregation_interval
        self.node_timeout = node_timeout
        self.max_nodes = max_nodes
        self.enable_federated_learning = enable_federated_learning
        
        # Node management
        self.fog_nodes: Dict[str, FogNodeInfo] = {}
        self.nodes_lock = threading.Lock()
        
        # Model management
        self.model_aggregator = ModelAggregator() if enable_federated_learning else None
        self.global_model = None
        self.model_version = "1.0.0"
        self.pending_model_updates = {}
        
        # Security and monitoring
        self.security_events: List[SecurityEvent] = []
        self.global_metrics = defaultdict(list)
        self.threat_intelligence = {}
        self.global_policy = self._create_default_policy()
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_nodes_registered': 0,
            'active_nodes': 0,
            'models_aggregated': 0,
            'security_events_processed': 0,
            'start_time': time.time()
        }
        
        logger.info("Central Orchestrator initialized")
    
    def _create_default_policy(self) -> Dict[str, Any]:
        """Create default global security policy."""
        return {
            'version': '1.0.0',
            'anomaly_threshold': 0.1,
            'block_threshold': 0.8,
            'throttle_threshold': 0.5,
            'rate_limits': {
                'max_blocks_per_minute': 100,
                'max_throttles_per_minute': 200
            },
            'whitelist_ips': [],
            'blacklist_ips': [],
            'emergency_mode': False,
            'updated_at': time.time()
        }
    
    async def start(self):
        """Start the orchestrator and background tasks."""
        if self.is_running:
            logger.warning("Orchestrator is already running")
            return
        
        self.is_running = True
        logger.info("Starting Central Orchestrator...")
        
        # Initialize global model
        if self.enable_federated_learning:
            self.global_model = HybridNIDSModel()
            logger.info("Global model initialized")
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._node_health_monitor()),
            asyncio.create_task(self._model_aggregation_loop()),
            asyncio.create_task(self._metrics_aggregation_loop()),
            asyncio.create_task(self._security_event_processor())
        ]
        
        logger.info("Central Orchestrator started successfully")
    
    async def stop(self):
        """Stop the orchestrator and cleanup."""
        if not self.is_running:
            return
        
        logger.info("Stopping Central Orchestrator...")
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Central Orchestrator stopped")
    
    async def register_node(self, registration_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new fog node.
        
        Args:
            registration_data: Node registration information
            
        Returns:
            Registration response
        """
        node_id = registration_data.get('node_id')
        if not node_id:
            raise ValueError("Node ID is required for registration")
        
        with self.nodes_lock:
            if len(self.fog_nodes) >= self.max_nodes:
                raise ValueError(f"Maximum number of nodes ({self.max_nodes}) reached")
            
            # Create node info
            node_info = FogNodeInfo(
                node_id=node_id,
                capabilities=registration_data.get('capabilities', {}),
                last_heartbeat=time.time(),
                registration_time=time.time(),
                model_version=self.model_version
            )
            
            self.fog_nodes[node_id] = node_info
            self.stats['total_nodes_registered'] += 1
            self.stats['active_nodes'] = len([n for n in self.fog_nodes.values() if n.status == 'active'])
        
        logger.info(f"Registered fog node: {node_id}")
        
        # Prepare response
        response = {
            'status': 'registered',
            'node_id': node_id,
            'model_version': self.model_version,
            'global_policy': self.global_policy,
            'orchestrator_info': {
                'version': '1.0.0',
                'capabilities': ['model_aggregation', 'policy_management', 'threat_intelligence']
            }
        }
        
        return response
    
    async def handle_heartbeat(self, heartbeat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle heartbeat from fog node.
        
        Args:
            heartbeat_data: Heartbeat information
            
        Returns:
            Heartbeat response
        """
        node_id = heartbeat_data.get('node_id')
        if not node_id:
            raise ValueError("Node ID is required for heartbeat")
        
        with self.nodes_lock:
            if node_id not in self.fog_nodes:
                raise ValueError(f"Node {node_id} is not registered")
            
            # Update heartbeat timestamp
            self.fog_nodes[node_id].last_heartbeat = time.time()
            self.fog_nodes[node_id].status = 'active'
        
        # Check if model update is needed
        node_model_version = heartbeat_data.get('model_version', '1.0.0')
        model_update_available = node_model_version != self.model_version
        
        response = {
            'status': 'acknowledged',
            'model_update_available': model_update_available,
            'global_policy_version': self.global_policy['version'],
            'timestamp': time.time()
        }
        
        return response
    
    async def collect_metrics(self, metrics_data: Dict[str, Any]):
        """
        Collect metrics from fog node.
        
        Args:
            metrics_data: Metrics data from fog node
        """
        node_id = metrics_data.get('node_id')
        if not node_id:
            return
        
        with self.nodes_lock:
            if node_id in self.fog_nodes:
                self.fog_nodes[node_id].metrics = metrics_data
        
        # Store metrics for aggregation
        timestamp = metrics_data.get('timestamp', time.time())
        self.global_metrics[node_id].append({
            'timestamp': timestamp,
            'metrics': metrics_data
        })
        
        # Keep only recent metrics (last 24 hours)
        cutoff_time = time.time() - 86400
        self.global_metrics[node_id] = [
            m for m in self.global_metrics[node_id]
            if m['timestamp'] >= cutoff_time
        ]
    
    async def report_security_event(self, event_data: Dict[str, Any]):
        """
        Process security event from fog node.
        
        Args:
            event_data: Security event data
        """
        node_id = event_data.get('node_id')
        event_details = event_data.get('event', {})
        
        # Create security event
        event = SecurityEvent(
            event_id=f"{node_id}_{int(time.time())}_{len(self.security_events)}",
            node_id=node_id,
            timestamp=event_data.get('timestamp', time.time()),
            event_type=event_details.get('type', 'unknown'),
            severity=event_details.get('severity', 'medium'),
            details=event_details
        )
        
        self.security_events.append(event)
        self.stats['security_events_processed'] += 1
        
        logger.info(f"Security event reported: {event.event_type} from {node_id}")
        
        # Check if immediate action is needed
        await self._process_security_event(event)
    
    async def _process_security_event(self, event: SecurityEvent):
        """Process and potentially respond to security event."""
        # High severity events might trigger policy updates
        if event.severity == 'critical':
            # Add to blacklist if it's an IP-based attack
            if 'src_ip' in event.details:
                src_ip = event.details['src_ip']
                if src_ip not in self.global_policy['blacklist_ips']:
                    self.global_policy['blacklist_ips'].append(src_ip)
                    self.global_policy['updated_at'] = time.time()
                    logger.info(f"Added {src_ip} to global blacklist due to critical event")
        
        # Coordinate response across nodes if needed
        if event.event_type == 'coordinated_attack':
            await self._coordinate_attack_response(event)
    
    async def _coordinate_attack_response(self, event: SecurityEvent):
        """Coordinate response to coordinated attacks across fog nodes."""
        # This could involve updating policies, sharing threat intelligence, etc.
        attack_indicators = event.details.get('indicators', {})
        
        # Update threat intelligence
        self.threat_intelligence[event.event_id] = {
            'type': 'coordinated_attack',
            'indicators': attack_indicators,
            'first_seen': event.timestamp,
            'affected_nodes': [event.node_id],
            'severity': event.severity
        }
        
        logger.info(f"Coordinated attack response initiated for event {event.event_id}")
    
    async def get_model_update(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model update for a specific node.
        
        Args:
            node_id: ID of the requesting node
            
        Returns:
            Model update data or None
        """
        with self.nodes_lock:
            if node_id not in self.fog_nodes:
                return None
            
            node = self.fog_nodes[node_id]
            if node.model_version == self.model_version:
                return None  # No update needed
        
        # Prepare model update
        if self.global_model:
            model_update = {
                'type': 'full',
                'version': self.model_version,
                'model_data': self.global_model.get_model_info(),
                'timestamp': time.time(),
                'changelog': 'Federated learning update'
            }
            
            # Update node's model version
            with self.nodes_lock:
                self.fog_nodes[node_id].model_version = self.model_version
            
            return model_update
        
        return None
    
    async def upload_local_model(self, node_id: str, model_data: Dict[str, Any]) -> bool:
        """
        Accept local model from fog node for aggregation.
        
        Args:
            node_id: ID of the uploading node
            model_data: Local model data
            
        Returns:
            True if accepted successfully
        """
        if not self.enable_federated_learning:
            return False
        
        with self.nodes_lock:
            if node_id not in self.fog_nodes:
                return False
        
        # Store model for aggregation
        self.pending_model_updates[node_id] = {
            'model_data': model_data,
            'timestamp': time.time(),
            'node_capabilities': self.fog_nodes[node_id].capabilities
        }
        
        logger.info(f"Received local model from node {node_id}")
        return True
    
    async def _node_health_monitor(self):
        """Monitor fog node health and update status."""
        while self.is_running:
            try:
                current_time = time.time()
                offline_nodes = []
                
                with self.nodes_lock:
                    for node_id, node_info in self.fog_nodes.items():
                        if current_time - node_info.last_heartbeat > self.node_timeout:
                            if node_info.status == 'active':
                                node_info.status = 'offline'
                                offline_nodes.append(node_id)
                    
                    # Update active node count
                    self.stats['active_nodes'] = len([n for n in self.fog_nodes.values() if n.status == 'active'])
                
                if offline_nodes:
                    logger.warning(f"Nodes went offline: {offline_nodes}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Node health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _model_aggregation_loop(self):
        """Periodically aggregate models from fog nodes."""
        while self.is_running:
            try:
                if self.enable_federated_learning and self.pending_model_updates:
                    logger.info("Starting model aggregation...")
                    
                    # Perform federated aggregation
                    aggregated_model = await self.model_aggregator.aggregate_models(
                        list(self.pending_model_updates.values())
                    )
                    
                    if aggregated_model:
                        # Update global model
                        self.global_model = aggregated_model
                        
                        # Increment version
                        version_parts = self.model_version.split('.')
                        version_parts[-1] = str(int(version_parts[-1]) + 1)
                        self.model_version = '.'.join(version_parts)
                        
                        self.stats['models_aggregated'] += 1
                        
                        logger.info(f"Model aggregation completed, new version: {self.model_version}")
                        
                        # Clear pending updates
                        self.pending_model_updates.clear()
                
                await asyncio.sleep(self.model_aggregation_interval)
                
            except Exception as e:
                logger.error(f"Model aggregation error: {e}")
                await asyncio.sleep(self.model_aggregation_interval)
    
    async def _metrics_aggregation_loop(self):
        """Aggregate metrics from all fog nodes."""
        while self.is_running:
            try:
                # Aggregate metrics every 5 minutes
                await asyncio.sleep(300)
                
                # Process and aggregate metrics
                aggregated_metrics = self._aggregate_global_metrics()
                
                # Log system-wide statistics
                if aggregated_metrics:
                    logger.info(f"Global metrics: {json.dumps(aggregated_metrics, indent=2)}")
                
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
                await asyncio.sleep(300)
    
    def _aggregate_global_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics from all fog nodes."""
        if not self.global_metrics:
            return {}
        
        # Collect recent metrics from all nodes
        current_time = time.time()
        recent_cutoff = current_time - 300  # Last 5 minutes
        
        aggregated = {
            'timestamp': current_time,
            'active_nodes': self.stats['active_nodes'],
            'total_packets_processed': 0,
            'total_anomalies_detected': 0,
            'avg_processing_time': 0,
            'total_actions': {'ALLOW': 0, 'BLOCK': 0, 'THROTTLE': 0}
        }
        
        processing_times = []
        
        for node_id, metrics_list in self.global_metrics.items():
            recent_metrics = [m for m in metrics_list if m['timestamp'] >= recent_cutoff]
            
            for metric_entry in recent_metrics:
                metrics = metric_entry['metrics']
                performance = metrics.get('performance', {})
                
                aggregated['total_packets_processed'] += performance.get('packets_processed', 0)
                aggregated['total_anomalies_detected'] += performance.get('anomalies_detected', 0)
                
                if 'avg_processing_time' in performance:
                    processing_times.append(performance['avg_processing_time'])
                
                actions = metrics.get('actions', {})
                for action, count in actions.items():
                    if action in aggregated['total_actions']:
                        aggregated['total_actions'][action] += count
        
        if processing_times:
            aggregated['avg_processing_time'] = sum(processing_times) / len(processing_times)
        
        return aggregated
    
    async def _security_event_processor(self):
        """Process security events and update threat intelligence."""
        while self.is_running:
            try:
                # Process security events every minute
                await asyncio.sleep(60)
                
                # Analyze recent security events for patterns
                recent_events = [
                    event for event in self.security_events
                    if time.time() - event.timestamp < 3600  # Last hour
                ]
                
                if recent_events:
                    await self._analyze_security_patterns(recent_events)
                
            except Exception as e:
                logger.error(f"Security event processor error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_security_patterns(self, events: List[SecurityEvent]):
        """Analyze security events for patterns and coordinated attacks."""
        # Group events by type and source
        event_groups = defaultdict(list)
        
        for event in events:
            src_ip = event.details.get('src_ip', 'unknown')
            event_type = event.event_type
            key = f"{event_type}_{src_ip}"
            event_groups[key].append(event)
        
        # Look for coordinated attacks (multiple events from same source)
        for key, group_events in event_groups.items():
            if len(group_events) >= 5:  # Threshold for coordinated attack
                logger.warning(f"Potential coordinated attack detected: {key} ({len(group_events)} events)")
                
                # Update threat intelligence
                self.threat_intelligence[key] = {
                    'type': 'coordinated_attack',
                    'event_count': len(group_events),
                    'first_seen': min(e.timestamp for e in group_events),
                    'last_seen': max(e.timestamp for e in group_events),
                    'affected_nodes': list(set(e.node_id for e in group_events))
                }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_time = time.time()
        uptime = current_time - self.stats['start_time']
        
        with self.nodes_lock:
            node_status = {
                node_id: {
                    'status': node.status,
                    'last_heartbeat': node.last_heartbeat,
                    'model_version': node.model_version,
                    'capabilities': node.capabilities
                }
                for node_id, node in self.fog_nodes.items()
            }
        
        return {
            'orchestrator': {
                'status': 'running' if self.is_running else 'stopped',
                'uptime': uptime,
                'version': '1.0.0'
            },
            'statistics': self.stats,
            'nodes': node_status,
            'global_model': {
                'version': self.model_version,
                'federated_learning_enabled': self.enable_federated_learning,
                'pending_updates': len(self.pending_model_updates)
            },
            'security': {
                'total_events': len(self.security_events),
                'recent_events': len([e for e in self.security_events if current_time - e.timestamp < 3600]),
                'threat_intelligence_entries': len(self.threat_intelligence)
            },
            'policy': {
                'version': self.global_policy['version'],
                'emergency_mode': self.global_policy['emergency_mode'],
                'last_updated': self.global_policy['updated_at']
            }
        }
