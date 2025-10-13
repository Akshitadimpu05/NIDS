"""
Fog Node implementation for edge-based NIDS deployment.
Handles real-time traffic analysis and mitigation decisions.
"""

import asyncio
import threading
import time
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from queue import Queue, Empty
import numpy as np
import torch

# Use absolute imports instead of relative imports
try:
    from models.ensemble import HybridNIDSModel
except ImportError:
    from src.models.ensemble import HybridNIDSModel

from .traffic_capture import TrafficCapture
from .mitigation import TrafficMitigation

try:
    from utils.communication import OrchestratorClient
    from utils.metrics import MetricsCollector
except ImportError:
    from src.utils.communication import OrchestratorClient
    from src.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class FogNodeConfig:
    """Configuration for fog node."""
    node_id: str
    orchestrator_url: str = "http://localhost:8000"
    capture_interface: str = "eth0"
    model_update_interval: int = 300  # seconds
    metrics_report_interval: int = 60  # seconds
    max_queue_size: int = 10000
    batch_size: int = 32
    processing_threads: int = 4
    enable_mitigation: bool = True
    log_level: str = "INFO"


class FogNode:
    """
    Fog Node for distributed NIDS deployment.
    
    Responsibilities:
    - Real-time traffic capture and analysis
    - Local decision making using hybrid model
    - Traffic mitigation (block/throttle)
    - Communication with central orchestrator
    - Model updates and synchronization
    """
    
    def __init__(self, config: FogNodeConfig):
        """
        Initialize Fog Node.
        
        Args:
            config: Fog node configuration
        """
        self.config = config
        self.node_id = config.node_id
        self.is_running = False
        
        # Initialize components
        self.joint_model = None
        self.traffic_capture = None
        self.mitigation = None
        self.orchestrator_client = None
        self.metrics_collector = None
        
        # Processing queues
        self.traffic_queue = Queue(maxsize=config.max_queue_size)
        self.decision_queue = Queue(maxsize=config.max_queue_size)
        
        # Threading
        self.threads = []
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.stats = {
            'packets_processed': 0,
            'decisions_made': 0,
            'anomalies_detected': 0,
            'actions_taken': {'ALLOW': 0, 'BLOCK': 0, 'THROTTLE': 0},
            'processing_times': [],
            'model_updates': 0,
            'start_time': time.time()
        }
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        logger.info(f"Fog Node {self.node_id} initialized")
    
    async def initialize(self):
        """Initialize all fog node components."""
        try:
            # Initialize joint model
            logger.info("Initializing NIDS model...")
            self._load_joint_model()
            
            # Initialize traffic capture
            logger.info("Initializing traffic capture...")
            self.traffic_capture = TrafficCapture(
                interface=self.config.capture_interface,
                queue=self.traffic_queue
            )
            
            # Initialize mitigation if enabled
            if self.config.enable_mitigation:
                logger.info("Initializing traffic mitigation...")
                self.mitigation = TrafficMitigation()
            
            # Initialize orchestrator client
            logger.info("Initializing orchestrator client...")
            self.orchestrator_client = OrchestratorClient(
                orchestrator_url=self.config.orchestrator_url,
                node_id=self.node_id
            )
            
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector(node_id=self.node_id)
            
            # Register with orchestrator
            await self.register_with_orchestrator()
            
            logger.info("Fog node initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize fog node: {e}")
            raise
    
    async def register_with_orchestrator(self):
        """Register this fog node with the central orchestrator."""
        registration_data = {
            'node_id': self.node_id,
            'capabilities': {
                'processing_threads': self.config.processing_threads,
                'max_queue_size': self.config.max_queue_size,
                'mitigation_enabled': self.config.enable_mitigation
            },
            'model_info': self.joint_model.get_model_info() if self.joint_model else None,
            'timestamp': time.time()
        }
        
        try:
            response = await self.orchestrator_client.register_node(registration_data)
            logger.info(f"Successfully registered with orchestrator: {response}")
        except Exception as e:
            logger.error(f"Failed to register with orchestrator: {e}")
    
    def start(self):
        """Start the fog node operation."""
        if self.is_running:
            logger.warning("Fog node is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        logger.info(f"Starting fog node {self.node_id}...")
        
        # Start traffic capture
        capture_thread = threading.Thread(
            target=self._run_traffic_capture,
            name=f"TrafficCapture-{self.node_id}"
        )
        capture_thread.start()
        self.threads.append(capture_thread)
        
        # Start processing threads
        for i in range(self.config.processing_threads):
            process_thread = threading.Thread(
                target=self._run_traffic_processing,
                name=f"TrafficProcessor-{self.node_id}-{i}"
            )
            process_thread.start()
            self.threads.append(process_thread)
        
        # Start mitigation thread
        if self.mitigation:
            mitigation_thread = threading.Thread(
                target=self._run_mitigation,
                name=f"Mitigation-{self.node_id}"
            )
            mitigation_thread.start()
            self.threads.append(mitigation_thread)
        
        # Start model update thread
        update_thread = threading.Thread(
            target=self._run_model_updates,
            name=f"ModelUpdate-{self.node_id}"
        )
        update_thread.start()
        self.threads.append(update_thread)
        
        # Start metrics reporting thread
        metrics_thread = threading.Thread(
            target=self._run_metrics_reporting,
            name=f"Metrics-{self.node_id}"
        )
        metrics_thread.start()
        self.threads.append(metrics_thread)
        
        logger.info(f"Fog node {self.node_id} started with {len(self.threads)} threads")
    
    def stop(self):
        """Stop the fog node operation."""
        if not self.is_running:
            logger.warning("Fog node is not running")
            return
        
        logger.info(f"Stopping fog node {self.node_id}...")
        
        self.is_running = False
        self.stop_event.set()
        
        # Stop traffic capture
        if self.traffic_capture:
            self.traffic_capture.stop()
        
        # Wait for all threads to complete
        for thread in self.threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"Thread {thread.name} did not stop gracefully")
        
        self.threads.clear()
        logger.info(f"Fog node {self.node_id} stopped")
    
    def _run_traffic_capture(self):
        """Run traffic capture in dedicated thread."""
        logger.info("Starting traffic capture thread")
        
        try:
            self.traffic_capture.start()
            
            while not self.stop_event.is_set():
                time.sleep(0.1)  # Keep thread alive
                
        except Exception as e:
            logger.error(f"Traffic capture thread error: {e}")
        finally:
            if self.traffic_capture:
                self.traffic_capture.stop()
            logger.info("Traffic capture thread stopped")
    
    def _run_traffic_processing(self):
        """Run traffic processing in dedicated thread."""
        logger.info("Starting traffic processing thread")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get traffic data from queue
                    traffic_data = self.traffic_queue.get(timeout=1.0)
                    
                    # Process traffic
                    start_time = time.time()
                    result = self._process_traffic(traffic_data)
                    processing_time = time.time() - start_time
                    
                    # Update statistics
                    self.stats['packets_processed'] += 1
                    self.stats['processing_times'].append(processing_time)
                    
                    # Keep only recent processing times
                    if len(self.stats['processing_times']) > 1000:
                        self.stats['processing_times'] = self.stats['processing_times'][-1000:]
                    
                    # Queue decision for mitigation
                    if result and self.mitigation:
                        self.decision_queue.put({
                            'traffic_data': traffic_data,
                            'decision': result,
                            'timestamp': time.time()
                        })
                    
                    self.traffic_queue.task_done()
                    
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Traffic processing error: {e}")
                    
        except Exception as e:
            logger.error(f"Traffic processing thread error: {e}")
        finally:
            logger.info("Traffic processing thread stopped")
    
    def _process_traffic(self, traffic_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process individual traffic sample.
        
        Args:
            traffic_data: Traffic features and metadata
            
        Returns:
            Processing result with decision
        """
        try:
            # Extract features
            features = traffic_data.get('features')
            if features is None:
                return None
            
            # Convert to numpy array
            if isinstance(features, list):
                features = np.array(features, dtype=np.float32)
            
            # Make prediction using joint model
            result = self.joint_model.predict(features, deterministic=True)
            
            # Update statistics
            self.stats['decisions_made'] += 1
            self.stats['actions_taken'][result['action_name']] += 1
            
            if result['is_anomaly']:
                self.stats['anomalies_detected'] += 1
            
            # Add metadata
            result.update({
                'node_id': self.node_id,
                'timestamp': time.time(),
                'traffic_metadata': traffic_data.get('metadata', {})
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing traffic: {e}")
            return None
    
    def _run_mitigation(self):
        """Run traffic mitigation in dedicated thread."""
        logger.info("Starting mitigation thread")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get decision from queue
                    decision_data = self.decision_queue.get(timeout=1.0)
                    
                    # Execute mitigation action
                    self.mitigation.execute_action(
                        action=decision_data['decision']['action'],
                        traffic_data=decision_data['traffic_data'],
                        metadata=decision_data['decision']
                    )
                    
                    self.decision_queue.task_done()
                    
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Mitigation error: {e}")
                    
        except Exception as e:
            logger.error(f"Mitigation thread error: {e}")
        finally:
            logger.info("Mitigation thread stopped")
    
    def _run_model_updates(self):
        """Run model updates in dedicated thread."""
        logger.info("Starting model update thread")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Check for model updates from orchestrator
                    update_available = loop.run_until_complete(
                        self.orchestrator_client.check_model_update()
                    )
                    
                    if update_available:
                        logger.info("Model update available, downloading...")
                        model_data = loop.run_until_complete(
                            self.orchestrator_client.download_model_update()
                        )
                        
                        if model_data:
                            self._apply_model_update(model_data)
                            self.stats['model_updates'] += 1
                            logger.info("Model update applied successfully")
                    
                except Exception as e:
                    logger.error(f"Model update error: {e}")
                
                # Wait for next update check
                self.stop_event.wait(self.config.model_update_interval)
                
        except Exception as e:
            logger.error(f"Model update thread error: {e}")
        finally:
            loop.close()
            logger.info("Model update thread stopped")
    
    def _apply_model_update(self, model_data: Dict[str, Any]):
        """Apply model update from orchestrator."""
        try:
            # Save current model as backup
            backup_path = f"data/models/backup_{self.node_id}_{int(time.time())}.pth"
            self.joint_model.save_model(backup_path)
            
            # Apply update based on type
            update_type = model_data.get('type', 'full')
            
            if update_type == 'full':
                # Full model replacement
                model_path = model_data['model_path']
                self.joint_model.load_model(model_path)
                
            elif update_type == 'weights':
                # Weight updates only
                weights = model_data['weights']
                self._update_model_weights(weights)
                
            elif update_type == 'rl_policy':
                # RL policy update only
                policy_data = model_data['policy_data']
                self.joint_model.rl_agent.load_model(policy_data)
            
            logger.info(f"Applied {update_type} model update")
            
        except Exception as e:
            logger.error(f"Failed to apply model update: {e}")
            # Restore from backup if needed
    
    def _update_model_weights(self, weights: Dict[str, Any]):
        """Update specific model weights."""
        try:
            if 'autoencoder' in weights:
                self.joint_model.autoencoder.load_state_dict(weights['autoencoder'])
            
            if 'capsnet' in weights:
                self.joint_model.capsnet.load_state_dict(weights['capsnet'])
            
            if 'rl_agent' in weights:
                self.joint_model.rl_agent.network.load_state_dict(weights['rl_agent'])
                
        except Exception as e:
            logger.error(f"Failed to update model weights: {e}")
            raise
    
    def _run_metrics_reporting(self):
        """Run metrics reporting in dedicated thread."""
        logger.info("Starting metrics reporting thread")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Collect current metrics
                    metrics = self._collect_metrics()
                    
                    # Send to orchestrator
                    loop.run_until_complete(
                        self.orchestrator_client.report_metrics(metrics)
                    )
                    
                    # Update local metrics collector
                    self.metrics_collector.update(metrics)
                    
                except Exception as e:
                    logger.error(f"Metrics reporting error: {e}")
                
                # Wait for next report
                self.stop_event.wait(self.config.metrics_report_interval)
                
        except Exception as e:
            logger.error(f"Metrics reporting thread error: {e}")
        finally:
            loop.close()
            logger.info("Metrics reporting thread stopped")
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current node metrics."""
        current_time = time.time()
        uptime = current_time - self.stats['start_time']
        
        # Calculate rates
        packets_per_second = self.stats['packets_processed'] / uptime if uptime > 0 else 0
        decisions_per_second = self.stats['decisions_made'] / uptime if uptime > 0 else 0
        
        # Calculate average processing time
        avg_processing_time = (
            np.mean(self.stats['processing_times']) 
            if self.stats['processing_times'] else 0
        )
        
        # Queue utilization
        traffic_queue_util = self.traffic_queue.qsize() / self.config.max_queue_size
        decision_queue_util = self.decision_queue.qsize() / self.config.max_queue_size
        
        metrics = {
            'node_id': self.node_id,
            'timestamp': current_time,
            'uptime': uptime,
            'performance': {
                'packets_processed': self.stats['packets_processed'],
                'decisions_made': self.stats['decisions_made'],
                'packets_per_second': packets_per_second,
                'decisions_per_second': decisions_per_second,
                'avg_processing_time': avg_processing_time,
                'anomalies_detected': self.stats['anomalies_detected']
            },
            'actions': self.stats['actions_taken'].copy(),
            'queues': {
                'traffic_queue_size': self.traffic_queue.qsize(),
                'decision_queue_size': self.decision_queue.qsize(),
                'traffic_queue_utilization': traffic_queue_util,
                'decision_queue_utilization': decision_queue_util
            },
            'model': {
                'updates_received': self.stats['model_updates'],
                'model_info': self.joint_model.get_model_info() if self.joint_model else None
            }
        }
        
        return metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get current fog node status."""
        return {
            'node_id': self.node_id,
            'is_running': self.is_running,
            'threads_active': len([t for t in self.threads if t.is_alive()]),
            'stats': self.stats.copy(),
            'config': {
                'orchestrator_url': self.config.orchestrator_url,
                'capture_interface': self.config.capture_interface,
                'processing_threads': self.config.processing_threads,
                'mitigation_enabled': self.config.enable_mitigation
            }
        }
    
    def _load_joint_model(self):
        """Load joint-trained NIDS model."""
        try:
            # Import joint model
            import sys
            sys.path.append('/app')
            from joint_training import JointNIDSModel
            from src.agents.ppo_agent import PPOAgent
            from src.utils.data_preprocessing import DataPreprocessor
            import yaml
            import os
            
            # Load model configuration
            config_path = '/app/config/docker_config.yaml'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # Fallback configuration
                config = {
                    'autoencoder': {'hidden_dims': [32, 16], 'latent_dim': 4, 'dropout_rate': 0.2},
                    'capsnet': {'primary_caps_dim': 8, 'primary_caps_num': 32, 'digit_caps_dim': 16, 'digit_caps_num': 4, 'routing_iterations': 3},
                    'rl_agent': {'lr': 3e-4, 'gamma': 0.99, 'eps_clip': 0.2, 'k_epochs': 4}
                }
            
            # Initialize joint model
            logger.info("Loading joint NIDS model...")
            self.joint_model = JointNIDSModel(54, config['autoencoder'], config['capsnet'])
            
            # Load trained weights
            model_path = '/app/data/models/docker_hybrid_model.pth'
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Create joint model state dict from separate components
                joint_state = {}
                
                # Add autoencoder weights
                for key, value in checkpoint['autoencoder_state_dict'].items():
                    joint_state[f'autoencoder.{key}'] = value
                
                # Add CapsNet weights
                for key, value in checkpoint['capsnet_state_dict'].items():
                    joint_state[f'capsnet.{key}'] = value
                
                self.joint_model.load_state_dict(joint_state)
                self.joint_model.eval()
                
                # Load RL agent
                logger.info("Loading RL agent...")
                self.rl_agent = PPOAgent(
                    state_dim=8,  # 4D autoencoder + 4D capsnet
                    action_dim=3,
                    **{k: v for k, v in config['rl_agent'].items() if k in ['lr', 'gamma', 'eps_clip', 'k_epochs']}
                )
                
                # Load RL agent weights
                if 'rl_agent_checkpoint' in checkpoint:
                    self.rl_agent.network.load_state_dict(checkpoint['rl_agent_checkpoint']['network_state_dict'])
                    self.rl_agent.optimizer.load_state_dict(checkpoint['rl_agent_checkpoint']['optimizer_state_dict'])
                
                # Load preprocessor
                preprocessor_path = '/app/data/models/joint_preprocessor.pkl'
                if os.path.exists(preprocessor_path):
                    self.preprocessor = DataPreprocessor()
                    self.preprocessor.load_preprocessor(preprocessor_path)
                    logger.info("✅ Preprocessor loaded successfully")
                else:
                    logger.warning("Preprocessor not found, using default")
                    self.preprocessor = DataPreprocessor()
                
                logger.info("✅ Joint NIDS model loaded successfully")
                return True
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load joint model: {e}")
            import traceback
            traceback.print_exc()
            return False


# Utility function for easy fog node deployment
def create_fog_node(node_id: str, **kwargs) -> FogNode:
    """
    Create and configure a fog node.
    
    Args:
        node_id: Unique identifier for the fog node
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured FogNode instance
    """
    config = FogNodeConfig(node_id=node_id, **kwargs)
    return FogNode(config)
