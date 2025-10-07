#!/usr/bin/env python3
"""
Demo script for NIDS-RL system.
Demonstrates the complete system functionality with simulated data.
"""

import os
import sys
import time
import logging
import asyncio
import threading
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ensemble import HybridNIDSModel
from utils.data_preprocessing import DataPreprocessor
from fog_node.fog_node import FogNode, FogNodeConfig
from orchestrator.orchestrator import CentralOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class NIDSDemo:
    """
    Demonstration of the complete NIDS-RL system.
    """
    
    def __init__(self):
        """Initialize demo components."""
        self.orchestrator = None
        self.fog_nodes = []
        self.demo_running = False
        
    async def setup_orchestrator(self):
        """Setup and start the central orchestrator."""
        logger.info("Setting up Central Orchestrator...")
        
        self.orchestrator = CentralOrchestrator(
            model_aggregation_interval=300,  # 5 minutes for demo
            node_timeout=60,  # 1 minute timeout
            enable_federated_learning=True
        )
        
        await self.orchestrator.start()
        logger.info("Central Orchestrator started")
    
    async def setup_fog_nodes(self, num_nodes: int = 3):
        """Setup and start fog nodes."""
        logger.info(f"Setting up {num_nodes} fog nodes...")
        
        for i in range(num_nodes):
            node_id = f"demo-fog-node-{i+1}"
            
            config = FogNodeConfig(
                node_id=node_id,
                orchestrator_url="http://localhost:8000",
                capture_interface="lo",  # Use loopback for demo
                enable_mitigation=False,  # Disable for demo safety
                model_update_interval=180,  # 3 minutes
                metrics_report_interval=30   # 30 seconds
            )
            
            fog_node = FogNode(config)
            await fog_node.initialize()
            
            # Start fog node in background
            fog_node.start()
            self.fog_nodes.append(fog_node)
            
            logger.info(f"Fog node {node_id} started")
            
            # Small delay between node startups
            await asyncio.sleep(2)
    
    def generate_demo_traffic(self):
        """Generate demo traffic for testing."""
        logger.info("Starting demo traffic generation...")
        
        import numpy as np
        import random
        
        def traffic_generator():
            """Generate simulated traffic data."""
            while self.demo_running:
                try:
                    # Generate random traffic features (78 features)
                    features = np.random.normal(0, 1, 78).astype(np.float32)
                    
                    # Add some realistic patterns
                    if random.random() < 0.1:  # 10% anomalous traffic
                        # Make some features more extreme for anomalies
                        anomaly_indices = random.sample(range(78), 10)
                        for idx in anomaly_indices:
                            features[idx] *= random.uniform(3, 5)
                    
                    # Simulate processing by fog nodes
                    for fog_node in self.fog_nodes:
                        if fog_node.model:
                            try:
                                result = fog_node.model.predict(features)
                                logger.debug(f"Node {fog_node.node_id}: Action={result['action_name']}, "
                                           f"Anomaly={result['is_anomaly']}")
                            except Exception as e:
                                logger.debug(f"Prediction error: {e}")
                    
                    time.sleep(0.1)  # 10 packets per second
                    
                except Exception as e:
                    logger.error(f"Traffic generation error: {e}")
                    time.sleep(1)
        
        # Start traffic generation in background thread
        self.traffic_thread = threading.Thread(target=traffic_generator, daemon=True)
        self.traffic_thread.start()
    
    async def run_demo_scenario(self):
        """Run a complete demo scenario."""
        logger.info("Starting NIDS-RL Demo Scenario...")
        
        try:
            # Phase 1: Setup
            logger.info("=== Phase 1: System Setup ===")
            await self.setup_orchestrator()
            await asyncio.sleep(2)
            
            await self.setup_fog_nodes(num_nodes=3)
            await asyncio.sleep(5)
            
            # Phase 2: Initial Operation
            logger.info("=== Phase 2: Initial Operation ===")
            self.demo_running = True
            self.generate_demo_traffic()
            
            # Let system run for a while
            logger.info("System running... generating traffic and collecting metrics")
            await asyncio.sleep(30)
            
            # Phase 3: Show system status
            logger.info("=== Phase 3: System Status ===")
            await self.show_system_status()
            
            # Phase 4: Simulate attack scenario
            logger.info("=== Phase 4: Attack Simulation ===")
            await self.simulate_attack_scenario()
            
            # Phase 5: Model aggregation demo
            logger.info("=== Phase 5: Model Aggregation ===")
            await self.demonstrate_model_aggregation()
            
            # Phase 6: Final status
            logger.info("=== Phase 6: Final Status ===")
            await self.show_system_status()
            
            logger.info("Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def show_system_status(self):
        """Display current system status."""
        if self.orchestrator:
            status = self.orchestrator.get_system_status()
            
            logger.info("=== System Status ===")
            logger.info(f"Orchestrator Status: {status['orchestrator']['status']}")
            logger.info(f"Active Nodes: {status['statistics']['active_nodes']}")
            logger.info(f"Total Nodes Registered: {status['statistics']['total_nodes_registered']}")
            logger.info(f"Security Events: {status['security']['total_events']}")
            logger.info(f"Model Version: {status['global_model']['version']}")
            
            # Show individual node status
            for node_id, node_info in status['nodes'].items():
                logger.info(f"Node {node_id}: {node_info['status']} "
                           f"(Model: {node_info['model_version']})")
    
    async def simulate_attack_scenario(self):
        """Simulate a coordinated attack scenario."""
        logger.info("Simulating coordinated attack...")
        
        # Generate attack events
        attack_events = []
        for i in range(5):
            event = {
                'type': 'coordinated_attack',
                'severity': 'critical',
                'src_ip': f"192.168.1.{100 + i}",
                'attack_type': 'ddos',
                'indicators': {
                    'packet_rate': 10000,
                    'target_ports': [80, 443, 22]
                }
            }
            attack_events.append(event)
        
        # Report events from different nodes
        for i, event in enumerate(attack_events):
            node_id = f"demo-fog-node-{(i % 3) + 1}"
            
            if self.orchestrator:
                await self.orchestrator.report_security_event({
                    'node_id': node_id,
                    'timestamp': time.time(),
                    'event': event
                })
            
            await asyncio.sleep(1)
        
        logger.info("Attack simulation completed")
        await asyncio.sleep(5)  # Let system process events
    
    async def demonstrate_model_aggregation(self):
        """Demonstrate federated model aggregation."""
        logger.info("Demonstrating model aggregation...")
        
        # Simulate model updates from fog nodes
        for i, fog_node in enumerate(self.fog_nodes):
            if fog_node.model:
                # Create mock model update
                model_update = {
                    'model_data': fog_node.model.get_model_info(),
                    'timestamp': time.time(),
                    'performance_metrics': {
                        'packets_processed': 1000 + i * 100,
                        'anomalies_detected': 50 + i * 10,
                        'avg_processing_time': 0.01 + i * 0.001
                    }
                }
                
                if self.orchestrator:
                    success = await self.orchestrator.upload_local_model(
                        fog_node.node_id, model_update
                    )
                    logger.info(f"Model upload from {fog_node.node_id}: {'Success' if success else 'Failed'}")
        
        # Wait for aggregation
        logger.info("Waiting for model aggregation...")
        await asyncio.sleep(10)
        
        # Check if new model version is available
        if self.orchestrator:
            logger.info(f"Current global model version: {self.orchestrator.model_version}")
    
    async def cleanup(self):
        """Clean up demo resources."""
        logger.info("Cleaning up demo resources...")
        
        self.demo_running = False
        
        # Stop fog nodes
        for fog_node in self.fog_nodes:
            try:
                fog_node.stop()
            except Exception as e:
                logger.error(f"Error stopping fog node: {e}")
        
        # Stop orchestrator
        if self.orchestrator:
            try:
                await self.orchestrator.stop()
            except Exception as e:
                logger.error(f"Error stopping orchestrator: {e}")
        
        logger.info("Cleanup completed")


async def main():
    """Main demo function."""
    print("=" * 60)
    print("NIDS-RL System Demonstration")
    print("=" * 60)
    print()
    print("This demo will showcase:")
    print("1. Central Orchestrator startup")
    print("2. Multiple Fog Nodes registration")
    print("3. Real-time traffic processing")
    print("4. Attack detection and response")
    print("5. Federated model aggregation")
    print()
    
    input("Press Enter to start the demo...")
    
    demo = NIDSDemo()
    
    try:
        await demo.run_demo_scenario()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    finally:
        await demo.cleanup()
    
    print()
    print("=" * 60)
    print("Demo completed. Thank you!")
    print("=" * 60)


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    # Run demo
    asyncio.run(main())
