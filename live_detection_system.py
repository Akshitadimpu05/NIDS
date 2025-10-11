#!/usr/bin/env python3
"""
Live Network Intrusion Detection and Mitigation System
Uses joint-trained NIDS model (Autoencoder + CapsNet) + RL agent for real-time decisions
"""

import os
import sys
import time
import threading
import queue
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import defaultdict, deque
import logging
import argparse

# Setup paths
sys.path.append('/home/tejasri/nids/NIDS')
sys.path.append('/home/tejasri/nids/NIDS/src')

from joint_training import JointNIDSModel
from src.agents.ppo_agent import PPOAgent
from src.utils.data_preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrafficAction:
    """Traffic action definitions."""
    ALLOW = 0
    BLOCK = 1
    THROTTLE = 2
    
    @classmethod
    def get_name(cls, action):
        names = {0: 'ALLOW', 1: 'BLOCK', 2: 'THROTTLE'}
        return names.get(action, 'UNKNOWN')

class NetworkFlowSimulator:
    """Simulates network traffic flows for demonstration."""
    
    def __init__(self, data_path='data/Darknet.CSV'):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_preprocessor('data/models/joint_preprocessor.pkl')
        self.flows = self._load_flows()
        self.current_idx = 0
        
    def _load_flows(self):
        """Load and preprocess network flows."""
        logger.info(f"Loading network flows from {self.data_path}")
        features_df, labels_df = self.preprocessor.load_cic_darknet2020(self.data_path)
        features = self.preprocessor.preprocess_features(features_df)
        labels = self.preprocessor.preprocess_labels(labels_df)
        
        # Create flow metadata
        flows = []
        class_names = ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
        
        for i in range(len(features)):
            flow = {
                'id': f"flow_{i:06d}",
                'timestamp': datetime.now().timestamp() + i * 0.1,  # Simulate timing
                'features': features[i],
                'true_label': labels[i],
                'true_class': class_names[labels[i]],
                'src_ip': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'dst_ip': f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'src_port': np.random.randint(1024, 65535),
                'dst_port': np.random.choice([80, 443, 22, 21, 25, 53, 993, 995])
            }
            flows.append(flow)
        
        logger.info(f"Loaded {len(flows)} network flows")
        return flows
    
    def get_next_flow(self):
        """Get next network flow."""
        if self.current_idx >= len(self.flows):
            self.current_idx = 0  # Loop back
        
        flow = self.flows[self.current_idx].copy()
        flow['timestamp'] = datetime.now().timestamp()  # Update to current time
        self.current_idx += 1
        return flow

class LiveNIDSDetector:
    """Live Network Intrusion Detection System."""
    
    def __init__(self, config_path='config/model_config.yaml'):
        self.config = self._load_config(config_path)
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_preprocessor('data/models/joint_preprocessor.pkl')
        
        # Load models
        self.joint_model = self._load_joint_model()
        self.rl_agent = self._load_rl_agent()
        
        # Detection statistics
        self.stats = {
            'total_flows': 0,
            'actions_taken': defaultdict(int),
            'class_predictions': defaultdict(int),
            'threats_detected': 0,
            'processing_times': deque(maxlen=1000),
            'start_time': datetime.now()
        }
        
        # Threat thresholds
        self.threat_threshold = 0.7  # Confidence threshold for threats
        
        logger.info("Live NIDS Detector initialized successfully")
    
    def _load_config(self, config_path):
        """Load configuration."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_joint_model(self):
        """Load joint-trained model."""
        logger.info("Loading joint NIDS model...")
        model = JointNIDSModel(54, self.config['autoencoder'], self.config['capsnet'])
        model.load_state_dict(torch.load('data/models/joint_nids_model.pth', map_location='cpu'))
        model.eval()
        logger.info("Joint model loaded successfully")
        return model
    
    def _load_rl_agent(self):
        """Load RL agent."""
        logger.info("Loading RL agent...")
        agent = PPOAgent(
            state_dim=8,  # 4D autoencoder + 4D capsnet
            action_dim=3,
            **{k: v for k, v in self.config['rl_agent'].items() if k in ['lr', 'gamma', 'eps_clip', 'k_epochs']}
        )
        agent.load_model('data/models/joint_rl_agent.pth')
        logger.info("RL agent loaded successfully")
        return agent
    
    def analyze_flow(self, flow):
        """Analyze a single network flow."""
        start_time = time.time()
        
        # Extract features
        features = flow['features'].reshape(1, -1)
        features_tensor = torch.FloatTensor(features)
        
        # Get joint model predictions
        with torch.no_grad():
            outputs = self.joint_model(features_tensor)
            
            # Classification prediction
            class_probs = torch.softmax(outputs['capsnet_pred'], dim=1).numpy()[0]
            predicted_class = np.argmax(class_probs)
            confidence = np.max(class_probs)
            
            # Anomaly score from autoencoder
            reconstruction_error = torch.nn.functional.mse_loss(
                outputs['ae_decoded'], features_tensor
            ).item()
            
            # Create RL state
            ae_features = outputs['ae_encoded'].numpy()[0]
            capsnet_features = outputs['capsnet_pred'].numpy()[0]
            rl_state = np.concatenate([ae_features, capsnet_features])
        
        # Get RL agent decision
        action, action_info = self.rl_agent.get_action(rl_state, deterministic=True)
        
        # Determine threat level
        class_names = ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
        predicted_class_name = class_names[predicted_class]
        
        is_threat = (
            predicted_class in [2, 3] and confidence > self.threat_threshold  # Tor or VPN with high confidence
            or reconstruction_error > 0.1  # High anomaly score
        )
        
        processing_time = time.time() - start_time
        
        # Create analysis result
        result = {
            'flow_id': flow['id'],
            'timestamp': flow['timestamp'],
            'src_ip': flow['src_ip'],
            'dst_ip': flow['dst_ip'],
            'src_port': flow['src_port'],
            'dst_port': flow['dst_port'],
            'predicted_class': predicted_class,
            'predicted_class_name': predicted_class_name,
            'confidence': float(confidence),
            'class_probabilities': class_probs.tolist(),
            'anomaly_score': float(reconstruction_error),
            'action': action,
            'action_name': TrafficAction.get_name(action),
            'is_threat': is_threat,
            'processing_time': processing_time,
            'true_class': flow['true_class'],
            'true_label': flow['true_label']
        }
        
        # Update statistics
        self.stats['total_flows'] += 1
        self.stats['actions_taken'][result['action_name']] += 1
        self.stats['class_predictions'][predicted_class_name] += 1
        self.stats['processing_times'].append(processing_time)
        
        if is_threat:
            self.stats['threats_detected'] += 1
        
        return result
    
    def apply_mitigation(self, result):
        """Apply mitigation action based on analysis result."""
        action_name = result['action_name']
        flow_info = f"{result['src_ip']}:{result['src_port']} -> {result['dst_ip']}:{result['dst_port']}"
        
        if action_name == 'BLOCK':
            logger.warning(f"🚫 BLOCKING flow {result['flow_id']} ({flow_info}) - "
                         f"Class: {result['predicted_class_name']}, "
                         f"Confidence: {result['confidence']:.3f}, "
                         f"Anomaly: {result['anomaly_score']:.3f}")
            # In real deployment: Add firewall rule, drop packets, etc.
            
        elif action_name == 'THROTTLE':
            logger.info(f"⚠️  THROTTLING flow {result['flow_id']} ({flow_info}) - "
                       f"Class: {result['predicted_class_name']}, "
                       f"Confidence: {result['confidence']:.3f}")
            # In real deployment: Apply rate limiting, QoS rules, etc.
            
        else:  # ALLOW
            logger.debug(f"✅ ALLOWING flow {result['flow_id']} ({flow_info}) - "
                        f"Class: {result['predicted_class_name']}")
            # In real deployment: Normal processing
    
    def get_statistics(self):
        """Get current detection statistics."""
        uptime = datetime.now() - self.stats['start_time']
        avg_processing_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_flows_processed': self.stats['total_flows'],
            'flows_per_second': self.stats['total_flows'] / max(uptime.total_seconds(), 1),
            'threats_detected': self.stats['threats_detected'],
            'threat_rate': self.stats['threats_detected'] / max(self.stats['total_flows'], 1),
            'actions_taken': dict(self.stats['actions_taken']),
            'class_predictions': dict(self.stats['class_predictions']),
            'avg_processing_time_ms': avg_processing_time * 1000,
            'total_processing_time_ms': sum(self.stats['processing_times']) * 1000
        }

class LiveDetectionSystem:
    """Main live detection system orchestrator."""
    
    def __init__(self, detection_rate=10):
        self.detector = LiveNIDSDetector()
        self.simulator = NetworkFlowSimulator()
        self.detection_rate = detection_rate  # flows per second
        self.running = False
        self.results_queue = queue.Queue()
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
    def start_detection(self, duration=None):
        """Start live detection system."""
        logger.info("🚀 Starting Live NIDS Detection System")
        logger.info(f"📊 Detection rate: {self.detection_rate} flows/second")
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Get next flow
                flow = self.simulator.get_next_flow()
                
                # Analyze flow
                result = self.detector.analyze_flow(flow)
                
                # Apply mitigation
                self.detector.apply_mitigation(result)
                
                # Store result
                self.results_queue.put(result)
                
                # Print periodic statistics
                if self.detector.stats['total_flows'] % 100 == 0:
                    self.print_statistics()
                
                # Rate limiting
                time.sleep(1.0 / self.detection_rate)
                
        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        finally:
            self.running = False
            logger.info("🛑 Live detection system stopped")
    
    def print_statistics(self):
        """Print current statistics."""
        stats = self.detector.get_statistics()
        
        print("\n" + "="*80)
        print("📊 LIVE NIDS DETECTION STATISTICS")
        print("="*80)
        print(f"⏱️  Uptime: {stats['uptime_seconds']:.1f}s")
        print(f"📈 Flows processed: {stats['total_flows_processed']}")
        print(f"🚀 Processing rate: {stats['flows_per_second']:.2f} flows/sec")
        print(f"⚠️  Threats detected: {stats['threats_detected']} ({stats['threat_rate']:.2%})")
        print(f"⚡ Avg processing time: {stats['avg_processing_time_ms']:.2f}ms")
        
        print("\n🎯 Actions Taken:")
        for action, count in stats['actions_taken'].items():
            percentage = count / stats['total_flows_processed'] * 100
            print(f"   {action}: {count} ({percentage:.1f}%)")
        
        print("\n🔍 Class Predictions:")
        for class_name, count in stats['class_predictions'].items():
            percentage = count / stats['total_flows_processed'] * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        print("="*80)
    
    def save_results(self, filename=None):
        """Save detection results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/detection_results_{timestamp}.json"
        
        results = []
        while not self.results_queue.empty():
            result = self.results_queue.get()
            # Convert numpy types to native Python types for JSON serialization
            json_result = {}
            for key, value in result.items():
                if isinstance(value, np.integer):
                    json_result[key] = int(value)
                elif isinstance(value, np.floating):
                    json_result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                else:
                    json_result[key] = value
            results.append(json_result)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Detection results saved to {filename}")
        return filename

def main():
    """Main function for live detection system."""
    parser = argparse.ArgumentParser(description='Live NIDS Detection System')
    parser.add_argument('--rate', type=int, default=10, help='Detection rate (flows/second)')
    parser.add_argument('--duration', type=int, help='Duration in seconds (default: run indefinitely)')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    # Create and start detection system
    system = LiveDetectionSystem(detection_rate=args.rate)
    
    try:
        system.start_detection(duration=args.duration)
    finally:
        # Print final statistics
        system.print_statistics()
        
        # Save results if requested
        if args.save_results:
            system.save_results()

if __name__ == "__main__":
    main()
