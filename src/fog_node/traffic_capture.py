"""
Traffic capture module for real-time network traffic analysis.
Captures encrypted traffic metadata without payload inspection.
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, List
from queue import Queue
import numpy as np
import json

try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not available, using simulated traffic capture")

logger = logging.getLogger(__name__)


class TrafficCapture:
    """
    Real-time traffic capture for NIDS analysis.
    Extracts flow-level metadata without inspecting payload content.
    """
    
    def __init__(self, 
                 interface: str = "eth0",
                 queue: Optional[Queue] = None,
                 capture_filter: str = "tcp or udp",
                 max_packets_per_flow: int = 100,
                 flow_timeout: int = 300):
        """
        Initialize traffic capture.
        
        Args:
            interface: Network interface to capture from
            queue: Queue to store captured traffic data
            capture_filter: BPF filter for packet capture
            max_packets_per_flow: Maximum packets to track per flow
            flow_timeout: Flow timeout in seconds
        """
        self.interface = interface
        self.queue = queue or Queue()
        self.capture_filter = capture_filter
        self.max_packets_per_flow = max_packets_per_flow
        self.flow_timeout = flow_timeout
        
        # Flow tracking
        self.flows = {}  # flow_id -> flow_data
        self.flow_lock = threading.Lock()
        
        # Capture state
        self.is_capturing = False
        self.capture_thread = None
        self.cleanup_thread = None
        
        # Statistics
        self.stats = {
            'packets_captured': 0,
            'flows_created': 0,
            'flows_completed': 0,
            'bytes_processed': 0,
            'start_time': None
        }
        
        if not SCAPY_AVAILABLE:
            logger.warning("Scapy not available, will use simulated traffic")
    
    def start(self):
        """Start traffic capture."""
        if self.is_capturing:
            logger.warning("Traffic capture already running")
            return
        
        self.is_capturing = True
        self.stats['start_time'] = time.time()
        
        logger.info(f"Starting traffic capture on interface {self.interface}")
        
        # Start packet capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_packets,
            name="PacketCapture"
        )
        self.capture_thread.start()
        
        # Start flow cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_flows,
            name="FlowCleanup"
        )
        self.cleanup_thread.start()
        
        logger.info("Traffic capture started")
    
    def stop(self):
        """Stop traffic capture."""
        if not self.is_capturing:
            logger.warning("Traffic capture not running")
            return
        
        logger.info("Stopping traffic capture...")
        self.is_capturing = False
        
        # Wait for threads to complete
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        logger.info("Traffic capture stopped")
    
    def _capture_packets(self):
        """Main packet capture loop."""
        try:
            if SCAPY_AVAILABLE:
                self._real_packet_capture()
            else:
                self._simulated_packet_capture()
        except Exception as e:
            logger.error(f"Packet capture error: {e}")
        finally:
            logger.info("Packet capture thread stopped")
    
    def _real_packet_capture(self):
        """Real packet capture using Scapy."""
        def packet_handler(packet):
            if not self.is_capturing:
                return
            
            try:
                self._process_packet(packet)
            except Exception as e:
                logger.error(f"Error processing packet: {e}")
        
        # Start packet sniffing
        scapy.sniff(
            iface=self.interface,
            filter=self.capture_filter,
            prn=packet_handler,
            stop_filter=lambda x: not self.is_capturing
        )
    
    def _simulated_packet_capture(self):
        """Simulated packet capture for testing."""
        logger.info("Using simulated traffic capture")
        
        while self.is_capturing:
            try:
                # Generate simulated packet
                simulated_packet = self._generate_simulated_packet()
                self._process_simulated_packet(simulated_packet)
                
                # Control capture rate
                time.sleep(0.01)  # 100 packets per second
                
            except Exception as e:
                logger.error(f"Simulated capture error: {e}")
                time.sleep(1.0)
    
    def _process_packet(self, packet):
        """Process captured packet and extract features."""
        try:
            # Extract basic packet info
            packet_info = self._extract_packet_info(packet)
            if not packet_info:
                return
            
            # Update flow information
            flow_id = self._get_flow_id(packet_info)
            self._update_flow(flow_id, packet_info)
            
            # Update statistics
            self.stats['packets_captured'] += 1
            self.stats['bytes_processed'] += packet_info.get('length', 0)
            
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
    
    def _extract_packet_info(self, packet) -> Optional[Dict[str, Any]]:
        """Extract relevant information from packet."""
        try:
            info = {
                'timestamp': time.time(),
                'length': len(packet)
            }
            
            # IP layer information
            if IP in packet:
                ip_layer = packet[IP]
                info.update({
                    'src_ip': ip_layer.src,
                    'dst_ip': ip_layer.dst,
                    'protocol': ip_layer.proto,
                    'ttl': ip_layer.ttl,
                    'ip_flags': ip_layer.flags
                })
            elif IPv6 in packet:
                ipv6_layer = packet[IPv6]
                info.update({
                    'src_ip': ipv6_layer.src,
                    'dst_ip': ipv6_layer.dst,
                    'protocol': ipv6_layer.nh,
                    'hop_limit': ipv6_layer.hlim
                })
            else:
                return None
            
            # Transport layer information
            if TCP in packet:
                tcp_layer = packet[TCP]
                info.update({
                    'src_port': tcp_layer.sport,
                    'dst_port': tcp_layer.dport,
                    'tcp_flags': tcp_layer.flags,
                    'tcp_window': tcp_layer.window,
                    'tcp_seq': tcp_layer.seq,
                    'tcp_ack': tcp_layer.ack
                })
            elif UDP in packet:
                udp_layer = packet[UDP]
                info.update({
                    'src_port': udp_layer.sport,
                    'dst_port': udp_layer.dport
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Error extracting packet info: {e}")
            return None
    
    def _get_flow_id(self, packet_info: Dict[str, Any]) -> str:
        """Generate flow identifier from packet information."""
        src_ip = packet_info.get('src_ip', '')
        dst_ip = packet_info.get('dst_ip', '')
        src_port = packet_info.get('src_port', 0)
        dst_port = packet_info.get('dst_port', 0)
        protocol = packet_info.get('protocol', 0)
        
        # Create bidirectional flow ID
        if (src_ip, src_port) < (dst_ip, dst_port):
            flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        else:
            flow_id = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        
        return flow_id
    
    def _update_flow(self, flow_id: str, packet_info: Dict[str, Any]):
        """Update flow statistics with new packet."""
        with self.flow_lock:
            current_time = time.time()
            
            if flow_id not in self.flows:
                # Create new flow
                self.flows[flow_id] = {
                    'flow_id': flow_id,
                    'start_time': current_time,
                    'last_seen': current_time,
                    'packet_count': 0,
                    'byte_count': 0,
                    'packets': [],
                    'src_ip': packet_info.get('src_ip'),
                    'dst_ip': packet_info.get('dst_ip'),
                    'src_port': packet_info.get('src_port'),
                    'dst_port': packet_info.get('dst_port'),
                    'protocol': packet_info.get('protocol'),
                    'direction_stats': {'forward': 0, 'backward': 0}
                }
                self.stats['flows_created'] += 1
            
            flow = self.flows[flow_id]
            
            # Update flow statistics
            flow['last_seen'] = current_time
            flow['packet_count'] += 1
            flow['byte_count'] += packet_info.get('length', 0)
            
            # Determine packet direction
            is_forward = (packet_info.get('src_ip') == flow['src_ip'] and 
                         packet_info.get('src_port') == flow['src_port'])
            
            if is_forward:
                flow['direction_stats']['forward'] += 1
            else:
                flow['direction_stats']['backward'] += 1
            
            # Store packet info (limited to prevent memory issues)
            if len(flow['packets']) < self.max_packets_per_flow:
                flow['packets'].append({
                    'timestamp': packet_info['timestamp'],
                    'length': packet_info.get('length', 0),
                    'direction': 'forward' if is_forward else 'backward',
                    'tcp_flags': packet_info.get('tcp_flags'),
                    'tcp_window': packet_info.get('tcp_window')
                })
            
            # Check if flow should be processed
            if self._should_process_flow(flow):
                self._process_flow(flow_id, flow)
    
    def _should_process_flow(self, flow: Dict[str, Any]) -> bool:
        """Determine if flow should be processed and sent for analysis."""
        # Process flow if it has enough packets or has been idle
        min_packets = 10
        max_idle_time = 30  # seconds
        
        current_time = time.time()
        idle_time = current_time - flow['last_seen']
        
        return (flow['packet_count'] >= min_packets or 
                idle_time > max_idle_time or
                flow['packet_count'] >= self.max_packets_per_flow)
    
    def _process_flow(self, flow_id: str, flow: Dict[str, Any]):
        """Process completed flow and extract features."""
        try:
            # Extract flow features
            features = self._extract_flow_features(flow)
            
            # Create traffic data for analysis
            traffic_data = {
                'flow_id': flow_id,
                'features': features,
                'metadata': {
                    'src_ip': flow['src_ip'],
                    'dst_ip': flow['dst_ip'],
                    'src_port': flow['src_port'],
                    'dst_port': flow['dst_port'],
                    'protocol': flow['protocol'],
                    'start_time': flow['start_time'],
                    'duration': flow['last_seen'] - flow['start_time'],
                    'packet_count': flow['packet_count'],
                    'byte_count': flow['byte_count']
                },
                'timestamp': time.time()
            }
            
            # Add to processing queue
            if not self.queue.full():
                self.queue.put(traffic_data)
            else:
                logger.warning("Traffic queue full, dropping flow data")
            
            # Remove processed flow
            del self.flows[flow_id]
            self.stats['flows_completed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing flow {flow_id}: {e}")
    
    def _extract_flow_features(self, flow: Dict[str, Any]) -> List[float]:
        """Extract numerical features from flow for ML analysis."""
        try:
            packets = flow['packets']
            duration = flow['last_seen'] - flow['start_time']
            
            # Basic flow statistics
            features = [
                duration,  # flow_duration
                flow['direction_stats']['forward'],  # total_fwd_packets
                flow['direction_stats']['backward'],  # total_bwd_packets
                sum(p['length'] for p in packets if p['direction'] == 'forward'),  # total_length_fwd_packets
                sum(p['length'] for p in packets if p['direction'] == 'backward'),  # total_length_bwd_packets
            ]
            
            # Packet length statistics
            fwd_lengths = [p['length'] for p in packets if p['direction'] == 'forward']
            bwd_lengths = [p['length'] for p in packets if p['direction'] == 'backward']
            
            if fwd_lengths:
                features.extend([
                    max(fwd_lengths),  # fwd_packet_length_max
                    min(fwd_lengths),  # fwd_packet_length_min
                    np.mean(fwd_lengths),  # fwd_packet_length_mean
                    np.std(fwd_lengths) if len(fwd_lengths) > 1 else 0  # fwd_packet_length_std
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            if bwd_lengths:
                features.extend([
                    max(bwd_lengths),  # bwd_packet_length_max
                    min(bwd_lengths),  # bwd_packet_length_min
                    np.mean(bwd_lengths),  # bwd_packet_length_mean
                    np.std(bwd_lengths) if len(bwd_lengths) > 1 else 0  # bwd_packet_length_std
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Flow rates
            if duration > 0:
                features.extend([
                    flow['byte_count'] / duration,  # flow_bytes_s
                    flow['packet_count'] / duration,  # flow_packets_s
                ])
            else:
                features.extend([0, 0])
            
            # Inter-arrival times
            timestamps = [p['timestamp'] for p in packets]
            if len(timestamps) > 1:
                iats = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                features.extend([
                    np.mean(iats),  # flow_iat_mean
                    np.std(iats),   # flow_iat_std
                    max(iats),      # flow_iat_max
                    min(iats)       # flow_iat_min
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # TCP flags (if TCP)
            tcp_packets = [p for p in packets if p.get('tcp_flags') is not None]
            if tcp_packets:
                features.extend([
                    sum(1 for p in tcp_packets if p['tcp_flags'] & 0x08),  # psh_flag_count
                    sum(1 for p in tcp_packets if p['tcp_flags'] & 0x20),  # urg_flag_count
                    sum(1 for p in tcp_packets if p['tcp_flags'] & 0x01),  # fin_flag_count
                    sum(1 for p in tcp_packets if p['tcp_flags'] & 0x02),  # syn_flag_count
                    sum(1 for p in tcp_packets if p['tcp_flags'] & 0x04),  # rst_flag_count
                    sum(1 for p in tcp_packets if p['tcp_flags'] & 0x10),  # ack_flag_count
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
            
            # Pad or truncate to expected feature count (78 features)
            target_features = 78
            if len(features) < target_features:
                features.extend([0] * (target_features - len(features)))
            elif len(features) > target_features:
                features = features[:target_features]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting flow features: {e}")
            return [0] * 78  # Return zero features on error
    
    def _cleanup_flows(self):
        """Clean up expired flows."""
        while self.is_capturing:
            try:
                current_time = time.time()
                expired_flows = []
                
                with self.flow_lock:
                    for flow_id, flow in self.flows.items():
                        if current_time - flow['last_seen'] > self.flow_timeout:
                            expired_flows.append(flow_id)
                    
                    # Process and remove expired flows
                    for flow_id in expired_flows:
                        if flow_id in self.flows:
                            self._process_flow(flow_id, self.flows[flow_id])
                
                # Sleep before next cleanup
                time.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Flow cleanup error: {e}")
                time.sleep(60)
    
    def _generate_simulated_packet(self) -> Dict[str, Any]:
        """Generate simulated packet for testing."""
        import random
        
        return {
            'timestamp': time.time(),
            'length': random.randint(64, 1500),
            'src_ip': f"192.168.1.{random.randint(1, 254)}",
            'dst_ip': f"10.0.0.{random.randint(1, 254)}",
            'src_port': random.randint(1024, 65535),
            'dst_port': random.choice([80, 443, 22, 21, 25, 53]),
            'protocol': 6,  # TCP
            'tcp_flags': random.randint(0, 63),
            'tcp_window': random.randint(1024, 65535)
        }
    
    def _process_simulated_packet(self, packet_info: Dict[str, Any]):
        """Process simulated packet."""
        flow_id = self._get_flow_id(packet_info)
        self._update_flow(flow_id, packet_info)
        
        self.stats['packets_captured'] += 1
        self.stats['bytes_processed'] += packet_info.get('length', 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        current_time = time.time()
        uptime = current_time - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            'is_capturing': self.is_capturing,
            'uptime': uptime,
            'packets_captured': self.stats['packets_captured'],
            'flows_created': self.stats['flows_created'],
            'flows_completed': self.stats['flows_completed'],
            'bytes_processed': self.stats['bytes_processed'],
            'active_flows': len(self.flows),
            'packets_per_second': self.stats['packets_captured'] / uptime if uptime > 0 else 0,
            'bytes_per_second': self.stats['bytes_processed'] / uptime if uptime > 0 else 0
        }
