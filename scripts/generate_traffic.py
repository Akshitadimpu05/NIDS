#!/usr/bin/env python3
"""
Traffic generator for testing NIDS system.
Generates both normal and attack traffic patterns.
"""

import os
import sys
import time
import random
import logging
import asyncio
import socket
import threading
from pathlib import Path
from typing import List, Dict, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class TrafficGenerator:
    """
    Generates realistic network traffic for testing NIDS.
    Includes both benign and malicious traffic patterns.
    """
    
    def __init__(self,
                 target_networks: List[str] = None,
                 generation_rate: int = 100,
                 attack_probability: float = 0.1):
        """
        Initialize traffic generator.
        
        Args:
            target_networks: List of target network ranges
            generation_rate: Packets per second to generate
            attack_probability: Probability of generating attack traffic
        """
        self.target_networks = target_networks or ['172.20.0.0/16']
        self.generation_rate = generation_rate
        self.attack_probability = attack_probability
        
        # Traffic patterns
        self.benign_patterns = [
            'web_browsing',
            'email',
            'file_transfer',
            'video_streaming',
            'social_media'
        ]
        
        self.attack_patterns = [
            'port_scan',
            'ddos',
            'brute_force',
            'malware_communication',
            'data_exfiltration'
        ]
        
        # Statistics
        self.stats = {
            'total_generated': 0,
            'benign_generated': 0,
            'attack_generated': 0,
            'start_time': time.time()
        }
        
        self.running = False
        
    def start(self):
        """Start traffic generation."""
        if self.running:
            logger.warning("Traffic generator already running")
            return
        
        self.running = True
        logger.info(f"Starting traffic generator at {self.generation_rate} pps")
        logger.info(f"Attack probability: {self.attack_probability}")
        
        # Start generation threads
        self.generation_thread = threading.Thread(
            target=self._generation_loop,
            name="TrafficGeneration"
        )
        self.generation_thread.start()
        
        # Start statistics thread
        self.stats_thread = threading.Thread(
            target=self._stats_loop,
            name="StatsReporting"
        )
        self.stats_thread.start()
    
    def stop(self):
        """Stop traffic generation."""
        if not self.running:
            return
        
        logger.info("Stopping traffic generator...")
        self.running = False
        
        if hasattr(self, 'generation_thread'):
            self.generation_thread.join(timeout=5.0)
        
        if hasattr(self, 'stats_thread'):
            self.stats_thread.join(timeout=5.0)
        
        logger.info("Traffic generator stopped")
    
    def _generation_loop(self):
        """Main traffic generation loop."""
        interval = 1.0 / self.generation_rate
        
        while self.running:
            try:
                # Generate traffic packet
                if random.random() < self.attack_probability:
                    self._generate_attack_traffic()
                    self.stats['attack_generated'] += 1
                else:
                    self._generate_benign_traffic()
                    self.stats['benign_generated'] += 1
                
                self.stats['total_generated'] += 1
                
                # Control generation rate
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Traffic generation error: {e}")
                time.sleep(1.0)
    
    def _generate_benign_traffic(self):
        """Generate benign traffic pattern."""
        pattern = random.choice(self.benign_patterns)
        
        if pattern == 'web_browsing':
            self._generate_http_traffic()
        elif pattern == 'email':
            self._generate_email_traffic()
        elif pattern == 'file_transfer':
            self._generate_ftp_traffic()
        elif pattern == 'video_streaming':
            self._generate_streaming_traffic()
        elif pattern == 'social_media':
            self._generate_social_traffic()
    
    def _generate_attack_traffic(self):
        """Generate attack traffic pattern."""
        pattern = random.choice(self.attack_patterns)
        
        if pattern == 'port_scan':
            self._generate_port_scan()
        elif pattern == 'ddos':
            self._generate_ddos_traffic()
        elif pattern == 'brute_force':
            self._generate_brute_force()
        elif pattern == 'malware_communication':
            self._generate_malware_traffic()
        elif pattern == 'data_exfiltration':
            self._generate_exfiltration_traffic()
    
    def _generate_http_traffic(self):
        """Generate HTTP web browsing traffic."""
        src_ip = self._random_ip()
        dst_ip = self._random_external_ip()
        
        # Simulate HTTP request/response
        self._send_tcp_packet(src_ip, dst_ip, 
                             src_port=random.randint(1024, 65535),
                             dst_port=80,
                             payload_size=random.randint(100, 1500))
    
    def _generate_email_traffic(self):
        """Generate email traffic (SMTP/IMAP)."""
        src_ip = self._random_ip()
        dst_ip = self._random_external_ip()
        
        # SMTP or IMAP
        dst_port = random.choice([25, 587, 993, 143])
        
        self._send_tcp_packet(src_ip, dst_ip,
                             src_port=random.randint(1024, 65535),
                             dst_port=dst_port,
                             payload_size=random.randint(200, 2000))
    
    def _generate_ftp_traffic(self):
        """Generate FTP file transfer traffic."""
        src_ip = self._random_ip()
        dst_ip = self._random_external_ip()
        
        self._send_tcp_packet(src_ip, dst_ip,
                             src_port=random.randint(1024, 65535),
                             dst_port=21,
                             payload_size=random.randint(500, 1500))
    
    def _generate_streaming_traffic(self):
        """Generate video streaming traffic."""
        src_ip = self._random_ip()
        dst_ip = self._random_external_ip()
        
        # High bandwidth, consistent flow
        for _ in range(random.randint(5, 15)):
            self._send_udp_packet(src_ip, dst_ip,
                                 src_port=random.randint(1024, 65535),
                                 dst_port=random.choice([1935, 8080]),
                                 payload_size=random.randint(1200, 1500))
    
    def _generate_social_traffic(self):
        """Generate social media traffic."""
        src_ip = self._random_ip()
        dst_ip = self._random_external_ip()
        
        self._send_tcp_packet(src_ip, dst_ip,
                             src_port=random.randint(1024, 65535),
                             dst_port=443,  # HTTPS
                             payload_size=random.randint(300, 800))
    
    def _generate_port_scan(self):
        """Generate port scanning attack."""
        src_ip = self._random_external_ip()
        dst_ip = self._random_ip()
        
        # Scan multiple ports rapidly
        for port in random.sample(range(1, 1024), random.randint(5, 20)):
            self._send_tcp_packet(src_ip, dst_ip,
                                 src_port=random.randint(1024, 65535),
                                 dst_port=port,
                                 payload_size=0,
                                 tcp_flags='S')  # SYN scan
    
    def _generate_ddos_traffic(self):
        """Generate DDoS attack traffic."""
        # Multiple sources targeting same destination
        dst_ip = self._random_ip()
        
        for _ in range(random.randint(10, 50)):
            src_ip = self._random_external_ip()
            self._send_udp_packet(src_ip, dst_ip,
                                 src_port=random.randint(1024, 65535),
                                 dst_port=random.choice([53, 80, 443]),
                                 payload_size=random.randint(64, 512))
    
    def _generate_brute_force(self):
        """Generate brute force attack."""
        src_ip = self._random_external_ip()
        dst_ip = self._random_ip()
        
        # Multiple login attempts
        for _ in range(random.randint(5, 15)):
            self._send_tcp_packet(src_ip, dst_ip,
                                 src_port=random.randint(1024, 65535),
                                 dst_port=22,  # SSH
                                 payload_size=random.randint(100, 300))
    
    def _generate_malware_traffic(self):
        """Generate malware communication traffic."""
        src_ip = self._random_ip()
        dst_ip = self._random_external_ip()
        
        # Suspicious communication patterns
        self._send_tcp_packet(src_ip, dst_ip,
                             src_port=random.randint(1024, 65535),
                             dst_port=random.choice([6667, 8080, 9999]),
                             payload_size=random.randint(50, 200))
    
    def _generate_exfiltration_traffic(self):
        """Generate data exfiltration traffic."""
        src_ip = self._random_ip()
        dst_ip = self._random_external_ip()
        
        # Large data transfers to external hosts
        for _ in range(random.randint(3, 8)):
            self._send_tcp_packet(src_ip, dst_ip,
                                 src_port=random.randint(1024, 65535),
                                 dst_port=random.choice([80, 443, 8080]),
                                 payload_size=random.randint(1000, 1500))
    
    def _send_tcp_packet(self, src_ip: str, dst_ip: str,
                        src_port: int, dst_port: int,
                        payload_size: int, tcp_flags: str = 'A'):
        """Simulate sending TCP packet."""
        # In a real implementation, this would use scapy or raw sockets
        # For container environment, we'll just log the packet
        logger.debug(f"TCP: {src_ip}:{src_port} -> {dst_ip}:{dst_port} "
                    f"[{tcp_flags}] size={payload_size}")
    
    def _send_udp_packet(self, src_ip: str, dst_ip: str,
                        src_port: int, dst_port: int,
                        payload_size: int):
        """Simulate sending UDP packet."""
        logger.debug(f"UDP: {src_ip}:{src_port} -> {dst_ip}:{dst_port} "
                    f"size={payload_size}")
    
    def _random_ip(self) -> str:
        """Generate random IP from target networks."""
        # Simplified - generate IPs in 172.20.x.x range
        return f"172.20.{random.randint(1, 254)}.{random.randint(1, 254)}"
    
    def _random_external_ip(self) -> str:
        """Generate random external IP address."""
        # Generate IPs outside target networks
        return f"{random.randint(1, 223)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
    
    def _stats_loop(self):
        """Report statistics periodically."""
        while self.running:
            try:
                time.sleep(60)  # Report every minute
                
                if self.stats['total_generated'] > 0:
                    uptime = time.time() - self.stats['start_time']
                    rate = self.stats['total_generated'] / uptime
                    attack_rate = self.stats['attack_generated'] / self.stats['total_generated']
                    
                    logger.info(f"Traffic Stats - Total: {self.stats['total_generated']}, "
                              f"Rate: {rate:.1f} pps, Attack Rate: {attack_rate:.2%}")
                
            except Exception as e:
                logger.error(f"Stats reporting error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        uptime = time.time() - self.stats['start_time']
        
        return {
            'total_generated': self.stats['total_generated'],
            'benign_generated': self.stats['benign_generated'],
            'attack_generated': self.stats['attack_generated'],
            'uptime': uptime,
            'generation_rate': self.stats['total_generated'] / uptime if uptime > 0 else 0,
            'attack_percentage': (self.stats['attack_generated'] / self.stats['total_generated'] * 100) 
                               if self.stats['total_generated'] > 0 else 0
        }


def main():
    """Main function to run traffic generator."""
    # Get configuration from environment
    target_networks = os.getenv('TARGET_NETWORKS', 'nids-network').split(',')
    generation_rate = int(os.getenv('GENERATION_RATE', '100'))
    attack_probability = float(os.getenv('ATTACK_PROBABILITY', '0.1'))
    
    logger.info("Starting NIDS Traffic Generator")
    logger.info(f"Target networks: {target_networks}")
    logger.info(f"Generation rate: {generation_rate} pps")
    logger.info(f"Attack probability: {attack_probability}")
    
    # Create and start generator
    generator = TrafficGenerator(
        target_networks=target_networks,
        generation_rate=generation_rate,
        attack_probability=attack_probability
    )
    
    try:
        generator.start()
        
        # Keep running
        while True:
            time.sleep(10)
            stats = generator.get_stats()
            logger.info(f"Generated {stats['total_generated']} packets "
                       f"({stats['attack_percentage']:.1f}% attacks)")
            
    except KeyboardInterrupt:
        logger.info("Traffic generator stopped by user")
    except Exception as e:
        logger.error(f"Traffic generator failed: {e}")
    finally:
        generator.stop()


if __name__ == "__main__":
    main()
