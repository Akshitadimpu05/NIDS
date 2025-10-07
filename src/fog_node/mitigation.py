"""
Traffic mitigation module for implementing security actions.
Handles blocking, throttling, and allowing traffic based on NIDS decisions.
"""

import time
import logging
import subprocess
import threading
from typing import Dict, Any, List, Optional
from enum import IntEnum
from dataclasses import dataclass
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MitigationAction(IntEnum):
    """Traffic mitigation actions."""
    ALLOW = 0
    BLOCK = 1
    THROTTLE = 2


@dataclass
class MitigationRule:
    """Represents a traffic mitigation rule."""
    rule_id: str
    src_ip: str
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    protocol: Optional[str] = None
    action: MitigationAction = MitigationAction.BLOCK
    duration: int = 300  # seconds
    created_at: float = 0.0
    reason: str = ""


class TrafficMitigation:
    """
    Traffic mitigation system for implementing security actions.
    
    Supports multiple mitigation mechanisms:
    - iptables rules for blocking/allowing
    - Traffic shaping for throttling
    - Application-level controls
    """
    
    def __init__(self,
                 enable_iptables: bool = True,
                 enable_traffic_shaping: bool = True,
                 max_rules: int = 10000,
                 rule_cleanup_interval: int = 60):
        """
        Initialize traffic mitigation system.
        
        Args:
            enable_iptables: Enable iptables-based blocking
            enable_traffic_shaping: Enable traffic shaping for throttling
            max_rules: Maximum number of active rules
            rule_cleanup_interval: Rule cleanup interval in seconds
        """
        self.enable_iptables = enable_iptables
        self.enable_traffic_shaping = enable_traffic_shaping
        self.max_rules = max_rules
        self.rule_cleanup_interval = rule_cleanup_interval
        
        # Active rules tracking
        self.active_rules = {}  # rule_id -> MitigationRule
        self.rules_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'actions_executed': defaultdict(int),
            'rules_created': 0,
            'rules_expired': 0,
            'iptables_commands': 0,
            'traffic_shaping_commands': 0,
            'errors': 0
        }
        
        # Rate limiting for actions
        self.action_history = defaultdict(lambda: deque(maxlen=100))
        
        # Cleanup thread
        self.cleanup_thread = None
        self.is_running = False
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize mitigation system components."""
        try:
            if self.enable_iptables:
                self._setup_iptables_chains()
            
            if self.enable_traffic_shaping:
                self._setup_traffic_shaping()
            
            # Start cleanup thread
            self.is_running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_expired_rules,
                name="MitigationCleanup"
            )
            self.cleanup_thread.start()
            
            logger.info("Traffic mitigation system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize mitigation system: {e}")
            raise
    
    def _setup_iptables_chains(self):
        """Setup custom iptables chains for NIDS."""
        try:
            # Create custom chains
            chains = ['NIDS_INPUT', 'NIDS_FORWARD', 'NIDS_OUTPUT']
            
            for chain in chains:
                # Create chain if it doesn't exist
                result = subprocess.run(
                    ['iptables', '-t', 'filter', '-N', chain],
                    capture_output=True, text=True
                )
                if result.returncode != 0 and "Chain already exists" not in result.stderr:
                    logger.warning(f"Failed to create chain {chain}: {result.stderr}")
            
            # Insert jump rules to custom chains
            jump_rules = [
                ['iptables', '-t', 'filter', '-I', 'INPUT', '-j', 'NIDS_INPUT'],
                ['iptables', '-t', 'filter', '-I', 'FORWARD', '-j', 'NIDS_FORWARD'],
                ['iptables', '-t', 'filter', '-I', 'OUTPUT', '-j', 'NIDS_OUTPUT']
            ]
            
            for rule in jump_rules:
                result = subprocess.run(rule, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"Failed to add jump rule: {result.stderr}")
            
            logger.info("iptables chains setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup iptables chains: {e}")
    
    def _setup_traffic_shaping(self):
        """Setup traffic shaping using tc (traffic control)."""
        try:
            # Check if tc is available
            result = subprocess.run(['which', 'tc'], capture_output=True)
            if result.returncode != 0:
                logger.warning("tc (traffic control) not available, disabling traffic shaping")
                self.enable_traffic_shaping = False
                return
            
            # Setup basic qdisc (queuing discipline)
            interfaces = ['eth0', 'lo']  # Common interfaces
            
            for interface in interfaces:
                # Check if interface exists
                result = subprocess.run(
                    ['ip', 'link', 'show', interface],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    # Setup HTB qdisc
                    subprocess.run([
                        'tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:',
                        'htb', 'default', '30'
                    ], capture_output=True)
                    
                    # Create classes for different priorities
                    subprocess.run([
                        'tc', 'class', 'add', 'dev', interface, 'parent', '1:',
                        'classid', '1:1', 'htb', 'rate', '1000mbit'
                    ], capture_output=True)
                    
                    subprocess.run([
                        'tc', 'class', 'add', 'dev', interface, 'parent', '1:1',
                        'classid', '1:10', 'htb', 'rate', '100mbit', 'ceil', '1000mbit'
                    ], capture_output=True)
                    
                    subprocess.run([
                        'tc', 'class', 'add', 'dev', interface, 'parent', '1:1',
                        'classid', '1:20', 'htb', 'rate', '10mbit', 'ceil', '100mbit'
                    ], capture_output=True)
            
            logger.info("Traffic shaping setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup traffic shaping: {e}")
            self.enable_traffic_shaping = False
    
    def execute_action(self, 
                      action: int,
                      traffic_data: Dict[str, Any],
                      metadata: Dict[str, Any]) -> bool:
        """
        Execute mitigation action on traffic.
        
        Args:
            action: Mitigation action (0=Allow, 1=Block, 2=Throttle)
            traffic_data: Traffic data and features
            metadata: Additional metadata including decision confidence
            
        Returns:
            True if action was executed successfully
        """
        try:
            action_enum = MitigationAction(action)
            
            # Extract traffic metadata
            traffic_meta = traffic_data.get('metadata', {})
            src_ip = traffic_meta.get('src_ip')
            dst_ip = traffic_meta.get('dst_ip')
            src_port = traffic_meta.get('src_port')
            dst_port = traffic_meta.get('dst_port')
            protocol = traffic_meta.get('protocol')
            
            if not src_ip:
                logger.warning("No source IP in traffic data, skipping action")
                return False
            
            # Check rate limiting
            if not self._check_rate_limit(src_ip, action_enum):
                logger.warning(f"Rate limit exceeded for {src_ip}, skipping action")
                return False
            
            # Create mitigation rule
            rule = self._create_mitigation_rule(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                action=action_enum,
                metadata=metadata
            )
            
            # Execute the action
            success = False
            if action_enum == MitigationAction.ALLOW:
                success = self._execute_allow(rule)
            elif action_enum == MitigationAction.BLOCK:
                success = self._execute_block(rule)
            elif action_enum == MitigationAction.THROTTLE:
                success = self._execute_throttle(rule)
            
            if success:
                # Store active rule
                with self.rules_lock:
                    self.active_rules[rule.rule_id] = rule
                
                # Update statistics
                self.stats['actions_executed'][action_enum.name] += 1
                self.stats['rules_created'] += 1
                
                # Update rate limiting
                self.action_history[src_ip].append(time.time())
                
                logger.info(f"Executed {action_enum.name} action for {src_ip}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute mitigation action: {e}")
            self.stats['errors'] += 1
            return False
    
    def _create_mitigation_rule(self,
                               src_ip: str,
                               dst_ip: Optional[str] = None,
                               src_port: Optional[int] = None,
                               dst_port: Optional[int] = None,
                               protocol: Optional[str] = None,
                               action: MitigationAction = MitigationAction.BLOCK,
                               metadata: Dict[str, Any] = None) -> MitigationRule:
        """Create a mitigation rule."""
        rule_id = f"{action.name}_{src_ip}_{int(time.time())}"
        
        # Determine rule duration based on action and confidence
        confidence = metadata.get('confidence', 0.5) if metadata else 0.5
        base_duration = 300  # 5 minutes
        
        if action == MitigationAction.BLOCK:
            duration = int(base_duration * (1 + confidence))  # Higher confidence = longer block
        elif action == MitigationAction.THROTTLE:
            duration = int(base_duration * 0.5)  # Shorter throttling duration
        else:  # ALLOW
            duration = 60  # Short allow rule duration
        
        reason = f"NIDS decision: {action.name} (confidence: {confidence:.2f})"
        if metadata:
            if metadata.get('is_anomaly'):
                reason += " - Anomaly detected"
            if metadata.get('anomaly_score', 0) > 0.5:
                reason += f" - High anomaly score: {metadata['anomaly_score']:.2f}"
        
        return MitigationRule(
            rule_id=rule_id,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=str(protocol) if protocol else None,
            action=action,
            duration=duration,
            created_at=time.time(),
            reason=reason
        )
    
    def _check_rate_limit(self, src_ip: str, action: MitigationAction) -> bool:
        """Check if action is rate limited for the source IP."""
        current_time = time.time()
        recent_actions = [
            t for t in self.action_history[src_ip] 
            if current_time - t < 60  # Last minute
        ]
        
        # Rate limits per minute
        limits = {
            MitigationAction.ALLOW: 100,
            MitigationAction.BLOCK: 10,
            MitigationAction.THROTTLE: 20
        }
        
        return len(recent_actions) < limits.get(action, 10)
    
    def _execute_allow(self, rule: MitigationRule) -> bool:
        """Execute allow action (remove any existing blocks)."""
        try:
            if self.enable_iptables:
                # Remove any existing block rules for this IP
                self._remove_iptables_rules(rule.src_ip)
            
            logger.debug(f"Allowed traffic from {rule.src_ip}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute allow action: {e}")
            return False
    
    def _execute_block(self, rule: MitigationRule) -> bool:
        """Execute block action using iptables."""
        try:
            if not self.enable_iptables:
                logger.debug(f"iptables disabled, simulating block for {rule.src_ip}")
                return True
            
            # Create iptables rule to block traffic
            iptables_cmd = [
                'iptables', '-t', 'filter', '-I', 'NIDS_INPUT',
                '-s', rule.src_ip, '-j', 'DROP',
                '-m', 'comment', '--comment', f"NIDS_BLOCK_{rule.rule_id}"
            ]
            
            # Add destination IP if specified
            if rule.dst_ip:
                iptables_cmd.extend(['-d', rule.dst_ip])
            
            # Add port restrictions if specified
            if rule.src_port:
                iptables_cmd.extend(['--sport', str(rule.src_port)])
            if rule.dst_port:
                iptables_cmd.extend(['--dport', str(rule.dst_port)])
            
            # Add protocol if specified
            if rule.protocol and rule.protocol.lower() in ['tcp', 'udp']:
                iptables_cmd.extend(['-p', rule.protocol.lower()])
            
            result = subprocess.run(iptables_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.stats['iptables_commands'] += 1
                logger.info(f"Blocked traffic from {rule.src_ip}")
                return True
            else:
                logger.error(f"Failed to add iptables rule: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute block action: {e}")
            return False
    
    def _execute_throttle(self, rule: MitigationRule) -> bool:
        """Execute throttle action using traffic shaping."""
        try:
            if not self.enable_traffic_shaping:
                logger.debug(f"Traffic shaping disabled, simulating throttle for {rule.src_ip}")
                return True
            
            # Create traffic shaping rule to limit bandwidth
            # This is a simplified implementation
            interface = 'eth0'  # Default interface
            
            # Add filter to classify traffic from this IP
            tc_filter_cmd = [
                'tc', 'filter', 'add', 'dev', interface, 'protocol', 'ip',
                'parent', '1:', 'prio', '1', 'u32',
                'match', 'ip', 'src', rule.src_ip,
                'flowid', '1:20'  # Low priority class
            ]
            
            result = subprocess.run(tc_filter_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.stats['traffic_shaping_commands'] += 1
                logger.info(f"Throttled traffic from {rule.src_ip}")
                return True
            else:
                logger.warning(f"Failed to add tc filter: {result.stderr}")
                # Fall back to iptables rate limiting
                return self._execute_rate_limit_iptables(rule)
                
        except Exception as e:
            logger.error(f"Failed to execute throttle action: {e}")
            return False
    
    def _execute_rate_limit_iptables(self, rule: MitigationRule) -> bool:
        """Execute rate limiting using iptables as fallback for throttling."""
        try:
            if not self.enable_iptables:
                return False
            
            # Use iptables rate limiting
            iptables_cmd = [
                'iptables', '-t', 'filter', '-I', 'NIDS_INPUT',
                '-s', rule.src_ip,
                '-m', 'limit', '--limit', '10/sec', '--limit-burst', '20',
                '-j', 'ACCEPT',
                '-m', 'comment', '--comment', f"NIDS_THROTTLE_{rule.rule_id}"
            ]
            
            result = subprocess.run(iptables_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Add a drop rule after the rate limit
                drop_cmd = [
                    'iptables', '-t', 'filter', '-I', 'NIDS_INPUT',
                    '-s', rule.src_ip, '-j', 'DROP',
                    '-m', 'comment', '--comment', f"NIDS_THROTTLE_DROP_{rule.rule_id}"
                ]
                subprocess.run(drop_cmd, capture_output=True, text=True)
                
                self.stats['iptables_commands'] += 2
                logger.info(f"Rate limited traffic from {rule.src_ip}")
                return True
            else:
                logger.error(f"Failed to add rate limit rule: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute rate limit: {e}")
            return False
    
    def _remove_iptables_rules(self, src_ip: str):
        """Remove all iptables rules for a specific source IP."""
        try:
            chains = ['NIDS_INPUT', 'NIDS_FORWARD', 'NIDS_OUTPUT']
            
            for chain in chains:
                # List rules with line numbers
                result = subprocess.run([
                    'iptables', '-t', 'filter', '-L', chain, '--line-numbers', '-n'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    rules_to_delete = []
                    
                    for line in lines:
                        if src_ip in line and ('NIDS_BLOCK' in line or 'NIDS_THROTTLE' in line):
                            parts = line.split()
                            if parts and parts[0].isdigit():
                                rules_to_delete.append(int(parts[0]))
                    
                    # Delete rules in reverse order to maintain line numbers
                    for rule_num in sorted(rules_to_delete, reverse=True):
                        subprocess.run([
                            'iptables', '-t', 'filter', '-D', chain, str(rule_num)
                        ], capture_output=True)
                        
        except Exception as e:
            logger.error(f"Failed to remove iptables rules for {src_ip}: {e}")
    
    def _cleanup_expired_rules(self):
        """Clean up expired mitigation rules."""
        while self.is_running:
            try:
                current_time = time.time()
                expired_rules = []
                
                with self.rules_lock:
                    for rule_id, rule in self.active_rules.items():
                        if current_time - rule.created_at > rule.duration:
                            expired_rules.append(rule_id)
                
                # Remove expired rules
                for rule_id in expired_rules:
                    self._remove_rule(rule_id)
                
                if expired_rules:
                    logger.info(f"Cleaned up {len(expired_rules)} expired rules")
                
                # Sleep until next cleanup
                time.sleep(self.rule_cleanup_interval)
                
            except Exception as e:
                logger.error(f"Rule cleanup error: {e}")
                time.sleep(self.rule_cleanup_interval)
    
    def _remove_rule(self, rule_id: str):
        """Remove a specific mitigation rule."""
        try:
            with self.rules_lock:
                if rule_id not in self.active_rules:
                    return
                
                rule = self.active_rules[rule_id]
                
                # Remove iptables rules
                if rule.action in [MitigationAction.BLOCK, MitigationAction.THROTTLE]:
                    self._remove_iptables_rules(rule.src_ip)
                
                # Remove traffic shaping rules
                if rule.action == MitigationAction.THROTTLE and self.enable_traffic_shaping:
                    self._remove_tc_rules(rule.src_ip)
                
                # Remove from active rules
                del self.active_rules[rule_id]
                self.stats['rules_expired'] += 1
                
        except Exception as e:
            logger.error(f"Failed to remove rule {rule_id}: {e}")
    
    def _remove_tc_rules(self, src_ip: str):
        """Remove traffic control rules for a specific IP."""
        try:
            interface = 'eth0'
            
            # List filters and remove matching ones
            result = subprocess.run([
                'tc', 'filter', 'show', 'dev', interface
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse and remove filters (simplified)
                # In a production system, you'd need more sophisticated parsing
                subprocess.run([
                    'tc', 'filter', 'del', 'dev', interface, 'protocol', 'ip'
                ], capture_output=True)
                
        except Exception as e:
            logger.error(f"Failed to remove tc rules for {src_ip}: {e}")
    
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Get list of active mitigation rules."""
        with self.rules_lock:
            return [
                {
                    'rule_id': rule.rule_id,
                    'src_ip': rule.src_ip,
                    'dst_ip': rule.dst_ip,
                    'action': rule.action.name,
                    'duration': rule.duration,
                    'created_at': rule.created_at,
                    'expires_at': rule.created_at + rule.duration,
                    'reason': rule.reason
                }
                for rule in self.active_rules.values()
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mitigation statistics."""
        return {
            'active_rules': len(self.active_rules),
            'actions_executed': dict(self.stats['actions_executed']),
            'rules_created': self.stats['rules_created'],
            'rules_expired': self.stats['rules_expired'],
            'iptables_commands': self.stats['iptables_commands'],
            'traffic_shaping_commands': self.stats['traffic_shaping_commands'],
            'errors': self.stats['errors'],
            'capabilities': {
                'iptables_enabled': self.enable_iptables,
                'traffic_shaping_enabled': self.enable_traffic_shaping
            }
        }
    
    def stop(self):
        """Stop the mitigation system."""
        logger.info("Stopping traffic mitigation system...")
        
        self.is_running = False
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        # Clean up all active rules
        with self.rules_lock:
            for rule_id in list(self.active_rules.keys()):
                self._remove_rule(rule_id)
        
        logger.info("Traffic mitigation system stopped")
