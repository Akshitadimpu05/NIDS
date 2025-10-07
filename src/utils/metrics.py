"""
Metrics collection and monitoring utilities for NIDS fog nodes.
Tracks performance, security, and system metrics.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import numpy as np
import json
import psutil

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and aggregates metrics for fog node monitoring.
    Tracks performance, security, and system health metrics.
    """
    
    def __init__(self, 
                 node_id: str,
                 history_size: int = 1000,
                 aggregation_window: int = 60):
        """
        Initialize metrics collector.
        
        Args:
            node_id: Unique identifier for the fog node
            history_size: Number of historical data points to keep
            aggregation_window: Time window for metric aggregation (seconds)
        """
        self.node_id = node_id
        self.history_size = history_size
        self.aggregation_window = aggregation_window
        
        # Metric storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=history_size))
        self.current_metrics = {}
        self.aggregated_metrics = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
        # Start time
        self.start_time = time.time()
        
        logger.info(f"Metrics collector initialized for node {node_id}")
    
    def update(self, metrics: Dict[str, Any]):
        """
        Update metrics with new data.
        
        Args:
            metrics: Dictionary of metric name -> value pairs
        """
        timestamp = time.time()
        
        with self.lock:
            # Store current metrics
            self.current_metrics.update(metrics)
            
            # Add to history with timestamp
            for metric_name, value in metrics.items():
                self.metrics_history[metric_name].append({
                    'timestamp': timestamp,
                    'value': value
                })
            
            # Update aggregated metrics
            self._update_aggregations()
    
    def _update_aggregations(self):
        """Update aggregated metrics based on recent history."""
        current_time = time.time()
        cutoff_time = current_time - self.aggregation_window
        
        for metric_name, history in self.metrics_history.items():
            # Get recent values within aggregation window
            recent_values = [
                entry['value'] for entry in history
                if entry['timestamp'] >= cutoff_time
            ]
            
            if recent_values:
                # Calculate aggregations for numeric values
                if all(isinstance(v, (int, float)) for v in recent_values):
                    self.aggregated_metrics[f"{metric_name}_avg"] = np.mean(recent_values)
                    self.aggregated_metrics[f"{metric_name}_min"] = np.min(recent_values)
                    self.aggregated_metrics[f"{metric_name}_max"] = np.max(recent_values)
                    self.aggregated_metrics[f"{metric_name}_std"] = np.std(recent_values)
                    self.aggregated_metrics[f"{metric_name}_count"] = len(recent_values)
                
                # Store latest value
                self.aggregated_metrics[f"{metric_name}_latest"] = recent_values[-1]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        with self.lock:
            return self.current_metrics.copy()
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        with self.lock:
            return self.aggregated_metrics.copy()
    
    def get_metric_history(self, metric_name: str, 
                          duration: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific metric.
        
        Args:
            metric_name: Name of the metric
            duration: Duration in seconds (None for all history)
            
        Returns:
            List of historical data points
        """
        with self.lock:
            history = list(self.metrics_history[metric_name])
            
            if duration is not None:
                cutoff_time = time.time() - duration
                history = [
                    entry for entry in history
                    if entry['timestamp'] >= cutoff_time
                ]
            
            return history
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return self.system_monitor.get_metrics()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Get recent performance metrics
        processing_times = self.get_metric_history('processing_time', duration=300)  # Last 5 minutes
        throughput_data = self.get_metric_history('packets_per_second', duration=300)
        
        summary = {
            'node_id': self.node_id,
            'uptime': uptime,
            'timestamp': current_time
        }
        
        # Processing performance
        if processing_times:
            times = [entry['value'] for entry in processing_times]
            summary['processing_performance'] = {
                'avg_processing_time': np.mean(times),
                'p95_processing_time': np.percentile(times, 95),
                'p99_processing_time': np.percentile(times, 99),
                'total_processed': len(times)
            }
        
        # Throughput performance
        if throughput_data:
            throughputs = [entry['value'] for entry in throughput_data]
            summary['throughput_performance'] = {
                'avg_throughput': np.mean(throughputs),
                'max_throughput': np.max(throughputs),
                'min_throughput': np.min(throughputs)
            }
        
        # System performance
        system_metrics = self.get_system_metrics()
        summary['system_performance'] = system_metrics
        
        return summary
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security-related metrics summary."""
        current_time = time.time()
        
        # Get recent security metrics
        anomalies = self.get_metric_history('anomalies_detected', duration=3600)  # Last hour
        actions = self.get_metric_history('actions_taken', duration=3600)
        
        summary = {
            'node_id': self.node_id,
            'timestamp': current_time
        }
        
        # Anomaly detection metrics
        if anomalies:
            anomaly_counts = [entry['value'] for entry in anomalies]
            summary['anomaly_detection'] = {
                'total_anomalies': sum(anomaly_counts),
                'anomaly_rate': sum(anomaly_counts) / len(anomaly_counts) if anomaly_counts else 0,
                'peak_anomalies': max(anomaly_counts) if anomaly_counts else 0
            }
        
        # Action metrics
        if actions:
            # Aggregate action counts
            action_totals = defaultdict(int)
            for entry in actions:
                if isinstance(entry['value'], dict):
                    for action, count in entry['value'].items():
                        action_totals[action] += count
            
            summary['mitigation_actions'] = dict(action_totals)
        
        return summary
    
    def export_metrics(self, format: str = 'json') -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ('json', 'prometheus')
            
        Returns:
            Formatted metrics string
        """
        if format == 'json':
            return self._export_json()
        elif format == 'prometheus':
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON."""
        export_data = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'current_metrics': self.get_current_metrics(),
            'aggregated_metrics': self.get_aggregated_metrics(),
            'system_metrics': self.get_system_metrics(),
            'performance_summary': self.get_performance_summary(),
            'security_summary': self.get_security_summary()
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        current_metrics = self.get_current_metrics()
        system_metrics = self.get_system_metrics()
        
        # Add node info
        lines.append(f'# HELP nids_node_info Node information')
        lines.append(f'# TYPE nids_node_info gauge')
        lines.append(f'nids_node_info{{node_id="{self.node_id}"}} 1')
        
        # Add current metrics
        for metric_name, value in current_metrics.items():
            if isinstance(value, (int, float)):
                safe_name = metric_name.replace('-', '_').replace('.', '_')
                lines.append(f'# HELP nids_{safe_name} {metric_name}')
                lines.append(f'# TYPE nids_{safe_name} gauge')
                lines.append(f'nids_{safe_name}{{node_id="{self.node_id}"}} {value}')
        
        # Add system metrics
        for metric_name, value in system_metrics.items():
            if isinstance(value, (int, float)):
                safe_name = metric_name.replace('-', '_').replace('.', '_')
                lines.append(f'# HELP nids_system_{safe_name} System {metric_name}')
                lines.append(f'# TYPE nids_system_{safe_name} gauge')
                lines.append(f'nids_system_{safe_name}{{node_id="{self.node_id}"}} {value}')
        
        return '\n'.join(lines)
    
    def clear_history(self):
        """Clear all metric history."""
        with self.lock:
            self.metrics_history.clear()
            self.aggregated_metrics.clear()
            logger.info("Metrics history cleared")


class SystemMonitor:
    """
    Monitors system-level metrics like CPU, memory, disk, and network usage.
    """
    
    def __init__(self):
        """Initialize system monitor."""
        self.last_network_stats = None
        self.last_network_time = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            metrics = {}
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            metrics.update({
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'load_avg_1m': load_avg[0],
                'load_avg_5m': load_avg[1],
                'load_avg_15m': load_avg[2]
            })
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.update({
                'memory_total': memory.total,
                'memory_available': memory.available,
                'memory_used': memory.used,
                'memory_percent': memory.percent,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_percent': swap.percent
            })
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics.update({
                'disk_total': disk_usage.total,
                'disk_used': disk_usage.used,
                'disk_free': disk_usage.free,
                'disk_percent': disk_usage.percent
            })
            
            if disk_io:
                metrics.update({
                    'disk_read_bytes': disk_io.read_bytes,
                    'disk_write_bytes': disk_io.write_bytes,
                    'disk_read_count': disk_io.read_count,
                    'disk_write_count': disk_io.write_count
                })
            
            # Network metrics
            network_io = psutil.net_io_counters()
            current_time = time.time()
            
            if network_io:
                metrics.update({
                    'network_bytes_sent': network_io.bytes_sent,
                    'network_bytes_recv': network_io.bytes_recv,
                    'network_packets_sent': network_io.packets_sent,
                    'network_packets_recv': network_io.packets_recv
                })
                
                # Calculate network rates
                if self.last_network_stats and self.last_network_time:
                    time_delta = current_time - self.last_network_time
                    if time_delta > 0:
                        bytes_sent_rate = (network_io.bytes_sent - self.last_network_stats.bytes_sent) / time_delta
                        bytes_recv_rate = (network_io.bytes_recv - self.last_network_stats.bytes_recv) / time_delta
                        
                        metrics.update({
                            'network_bytes_sent_rate': bytes_sent_rate,
                            'network_bytes_recv_rate': bytes_recv_rate
                        })
                
                self.last_network_stats = network_io
                self.last_network_time = current_time
            
            # Process metrics
            process = psutil.Process()
            metrics.update({
                'process_cpu_percent': process.cpu_percent(),
                'process_memory_rss': process.memory_info().rss,
                'process_memory_vms': process.memory_info().vms,
                'process_num_threads': process.num_threads(),
                'process_num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}


class AlertManager:
    """
    Manages alerts based on metric thresholds.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize alert manager.
        
        Args:
            metrics_collector: MetricsCollector instance to monitor
        """
        self.metrics_collector = metrics_collector
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
    def add_alert_rule(self, 
                      rule_name: str,
                      metric_name: str,
                      threshold: float,
                      operator: str = 'gt',
                      duration: int = 60,
                      severity: str = 'warning'):
        """
        Add alert rule.
        
        Args:
            rule_name: Unique name for the rule
            metric_name: Metric to monitor
            threshold: Alert threshold value
            operator: Comparison operator ('gt', 'lt', 'eq', 'gte', 'lte')
            duration: Duration threshold must be exceeded (seconds)
            severity: Alert severity ('info', 'warning', 'critical')
        """
        self.alert_rules[rule_name] = {
            'metric_name': metric_name,
            'threshold': threshold,
            'operator': operator,
            'duration': duration,
            'severity': severity,
            'triggered_at': None
        }
        
        logger.info(f"Added alert rule: {rule_name}")
    
    def check_alerts(self):
        """Check all alert rules and trigger alerts if necessary."""
        current_time = time.time()
        current_metrics = self.metrics_collector.get_current_metrics()
        
        for rule_name, rule in self.alert_rules.items():
            metric_value = current_metrics.get(rule['metric_name'])
            
            if metric_value is None:
                continue
            
            # Check if threshold is exceeded
            threshold_exceeded = self._check_threshold(
                metric_value, rule['threshold'], rule['operator']
            )
            
            if threshold_exceeded:
                if rule['triggered_at'] is None:
                    rule['triggered_at'] = current_time
                elif current_time - rule['triggered_at'] >= rule['duration']:
                    # Trigger alert
                    self._trigger_alert(rule_name, rule, metric_value, current_time)
            else:
                # Reset trigger time
                rule['triggered_at'] = None
                
                # Clear active alert if it exists
                if rule_name in self.active_alerts:
                    self._clear_alert(rule_name, current_time)
    
    def _check_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Check if value exceeds threshold based on operator."""
        if operator == 'gt':
            return value > threshold
        elif operator == 'lt':
            return value < threshold
        elif operator == 'eq':
            return value == threshold
        elif operator == 'gte':
            return value >= threshold
        elif operator == 'lte':
            return value <= threshold
        else:
            return False
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], 
                      value: float, timestamp: float):
        """Trigger an alert."""
        if rule_name not in self.active_alerts:
            alert = {
                'rule_name': rule_name,
                'metric_name': rule['metric_name'],
                'threshold': rule['threshold'],
                'current_value': value,
                'severity': rule['severity'],
                'triggered_at': timestamp,
                'node_id': self.metrics_collector.node_id
            }
            
            self.active_alerts[rule_name] = alert
            self.alert_history.append(alert.copy())
            
            logger.warning(f"Alert triggered: {rule_name} - {rule['metric_name']} = {value} "
                          f"(threshold: {rule['threshold']})")
    
    def _clear_alert(self, rule_name: str, timestamp: float):
        """Clear an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert['cleared_at'] = timestamp
            
            del self.active_alerts[rule_name]
            
            logger.info(f"Alert cleared: {rule_name}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, duration: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get alert history."""
        if duration is None:
            return list(self.alert_history)
        
        cutoff_time = time.time() - duration
        return [
            alert for alert in self.alert_history
            if alert['triggered_at'] >= cutoff_time
        ]
