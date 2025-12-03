#!/usr/bin/env python3
"""
Distributed NIDS System Status Checker
Monitors the performance of orchestrator and fog nodes
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any

class DistributedNIDSMonitor:
    """Monitor for distributed NIDS system."""
    
    def __init__(self):
        self.orchestrator_url = "http://localhost:8000"
        self.fog_nodes = [
            {"id": "fog-node-1", "url": "http://localhost:8080"},
            {"id": "fog-node-2", "url": "http://localhost:8081"},
            {"id": "fog-node-3", "url": "http://localhost:8082"}
        ]
        self.dashboard_url = "http://localhost:3000"
        self.prometheus_url = "http://localhost:9090"
    
    def check_service_health(self, url: str, service_name: str) -> Dict[str, Any]:
        """Check health of a service."""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return {
                    "status": "✅ HEALTHY",
                    "response_time": response.elapsed.total_seconds(),
                    "details": response.json() if response.headers.get('content-type', '').startswith('application/json') else "OK"
                }
            else:
                return {
                    "status": f"⚠️ UNHEALTHY ({response.status_code})",
                    "response_time": response.elapsed.total_seconds(),
                    "details": response.text[:100]
                }
        except requests.exceptions.ConnectionError:
            return {
                "status": "❌ UNREACHABLE",
                "response_time": None,
                "details": "Connection refused"
            }
        except Exception as e:
            return {
                "status": f"❌ ERROR",
                "response_time": None,
                "details": str(e)[:100]
            }
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        try:
            response = requests.get(f"{self.orchestrator_url}/api/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_fog_node_metrics(self, node_url: str) -> Dict[str, Any]:
        """Get fog node metrics."""
        try:
            response = requests.get(f"{node_url}/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get complete system overview."""
        try:
            response = requests.get(f"{self.orchestrator_url}/api/nodes", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def display_system_status(self):
        """Display complete system status."""
        print("🐳 DISTRIBUTED NIDS SYSTEM STATUS")
        print("=" * 80)
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check Orchestrator
        print("🎛️  CENTRAL ORCHESTRATOR")
        print("-" * 40)
        orchestrator_health = self.check_service_health(self.orchestrator_url, "orchestrator")
        print(f"Status: {orchestrator_health['status']}")
        if orchestrator_health['response_time']:
            print(f"Response Time: {orchestrator_health['response_time']:.3f}s")
        print(f"Details: {orchestrator_health['details']}")
        print()
        
        # Get orchestrator metrics
        if orchestrator_health['status'].startswith("✅"):
            orchestrator_metrics = self.get_orchestrator_metrics()
            if 'error' not in orchestrator_metrics:
                print("📊 Orchestrator Metrics:")
                for key, value in orchestrator_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value}")
                    elif isinstance(value, dict):
                        print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"    {sub_key}: {sub_value}")
                print()
        
        # Check Fog Nodes
        print("🌫️  FOG NODES")
        print("-" * 40)
        
        total_processing_rate = 0
        active_nodes = 0
        
        for node in self.fog_nodes:
            print(f"Node: {node['id']}")
            health = self.check_service_health(node['url'], node['id'])
            print(f"  Status: {health['status']}")
            
            if health['response_time']:
                print(f"  Response Time: {health['response_time']:.3f}s")
            
            # Get node metrics if healthy
            if health['status'].startswith("✅"):
                active_nodes += 1
                metrics = self.get_fog_node_metrics(node['url'])
                if 'error' not in metrics:
                    print(f"  📈 Metrics:")
                    for key, value in metrics.items():
                        if key == 'processing_rate' and isinstance(value, (int, float)):
                            total_processing_rate += value
                            print(f"    {key}: {value} flows/sec")
                        elif isinstance(value, (int, float)):
                            print(f"    {key}: {value}")
                        elif isinstance(value, dict) and len(value) < 5:
                            for sub_key, sub_value in value.items():
                                print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  ⚠️ Metrics Error: {metrics['error']}")
            else:
                print(f"  Details: {health['details']}")
            print()
        
        # System Summary
        print("📊 SYSTEM SUMMARY")
        print("-" * 40)
        print(f"Active Fog Nodes: {active_nodes}/{len(self.fog_nodes)}")
        if total_processing_rate > 0:
            print(f"Total Processing Rate: {total_processing_rate:.2f} flows/sec")
        print(f"Orchestrator: {'Online' if orchestrator_health['status'].startswith('✅') else 'Offline'}")
        
        # Check additional services
        print()
        print("🔧 ADDITIONAL SERVICES")
        print("-" * 40)
        
        # Dashboard
        dashboard_health = self.check_service_health(self.dashboard_url, "dashboard")
        print(f"Grafana Dashboard: {dashboard_health['status']}")
        if dashboard_health['status'].startswith("✅"):
            print(f"  URL: {self.dashboard_url} (admin/admin)")
        
        # Prometheus
        prometheus_health = self.check_service_health(self.prometheus_url, "prometheus")
        print(f"Prometheus: {prometheus_health['status']}")
        if prometheus_health['status'].startswith("✅"):
            print(f"  URL: {self.prometheus_url}")
        
        print()
        print("=" * 80)
    
    def get_real_time_metrics(self, duration: int = 30):
        """Monitor real-time metrics for specified duration."""
        print(f"📊 REAL-TIME MONITORING ({duration} seconds)")
        print("=" * 80)
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Clear screen (optional)
            print("\033[H\033[J", end="")
            
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Monitoring...")
            print("-" * 50)
            
            # Quick health check
            orchestrator_health = self.check_service_health(self.orchestrator_url, "orchestrator")
            print(f"Orchestrator: {orchestrator_health['status']}")
            
            total_rate = 0
            for i, node in enumerate(self.fog_nodes, 1):
                health = self.check_service_health(node['url'], node['id'])
                status_icon = "🟢" if health['status'].startswith("✅") else "🔴"
                print(f"Fog Node {i}: {status_icon}")
                
                if health['status'].startswith("✅"):
                    metrics = self.get_fog_node_metrics(node['url'])
                    if 'error' not in metrics and 'processing_rate' in metrics:
                        rate = metrics['processing_rate']
                        total_rate += rate
                        print(f"  Rate: {rate:.2f} flows/sec")
            
            print(f"Total System Rate: {total_rate:.2f} flows/sec")
            print("-" * 50)
            
            time.sleep(2)
        
        print("\n✅ Monitoring completed!")
    
    def run_diagnostics(self):
        """Run comprehensive system diagnostics."""
        print("🔧 DISTRIBUTED NIDS DIAGNOSTICS")
        print("=" * 80)
        
        # Test orchestrator API endpoints
        print("🧪 Testing Orchestrator APIs...")
        endpoints = [
            "/health",
            "/api/nodes",
            "/api/metrics",
            "/api/status"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.orchestrator_url}{endpoint}", timeout=5)
                status = "✅" if response.status_code == 200 else f"❌ ({response.status_code})"
                print(f"  {endpoint}: {status}")
            except Exception as e:
                print(f"  {endpoint}: ❌ ({str(e)[:30]})")
        
        print()
        
        # Test fog node connectivity
        print("🌫️  Testing Fog Node Connectivity...")
        for node in self.fog_nodes:
            try:
                response = requests.get(f"{node['url']}/health", timeout=5)
                status = "✅" if response.status_code == 200 else f"❌ ({response.status_code})"
                print(f"  {node['id']}: {status}")
            except Exception as e:
                print(f"  {node['id']}: ❌ ({str(e)[:30]})")
        
        print()
        print("✅ Diagnostics completed!")

def main():
    """Main function."""
    monitor = DistributedNIDSMonitor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            monitor.display_system_status()
        elif command == "monitor":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            monitor.get_real_time_metrics(duration)
        elif command == "diagnostics":
            monitor.run_diagnostics()
        else:
            print("Usage: python check_distributed_system.py [status|monitor|diagnostics] [duration]")
    else:
        # Default: show status
        monitor.display_system_status()

if __name__ == "__main__":
    main()
