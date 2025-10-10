#!/usr/bin/env python3
"""
Web Dashboard for Live NIDS Detection System
Real-time monitoring and visualization of network intrusion detection
"""

import os
import sys
import json
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from collections import defaultdict, deque
import numpy as np

# Setup paths
sys.path.append('/home/tejasri/nids/NIDS')
sys.path.append('/home/tejasri/nids/NIDS/src')

from live_detection_system import LiveDetectionSystem

app = Flask(__name__)

# Global variables for dashboard
dashboard_data = {
    'detection_system': None,
    'is_running': False,
    'recent_detections': deque(maxlen=100),
    'threat_timeline': deque(maxlen=50),
    'performance_metrics': deque(maxlen=30)
}

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current system status."""
    if dashboard_data['detection_system']:
        stats = dashboard_data['detection_system'].detector.get_statistics()
        return jsonify({
            'status': 'running' if dashboard_data['is_running'] else 'stopped',
            'uptime': stats['uptime_seconds'],
            'total_flows': stats['total_flows_processed'],
            'threats_detected': stats['threats_detected'],
            'threat_rate': stats['threat_rate'],
            'processing_rate': stats['flows_per_second'],
            'avg_processing_time': stats['avg_processing_time_ms']
        })
    else:
        return jsonify({
            'status': 'not_initialized',
            'uptime': 0,
            'total_flows': 0,
            'threats_detected': 0,
            'threat_rate': 0,
            'processing_rate': 0,
            'avg_processing_time': 0
        })

@app.route('/api/statistics')
def get_statistics():
    """Get detailed statistics."""
    if dashboard_data['detection_system']:
        stats = dashboard_data['detection_system'].detector.get_statistics()
        return jsonify(stats)
    else:
        return jsonify({})

@app.route('/api/recent_detections')
def get_recent_detections():
    """Get recent detection results."""
    return jsonify(list(dashboard_data['recent_detections']))

@app.route('/api/threat_timeline')
def get_threat_timeline():
    """Get threat detection timeline."""
    return jsonify(list(dashboard_data['threat_timeline']))

@app.route('/api/performance_metrics')
def get_performance_metrics():
    """Get performance metrics over time."""
    return jsonify(list(dashboard_data['performance_metrics']))

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start the detection system."""
    try:
        rate = request.json.get('rate', 10)
        
        if not dashboard_data['detection_system']:
            dashboard_data['detection_system'] = LiveDetectionSystem(detection_rate=rate)
        
        # Start detection in background thread
        if not dashboard_data['is_running']:
            detection_thread = threading.Thread(
                target=run_detection_with_monitoring,
                daemon=True
            )
            detection_thread.start()
            dashboard_data['is_running'] = True
        
        return jsonify({'success': True, 'message': 'Detection started'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop the detection system."""
    try:
        if dashboard_data['detection_system']:
            dashboard_data['detection_system'].running = False
            dashboard_data['is_running'] = False
        
        return jsonify({'success': True, 'message': 'Detection stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

def run_detection_with_monitoring():
    """Run detection system with monitoring for dashboard."""
    system = dashboard_data['detection_system']
    
    while dashboard_data['is_running']:
        try:
            # Get next flow
            flow = system.simulator.get_next_flow()
            
            # Analyze flow
            result = system.detector.analyze_flow(flow)
            
            # Apply mitigation
            system.detector.apply_mitigation(result)
            
            # Update dashboard data
            update_dashboard_data(result)
            
            # Rate limiting
            time.sleep(1.0 / system.detection_rate)
            
        except Exception as e:
            print(f"Error in detection: {e}")
            break
    
    dashboard_data['is_running'] = False

def update_dashboard_data(result):
    """Update dashboard data with new detection result."""
    current_time = datetime.now()
    
    # Add to recent detections
    dashboard_data['recent_detections'].append({
        'timestamp': current_time.isoformat(),
        'flow_id': result['flow_id'],
        'src_ip': result['src_ip'],
        'dst_ip': result['dst_ip'],
        'predicted_class': result['predicted_class_name'],
        'confidence': result['confidence'],
        'action': result['action_name'],
        'is_threat': result['is_threat'],
        'anomaly_score': result['anomaly_score']
    })
    
    # Update threat timeline
    if result['is_threat']:
        dashboard_data['threat_timeline'].append({
            'timestamp': current_time.isoformat(),
            'threat_type': result['predicted_class_name'],
            'confidence': result['confidence'],
            'action': result['action_name'],
            'src_ip': result['src_ip']
        })
    
    # Update performance metrics (every 10 flows)
    if len(dashboard_data['recent_detections']) % 10 == 0:
        stats = dashboard_data['detection_system'].detector.get_statistics()
        dashboard_data['performance_metrics'].append({
            'timestamp': current_time.isoformat(),
            'flows_per_second': stats['flows_per_second'],
            'threat_rate': stats['threat_rate'],
            'avg_processing_time': stats['avg_processing_time_ms'],
            'total_flows': stats['total_flows_processed']
        })

# Create templates directory and HTML template
def create_dashboard_template():
    """Create the HTML template for the dashboard."""
    os.makedirs('templates', exist_ok=True)
    
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live NIDS Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .controls {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .detections-table {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
        }
        .threat {
            background-color: #ffe6e6;
        }
        .status-running {
            color: #28a745;
        }
        .status-stopped {
            color: #dc3545;
        }
        button {
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn-start {
            background-color: #28a745;
            color: white;
        }
        .btn-stop {
            background-color: #dc3545;
            color: white;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Live NIDS Detection Dashboard</h1>
        <p>Real-time Network Intrusion Detection & Mitigation System</p>
    </div>

    <div class="controls">
        <h3>System Controls</h3>
        <button class="btn-start" onclick="startDetection()">Start Detection</button>
        <button class="btn-stop" onclick="stopDetection()">Stop Detection</button>
        <span id="status" class="status-stopped">System Stopped</span>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value" id="total-flows">0</div>
            <div class="metric-label">Total Flows Processed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="threats-detected">0</div>
            <div class="metric-label">Threats Detected</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="processing-rate">0</div>
            <div class="metric-label">Flows/Second</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="avg-processing-time">0</div>
            <div class="metric-label">Avg Processing Time (ms)</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-container">
            <h3>Threat Detection Timeline</h3>
            <canvas id="threatChart"></canvas>
        </div>
        <div class="chart-container">
            <h3>Performance Metrics</h3>
            <canvas id="performanceChart"></canvas>
        </div>
    </div>

    <div class="detections-table">
        <h3>Recent Detections</h3>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Source IP</th>
                    <th>Destination IP</th>
                    <th>Class</th>
                    <th>Confidence</th>
                    <th>Action</th>
                    <th>Threat</th>
                </tr>
            </thead>
            <tbody id="detections-tbody">
            </tbody>
        </table>
    </div>

    <script>
        let threatChart, performanceChart;
        
        // Initialize charts
        function initCharts() {
            const threatCtx = document.getElementById('threatChart').getContext('2d');
            threatChart = new Chart(threatCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Threats per Minute',
                        data: [],
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            const perfCtx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Processing Rate (flows/sec)',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        // Update dashboard data
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-flows').textContent = data.total_flows;
                    document.getElementById('threats-detected').textContent = data.threats_detected;
                    document.getElementById('processing-rate').textContent = data.processing_rate.toFixed(2);
                    document.getElementById('avg-processing-time').textContent = data.avg_processing_time.toFixed(2);
                    
                    const statusElement = document.getElementById('status');
                    if (data.status === 'running') {
                        statusElement.textContent = 'System Running';
                        statusElement.className = 'status-running';
                    } else {
                        statusElement.textContent = 'System Stopped';
                        statusElement.className = 'status-stopped';
                    }
                });

            // Update recent detections
            fetch('/api/recent_detections')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('detections-tbody');
                    tbody.innerHTML = '';
                    data.slice(-10).reverse().forEach(detection => {
                        const row = tbody.insertRow();
                        row.className = detection.is_threat ? 'threat' : '';
                        row.innerHTML = `
                            <td>${new Date(detection.timestamp).toLocaleTimeString()}</td>
                            <td>${detection.src_ip}</td>
                            <td>${detection.dst_ip}</td>
                            <td>${detection.predicted_class}</td>
                            <td>${(detection.confidence * 100).toFixed(1)}%</td>
                            <td>${detection.action}</td>
                            <td>${detection.is_threat ? '⚠️' : '✅'}</td>
                        `;
                    });
                });
        }

        // Control functions
        function startDetection() {
            fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rate: 10})
            });
        }

        function stopDetection() {
            fetch('/api/stop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
        }

        // Initialize
        initCharts();
        updateDashboard();
        setInterval(updateDashboard, 2000); // Update every 2 seconds
    </script>
</body>
</html>
'''
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(html_template)

if __name__ == '__main__':
    # Create template
    create_dashboard_template()
    
    print("🌐 Starting NIDS Web Dashboard...")
    print("📊 Dashboard URL: http://localhost:7000")
    print("🛡️ Live detection and monitoring interface")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=7000, debug=False)
