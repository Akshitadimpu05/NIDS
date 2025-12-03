#!/bin/bash
# Comprehensive Docker Fix - Stop containers, fix permissions, recreate files

echo "🔧 Comprehensive Docker Fix"
echo "============================"

# Step 1: Stop all containers to release file locks
echo "🛑 Stopping all NIDS containers..."
docker-compose -f docker/docker-compose.yml down 2>/dev/null || echo "Containers already stopped"

# Step 2: Remove problematic directories/files created by Docker
echo "🗑️  Cleaning up Docker-created directories..."
sudo rm -rf docker/prometheus/prometheus.yml 2>/dev/null || rm -rf docker/prometheus/prometheus.yml 2>/dev/null
sudo rm -rf docker/grafana/provisioning 2>/dev/null || rm -rf docker/grafana/provisioning 2>/dev/null

# Step 3: Create directories with proper structure
echo "📁 Creating directory structure..."
mkdir -p docker/prometheus
mkdir -p docker/grafana/dashboards
mkdir -p docker/grafana/provisioning/dashboards
mkdir -p docker/grafana/provisioning/datasources

# Step 4: Create Prometheus configuration file
echo "📊 Creating Prometheus configuration..."
cat > docker/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  # NIDS Orchestrator metrics
  - job_name: 'nids-orchestrator'
    static_configs:
      - targets: ['orchestrator:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  # NIDS Fog Nodes metrics
  - job_name: 'nids-fog-nodes'
    static_configs:
      - targets: 
          - 'fog-node-1:8080'
          - 'fog-node-2:8080'
          - 'fog-node-3:8080'
    metrics_path: '/metrics'
    scrape_interval: 5s

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

# Step 5: Create Grafana datasource configuration
echo "📈 Creating Grafana datasource configuration..."
cat > docker/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Step 6: Create Grafana dashboard provisioning configuration
echo "📊 Creating Grafana dashboard configuration..."
cat > docker/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'NIDS Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# Step 7: Create a sample NIDS dashboard
echo "🎛️ Creating NIDS dashboard..."
cat > docker/grafana/dashboards/nids-overview.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "NIDS System Overview",
    "tags": ["nids", "security"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(nids_flows_processed_total[5m]))",
            "legendFormat": "Total Flows/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "5s"
  }
}
EOF

# Step 8: Set proper permissions
echo "🔐 Setting proper permissions..."
chmod 644 docker/prometheus/prometheus.yml
chmod -R 644 docker/grafana/provisioning/
chmod -R 644 docker/grafana/dashboards/
chmod 755 docker/prometheus
chmod -R 755 docker/grafana

# Step 9: Verify files exist and are correct type
echo "✅ Verifying files..."
if [ -f "docker/prometheus/prometheus.yml" ]; then
    echo "  ✅ prometheus.yml is a file"
else
    echo "  ❌ prometheus.yml is missing or not a file"
fi

if [ -f "docker/grafana/provisioning/datasources/prometheus.yml" ]; then
    echo "  ✅ Grafana datasource config exists"
else
    echo "  ❌ Grafana datasource config missing"
fi

# Step 10: Show file structure
echo ""
echo "📁 Created file structure:"
find docker/ -type f 2>/dev/null | sort

echo ""
echo "🚀 Ready to restart Docker services!"
echo ""
echo "Run these commands:"
echo "  docker-compose -f docker/docker-compose.yml up -d"
echo ""
echo "Or use the simplified version:"
echo "  docker-compose -f docker/docker-compose-simple.yml up -d"
echo ""
echo "🌐 Access URLs:"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Orchestrator: http://localhost:8000"
