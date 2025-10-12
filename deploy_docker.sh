#!/bin/bash
set -e  # Exit on error

echo "🚀 NIDS Docker Deployment - Portable Version"
echo "=============================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Step 1: Verify required files exist
echo ""
echo "✅ Step 1: Verifying required files..."
required_files=(
    "docker/docker-compose.yml"
    "docker/Dockerfile.orchestrator"
    "docker/Dockerfile.fog-node"
    "docker/requirements-docker.txt"
    "docker/prometheus/prometheus.yml"
    "scripts/run_orchestrator.py"
    "scripts/run_fog_node.py"
    "src/orchestrator/api_server.py"
    "src/fog_node/fog_node.py"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ] && [ ! -d "$file" ]; then
        echo "❌ Missing required file: $file"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -gt 0 ]; then
    echo "❌ $missing_files required files are missing. Please check your project structure."
    exit 1
fi
echo "✅ All required files present"

# Step 2: Verify model files (optional but recommended)
echo ""
echo "✅ Step 2: Checking model files..."
if [ -f "data/models/joint_nids_model.pth" ]; then
    echo "✅ Found joint_nids_model.pth"
else
    echo "⚠️ Warning: joint_nids_model.pth not found (fog nodes may fail to start)"
fi

if [ -f "data/models/joint_rl_agent.pth" ]; then
    echo "✅ Found joint_rl_agent.pth"
else
    echo "⚠️ Warning: joint_rl_agent.pth not found (RL agent may not work)"
fi

# Step 3: Verify __init__.py files
echo ""
echo "✅ Step 3: Verifying __init__.py files..."
init_files=(
    "src/__init__.py"
    "src/orchestrator/__init__.py"
    "src/fog_node/__init__.py"
    "src/models/__init__.py"
    "src/agents/__init__.py"
    "src/utils/__init__.py"
)

for file in "${init_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️ Creating missing $file"
        mkdir -p "$(dirname "$file")"
        echo '"""NIDS Module"""' > "$file"
    fi
done
echo "✅ All __init__.py files present"

# Step 4: Stop existing containers
echo ""
echo "🛑 Step 4: Stopping existing containers..."
docker-compose -f docker/docker-compose.yml down 2>/dev/null || true
echo "✅ Stopped existing containers"

# Step 5: Clean up old images (optional)
echo ""
read -p "🗑️ Clean up old Docker images and build cache? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Cleaning Docker system..."
    docker system prune -f
    echo "✅ Cleanup complete"
fi

# Step 6: Build images
echo ""
echo "🔨 Step 6: Building Docker images (this may take a few minutes)..."
docker-compose -f docker/docker-compose.yml build --no-cache

if [ $? -eq 0 ]; then
    echo "✅ Docker images built successfully"
else
    echo "❌ Docker build failed. Check the error messages above."
    exit 1
fi

# Step 7: Start services
echo ""
echo "▶️ Step 7: Starting services..."
docker-compose -f docker/docker-compose.yml up -d

if [ $? -eq 0 ]; then
    echo "✅ Services started"
else
    echo "❌ Failed to start services"
    exit 1
fi

# Step 8: Wait for services to initialize
echo ""
echo "⏳ Step 8: Waiting for services to initialize (30 seconds)..."
for i in {1..30}; do
    echo -n "."
    sleep 1
done
echo ""
echo "✅ Initialization period complete"

# Step 9: Check container status
echo ""
echo "📊 Step 9: Checking container status..."
docker-compose -f docker/docker-compose.yml ps

# Step 10: Check logs for errors
echo ""
echo "📋 Step 10: Checking logs for import errors..."
echo ""
echo "--- Orchestrator Logs (last 20 lines) ---"
docker logs nids-orchestrator --tail 20 2>&1 | grep -E "(✅|❌|Error|Failed|import|Starting)" || echo "No significant messages"

echo ""
echo "--- Fog Node 1 Logs (last 20 lines) ---"
docker logs nids-fog-node-1 --tail 20 2>&1 | grep -E "(✅|❌|Error|Failed|import|Starting)" || echo "No significant messages"

# Step 11: Test endpoints
echo ""
echo "🧪 Step 11: Testing endpoints..."
sleep 5

echo -n "Testing orchestrator health... "
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ PASSED"
else
    echo "❌ FAILED"
fi

echo -n "Testing orchestrator metrics... "
if curl -f -s http://localhost:8000/metrics > /dev/null 2>&1; then
    echo "✅ PASSED"
else
    echo "❌ FAILED (may not be implemented yet)"
fi

# Step 12: Display access information
echo ""
echo "=============================================="
echo "🎉 Deployment Complete!"
echo "=============================================="
echo ""
echo "📊 Access Points:"
echo "  • Orchestrator API:    http://localhost:8000"
echo "  • Orchestrator Health: http://localhost:8000/health"
echo "  • Orchestrator Metrics: http://localhost:8000/metrics"
echo "  • Grafana Dashboard:   http://localhost:3000 (admin/admin)"
echo "  • Prometheus:          http://localhost:9090"
echo ""
echo "📋 Useful Commands:"
echo "  • View all logs:       docker-compose -f docker/docker-compose.yml logs -f"
echo "  • View orchestrator:   docker logs nids-orchestrator -f"
echo "  • View fog node:       docker logs nids-fog-node-1 -f"
echo "  • Stop services:       docker-compose -f docker/docker-compose.yml down"
echo "  • Restart services:    docker-compose -f docker/docker-compose.yml restart"
echo ""
echo "🔍 Debug Commands:"
echo "  • Check status:        docker-compose -f docker/docker-compose.yml ps"
echo "  • Exec into container: docker exec -it nids-orchestrator /bin/bash"
echo "  • View full logs:      docker logs nids-orchestrator --tail 100"
echo ""
echo "⚠️ If containers are restarting:"
echo "  1. Check logs: docker logs nids-orchestrator --tail 50"
echo "  2. Look for import errors or missing files"
echo "  3. Verify model files exist in data/models/"
echo ""
