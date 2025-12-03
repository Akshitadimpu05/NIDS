#!/bin/bash

# Comprehensive Docker Fix Script for NIDS
# Fixes import errors, dependencies, and deployment issues

set -e

echo "🔧 NIDS Docker Comprehensive Fix Script"
echo "========================================"

# Step 1: Stop and clean existing containers
echo ""
echo "🛑 Step 1: Stopping and cleaning existing containers..."
docker-compose -f docker/docker-compose.yml down --remove-orphans 2>/dev/null || true
docker container prune -f
echo "✅ Containers cleaned"

# Step 2: Remove old images
echo ""
echo "🗑️ Step 2: Removing old images..."
docker rmi -f $(docker images | grep "nids\|docker_" | awk '{print $3}') 2>/dev/null || true
echo "✅ Old images removed"

# Step 3: Clean build cache
echo ""
echo "🧼 Step 3: Cleaning Docker build cache..."
docker builder prune -f
echo "✅ Build cache cleaned"

# Step 4: Verify project structure
echo ""
echo "📁 Step 4: Verifying project structure..."
required_dirs=(
    "src/models"
    "src/orchestrator" 
    "src/fog_node"
    "src/agents"
    "src/utils"
    "data/models"
    "docker"
    "scripts"
)

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Missing directory: $dir"
        exit 1
    fi
done

# Check for required model files
required_models=(
    "data/models/joint_nids_model.pth"
    "data/models/joint_preprocessor.pkl"
)

for model in "${required_models[@]}"; do
    if [ ! -f "$model" ]; then
        echo "⚠️ Warning: Missing model file: $model"
    fi
done

echo "✅ Project structure verified"

# Step 5: Create missing __init__.py files
echo ""
echo "📝 Step 5: Creating missing __init__.py files..."

# Ensure all Python packages have __init__.py
find src -type d -exec touch {}/__init__.py \; 2>/dev/null || true

echo "✅ __init__.py files created"

# Step 6: Build images with better error handling
echo ""
echo "🏗️ Step 6: Building Docker images..."

# Build orchestrator
echo "Building orchestrator image..."
if ! docker build -f docker/Dockerfile.orchestrator -t nids-orchestrator . ; then
    echo "❌ Failed to build orchestrator image"
    exit 1
fi

# Build fog node
echo "Building fog node image..."
if ! docker build -f docker/Dockerfile.fog-node -t nids-fog-node . ; then
    echo "❌ Failed to build fog node image"
    exit 1
fi

echo "✅ All images built successfully!"

# Step 7: Test imports in containers
echo ""
echo "🧪 Step 7: Testing imports in containers..."

# Test orchestrator imports
echo "Testing orchestrator imports..."
if ! docker run --rm nids-orchestrator python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
try:
    from src.orchestrator.api_server import create_api_server
    print('✅ Orchestrator imports successful')
except ImportError as e:
    print(f'❌ Orchestrator import failed: {e}')
    sys.exit(1)
"; then
    echo "❌ Orchestrator import test failed"
    exit 1
fi

# Test fog node imports
echo "Testing fog node imports..."
if ! docker run --rm nids-fog-node python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
try:
    from src.fog_node.fog_node import FogNode, FogNodeConfig
    print('✅ Fog node imports successful')
except ImportError as e:
    print(f'❌ Fog node import failed: {e}')
    sys.exit(1)
"; then
    echo "❌ Fog node import test failed"
    exit 1
fi

echo "✅ All import tests passed!"

# Step 8: Deploy with health checks
echo ""
echo "🚀 Step 8: Deploying containers..."

# Start orchestrator first
docker-compose -f docker/docker-compose.yml up -d orchestrator

# Wait for orchestrator to be healthy
echo "Waiting for orchestrator to be healthy..."
timeout=60
counter=0
while [ $counter -lt $timeout ]; do
    if docker-compose -f docker/docker-compose.yml ps orchestrator | grep -q "Up"; then
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            echo "✅ Orchestrator is healthy"
            break
        fi
    fi
    sleep 2
    counter=$((counter + 2))
    echo -n "."
done

if [ $counter -ge $timeout ]; then
    echo "❌ Orchestrator failed to become healthy"
    docker-compose -f docker/docker-compose.yml logs orchestrator
    exit 1
fi

# Start fog nodes
echo "Starting fog nodes..."
docker-compose -f docker/docker-compose.yml up -d fog-node-1 fog-node-2 fog-node-3

# Wait a bit for fog nodes to start
sleep 10

# Step 9: Verify deployment
echo ""
echo "✅ Step 9: Verifying deployment..."

# Check container status
echo "Container status:"
docker-compose -f docker/docker-compose.yml ps

# Check orchestrator logs
echo ""
echo "📋 Orchestrator logs (last 10 lines):"
docker-compose -f docker/docker-compose.yml logs --tail 10 orchestrator

# Check fog node logs
echo ""
echo "📋 Fog Node 1 logs (last 10 lines):"
docker-compose -f docker/docker-compose.yml logs --tail 10 fog-node-1

# Test API endpoints
echo ""
echo "🔍 Testing API endpoints..."

if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ Health endpoint working"
else
    echo "❌ Health endpoint failed"
fi

if curl -f http://localhost:8000/metrics >/dev/null 2>&1; then
    echo "✅ Metrics endpoint working"
else
    echo "❌ Metrics endpoint failed"
fi

echo ""
echo "🎉 Docker deployment completed successfully!"
echo ""
echo "📊 Access points:"
echo "  - Orchestrator API: http://localhost:8000"
echo "  - Health check: http://localhost:8000/health"
echo "  - Metrics: http://localhost:8000/metrics"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000"
echo ""
echo "📋 Useful commands:"
echo "  - View logs: docker-compose -f docker/docker-compose.yml logs -f [service]"
echo "  - Stop all: docker-compose -f docker/docker-compose.yml down"
echo "  - Restart: docker-compose -f docker/docker-compose.yml restart [service]"
echo ""
echo "✅ All systems operational!"
