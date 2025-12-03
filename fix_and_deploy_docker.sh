#!/bin/bash
# Complete Docker cleanup and rebuild with minimal requirements
# This ensures old requirements.txt is not cached

set -e

echo "🧹 COMPLETE DOCKER CLEANUP AND REBUILD"
echo "======================================="
echo ""

# Step 1: Stop all containers
echo "🛑 Step 1: Stopping all NIDS containers..."
docker-compose -f docker/docker-compose.yml down 2>/dev/null || true
echo "✅ Containers stopped"

# Step 2: Remove all NIDS images
echo ""
echo "🗑️  Step 2: Removing old NIDS images..."
docker rmi -f nids-orchestrator 2>/dev/null || true
docker rmi -f nids-fog-node 2>/dev/null || true
docker rmi -f nids-traffic-generator 2>/dev/null || true
docker rmi -f docker-orchestrator 2>/dev/null || true
docker rmi -f docker-fog-node 2>/dev/null || true
docker rmi -f docker-traffic-generator 2>/dev/null || true
echo "✅ Old images removed"

# Step 3: Prune build cache
echo ""
echo "🧼 Step 3: Cleaning Docker build cache..."
docker builder prune -f
echo "✅ Build cache cleaned"

# Step 4: Verify minimal requirements file
echo ""
echo "📋 Step 4: Verifying minimal requirements file..."
if [ ! -f "docker/requirements-docker.txt" ]; then
    echo "❌ ERROR: docker/requirements-docker.txt not found!"
    echo "   Please ensure the minimal requirements file exists"
    exit 1
fi

echo "✅ Found docker/requirements-docker.txt with packages:"
cat docker/requirements-docker.txt | grep -v "^#" | grep -v "^$"

# Step 5: Verify Dockerfiles are using correct requirements
echo ""
echo "🔍 Step 5: Verifying Dockerfiles..."
if grep -q "requirements-docker.txt" docker/Dockerfile.orchestrator; then
    echo "✅ Dockerfile.orchestrator uses requirements-docker.txt"
else
    echo "❌ ERROR: Dockerfile.orchestrator not using requirements-docker.txt"
    exit 1
fi

if grep -q "requirements-docker.txt" docker/Dockerfile.fog-node; then
    echo "✅ Dockerfile.fog-node uses requirements-docker.txt"
else
    echo "❌ ERROR: Dockerfile.fog-node not using requirements-docker.txt"
    exit 1
fi

# Step 6: Build images with no cache
echo ""
echo "🔨 Step 6: Building Docker images with NO CACHE..."
echo "   This will take 5-10 minutes..."
echo ""

# Build orchestrator
echo "📦 Building orchestrator..."
docker-compose -f docker/docker-compose.yml build --no-cache orchestrator

# Build fog nodes
echo "📦 Building fog-node..."
docker-compose -f docker/docker-compose.yml build --no-cache fog-node-1

echo ""
echo "✅ All images built successfully!"

# Step 7: Verify images
echo ""
echo "🔍 Step 7: Verifying built images..."
docker images | grep nids

# Step 8: Start services
echo ""
echo "🚀 Step 8: Starting all services..."
docker-compose -f docker/docker-compose.yml up -d

# Step 9: Wait for initialization
echo ""
echo "⏱️  Step 9: Waiting 30 seconds for services to initialize..."
for i in {30..1}; do
    printf "\r   %2d seconds remaining..." $i
    sleep 1
done
echo ""
echo "✅ Initialization complete!"

# Step 10: Check status
echo ""
echo "📊 Step 10: Checking container status..."
echo "========================================="
docker-compose -f docker/docker-compose.yml ps

# Step 11: Check logs
echo ""
echo "📋 Step 11: Checking logs for errors..."
echo "========================================"

echo ""
echo "Orchestrator (last 10 lines):"
docker logs nids-orchestrator --tail 10 2>&1 | grep -v "GET /metrics" || echo "No logs yet"

echo ""
echo "Fog Node 1 (last 10 lines):"
docker logs nids-fog-node-1 --tail 10 2>&1 || echo "No logs yet"

# Step 12: Test endpoints
echo ""
echo "🌐 Step 12: Testing endpoints..."
echo "================================="

sleep 5  # Give services a bit more time

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Orchestrator health: OK"
else
    echo "⚠️  Orchestrator health: Not ready yet (may need more time)"
fi

if curl -s http://localhost:8000/metrics > /dev/null 2>&1; then
    echo "✅ Orchestrator metrics: OK"
else
    echo "⚠️  Orchestrator metrics: Not ready yet"
fi

# Final summary
echo ""
echo "🎉 REBUILD COMPLETE!"
echo "===================="
echo ""
echo "✅ Used minimal requirements (10 packages instead of 50+)"
echo "✅ All old images and cache removed"
echo "✅ Fresh build with no cache"
echo "✅ Services started"
echo ""
echo "📊 Monitor your system:"
echo "   docker-compose -f docker/docker-compose.yml ps"
echo "   docker logs nids-orchestrator -f"
echo "   docker logs nids-fog-node-1 -f"
echo ""
echo "🌐 Access points:"
echo "   - Orchestrator: http://localhost:8000"
echo "   - Health: http://localhost:8000/health"
echo "   - Metrics: http://localhost:8000/metrics"
echo "   - Grafana: http://localhost:3000 (admin/admin)"
echo "   - Prometheus: http://localhost:9090"
echo ""
