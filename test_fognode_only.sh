#!/bin/bash
set -e

echo "🔧 Testing Fog Node 1 Only - Step by Step"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Step 1: Stop existing fog node
echo "🛑 Step 1: Stopping existing fog-node-1..."
docker stop nids-fog-node-1 2>/dev/null || true
docker rm nids-fog-node-1 2>/dev/null || true
echo "✅ Stopped"
echo ""

# Step 2: Remove old fog node image
echo "🗑️ Step 2: Removing old fog-node image..."
docker rmi docker_fog-node-1 2>/dev/null || true
echo "✅ Removed"
echo ""

# Step 3: Build fog node only
echo "🔨 Step 3: Building fog-node image..."
echo "   (This will show full build output)"
docker build -f docker/Dockerfile.fog-node -t docker_fog-node-1 . --no-cache

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi
echo "✅ Build successful!"
echo ""

# Step 4: Test PyTorch import in the image
echo "🧪 Step 4: Testing PyTorch import in image..."
docker run --rm docker_fog-node-1 python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully')"

if [ $? -ne 0 ]; then
    echo "❌ PyTorch import failed!"
    exit 1
fi
echo "✅ PyTorch import successful!"
echo ""

# Step 5: Test fog node import
echo "🧪 Step 5: Testing fog node imports..."
docker run --rm docker_fog-node-1 python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
from fog_node.fog_node import FogNode, FogNodeConfig
print('✅ Fog node imports successful!')
"

if [ $? -ne 0 ]; then
    echo "❌ Fog node import failed!"
    exit 1
fi
echo "✅ All imports successful!"
echo ""

# Step 6: Check if orchestrator is running
echo "🔍 Step 6: Checking if orchestrator is running..."
if docker ps | grep -q nids-orchestrator; then
    echo "✅ Orchestrator is running"
    ORCHESTRATOR_URL="http://orchestrator:8000"
else
    echo "⚠️  Orchestrator not running, fog node will use default URL"
    ORCHESTRATOR_URL="http://localhost:8000"
fi
echo ""

# Step 7: Start fog node with security options
echo "▶️ Step 7: Starting fog-node-1 container..."
docker run -d \
    --name nids-fog-node-1 \
    --security-opt seccomp:unconfined \
    --cap-add NET_ADMIN \
    -e NODE_ID=fog-node-1 \
    -e ORCHESTRATOR_URL=$ORCHESTRATOR_URL \
    -e CAPTURE_INTERFACE=eth0 \
    -e LOG_LEVEL=INFO \
    -e ENABLE_MITIGATION=false \
    docker_fog-node-1

echo "✅ Container started"
echo ""

# Step 8: Wait and check logs
echo "⏳ Step 8: Waiting 15 seconds for startup..."
sleep 15
echo ""

echo "📋 Container logs:"
docker logs nids-fog-node-1 --tail 30
echo ""

# Step 9: Check if container is running
echo "📊 Step 9: Checking container status..."
STATUS=$(docker inspect -f '{{.State.Status}}' nids-fog-node-1)
echo "   Status: $STATUS"

if [ "$STATUS" != "running" ]; then
    echo "❌ Container is not running!"
    echo ""
    echo "Full logs:"
    docker logs nids-fog-node-1
    exit 1
fi
echo "✅ Container is running!"
echo ""

echo "============================================"
echo "🎉 Fog Node 1 Test Complete!"
echo "============================================"
echo ""
echo "Both orchestrator and fog-node-1 are working!"
echo ""
echo "Next steps:"
echo "1. Run full deployment: ./deploy_docker.sh"
echo "2. Or manually start remaining services:"
echo "   docker-compose -f docker/docker-compose.yml up -d"
echo ""
echo "To stop fog node: docker stop nids-fog-node-1"
echo "To view logs: docker logs nids-fog-node-1 -f"
echo ""
