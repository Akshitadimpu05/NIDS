#!/bin/bash
set -e

echo "🔧 Quick Fog Node Rebuild & Test"
echo "================================="
echo ""

cd "$(dirname "$0")"

# Stop and remove
echo "🛑 Stopping fog node..."
docker stop nids-fog-node-1 2>/dev/null || true
docker rm nids-fog-node-1 2>/dev/null || true
docker rmi docker_fog-node-1 2>/dev/null || true
echo ""

# Rebuild
echo "🔨 Rebuilding fog node..."
docker build -f docker/Dockerfile.fog-node -t docker_fog-node-1 . --no-cache --quiet

echo "✅ Build complete!"
echo ""

# Test imports
echo "🧪 Testing imports..."
docker run --rm docker_fog-node-1 python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

print('Testing PyTorch...')
import torch
print(f'✅ PyTorch {torch.__version__}')

print('Testing fog node imports...')
from fog_node.fog_node import FogNode, FogNodeConfig
print('✅ FogNode imported')

print('Testing traffic capture...')
from fog_node.traffic_capture import TrafficCapture
print('✅ TrafficCapture imported')

print('Testing mitigation...')
from fog_node.mitigation import TrafficMitigation
print('✅ TrafficMitigation imported')

print('')
print('🎉 All imports successful!')
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed!"
    exit 1
fi
echo ""

# Check if orchestrator is running
echo "🔍 Checking orchestrator..."
if docker ps | grep -q nids-orchestrator; then
    echo "✅ Orchestrator is running"
    ORCH_URL="http://orchestrator:8000"
    NETWORK="--network docker_nids-network"
else
    echo "⚠️  Orchestrator not running"
    ORCH_URL="http://localhost:8000"
    NETWORK=""
fi
echo ""

# Start container
echo "▶️ Starting fog node..."
docker run -d \
    --name nids-fog-node-1 \
    --security-opt seccomp:unconfined \
    --cap-add NET_ADMIN \
    $NETWORK \
    -e NODE_ID=fog-node-1 \
    -e ORCHESTRATOR_URL=$ORCH_URL \
    -e CAPTURE_INTERFACE=eth0 \
    -e LOG_LEVEL=INFO \
    -e ENABLE_MITIGATION=false \
    docker_fog-node-1

echo "✅ Container started"
echo ""

# Wait and check
echo "⏳ Waiting 10 seconds..."
sleep 10
echo ""

echo "📋 Logs:"
docker logs nids-fog-node-1 --tail 20
echo ""

# Check status
STATUS=$(docker inspect -f '{{.State.Status}}' nids-fog-node-1)
echo "📊 Status: $STATUS"
echo ""

if [ "$STATUS" = "running" ]; then
    echo "🎉 SUCCESS! Fog node is running!"
    echo ""
    echo "Both orchestrator and fog node are working!"
else
    echo "❌ Container not running"
    docker logs nids-fog-node-1
    exit 1
fi
