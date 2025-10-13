#!/bin/bash
set -e

echo "🔧 Testing Orchestrator Only - Step by Step"
echo "============================================"
echo ""

cd "$(dirname "$0")"

# Step 1: Stop existing orchestrator
echo "🛑 Step 1: Stopping existing orchestrator..."
docker stop nids-orchestrator 2>/dev/null || true
docker rm nids-orchestrator 2>/dev/null || true
echo "✅ Stopped"
echo ""

# Step 2: Remove old orchestrator image
echo "🗑️ Step 2: Removing old orchestrator image..."
docker rmi docker_orchestrator 2>/dev/null || true
echo "✅ Removed"
echo ""

# Step 3: Build orchestrator only
echo "🔨 Step 3: Building orchestrator image..."
echo "   (This will show full build output)"
docker build -f docker/Dockerfile.orchestrator -t docker_orchestrator . --no-cache

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi
echo "✅ Build successful!"
echo ""

# Step 4: Test PyTorch import in the image
echo "🧪 Step 4: Testing PyTorch import in image..."
docker run --rm docker_orchestrator python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully')"

if [ $? -ne 0 ]; then
    echo "❌ PyTorch import failed!"
    exit 1
fi
echo "✅ PyTorch import successful!"
echo ""

# Step 5: Test orchestrator import
echo "🧪 Step 5: Testing orchestrator imports..."
docker run --rm docker_orchestrator python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
from orchestrator.api_server import create_api_server
print('✅ Orchestrator imports successful!')
"

if [ $? -ne 0 ]; then
    echo "❌ Orchestrator import failed!"
    exit 1
fi
echo "✅ All imports successful!"
echo ""

# Step 6: Start orchestrator with security options
echo "▶️ Step 6: Starting orchestrator container..."
docker run -d \
    --name nids-orchestrator \
    --security-opt seccomp:unconfined \
    -p 8000:8000 \
    -e ORCHESTRATOR_HOST=0.0.0.0 \
    -e ORCHESTRATOR_PORT=8000 \
    -e LOG_LEVEL=INFO \
    docker_orchestrator

echo "✅ Container started"
echo ""

# Step 7: Wait and check logs
echo "⏳ Step 7: Waiting 15 seconds for startup..."
sleep 15
echo ""

echo "📋 Container logs:"
docker logs nids-orchestrator --tail 30
echo ""

# Step 8: Check if container is running
echo "📊 Step 8: Checking container status..."
STATUS=$(docker inspect -f '{{.State.Status}}' nids-orchestrator)
echo "   Status: $STATUS"

if [ "$STATUS" != "running" ]; then
    echo "❌ Container is not running!"
    echo ""
    echo "Full logs:"
    docker logs nids-orchestrator
    exit 1
fi
echo "✅ Container is running!"
echo ""

# Step 9: Test health endpoint
echo "🧪 Step 9: Testing health endpoint..."
sleep 5

if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    RESPONSE=$(curl -s http://localhost:8000/health)
    echo "✅ Health endpoint working!"
    echo "   Response: $RESPONSE"
else
    echo "⚠️  Health endpoint not responding yet (may need more time)"
    echo "   Check logs: docker logs nids-orchestrator"
fi
echo ""

echo "============================================"
echo "🎉 Orchestrator Test Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. If successful, run: ./test_fognode_only.sh"
echo "2. Then run full deployment: ./deploy_docker.sh"
echo ""
echo "To stop orchestrator: docker stop nids-orchestrator"
echo "To view logs: docker logs nids-orchestrator -f"
echo ""
