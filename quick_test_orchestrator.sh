#!/bin/bash
set -e

echo "🔧 Quick Orchestrator Rebuild & Test"
echo "====================================="
echo ""

cd "$(dirname "$0")"

# Stop and remove
echo "🛑 Stopping orchestrator..."
docker stop nids-orchestrator 2>/dev/null || true
docker rm nids-orchestrator 2>/dev/null || true
docker rmi docker_orchestrator 2>/dev/null || true
echo ""

# Rebuild
echo "🔨 Rebuilding orchestrator..."
docker build -f docker/Dockerfile.orchestrator -t docker_orchestrator . --no-cache --quiet

echo "✅ Build complete!"
echo ""

# Test imports
echo "🧪 Testing imports..."
docker run --rm docker_orchestrator python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

print('Testing PyTorch...')
import torch
print(f'✅ PyTorch {torch.__version__}')

print('Testing orchestrator imports...')
from orchestrator.api_server import create_api_server
print('✅ api_server imported')

print('Testing orchestrator class...')
from orchestrator.orchestrator import CentralOrchestrator
print('✅ CentralOrchestrator imported')

print('Testing model aggregator...')
from orchestrator.model_aggregator import ModelAggregator
print('✅ ModelAggregator imported')

print('')
print('🎉 All imports successful!')
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed!"
    exit 1
fi
echo ""

# Start container
echo "▶️ Starting orchestrator..."
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

# Wait and check
echo "⏳ Waiting 10 seconds..."
sleep 10
echo ""

echo "📋 Logs:"
docker logs nids-orchestrator --tail 20
echo ""

# Check status
STATUS=$(docker inspect -f '{{.State.Status}}' nids-orchestrator)
echo "📊 Status: $STATUS"
echo ""

if [ "$STATUS" = "running" ]; then
    echo "🎉 SUCCESS! Orchestrator is running!"
    echo ""
    echo "Test health: curl http://localhost:8000/health"
else
    echo "❌ Container not running"
    docker logs nids-orchestrator
    exit 1
fi
