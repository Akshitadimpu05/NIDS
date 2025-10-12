#!/bin/bash
set -e

echo "🔧 Fixing PyTorch libtorch_cpu.so Issue and Redeploying"
echo "========================================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# Step 1: Stop all containers
echo "🛑 Step 1: Stopping all containers..."
docker-compose -f docker/docker-compose.yml down
echo "✅ Containers stopped"
echo ""

# Step 2: Remove old images to force rebuild
echo "🗑️ Step 2: Removing old images..."
docker rmi docker_orchestrator docker_fog-node-1 docker_fog-node-2 docker_fog-node-3 2>/dev/null || true
echo "✅ Old images removed"
echo ""

# Step 3: Rebuild with PyTorch fix
echo "🔨 Step 3: Rebuilding images with PyTorch fix..."
echo "   (This includes execstack utility and security_opt settings)"
docker-compose -f docker/docker-compose.yml build --no-cache orchestrator fog-node-1

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Check error messages above."
    exit 1
fi
echo "✅ Images rebuilt successfully"
echo ""

# Step 4: Start services
echo "▶️ Step 4: Starting services..."
docker-compose -f docker/docker-compose.yml up -d
echo "✅ Services started"
echo ""

# Step 5: Wait for initialization
echo "⏳ Step 5: Waiting 30 seconds for initialization..."
for i in {1..30}; do
    echo -n "."
    sleep 1
done
echo ""
echo "✅ Wait complete"
echo ""

# Step 6: Check status
echo "📊 Step 6: Checking container status..."
docker-compose -f docker/docker-compose.yml ps
echo ""

# Step 7: Check logs
echo "📋 Step 7: Checking logs for PyTorch errors..."
echo ""
echo "--- Orchestrator Logs ---"
docker logs nids-orchestrator --tail 15 2>&1 | grep -E "(✅|❌|libtorch|Starting|Error)" || echo "No errors found"
echo ""
echo "--- Fog Node 1 Logs ---"
docker logs nids-fog-node-1 --tail 15 2>&1 | grep -E "(✅|❌|libtorch|Starting|Error)" || echo "No errors found"
echo ""

# Step 8: Test endpoints
echo "🧪 Step 8: Testing endpoints..."
sleep 5

echo -n "Orchestrator health: "
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ WORKING"
else
    echo "❌ NOT RESPONDING"
fi

echo ""
echo "========================================================"
echo "🎉 Redeployment Complete!"
echo "========================================================"
echo ""
echo "If containers are still restarting, check full logs:"
echo "  docker logs nids-orchestrator --tail 50"
echo "  docker logs nids-fog-node-1 --tail 50"
echo ""
