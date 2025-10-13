#!/bin/bash
set -e

echo "🔧 Final PyTorch Fix - Comprehensive Solution"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# Step 1: Verify security_opt is in docker-compose.yml
echo "✅ Step 1: Verifying security_opt in docker-compose.yml..."
if grep -q "security_opt:" docker/docker-compose.yml; then
    echo "✅ security_opt found in docker-compose.yml"
    grep -A 1 "security_opt:" docker/docker-compose.yml | head -6
else
    echo "❌ security_opt NOT found in docker-compose.yml"
    echo "⚠️  This is required for PyTorch to work!"
    exit 1
fi
echo ""

# Step 2: Stop and remove ALL containers and volumes
echo "🛑 Step 2: Stopping and removing all containers..."
docker-compose -f docker/docker-compose.yml down -v
docker container prune -f
echo "✅ All containers stopped and removed"
echo ""

# Step 3: Remove ALL images
echo "🗑️ Step 3: Removing all NIDS images..."
docker rmi -f $(docker images | grep -E "docker_|nids-" | awk '{print $3}') 2>/dev/null || true
echo "✅ All images removed"
echo ""

# Step 4: Clean Docker build cache
echo "🧹 Step 4: Cleaning Docker build cache..."
docker builder prune -f
echo "✅ Build cache cleaned"
echo ""

# Step 5: Rebuild with updated PyTorch version
echo "🔨 Step 5: Rebuilding with PyTorch 1.13.1 (better compatibility)..."
echo "   This may take 5-10 minutes..."
docker-compose -f docker/docker-compose.yml build --no-cache --progress=plain orchestrator fog-node-1 2>&1 | tee build.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Build failed. Check build.log for details."
    exit 1
fi
echo "✅ Images rebuilt successfully"
echo ""

# Step 6: Start services with security options
echo "▶️ Step 6: Starting services..."
docker-compose -f docker/docker-compose.yml up -d
echo "✅ Services started"
echo ""

# Step 7: Verify security options are applied
echo "🔍 Step 7: Verifying security options..."
sleep 5
SECURITY_CHECK=$(docker inspect nids-orchestrator 2>/dev/null | grep -A 2 "SecurityOpt" || echo "NOT_FOUND")
if echo "$SECURITY_CHECK" | grep -q "seccomp:unconfined"; then
    echo "✅ Security options correctly applied"
else
    echo "⚠️  Security options may not be applied correctly"
    echo "$SECURITY_CHECK"
fi
echo ""

# Step 8: Wait for initialization
echo "⏳ Step 8: Waiting 45 seconds for initialization..."
for i in {1..45}; do
    echo -n "."
    sleep 1
done
echo ""
echo "✅ Initialization period complete"
echo ""

# Step 9: Check container status
echo "📊 Step 9: Checking container status..."
docker-compose -f docker/docker-compose.yml ps
echo ""

# Step 10: Check logs for PyTorch errors
echo "📋 Step 10: Checking logs..."
echo ""
echo "--- Orchestrator Logs (last 25 lines) ---"
docker logs nids-orchestrator --tail 25 2>&1
echo ""
echo "--- Fog Node 1 Logs (last 25 lines) ---"
docker logs nids-fog-node-1 --tail 25 2>&1
echo ""

# Step 11: Test endpoints
echo "🧪 Step 11: Testing endpoints..."
sleep 10

echo -n "Orchestrator health: "
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ WORKING"
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    echo "   Response: $HEALTH_RESPONSE"
else
    echo "❌ NOT RESPONDING"
fi
echo ""

# Step 12: Final status
echo "=============================================="
echo "📊 Final Status"
echo "=============================================="
docker-compose -f docker/docker-compose.yml ps
echo ""

# Check if any containers are restarting
RESTARTING=$(docker-compose -f docker/docker-compose.yml ps | grep -c "Restarting" || true)
if [ "$RESTARTING" -gt 0 ]; then
    echo "⚠️  WARNING: $RESTARTING container(s) still restarting"
    echo ""
    echo "🔍 Troubleshooting steps:"
    echo "1. Check full logs: docker logs nids-orchestrator --tail 100"
    echo "2. Verify PyTorch import: docker exec nids-orchestrator python -c 'import torch; print(torch.__version__)'"
    echo "3. Check security opts: docker inspect nids-orchestrator | grep -A 5 SecurityOpt"
    echo ""
else
    echo "🎉 All containers running successfully!"
    echo ""
    echo "📊 Access Points:"
    echo "  • Orchestrator: http://localhost:8000"
    echo "  • Grafana: http://localhost:3000 (admin/admin)"
    echo "  • Prometheus: http://localhost:9090"
    echo ""
fi
