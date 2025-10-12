#!/bin/bash
# Restart Docker NIDS with all asyncio fixes applied

echo "🔧 RESTARTING DOCKER NIDS WITH FIXES"
echo "====================================="

# Step 1: Stop all containers
echo "🛑 Stopping all containers..."
docker-compose -f docker/docker-compose.yml down

# Step 2: Rebuild images with no cache
echo "🔨 Rebuilding Docker images (this may take a few minutes)..."
docker-compose -f docker/docker-compose.yml build --no-cache

# Step 3: Start all services
echo "🚀 Starting all services..."
docker-compose -f docker/docker-compose.yml up -d

# Step 4: Wait for services to initialize
echo "⏱️  Waiting 30 seconds for services to initialize..."
sleep 30

# Step 5: Check status
echo ""
echo "📊 CHECKING SYSTEM STATUS"
echo "========================="
docker-compose -f docker/docker-compose.yml ps

echo ""
echo "🔍 CHECKING NIDS SYSTEM"
echo "======================="
python check_distributed_system.py status

echo ""
echo "✅ RESTART COMPLETE!"
echo ""
echo "🌐 Access Points:"
echo "  - Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Orchestrator API: http://localhost:8000/health"
echo ""
echo "📋 View Logs:"
echo "  docker logs nids-orchestrator"
echo "  docker logs nids-fog-node-1"
echo "  docker logs nids-fog-node-2"
echo "  docker logs nids-fog-node-3"
