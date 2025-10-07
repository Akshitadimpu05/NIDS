"""
FastAPI server for the Central Orchestrator.
Provides REST API endpoints for fog node communication.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import asyncio
from contextlib import asynccontextmanager

from .orchestrator import CentralOrchestrator

logger = logging.getLogger(__name__)


# Pydantic models for API
class NodeRegistration(BaseModel):
    node_id: str
    capabilities: Dict[str, Any]
    model_info: Optional[Dict[str, Any]] = None
    timestamp: float


class Heartbeat(BaseModel):
    node_id: str
    status: str
    timestamp: float
    model_version: Optional[str] = "1.0.0"


class MetricsReport(BaseModel):
    node_id: str
    timestamp: float
    performance: Dict[str, Any]
    actions: Dict[str, int]
    queues: Dict[str, Any]
    model: Dict[str, Any]


class SecurityEvent(BaseModel):
    node_id: str
    timestamp: float
    event: Dict[str, Any]


class ModelUpload(BaseModel):
    node_id: str
    model_data: Dict[str, Any]
    timestamp: float


# Global orchestrator instance
orchestrator: Optional[CentralOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global orchestrator
    
    # Startup
    logger.info("Starting Central Orchestrator API server...")
    orchestrator = CentralOrchestrator(
        model_aggregation_interval=3600,  # 1 hour
        node_timeout=300,  # 5 minutes
        enable_federated_learning=True
    )
    await orchestrator.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Central Orchestrator...")
    if orchestrator:
        await orchestrator.stop()


def create_api_server() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="NIDS Central Orchestrator API",
        description="REST API for NIDS fog computing orchestration",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "NIDS Central Orchestrator",
            "version": "1.0.0",
            "status": "running" if orchestrator and orchestrator.is_running else "stopped"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        if not orchestrator or not orchestrator.is_running:
            raise HTTPException(status_code=503, detail="Orchestrator not running")
        
        return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}
    
    @app.post("/api/nodes/register")
    async def register_node(registration: NodeRegistration):
        """Register a new fog node."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        try:
            response = await orchestrator.register_node(registration.dict())
            return response
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Node registration error: {e}")
            raise HTTPException(status_code=500, detail="Registration failed")
    
    @app.post("/api/nodes/heartbeat")
    async def handle_heartbeat(heartbeat: Heartbeat):
        """Handle heartbeat from fog node."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        try:
            response = await orchestrator.handle_heartbeat(heartbeat.dict())
            return response
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            raise HTTPException(status_code=500, detail="Heartbeat failed")
    
    @app.get("/api/nodes")
    async def list_nodes():
        """List all registered fog nodes."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        with orchestrator.nodes_lock:
            nodes = {
                node_id: {
                    "status": node.status,
                    "last_heartbeat": node.last_heartbeat,
                    "registration_time": node.registration_time,
                    "model_version": node.model_version,
                    "capabilities": node.capabilities
                }
                for node_id, node in orchestrator.fog_nodes.items()
            }
        
        return {"nodes": nodes, "total_count": len(nodes)}
    
    @app.get("/api/nodes/{node_id}")
    async def get_node_info(node_id: str):
        """Get information about a specific fog node."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        with orchestrator.nodes_lock:
            if node_id not in orchestrator.fog_nodes:
                raise HTTPException(status_code=404, detail="Node not found")
            
            node = orchestrator.fog_nodes[node_id]
            return {
                "node_id": node_id,
                "status": node.status,
                "last_heartbeat": node.last_heartbeat,
                "registration_time": node.registration_time,
                "model_version": node.model_version,
                "capabilities": node.capabilities,
                "metrics": node.metrics
            }
    
    @app.post("/api/metrics/report")
    async def report_metrics(metrics: MetricsReport):
        """Receive metrics report from fog node."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        try:
            await orchestrator.collect_metrics(metrics.dict())
            return {"status": "received", "timestamp": asyncio.get_event_loop().time()}
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            raise HTTPException(status_code=500, detail="Metrics collection failed")
    
    @app.get("/api/metrics/global")
    async def get_global_metrics():
        """Get aggregated global metrics."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        try:
            global_metrics = orchestrator._aggregate_global_metrics()
            return {"metrics": global_metrics}
        except Exception as e:
            logger.error(f"Global metrics error: {e}")
            raise HTTPException(status_code=500, detail="Failed to get global metrics")
    
    @app.post("/api/events/security")
    async def report_security_event(event: SecurityEvent):
        """Report security event from fog node."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        try:
            await orchestrator.report_security_event(event.dict())
            return {"status": "processed", "timestamp": asyncio.get_event_loop().time()}
        except Exception as e:
            logger.error(f"Security event processing error: {e}")
            raise HTTPException(status_code=500, detail="Event processing failed")
    
    @app.get("/api/events/security")
    async def get_security_events(limit: int = 100, severity: Optional[str] = None):
        """Get recent security events."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        events = orchestrator.security_events[-limit:]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        return {
            "events": [
                {
                    "event_id": e.event_id,
                    "node_id": e.node_id,
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "severity": e.severity,
                    "details": e.details,
                    "resolved": e.resolved
                }
                for e in events
            ],
            "total_count": len(events)
        }
    
    @app.get("/api/models/check-update")
    async def check_model_update(node_id: str):
        """Check if model update is available for node."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        with orchestrator.nodes_lock:
            if node_id not in orchestrator.fog_nodes:
                raise HTTPException(status_code=404, detail="Node not found")
            
            node = orchestrator.fog_nodes[node_id]
            update_available = node.model_version != orchestrator.model_version
        
        return {
            "update_available": update_available,
            "current_version": orchestrator.model_version,
            "node_version": node.model_version if node_id in orchestrator.fog_nodes else "unknown"
        }
    
    @app.get("/api/models/download")
    async def download_model_update(node_id: str):
        """Download model update for node."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        try:
            model_update = await orchestrator.get_model_update(node_id)
            if model_update is None:
                raise HTTPException(status_code=404, detail="No update available")
            
            return model_update
        except Exception as e:
            logger.error(f"Model download error: {e}")
            raise HTTPException(status_code=500, detail="Model download failed")
    
    @app.post("/api/models/upload")
    async def upload_local_model(model_upload: ModelUpload):
        """Upload local model from fog node."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        try:
            success = await orchestrator.upload_local_model(
                model_upload.node_id, 
                model_upload.model_data
            )
            
            if not success:
                raise HTTPException(status_code=400, detail="Model upload rejected")
            
            return {"status": "accepted", "timestamp": asyncio.get_event_loop().time()}
        except Exception as e:
            logger.error(f"Model upload error: {e}")
            raise HTTPException(status_code=500, detail="Model upload failed")
    
    @app.get("/api/policy/global")
    async def get_global_policy():
        """Get global security policy."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        return {"policy": orchestrator.global_policy}
    
    @app.put("/api/policy/global")
    async def update_global_policy(policy_update: Dict[str, Any]):
        """Update global security policy."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        try:
            # Update policy
            orchestrator.global_policy.update(policy_update)
            orchestrator.global_policy['updated_at'] = asyncio.get_event_loop().time()
            
            return {
                "status": "updated",
                "policy": orchestrator.global_policy,
                "timestamp": orchestrator.global_policy['updated_at']
            }
        except Exception as e:
            logger.error(f"Policy update error: {e}")
            raise HTTPException(status_code=500, detail="Policy update failed")
    
    @app.get("/api/intelligence/threats")
    async def get_threat_intelligence():
        """Get current threat intelligence."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        return {"threat_intelligence": orchestrator.threat_intelligence}
    
    @app.get("/api/status")
    async def get_system_status():
        """Get comprehensive system status."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        try:
            status = orchestrator.get_system_status()
            return status
        except Exception as e:
            logger.error(f"Status retrieval error: {e}")
            raise HTTPException(status_code=500, detail="Failed to get system status")
    
    @app.get("/api/stats/aggregation")
    async def get_aggregation_stats():
        """Get model aggregation statistics."""
        if not orchestrator or not orchestrator.model_aggregator:
            raise HTTPException(status_code=503, detail="Aggregator not available")
        
        try:
            stats = orchestrator.model_aggregator.get_aggregation_stats()
            return {"aggregation_stats": stats}
        except Exception as e:
            logger.error(f"Aggregation stats error: {e}")
            raise HTTPException(status_code=500, detail="Failed to get aggregation stats")
    
    # WebSocket endpoint for real-time updates (optional)
    @app.websocket("/ws/status")
    async def websocket_status(websocket):
        """WebSocket endpoint for real-time status updates."""
        await websocket.accept()
        
        try:
            while True:
                if orchestrator:
                    status = orchestrator.get_system_status()
                    await websocket.send_json(status)
                
                await asyncio.sleep(30)  # Send updates every 30 seconds
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await websocket.close()
    
    return app


# For running with uvicorn
app = create_api_server()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
