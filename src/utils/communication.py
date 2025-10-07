"""
Communication utilities for fog node and orchestrator interaction.
Handles model updates, metrics reporting, and coordination.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional, List
import time

logger = logging.getLogger(__name__)


class OrchestratorClient:
    """
    Client for communicating with the central orchestrator.
    Handles registration, model updates, and metrics reporting.
    """
    
    def __init__(self, 
                 orchestrator_url: str,
                 node_id: str,
                 timeout: int = 30,
                 retry_attempts: int = 3):
        """
        Initialize orchestrator client.
        
        Args:
            orchestrator_url: URL of the central orchestrator
            node_id: Unique identifier for this fog node
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
        """
        self.orchestrator_url = orchestrator_url.rstrip('/')
        self.node_id = node_id
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        
        # Session for connection pooling
        self.session = None
        
        # Connection state
        self.is_connected = False
        self.last_heartbeat = 0
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _make_request(self, 
                           method: str, 
                           endpoint: str, 
                           data: Optional[Dict[str, Any]] = None,
                           params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to orchestrator with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            endpoint: API endpoint
            data: Request data (for POST/PUT)
            params: Query parameters
            
        Returns:
            Response data or None if failed
        """
        url = f"{self.orchestrator_url}{endpoint}"
        
        for attempt in range(self.retry_attempts):
            try:
                session = await self._get_session()
                
                kwargs = {
                    'params': params,
                    'headers': {
                        'Content-Type': 'application/json',
                        'X-Node-ID': self.node_id
                    }
                }
                
                if data is not None:
                    kwargs['json'] = data
                
                async with session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.is_connected = True
                        return result
                    else:
                        logger.warning(f"Request failed with status {response.status}: {await response.text()}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.retry_attempts})")
            except aiohttp.ClientError as e:
                logger.warning(f"Client error (attempt {attempt + 1}/{self.retry_attempts}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{self.retry_attempts}): {e}")
            
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        self.is_connected = False
        return None
    
    async def register_node(self, registration_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Register fog node with orchestrator.
        
        Args:
            registration_data: Node registration information
            
        Returns:
            Registration response or None if failed
        """
        logger.info(f"Registering node {self.node_id} with orchestrator")
        
        response = await self._make_request(
            'POST', 
            '/api/nodes/register',
            data=registration_data
        )
        
        if response:
            logger.info(f"Node registration successful: {response}")
        else:
            logger.error("Node registration failed")
        
        return response
    
    async def send_heartbeat(self) -> bool:
        """
        Send heartbeat to orchestrator.
        
        Returns:
            True if heartbeat was successful
        """
        heartbeat_data = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'status': 'active'
        }
        
        response = await self._make_request(
            'POST',
            '/api/nodes/heartbeat',
            data=heartbeat_data
        )
        
        if response:
            self.last_heartbeat = time.time()
            return True
        
        return False
    
    async def report_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Report node metrics to orchestrator.
        
        Args:
            metrics: Node performance and security metrics
            
        Returns:
            True if metrics were reported successfully
        """
        response = await self._make_request(
            'POST',
            '/api/metrics/report',
            data=metrics
        )
        
        return response is not None
    
    async def check_model_update(self) -> bool:
        """
        Check if model update is available.
        
        Returns:
            True if update is available
        """
        response = await self._make_request(
            'GET',
            '/api/models/check-update',
            params={'node_id': self.node_id}
        )
        
        if response:
            return response.get('update_available', False)
        
        return False
    
    async def download_model_update(self) -> Optional[Dict[str, Any]]:
        """
        Download model update from orchestrator.
        
        Returns:
            Model update data or None if failed
        """
        response = await self._make_request(
            'GET',
            '/api/models/download',
            params={'node_id': self.node_id}
        )
        
        if response:
            logger.info("Model update downloaded successfully")
        else:
            logger.error("Failed to download model update")
        
        return response
    
    async def upload_local_model(self, model_data: Dict[str, Any]) -> bool:
        """
        Upload local model improvements to orchestrator.
        
        Args:
            model_data: Local model data and improvements
            
        Returns:
            True if upload was successful
        """
        response = await self._make_request(
            'POST',
            '/api/models/upload',
            data=model_data
        )
        
        return response is not None
    
    async def get_global_policy(self) -> Optional[Dict[str, Any]]:
        """
        Get global security policy from orchestrator.
        
        Returns:
            Global policy configuration or None if failed
        """
        response = await self._make_request(
            'GET',
            '/api/policy/global'
        )
        
        return response
    
    async def report_security_event(self, event: Dict[str, Any]) -> bool:
        """
        Report security event to orchestrator.
        
        Args:
            event: Security event details
            
        Returns:
            True if event was reported successfully
        """
        event_data = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'event': event
        }
        
        response = await self._make_request(
            'POST',
            '/api/events/security',
            data=event_data
        )
        
        return response is not None
    
    async def get_threat_intelligence(self) -> Optional[Dict[str, Any]]:
        """
        Get latest threat intelligence from orchestrator.
        
        Returns:
            Threat intelligence data or None if failed
        """
        response = await self._make_request(
            'GET',
            '/api/intelligence/threats'
        )
        
        return response
    
    async def close(self):
        """Close the client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None


class MessageQueue:
    """
    Simple message queue for asynchronous communication.
    Used for buffering messages when orchestrator is unavailable.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize message queue.
        
        Args:
            max_size: Maximum number of messages to buffer
        """
        self.max_size = max_size
        self.queue = asyncio.Queue(maxsize=max_size)
        self.failed_messages = []
        
    async def put(self, message: Dict[str, Any]):
        """Add message to queue."""
        try:
            await self.queue.put(message)
        except asyncio.QueueFull:
            logger.warning("Message queue full, dropping oldest message")
            try:
                await self.queue.get_nowait()
                await self.queue.put(message)
            except asyncio.QueueEmpty:
                pass
    
    async def get(self) -> Optional[Dict[str, Any]]:
        """Get message from queue."""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def add_failed_message(self, message: Dict[str, Any]):
        """Add failed message for retry."""
        if len(self.failed_messages) < self.max_size:
            self.failed_messages.append(message)
    
    def get_failed_messages(self) -> List[Dict[str, Any]]:
        """Get and clear failed messages."""
        messages = self.failed_messages.copy()
        self.failed_messages.clear()
        return messages
    
    def size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()


class CommunicationManager:
    """
    Manages communication between fog node and orchestrator.
    Handles message queuing, retries, and connection management.
    """
    
    def __init__(self, 
                 orchestrator_client: OrchestratorClient,
                 heartbeat_interval: int = 60,
                 metrics_interval: int = 300):
        """
        Initialize communication manager.
        
        Args:
            orchestrator_client: Orchestrator client instance
            heartbeat_interval: Heartbeat interval in seconds
            metrics_interval: Metrics reporting interval in seconds
        """
        self.client = orchestrator_client
        self.heartbeat_interval = heartbeat_interval
        self.metrics_interval = metrics_interval
        
        # Message queues
        self.metrics_queue = MessageQueue()
        self.events_queue = MessageQueue()
        
        # Background tasks
        self.tasks = []
        self.is_running = False
        
    async def start(self):
        """Start communication manager."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._metrics_sender()),
            asyncio.create_task(self._events_sender())
        ]
        
        logger.info("Communication manager started")
    
    async def stop(self):
        """Stop communication manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close client session
        await self.client.close()
        
        logger.info("Communication manager stopped")
    
    async def queue_metrics(self, metrics: Dict[str, Any]):
        """Queue metrics for sending."""
        await self.metrics_queue.put(metrics)
    
    async def queue_event(self, event: Dict[str, Any]):
        """Queue security event for sending."""
        await self.events_queue.put(event)
    
    async def _heartbeat_loop(self):
        """Background heartbeat loop."""
        while self.is_running:
            try:
                success = await self.client.send_heartbeat()
                if not success:
                    logger.warning("Heartbeat failed")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _metrics_sender(self):
        """Background metrics sender."""
        while self.is_running:
            try:
                # Send queued metrics
                message = await self.metrics_queue.get()
                if message:
                    success = await self.client.report_metrics(message)
                    if not success:
                        self.metrics_queue.add_failed_message(message)
                
                # Retry failed messages
                failed_messages = self.metrics_queue.get_failed_messages()
                for message in failed_messages:
                    success = await self.client.report_metrics(message)
                    if not success:
                        self.metrics_queue.add_failed_message(message)
                        break  # Don't retry all if one fails
                
                await asyncio.sleep(1.0)  # Check queue every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics sender error: {e}")
                await asyncio.sleep(5.0)
    
    async def _events_sender(self):
        """Background events sender."""
        while self.is_running:
            try:
                # Send queued events
                message = await self.events_queue.get()
                if message:
                    success = await self.client.report_security_event(message)
                    if not success:
                        self.events_queue.add_failed_message(message)
                
                # Retry failed messages
                failed_messages = self.events_queue.get_failed_messages()
                for message in failed_messages:
                    success = await self.client.report_security_event(message)
                    if not success:
                        self.events_queue.add_failed_message(message)
                        break
                
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Events sender error: {e}")
                await asyncio.sleep(5.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get communication manager status."""
        return {
            'is_running': self.is_running,
            'is_connected': self.client.is_connected,
            'last_heartbeat': self.client.last_heartbeat,
            'metrics_queue_size': self.metrics_queue.size(),
            'events_queue_size': self.events_queue.size(),
            'failed_metrics': len(self.metrics_queue.failed_messages),
            'failed_events': len(self.events_queue.failed_messages)
        }
