# NIDS-RL Deployment Guide

## Quick Start

### 1. Environment Setup
```bash
# Make setup script executable and run it
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Download Dataset
Download the CIC-Darknet2020 dataset from:
https://www.kaggle.com/datasets/peterfriedrich1/cicdarknet2020-internet-traffic

Extract CSV files to: `data/raw/cic-darknet2020/`

### 3. Train Models
```bash
# Train all components
python scripts/train_models.py --data data/raw/cic-darknet2020

# Or train individual components
python scripts/train_models.py --data data/raw/cic-darknet2020 --component autoencoder
python scripts/train_models.py --data data/raw/cic-darknet2020 --component capsnet
python scripts/train_models.py --data data/raw/cic-darknet2020 --component rl
```

### 4. Run Demo
```bash
# Quick demonstration
python scripts/demo.py
```

### 5. Deploy with Docker
```bash
# Build and start all services
docker-compose -f docker/docker-compose.yml up --build

# Or start individual services
docker-compose -f docker/docker-compose.yml up orchestrator
docker-compose -f docker/docker-compose.yml up fog-node-1 fog-node-2 fog-node-3
```

## System Architecture

### Core Components

1. **Autoencoder (AE)**
   - Learns normal encrypted traffic behavior (unsupervised)
   - Detects anomalies through reconstruction error
   - Lightweight design for fog deployment

2. **Capsule Network (CapsNet)**
   - Captures hierarchical feature relationships
   - Provides robust classification capabilities
   - Handles spatial hierarchies in traffic patterns

3. **Reinforcement Learning Agent (PPO)**
   - Makes dynamic traffic control decisions
   - Actions: Allow, Block, Throttle
   - Learns optimal policies through reward feedback

4. **Fog Nodes**
   - Edge deployment for real-time processing
   - Local decision making with global coordination
   - Traffic capture, analysis, and mitigation

5. **Central Orchestrator**
   - Federated learning coordination
   - Global policy management
   - Model aggregation and distribution

### Key Features

- **Privacy-Preserving**: No payload inspection, metadata only
- **Lightweight**: Optimized for resource-constrained fog nodes
- **Real-time**: Live traffic analysis and response
- **Adaptive**: Continuous learning and policy improvement
- **Scalable**: Distributed fog computing architecture

## API Endpoints

### Orchestrator API (Port 8000)

- `GET /` - Service information
- `GET /health` - Health check
- `POST /api/nodes/register` - Register fog node
- `POST /api/nodes/heartbeat` - Node heartbeat
- `GET /api/nodes` - List all nodes
- `POST /api/metrics/report` - Report metrics
- `GET /api/metrics/global` - Global metrics
- `POST /api/events/security` - Report security events
- `GET /api/models/check-update` - Check model updates
- `GET /api/models/download` - Download model updates
- `POST /api/models/upload` - Upload local models
- `GET /api/policy/global` - Get global policy
- `GET /api/status` - System status

### Monitoring

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090
- **Orchestrator API**: http://localhost:8000

## Configuration

### Model Configuration (`config/model_config.yaml`)
- Autoencoder parameters (hidden layers, latent dimension)
- CapsNet parameters (capsule dimensions, routing iterations)
- RL agent parameters (state/action space, learning rate)
- Training parameters (epochs, batch size, device)

### Environment Variables

#### Orchestrator
- `ORCHESTRATOR_HOST`: Host address (default: 0.0.0.0)
- `ORCHESTRATOR_PORT`: Port number (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)

#### Fog Nodes
- `NODE_ID`: Unique node identifier
- `ORCHESTRATOR_URL`: Orchestrator endpoint
- `CAPTURE_INTERFACE`: Network interface for traffic capture
- `ENABLE_MITIGATION`: Enable traffic mitigation (default: true)
- `LOG_LEVEL`: Logging level

## Security Considerations

### Traffic Mitigation
- **iptables Integration**: Automatic firewall rule management
- **Traffic Shaping**: Bandwidth throttling for suspicious flows
- **Rate Limiting**: Prevent action flooding
- **Whitelist/Blacklist**: IP-based access control

### Privacy Protection
- **No Payload Inspection**: Only flow-level metadata analyzed
- **Encrypted Communication**: Secure fog-orchestrator communication
- **Local Processing**: Sensitive decisions made at edge

## Performance Optimization

### Fog Node Optimization
- **Multi-threading**: Parallel traffic processing
- **Queue Management**: Efficient packet buffering
- **Model Compression**: Lightweight model deployment
- **Batch Processing**: Optimized inference

### Orchestrator Optimization
- **Federated Learning**: Distributed model training
- **Asynchronous Processing**: Non-blocking operations
- **Connection Pooling**: Efficient node communication
- **Caching**: Model and policy caching

## Troubleshooting

### Common Issues

1. **Permission Denied (Traffic Capture)**
   ```bash
   # Run with appropriate privileges or use simulation mode
   sudo python scripts/run_fog_node.py
   ```

2. **Docker Permission Issues**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **Port Already in Use**
   ```bash
   # Check and kill processes using ports
   sudo lsof -i :8000
   sudo kill -9 <PID>
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch size in config
   # Use CPU instead of GPU if memory limited
   ```

### Logs
- Orchestrator logs: `logs/orchestrator.log`
- Fog node logs: `logs/fog_node.log`
- Training logs: `logs/training.log`
- Docker logs: `docker-compose logs <service>`

## Development

### Adding New Features

1. **New Attack Detection**
   - Extend `TrafficCapture` for new features
   - Update model input dimensions
   - Retrain with new attack samples

2. **New Mitigation Actions**
   - Extend `TrafficMitigation` class
   - Add new action types to RL environment
   - Update reward function

3. **New Aggregation Strategies**
   - Implement in `ModelAggregator`
   - Add configuration options
   - Test with different node configurations

### Testing

```bash
# Run unit tests (when implemented)
python -m pytest tests/

# Run integration tests
python scripts/demo.py

# Load testing
# Use traffic generator with high rates
```

## Production Deployment

### Requirements
- **Minimum Hardware**: 2 CPU cores, 4GB RAM per fog node
- **Network**: Low-latency connection to orchestrator
- **Storage**: 10GB for models and logs
- **OS**: Linux with iptables support

### Security Hardening
- Use non-root users in containers
- Enable firewall rules
- Secure orchestrator API with authentication
- Regular security updates
- Monitor system logs

### Scaling
- Deploy multiple orchestrators for redundancy
- Use load balancers for fog node distribution
- Implement horizontal pod autoscaling in Kubernetes
- Monitor resource usage and scale accordingly

## Support

For issues and questions:
1. Check logs for error messages
2. Review configuration files
3. Ensure all dependencies are installed
4. Verify network connectivity between components
5. Check system resources (CPU, memory, disk)

## License

MIT License - See LICENSE file for details.
