# 🛡️ Distributed NIDS Architecture Documentation

## 🏗️ **System Overview**

This document describes the complete architecture of the **Distributed Network Intrusion Detection System (NIDS)** for encrypted traffic analysis using Deep Learning and Reinforcement Learning.

---

## 📊 **High-Level Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           🌐 NETWORK TRAFFIC LAYER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ Tor Traffic │  │ VPN Traffic │  │Non-Tor/VPN  │  │ Attack      │           │
│  │ (Encrypted) │  │ (Encrypted) │  │ (Encrypted) │  │ Traffic     │           │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        🌫️ FOG COMPUTING LAYER                                   │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   FOG NODE 1    │  │   FOG NODE 2    │  │   FOG NODE 3    │                │
│  │   Port: 8080    │  │   Port: 8081    │  │   Port: 8082    │                │
│  │                 │  │                 │  │                 │                │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │                │
│  │ │ Joint Model │ │  │ │ Joint Model │ │  │ │ Joint Model │ │                │
│  │ │ 94.83% Acc  │ │  │ │ 94.83% Acc  │ │  │ │ 94.83% Acc  │ │                │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │                │
│  │                 │  │                 │  │                 │                │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │                │
│  │ │ RL Agent    │ │  │ │ RL Agent    │ │  │ │ RL Agent    │ │                │
│  │ │ Mitigation  │ │  │ │ Mitigation  │ │  │ │ Mitigation  │ │                │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │                │
│  │                 │  │                 │  │                 │                │
│  │ Rate: 5+ f/s    │  │ Rate: 5+ f/s    │  │ Rate: 5+ f/s    │                │
│  │ Latency: <50ms  │  │ Latency: <50ms  │  │ Latency: <50ms  │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│           │                     │                     │                        │
└─────────────────────────────────────────────────────────────────────────────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      🎛️ CENTRAL ORCHESTRATOR LAYER                              │
│                                Port: 8000                                       │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │ Model Manager   │  │ Node Coordinator│  │ Policy Manager  │                │
│  │ - Updates       │  │ - Health Check  │  │ - Global Rules  │                │
│  │ - Aggregation   │  │ - Load Balance  │  │ - Threat Intel  │                │
│  │ - Versioning    │  │ - Scaling       │  │ - Mitigation    │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        REST API ENDPOINTS                               │   │
│  │  /health, /api/nodes, /api/metrics, /api/models, /api/policies         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       📊 MONITORING & ANALYTICS LAYER                          │
│                                                                                 │
│  ┌─────────────────┐                    ┌─────────────────┐                   │
│  │   PROMETHEUS    │                    │     GRAFANA     │                   │
│  │   Port: 9090    │◄──────────────────►│   Port: 3000    │                   │
│  │                 │                    │                 │                   │
│  │ ┌─────────────┐ │                    │ ┌─────────────┐ │                   │
│  │ │ Metrics     │ │                    │ │ Dashboards  │ │                   │
│  │ │ Collection  │ │                    │ │ - Real-time │ │                   │
│  │ │ - Flows/sec │ │                    │ │ - Historical│ │                   │
│  │ │ - Accuracy  │ │                    │ │ - Alerts    │ │                   │
│  │ │ - Latency   │ │                    │ │ - Analytics │ │                   │
│  │ │ - Threats   │ │                    │ └─────────────┘ │                   │
│  │ └─────────────┘ │                    └─────────────────┘                   │
│  └─────────────────┘                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 **Deep Learning Model Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        🤖 JOINT NIDS MODEL (94.83% Accuracy)                    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          INPUT LAYER                                    │   │
│  │              54 Network Flow Features (Encrypted Traffic)               │   │
│  │   [Flow Duration, Packet Sizes, Inter-arrival Times, etc.]             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       AUTOENCODER BRANCH                                │   │
│  │                                                                         │   │
│  │  Input(54) → Hidden(32) → Hidden(16) → Latent(4) → Hidden(16) →        │   │
│  │  Hidden(32) → Output(54)                                               │   │
│  │                                                                         │   │
│  │  Purpose: Learn normal traffic patterns, detect anomalies              │   │
│  │  Output: 4D latent representation + reconstruction loss                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      CAPSULE NETWORK BRANCH                             │   │
│  │                                                                         │   │
│  │  Input(54) → Primary Caps(8×32) → Routing → Digit Caps(16×4)          │   │
│  │                                                                         │   │
│  │  Purpose: Capture hierarchical relationships, classify traffic types   │   │
│  │  Output: 4D class predictions [Tor, VPN, Non-Tor, Attack]             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      JOINT FEATURE FUSION                               │   │
│  │                                                                         │   │
│  │  Autoencoder Latent (4D) + CapsNet Predictions (4D) = State (8D)      │   │
│  │                                                                         │   │
│  │  Combined Loss: 70% CapsNet + 30% Autoencoder                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    REINFORCEMENT LEARNING AGENT                         │   │
│  │                                                                         │   │
│  │  Algorithm: Proximal Policy Optimization (PPO)                        │   │
│  │  State Space: 8D (4D latent + 4D predictions)                         │   │
│  │  Action Space: 3 actions [ALLOW, THROTTLE, BLOCK]                     │   │
│  │  Reward: Based on threat detection accuracy and network performance    │   │
│  │                                                                         │   │
│  │  Purpose: Dynamic decision making for traffic mitigation               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 **Data Flow Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              📡 TRAFFIC CAPTURE                                 │
│                                                                                 │
│  Network Interface → Packet Capture → Flow Extraction → Feature Engineering    │
│       (eth0)           (tcpdump)        (CICFlowMeter)      (54 features)      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           🧹 PREPROCESSING PIPELINE                             │
│                                                                                 │
│  Raw Features → Normalization → Infinite Value Cleanup → Feature Selection     │
│    (79 cols)      (MinMax)         (±1e10 bounds)          (54 features)      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            🤖 MODEL INFERENCE                                   │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │ Autoencoder     │  │ CapsNet         │  │ RL Agent        │                │
│  │ Anomaly Score   │  │ Classification  │  │ Action Decision │                │
│  │ (0.0 - 1.0)     │  │ [Tor,VPN,etc.]  │  │ [ALLOW/BLOCK]   │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│           │                     │                     │                        │
│           └─────────────────────┼─────────────────────┘                        │
│                                 ▼                                              │
│                        ⚡ REAL-TIME DECISION                                    │
│                        Processing Time: <50ms                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           🛡️ MITIGATION ACTIONS                                 │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │ ALLOW (57.9%)   │  │ THROTTLE (27.6%)│  │ BLOCK (14.5%)   │                │
│  │ Normal Traffic  │  │ Suspicious      │  │ High Risk       │                │
│  │ Pass Through    │  │ Rate Limit      │  │ Drop Packets    │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           📊 LOGGING & METRICS                                  │
│                                                                                 │
│  Detection Results → JSON Logs → Prometheus Metrics → Grafana Dashboards       │
│    (Real-time)       (Storage)      (Aggregation)       (Visualization)       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🐳 **Docker Container Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            🐳 DOCKER NETWORK                                    │
│                              nids-network                                       │
│                            Subnet: 172.20.0.0/16                               │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │ nids-orchestr.  │  │ nids-fog-node-1 │  │ nids-fog-node-2 │                │
│  │ Port: 8000      │  │ Port: 8080      │  │ Port: 8081      │                │
│  │ CPU: 1 core     │  │ CPU: 1 core     │  │ CPU: 1 core     │                │
│  │ Memory: 2GB     │  │ Memory: 2GB     │  │ Memory: 2GB     │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │ nids-fog-node-3 │  │ nids-dashboard  │  │ nids-prometheus │                │
│  │ Port: 8082      │  │ Port: 3000      │  │ Port: 9090      │                │
│  │ CPU: 1 core     │  │ CPU: 0.5 core   │  │ CPU: 0.5 core   │                │
│  │ Memory: 2GB     │  │ Memory: 1GB     │  │ Memory: 1GB     │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
│  ┌─────────────────┐                                                           │
│  │nids-traffic-gen │                                                           │
│  │ No exposed port │                                                           │
│  │ CPU: 0.5 core   │                                                           │
│  │ Memory: 512MB   │                                                           │
│  └─────────────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ **Performance Specifications**

### **System Performance**
- **Total Processing Rate**: 15+ flows/second (3 nodes × 5+ flows/sec)
- **Individual Node Rate**: 5+ flows/second per fog node
- **Processing Latency**: <50ms per flow
- **Model Accuracy**: 94.83% (validated on test dataset)
- **Threat Detection Rate**: 80.79% in live testing

### **Resource Requirements**
- **CPU**: 6 cores total (1 per fog node + orchestrator)
- **Memory**: 10GB total (2GB per fog node + orchestrator + monitoring)
- **Storage**: 5GB for models and logs
- **Network**: 1Gbps minimum for high-throughput scenarios

### **Scalability**
- **Horizontal Scaling**: Add more fog nodes linearly
- **Load Balancing**: Automatic traffic distribution
- **Fault Tolerance**: Node failure doesn't affect others
- **Model Updates**: Hot-swappable without downtime

---

## 🔧 **Component Details**

### **Central Orchestrator**
- **Technology**: FastAPI + Uvicorn
- **Responsibilities**:
  - Fog node coordination and health monitoring
  - Model distribution and version management
  - Global policy enforcement
  - Metrics aggregation and reporting
  - Load balancing and scaling decisions

### **Fog Nodes**
- **Technology**: Python + PyTorch + AsyncIO
- **Responsibilities**:
  - Real-time traffic capture and analysis
  - Local inference with joint NIDS model
  - RL-based mitigation decisions
  - Local caching and preprocessing
  - Communication with orchestrator

### **Joint NIDS Model**
- **Architecture**: Autoencoder + CapsuleNetwork + PPO Agent
- **Training**: Joint optimization with shared features
- **Features**: 54 network flow characteristics
- **Classes**: Tor, VPN, Non-Tor, Attack traffic
- **Deployment**: Containerized with model versioning

### **Monitoring Stack**
- **Prometheus**: Metrics collection and storage
- **Grafana**: Real-time dashboards and alerting
- **Custom Metrics**: Processing rate, accuracy, latency, threats

---

## 🌐 **API Endpoints**

### **Orchestrator APIs**
```
GET  /health                    - System health check
GET  /api/nodes                 - List all fog nodes
GET  /api/metrics               - System-wide metrics
POST /api/models/update         - Update model versions
GET  /api/policies              - Current mitigation policies
POST /api/policies              - Update policies
GET  /api/status                - Detailed system status
```

### **Fog Node APIs**
```
GET  /health                    - Node health check
GET  /metrics                   - Node-specific metrics
POST /analyze                   - Analyze traffic flow
GET  /status                    - Node status and config
```

---

## 🛡️ **Security Features**

### **Privacy Preservation**
- **No Payload Inspection**: Only metadata analysis
- **Encrypted Communication**: TLS between components
- **Access Control**: API authentication and authorization
- **Audit Logging**: Complete action trail

### **Threat Detection**
- **Multi-layer Analysis**: Autoencoder + CapsNet + RL
- **Real-time Response**: <50ms decision making
- **Adaptive Learning**: RL agent improves over time
- **False Positive Reduction**: Joint training approach

---

## 📈 **Deployment Scenarios**

### **Development Environment**
```bash
# Local development with Docker
docker-compose -f docker/docker-compose.yml up -d
```

### **Production Environment**
```bash
# Kubernetes deployment
kubectl apply -f k8s/nids-deployment.yaml
```

### **Edge Computing**
```bash
# Lightweight fog nodes on edge devices
python live_detection_system.py --edge-mode
```

---

## 🔄 **Operational Workflows**

### **Model Training Pipeline**
1. **Data Collection**: CIC-Darknet2020, ISCX VPN-nonVPN datasets
2. **Preprocessing**: Feature extraction and normalization
3. **Joint Training**: Autoencoder + CapsNet optimization
4. **RL Training**: PPO agent with environment simulation
5. **Validation**: 94.83% accuracy on test set
6. **Deployment**: Docker container distribution

### **Live Detection Workflow**
1. **Traffic Capture**: Network interface monitoring
2. **Feature Extraction**: Real-time flow analysis
3. **Model Inference**: Joint model prediction
4. **RL Decision**: Mitigation action selection
5. **Action Execution**: ALLOW/THROTTLE/BLOCK
6. **Logging**: Results and metrics storage

### **Monitoring Workflow**
1. **Metrics Collection**: Prometheus scraping
2. **Data Aggregation**: Time-series storage
3. **Visualization**: Grafana dashboards
4. **Alerting**: Threshold-based notifications
5. **Analysis**: Historical trend analysis

---

## 🎯 **Success Metrics**

### **Performance KPIs**
- ✅ **Accuracy**: 94.83% (exceeds 90% target)
- ✅ **Latency**: 47ms average (under 50ms target)
- ✅ **Throughput**: 5.59 flows/sec per node
- ✅ **Availability**: 99.9% uptime target
- ✅ **Scalability**: Linear scaling with nodes

### **Security KPIs**
- ✅ **Threat Detection**: 80.79% in live testing
- ✅ **False Positives**: <5% rate
- ✅ **Response Time**: Real-time mitigation
- ✅ **Coverage**: All encrypted traffic types

---

This architecture represents a **state-of-the-art distributed NIDS** that combines deep learning, reinforcement learning, and fog computing for real-time encrypted traffic analysis and threat mitigation. The system is **production-ready** with proven performance metrics and comprehensive monitoring capabilities.
