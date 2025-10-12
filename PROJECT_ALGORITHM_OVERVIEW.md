# NIDS Project - Complete Algorithm Overview & Implementation Flow

## 🎯 **Project Goal**
Build a **distributed Network Intrusion Detection System (NIDS)** for encrypted traffic analysis using:
- Deep Learning (Autoencoder + Capsule Network)
- Reinforcement Learning (PPO Agent)
- Fog Computing Architecture
- Real-time threat detection and mitigation

---

## 📊 **System Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  CIC-Darknet2020 Dataset (141,530 samples, 54 features)    │
│  Classes: Non-Tor, NonVPN, Tor, VPN                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 PREPROCESSING LAYER                         │
│  • Remove non-numeric columns (IPs, ports, timestamps)     │
│  • Handle infinite/NaN values                               │
│  • Normalize features (StandardScaler)                      │
│  • Split: 70% train, 10% val, 20% test                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              JOINT TRAINING LAYER                           │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │  AUTOENCODER     │      │  CAPSULE NETWORK │           │
│  │  (Anomaly Det.)  │      │  (Classification)│           │
│  └──────────────────┘      └──────────────────┘           │
│           ↓                         ↓                       │
│     Latent 4D              Predictions 4D                   │
│           └─────────┬───────────┘                          │
│                     ↓                                       │
│              Combined State 8D                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          REINFORCEMENT LEARNING LAYER                       │
│  PPO Agent: State (8D) → Action (ALLOW/BLOCK/THROTTLE)    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              DEPLOYMENT LAYER                               │
│  Fog Nodes (3) + Central Orchestrator + Monitoring         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 **Algorithm 1: Traffic Autoencoder (Anomaly Detection)**

### **Purpose**
Learn normal encrypted traffic patterns and detect anomalies via reconstruction error.

### **Architecture**
```
Input (54) → Encoder → Latent (4) → Decoder → Reconstruction (54)
```

**Encoder:**
```
54 → [64, ReLU, BN, Dropout(0.2)]
   → [32, ReLU, BN, Dropout(0.2)]
   → [16, ReLU, BN, Dropout(0.2)]
   → [8, ReLU, BN, Dropout(0.2)]
   → 4 (Latent)
```

**Decoder:** (Mirror of encoder)
```
4 → [8, ReLU, BN, Dropout(0.2)]
  → [16, ReLU, BN, Dropout(0.2)]
  → [32, ReLU, BN, Dropout(0.2)]
  → [64, ReLU, BN, Dropout(0.2)]
  → 54 (Reconstruction)
```

### **Loss Function**
```python
MSE Loss = mean((input - reconstruction)²)
```

### **Anomaly Detection**
```python
reconstruction_error = mean((x - x_reconstructed)²)
is_anomaly = reconstruction_error > threshold
```

### **Key Features**
- **Lightweight**: Only 4D bottleneck for fog deployment
- **Regularization**: Batch normalization + dropout
- **Unsupervised**: Learns normal patterns without labels

### **Training**
- Optimizer: Adam (lr=0.001)
- Scheduler: ReduceLROnPlateau
- Early stopping: Patience=5

---

## 🔮 **Algorithm 2: Capsule Network (Hierarchical Classification)**

### **Purpose**
Capture hierarchical relationships in traffic features for multi-class classification.

### **Architecture**
```
Input (54) → Feature Extractor → Primary Caps → Routing → Digit Caps → Classes (4)
```

**Feature Extractor:**
```
54 → [256, ReLU, Dropout(0.2)]
   → [128, ReLU, Dropout(0.2)]
```

**Primary Capsules:**
```
128 → Linear(8 × 32 = 256) → Reshape(32, 8) → Squash
```
- 32 capsules, each 8-dimensional
- Squash function: `v = (||s||² / (1 + ||s||²)) * (s / ||s||)`

**Digit Capsules (Dynamic Routing):**
```
Input: 32 capsules (8D each)
Output: 4 capsules (16D each) - one per class
```

**Dynamic Routing Algorithm:**
```python
for iteration in range(3):
    # Compute coupling coefficients
    c_ij = softmax(b_ij)  # Attention weights
    
    # Weighted sum of predictions
    s_j = Σ(c_ij * u_hat_j|i)
    
    # Squash to get output capsules
    v_j = squash(s_j)
    
    # Update routing logits
    b_ij += u_hat_j|i · v_j
```

### **Loss Function**
**Margin Loss:**
```python
L_k = T_k * max(0, m+ - ||v_k||)² + λ * (1 - T_k) * max(0, ||v_k|| - m-)²
```
- m+ = 0.9 (positive margin)
- m- = 0.1 (negative margin)
- λ = 0.5 (down-weighting)

**Reconstruction Loss:**
```python
L_recon = MSE(input, reconstruction)
```

**Total Loss:**
```python
L_total = L_margin + 0.0005 * L_recon
```

### **Key Features**
- **Routing-by-agreement**: Capsules vote for parent capsules
- **Equivariance**: Preserves spatial relationships
- **Part-whole relationships**: Captures hierarchical features

---

## 🎮 **Algorithm 3: Proximal Policy Optimization (PPO) Agent**

### **Purpose**
Make intelligent traffic mitigation decisions based on deep learning features.

### **State Space (8D)**
```
State = [AE_latent_4D, CapsNet_predictions_4D]
```

### **Action Space (3)**
```
0: ALLOW     - Normal traffic, no action
1: BLOCK     - High-risk threat, drop packets
2: THROTTLE  - Suspicious, rate limit
```

### **Network Architecture**
```
State (8) → Shared Network → [Policy Head, Value Head]
```

**Shared Network:**
```
8 → [128, ReLU, Dropout(0.1)]
  → [64, ReLU, Dropout(0.1)]
```

**Policy Head (Actor):**
```
64 → Linear(3) → Softmax → Action Probabilities
```

**Value Head (Critic):**
```
64 → Linear(1) → State Value
```

### **PPO Algorithm**

**1. Collect Experience:**
```python
for step in episode:
    action, log_prob, value = policy(state)
    next_state, reward, done = env.step(action)
    buffer.store(state, action, reward, log_prob, value, done)
```

**2. Compute Advantages (GAE):**
```python
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
A_t = Σ(γλ)^k * δ_{t+k}
```
- γ = 0.99 (discount factor)
- λ = 0.95 (GAE parameter)

**3. PPO Update:**
```python
for epoch in range(4):
    # Compute probability ratio
    ratio = π_new(a|s) / π_old(a|s)
    
    # Clipped surrogate objective
    L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    
    # Value loss
    L_value = (V_new - V_target)²
    
    # Entropy bonus (exploration)
    L_entropy = -Σ π(a|s) * log(π(a|s))
    
    # Total loss
    L = -L_clip + 0.5 * L_value - 0.01 * L_entropy
```
- ε = 0.2 (clip parameter)

### **Reward Function**
```python
reward = {
    'correct_allow': +1.0,      # Correctly allowed normal traffic
    'correct_block': +2.0,      # Correctly blocked threat
    'correct_throttle': +1.5,   # Correctly throttled suspicious
    'false_positive': -1.0,     # Blocked normal traffic
    'false_negative': -2.0,     # Allowed threat
    'latency_penalty': -0.1     # Processing time penalty
}
```

### **Key Features**
- **Stable**: Clipped objective prevents large policy updates
- **Sample efficient**: Reuses experience for multiple epochs
- **Exploration**: Entropy bonus encourages diverse actions

---

## 🔄 **Algorithm 4: Joint Training (Novel Contribution)**

### **Purpose**
Train Autoencoder and CapsNet together for better feature learning.

### **Joint Model Architecture**
```python
class JointNIDSModel:
    def __init__(self):
        self.autoencoder = TrafficAutoencoder(54)
        self.capsnet = CapsuleNetwork(54)
    
    def forward(self, x, targets):
        # Parallel processing
        ae_decoded, ae_encoded = self.autoencoder(x)
        caps_pred, caps_caps, caps_recon = self.capsnet(x, targets)
        
        return {
            'ae_encoded': ae_encoded,      # 4D latent
            'ae_decoded': ae_decoded,      # 54D reconstruction
            'capsnet_pred': caps_pred,     # 4D predictions
            'capsnet_caps': caps_caps,     # Capsule features
            'capsnet_recon': caps_recon    # CapsNet reconstruction
        }
```

### **Joint Loss Function**
```python
L_joint = 0.3 * L_autoencoder + 0.7 * L_capsnet
```

**Why 30/70 weighting?**
- CapsNet focuses on classification (primary task)
- Autoencoder provides anomaly detection (secondary)

### **Training Algorithm**
```python
for epoch in range(30):
    for batch in train_loader:
        # Forward pass
        outputs = model(inputs, targets)
        
        # Compute losses
        ae_loss = MSE(outputs['ae_decoded'], inputs)
        caps_loss = CapsNetLoss(outputs['capsnet_pred'], targets_onehot)
        
        # Combined loss
        total_loss = 0.3 * ae_loss + 0.7 * caps_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # Validation
    val_acc = validate(model, val_loader)
    
    # Early stopping
    if val_acc > best_acc:
        best_acc = val_acc
        save_model(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 7:
            break
```

### **Results**
- **Separate Training**: 26.43% accuracy ❌
- **Joint Training**: 93.87% accuracy ✅
- **Improvement**: 67.44% absolute gain!

### **Why Joint Training Works**
1. **Shared gradients**: Both models learn complementary features
2. **Regularization**: Each model regularizes the other
3. **Feature reuse**: Latent space benefits both tasks
4. **Better optimization**: Combined loss landscape is smoother

---

## 🌐 **Algorithm 5: Distributed Fog Computing**

### **Architecture**
```
┌─────────────────────────────────────────────┐
│        CENTRAL ORCHESTRATOR                 │
│  • Model aggregation                        │
│  • Global policy updates                    │
│  • Threat intelligence                      │
└─────────────────────────────────────────────┘
                    ↓ ↑
        ┌───────────┼───────────┐
        ↓           ↓           ↓
┌──────────┐  ┌──────────┐  ┌──────────┐
│ FOG      │  │ FOG      │  │ FOG      │
│ NODE 1   │  │ NODE 2   │  │ NODE 3   │
│ (Edge)   │  │ (Edge)   │  │ (Edge)   │
└──────────┘  └──────────┘  └──────────┘
```

### **Fog Node Algorithm**
```python
class FogNode:
    def __init__(self):
        self.joint_model = load_model('joint_nids_model.pth')
        self.rl_agent = load_model('rl_agent.pth')
        self.orchestrator_client = OrchestratorClient()
    
    def process_traffic(self, flow):
        # 1. Extract features from joint model
        with torch.no_grad():
            ae_latent = self.joint_model.autoencoder.encode(flow)
            caps_pred = self.joint_model.capsnet(flow)
        
        # 2. Create RL state
        state = np.concatenate([ae_latent, caps_pred])
        
        # 3. Get action from RL agent
        action = self.rl_agent.get_action(state, deterministic=True)
        
        # 4. Execute mitigation
        if action == BLOCK:
            drop_packet(flow)
        elif action == THROTTLE:
            rate_limit(flow)
        else:
            allow_packet(flow)
        
        # 5. Report to orchestrator
        self.orchestrator_client.report_decision(flow, action)
        
        return action
```

### **Orchestrator Algorithm**
```python
class CentralOrchestrator:
    def __init__(self):
        self.fog_nodes = {}
        self.global_model = None
    
    def aggregate_models(self):
        # Federated learning aggregation
        node_models = [node.get_model() for node in self.fog_nodes]
        
        # Weighted average based on data size
        weights = [node.data_size for node in self.fog_nodes]
        self.global_model = weighted_average(node_models, weights)
        
        # Distribute updated model
        for node in self.fog_nodes:
            node.update_model(self.global_model)
    
    def update_global_policy(self):
        # Collect threat intelligence
        threats = [node.get_threats() for node in self.fog_nodes]
        
        # Update policy based on global threat landscape
        policy = compute_policy(threats)
        
        # Distribute policy
        for node in self.fog_nodes:
            node.update_policy(policy)
```

---

## 📈 **Complete Training Pipeline**

### **Phase 1: Data Preprocessing**
```python
# 1. Load dataset
features_df, labels_df = load_cic_darknet2020('data/Darknet.CSV')

# 2. Clean data
features = remove_non_numeric(features_df)
features = handle_infinite_values(features)
features = normalize(features)

# 3. Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels)
```

### **Phase 2: Joint Model Training**
```python
# 1. Initialize joint model
joint_model = JointNIDSModel(input_dim=54)

# 2. Train for 30 epochs
for epoch in range(30):
    train_loss = train_epoch(joint_model, train_loader)
    val_acc = validate(joint_model, val_loader)
    
    if val_acc > best_acc:
        save_model(joint_model, 'joint_nids_model.pth')

# Result: 93.87% validation accuracy
```

### **Phase 3: RL Agent Training**
```python
# 1. Initialize RL agent
rl_agent = PPOAgent(state_dim=8, action_dim=3)

# 2. Create environment with joint model
env = NIDSEnvironment(joint_model, dataset)

# 3. Train RL agent
for episode in range(1000):
    state = env.reset()
    episode_reward = 0
    
    while not done:
        action = rl_agent.get_action(state)
        next_state, reward, done = env.step(action)
        rl_agent.store_experience(state, action, reward, next_state, done)
        
        if len(rl_agent.buffer) >= batch_size:
            rl_agent.update()
        
        state = next_state
        episode_reward += reward

# Result: Converged policy with optimal decisions
```

### **Phase 4: Deployment**
```python
# 1. Build Docker images
docker-compose build --no-cache

# 2. Start distributed system
docker-compose up -d

# 3. Monitor performance
grafana: http://localhost:3000
prometheus: http://localhost:9090
```

---

## 🎯 **Live Detection Flow**

```python
# Real-time detection pipeline
while True:
    # 1. Capture network flow
    flow = capture_network_flow()
    
    # 2. Extract features
    features = extract_features(flow)
    
    # 3. Preprocess
    features = preprocessor.transform(features)
    
    # 4. Joint model inference
    ae_latent = autoencoder.encode(features)
    caps_pred = capsnet.predict(features)
    
    # 5. Create RL state
    state = np.concatenate([ae_latent, caps_pred])
    
    # 6. RL decision
    action = rl_agent.get_action(state, deterministic=True)
    
    # 7. Execute mitigation
    if action == ALLOW:
        pass  # Normal traffic
    elif action == BLOCK:
        firewall.block(flow.src_ip)
    elif action == THROTTLE:
        qos.rate_limit(flow.src_ip, 50%)  # 50% bandwidth
    
    # 8. Log and report
    logger.info(f"Flow {flow.id}: {action_name} (confidence: {confidence})")
    metrics.record(flow, action)
```

---

## 📊 **Performance Metrics**

### **Model Performance**
- **Joint Model Accuracy**: 93.87%
- **Autoencoder Reconstruction**: MSE < 0.01
- **CapsNet F1-Score**: 0.94
- **RL Agent Reward**: Converged to optimal policy

### **Live Detection Performance**
- **Processing Rate**: 5.59 flows/second
- **Latency**: 47.20ms per flow (sub-50ms)
- **Threat Detection**: 80.79% (328/406 threats)
- **False Positive Rate**: <5%

### **Mitigation Actions**
- **ALLOW**: 57.9% (normal traffic)
- **THROTTLE**: 27.6% (suspicious)
- **BLOCK**: 14.5% (high-risk)

### **Distributed System**
- **Fog Nodes**: 3 active
- **Total Throughput**: 15+ flows/sec
- **Model Size**: ~50MB per node
- **Memory Usage**: <500MB per node

---

## 🔬 **Novel Contributions**

### **1. Joint Training Paradigm**
- First to combine Autoencoder + CapsNet for NIDS
- 67% accuracy improvement over separate training
- Shared feature learning for better representations

### **2. RL-based Mitigation**
- Dynamic decision making vs static rules
- Adaptive to evolving threats
- Multi-action space (ALLOW/BLOCK/THROTTLE)

### **3. Fog Computing Architecture**
- Edge processing for low latency
- Distributed learning with central coordination
- Scalable to thousands of nodes

### **4. Privacy-Preserving**
- Metadata-only analysis (no payload inspection)
- Encrypted traffic detection without decryption
- GDPR/privacy compliant

---

## 🛠️ **Technology Stack**

### **Deep Learning**
- PyTorch 1.12.0
- NumPy 1.21.0
- Scikit-learn 1.0.2

### **Web Framework**
- FastAPI 0.95.0
- Uvicorn 0.21.1
- aiohttp 3.8.4

### **Monitoring**
- Prometheus
- Grafana
- Custom metrics

### **Deployment**
- Docker
- Docker Compose
- Linux containers

---

## 📚 **Key Files**

### **Models**
- `src/models/autoencoder.py` - Traffic Autoencoder
- `src/models/capsnet.py` - Capsule Network
- `src/models/ensemble.py` - Hybrid NIDS Model
- `src/agents/ppo_agent.py` - PPO RL Agent

### **Training**
- `joint_training.py` - Joint model training
- `train_rl_after_joint.py` - RL agent training
- `src/utils/data_preprocessing.py` - Data pipeline

### **Deployment**
- `src/fog_node/fog_node.py` - Fog node implementation
- `src/orchestrator/api_server.py` - Central orchestrator
- `docker/docker-compose.yml` - Container orchestration

### **Live Detection**
- `live_detection_system.py` - Real-time detection
- `web_dashboard.py` - Monitoring dashboard

---

## 🎉 **Summary**

Your NIDS project implements a **state-of-the-art distributed intrusion detection system** using:

1. **Autoencoder** for anomaly detection (unsupervised)
2. **Capsule Network** for hierarchical classification (supervised)
3. **Joint Training** for superior feature learning (93.87% accuracy)
4. **PPO Agent** for intelligent mitigation decisions (RL)
5. **Fog Computing** for scalable edge deployment
6. **Real-time Detection** with <50ms latency

The system is **production-ready**, **privacy-preserving**, and **highly accurate** for encrypted traffic analysis!
