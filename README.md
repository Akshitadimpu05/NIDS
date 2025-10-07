# Network Intrusion Detection System (NIDS) with Hybrid Deep Learning + Reinforcement Learning

A privacy-preserving Network Intrusion Detection System that works on encrypted traffic using a lightweight hybrid approach combining Autoencoders, Capsule Networks, and Reinforcement Learning.

## Architecture Overview

### Components
1. **Autoencoder (AE)**: Learns normal encrypted traffic behavior patterns (unsupervised)
2. **Capsule Network (CapsNet)**: Captures hierarchical feature relationships in flow-level metadata
3. **Reinforcement Learning Agent**: Dynamically decides traffic actions (allow/block/throttle) using PPO/DQN
4. **Fog Layer**: Lightweight deployment on edge nodes for real-time processing
5. **Central Orchestrator**: Aggregates learning and redistributes improved models

### Key Features
- **Privacy-Preserving**: No payload decryption, works on metadata only
- **Lightweight**: Optimized for fog computing deployment
- **Real-time**: Live traffic analysis and mitigation
- **Adaptive**: RL-based policy learning with continuous improvement

## Project Structure

```
NIDS-RL/
├── src/
│   ├── models/           # Deep learning models (AE, CapsNet)
│   ├── agents/           # RL agents (PPO, DQN)
│   ├── fog_node/         # Fog node implementation
│   ├── orchestrator/     # Central orchestrator
│   ├── data_processing/  # Data preprocessing and feature extraction
│   └── utils/           # Utility functions
├── data/
│   ├── raw/             # Raw datasets (CIC-Darknet2020, etc.)
│   ├── processed/       # Preprocessed data
│   └── models/          # Trained model weights
├── docker/              # Docker configurations
├── config/              # Configuration files
├── scripts/             # Training and deployment scripts
└── tests/               # Unit tests
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download CIC-Darknet2020 dataset:
- Download from: https://www.kaggle.com/datasets/peterfriedrich1/cicdarknet2020-internet-traffic
- Extract to `data/raw/cic-darknet2020/`

## Usage

### Training
```bash
python scripts/train_models.py --config config/training_config.yaml
```

### Fog Node Deployment
```bash
docker-compose up fog-nodes
```

### Real-time Traffic Generation and Detection
```bash
python scripts/generate_traffic.py
python scripts/run_detection.py
```

## Datasets Supported
- CIC-Darknet2020
- ISCX VPN-nonVPN
- USTC-TFC2016
- Real-time traffic via Zeek/CICFlowMeter

## License
MIT License
