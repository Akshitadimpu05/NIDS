#!/bin/bash

# Setup script for NIDS-RL environment
# This script sets up the complete environment for running the NIDS system

set -e

echo "Setting up NIDS-RL Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root for system dependencies
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. This is not recommended for development."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y \
            python3 \
            python3-pip \
            python3-venv \
            python3-dev \
            gcc \
            g++ \
            libpcap-dev \
            tcpdump \
            iptables \
            iproute2 \
            net-tools \
            curl \
            wget \
            git \
            docker.io \
            docker-compose
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum update -y
        sudo yum install -y \
            python3 \
            python3-pip \
            python3-devel \
            gcc \
            gcc-c++ \
            libpcap-devel \
            tcpdump \
            iptables \
            iproute \
            net-tools \
            curl \
            wget \
            git \
            docker \
            docker-compose
    else
        print_error "Unsupported package manager. Please install dependencies manually."
        exit 1
    fi
    
    print_status "System dependencies installed successfully"
}

# Setup Python virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_status "Python environment setup completed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/{raw,processed,models}
    mkdir -p logs
    mkdir -p docker/grafana/{dashboards,datasources}
    mkdir -p docker/prometheus
    
    print_status "Directories created successfully"
}

# Setup Docker
setup_docker() {
    print_status "Setting up Docker..."
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Start Docker service
    sudo systemctl start docker
    sudo systemctl enable docker
    
    print_status "Docker setup completed"
    print_warning "You may need to log out and back in for Docker group changes to take effect"
}

# Download sample data
download_sample_data() {
    print_status "Setting up sample data directory..."
    
    # Create sample data structure
    mkdir -p data/raw/cic-darknet2020
    
    # Create a sample dataset info file
    cat > data/raw/cic-darknet2020/README.md << EOF
# CIC-Darknet2020 Dataset

This directory should contain the CIC-Darknet2020 dataset files.

## Download Instructions

1. Visit: https://www.kaggle.com/datasets/peterfriedrich1/cicdarknet2020-internet-traffic
2. Download the dataset
3. Extract CSV files to this directory

## Expected Files

- Various CSV files containing network traffic data
- Each file should have flow-level features and labels

## Usage

Run the training script with:
\`\`\`bash
python scripts/train_models.py --data data/raw/cic-darknet2020
\`\`\`
EOF
    
    print_status "Sample data directory created"
    print_warning "Please download the actual CIC-Darknet2020 dataset manually"
}

# Setup monitoring configuration
setup_monitoring() {
    print_status "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > docker/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'orchestrator'
    static_configs:
      - targets: ['orchestrator:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'fog-nodes'
    static_configs:
      - targets: ['fog-node-1:8080', 'fog-node-2:8080', 'fog-node-3:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
EOF
    
    # Grafana datasource configuration
    mkdir -p docker/grafana/datasources
    cat > docker/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    print_status "Monitoring configuration created"
}

# Setup configuration files
setup_config() {
    print_status "Setting up configuration files..."
    
    # Create training configuration if it doesn't exist
    if [ ! -f "config/training_config.yaml" ]; then
        cat > config/training_config.yaml << EOF
# Training Configuration for NIDS-RL System

# Data Configuration
data:
  dataset_path: "data/raw/cic-darknet2020"
  validation_split: 0.2
  test_split: 0.1
  batch_size: 256
  num_workers: 4

# Model Training
training:
  device: "cpu"  # Change to "cuda" if GPU available
  epochs:
    autoencoder: 100
    capsnet: 50
    rl_agent: 1000
  learning_rates:
    autoencoder: 0.001
    capsnet: 0.001
    rl_agent: 0.0003
  early_stopping_patience: 10
  save_frequency: 10

# Evaluation
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1_score"]
  anomaly_threshold: 0.1
EOF
    fi
    
    print_status "Configuration files setup completed"
}

# Make scripts executable
make_scripts_executable() {
    print_status "Making scripts executable..."
    
    chmod +x scripts/*.py
    chmod +x scripts/*.sh
    
    print_status "Scripts made executable"
}

# Main setup function
main() {
    echo "=========================================="
    echo "NIDS-RL Environment Setup"
    echo "=========================================="
    
    check_root
    
    # Install system dependencies
    read -p "Install system dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_system_deps
    fi
    
    # Setup Python environment
    read -p "Setup Python virtual environment? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        setup_venv
    fi
    
    # Create directories
    create_directories
    
    # Setup Docker
    read -p "Setup Docker? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_docker
    fi
    
    # Setup monitoring
    setup_monitoring
    
    # Setup configuration
    setup_config
    
    # Download sample data
    download_sample_data
    
    # Make scripts executable
    make_scripts_executable
    
    echo "=========================================="
    print_status "NIDS-RL Environment Setup Completed!"
    echo "=========================================="
    
    echo ""
    echo "Next steps:"
    echo "1. Download the CIC-Darknet2020 dataset to data/raw/cic-darknet2020/"
    echo "2. Activate the virtual environment: source venv/bin/activate"
    echo "3. Train the models: python scripts/train_models.py --data data/raw/cic-darknet2020"
    echo "4. Start the system: docker-compose -f docker/docker-compose.yml up"
    echo ""
    echo "For more information, see README.md"
}

# Run main function
main "$@"
