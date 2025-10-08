#!/bin/bash

# NIDS-RL Training Script
# This script sets up the environment and runs the training

echo "NIDS-RL Training Script"
echo "=========================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
SRC_PATH="$PROJECT_ROOT/src"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$SRC_PATH:$PYTHONPATH"

echo "Project root: $PROJECT_ROOT"
echo "Source path: $SRC_PATH"
echo "PYTHONPATH: $PYTHONPATH"

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/results"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment is active: $VIRTUAL_ENV"
else
    echo "No virtual environment detected. Consider activating one."
fi

# Check if config file exists
CONFIG_FILE="$PROJECT_ROOT/config/model_config.yaml"
if [[ -f "$CONFIG_FILE" ]]; then
    echo "Config file found: $CONFIG_FILE"
else
    echo "Config file not found: $CONFIG_FILE"
    echo "Please create the config file first."
    exit 1
fi

# Run the training
echo ""
echo "Starting training..."
echo "=========================="

cd "$PROJECT_ROOT"

# Try the launcher script first
if [[ -f "$PROJECT_ROOT/run_training.py" ]]; then
    echo "Using launcher script..."
    python run_training.py "$@"
else
    echo "Using direct script execution..."
    python scripts/train_models.py --config config/training_config.yaml "$@"
fi

echo ""
echo "Training script completed."
