#!/bin/bash
# setup.sh - Run this once after cloning on vast.ai or local environment
# This script creates all required directories and installs dependencies

set -e  # Exit on error

echo "============================================================"
echo "🚀 Setting up CAN-LSS-Mamba environment..."
echo "============================================================"

# Detect environment
if [ -d "/workspace" ]; then
    echo "📍 Detected vast.ai environment (using /workspace)"
    BASE_DIR="/workspace"
    CONFIG_FILE="configs/vastai.yaml"
else
    echo "📍 Detected local/Codespaces environment (using ./)"
    BASE_DIR="."
    CONFIG_FILE="configs/codespaces.yaml"
fi

echo ""
echo "============================================================"
echo "📁 Creating directory structure..."
echo "============================================================"

# Create data directories
mkdir -p "${BASE_DIR}/data/can-train-and-test-v1.5"
mkdir -p "${BASE_DIR}/data/processed_data"
echo "✅ Created: ${BASE_DIR}/data/"

# Create checkpoint directories
mkdir -p "${BASE_DIR}/checkpoints/set_01"
echo "✅ Created: ${BASE_DIR}/checkpoints/set_01/"

# Create results directory
mkdir -p "${BASE_DIR}/results"
echo "✅ Created: ${BASE_DIR}/results/"

# Create local directories (always needed for scripts)
mkdir -p ./data ./checkpoints ./results
echo "✅ Created local directories (./data, ./checkpoints, ./results)"

echo ""
echo "============================================================"
echo "📦 Installing dependencies..."
echo "============================================================"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ ERROR: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "✅ Dependencies installed successfully"

echo ""
echo "============================================================"
echo "🔍 Verifying environment..."
echo "============================================================"

# Run environment verification
if [ -f "verify_environment.py" ]; then
    python verify_environment.py
else
    echo "⚠️  Warning: verify_environment.py not found. Skipping verification."
fi

echo ""
echo "============================================================"
echo "✅ Setup complete!"
echo "============================================================"
echo ""
echo "📋 Next steps (Terminal Workflow):"
echo ""
echo "  1. Download your dataset to:"
echo "     ${BASE_DIR}/data/can-train-and-test-v1.5/set_01/"
echo ""
echo "  2. (Optional) Configure environment variables:"
echo "     cp .env.example .env"
echo "     # Edit .env with your settings"
echo ""
echo "  3. Run preprocessing:"
echo "     python preprocessing/CAN_preprocess.py"
echo ""
echo "  4. Train the model:"
echo "     python train.py"
echo "     # Or with custom settings:"
echo "     # BATCH_SIZE=64 EPOCHS=50 python train.py"
echo ""
echo "  5. Evaluate the model:"
echo "     python evaluate.py"
echo ""
echo "  6. (Optional) Enable WandB tracking:"
echo "     WANDB_ENABLED=true python train.py"
echo ""
echo "💡 Tip: For detailed terminal instructions, see:"
echo "     cat TERMINAL_QUICKSTART.md"
echo ""
echo "📝 Configuration file in use: ${CONFIG_FILE}"
echo "============================================================"
