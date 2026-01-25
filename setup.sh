#!/bin/bash
# setup.sh - Run this once after cloning on vast.ai or local environment
# This script creates all required directories and installs dependencies
# Smart dependency checking for Docker environments

set -e  # Exit on error

echo "============================================================"
echo "🚀 Setting up CAN-LSS-Mamba environment..."
echo "============================================================"

# 1. Detect environment
if [ -f /.dockerenv ]; then
    echo "✓ Running in Docker container"
    IN_DOCKER=true
else
    echo "✓ Running on host system"
    IN_DOCKER=false
fi

# Detect vast.ai vs local
if [ -d "/workspace" ]; then
    echo "📍 Detected vast.ai environment (using /workspace)"
    BASE_DIR="/workspace"
    CONFIG_FILE="configs/vastai.yaml"
else
    echo "📍 Detected local/Codespaces environment (using ./)"
    BASE_DIR="."
    CONFIG_FILE="configs/codespaces.yaml"
fi

# 2. Check if using jhoshcinco/can-mamba Docker image
if command -v docker &> /dev/null; then
    if docker images 2>/dev/null | grep -q "jhoshcinco/can-mamba"; then
        echo "✓ Detected jhoshcinco/can-mamba Docker image"
        USING_CUSTOM_IMAGE=true
    else
        USING_CUSTOM_IMAGE=false
    fi
else
    USING_CUSTOM_IMAGE=false
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

# 3. Smart dependency checking
echo "Checking dependencies..."

check_package() {
    python -c "import $1" 2>/dev/null && return 0 || return 1
}

NEED_INSTALL=false

# Check core packages
for pkg in torch pandas sklearn mamba_ssm wandb omegaconf; do
    if check_package $pkg; then
        echo "✓ $pkg already installed"
    else
        echo "⚠️  $pkg not found"
        NEED_INSTALL=true
    fi
done

# 4. Install only if needed
if [ "$NEED_INSTALL" = true ]; then
    echo ""
    echo "⚙️  Installing missing dependencies..."
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    
    echo "✅ Dependencies installed"
else
    echo "✅ All dependencies already installed (skipping)"
fi

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
echo "     python -m can_lss_mamba.train"
echo "     # Or with custom settings:"
echo "     # BATCH_SIZE=64 EPOCHS=50 python -m can_lss_mamba.train"
echo ""
echo "  5. Evaluate the model:"
echo "     python -m can_lss_mamba.evaluate"
echo ""
echo "  6. (Optional) Enable WandB tracking:"
echo "     WANDB_ENABLED=true python -m can_lss_mamba.train"
echo ""
echo "💡 Tip: For detailed terminal instructions, see:"
echo "     cat docs/TERMINAL_QUICKSTART.md"
echo ""
echo "📝 Configuration file in use: ${CONFIG_FILE}"
echo "============================================================"
