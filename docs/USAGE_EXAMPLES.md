# Usage Examples

This document provides practical examples for using the CAN-LSS-Mamba workflow.

## Terminal Workflow (Recommended)

### Basic Pipeline
```bash
# 1. Setup (one time)
bash setup.sh

# 2. Run pipeline
python preprocessing/CAN_preprocess.py
python train.py
python evaluate.py
```

### With Custom Hyperparameters
```bash
# Train with larger batch size and more epochs
BATCH_SIZE=64 EPOCHS=50 LR=0.0001 python train.py
```

### With WandB Tracking
```bash
# Enable WandB for experiment tracking
WANDB_ENABLED=true WANDB_API_KEY=your_key python train.py
```

### Background Training
```bash
# Run training in background
nohup python train.py > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## Configuration-Based Workflow

### Using Config Files
```bash
# Use vast.ai config (default /workspace paths)
CONFIG_PATH=configs/vastai.yaml python scripts/train.py

# Use Codespaces config (relative paths)
CONFIG_PATH=configs/codespaces.yaml python scripts/train.py

# Use custom config
CONFIG_PATH=my_config.yaml python scripts/train.py
```

### Override Config with Environment Variables
```bash
# Use vastai config but override batch size
CONFIG_PATH=configs/vastai.yaml BATCH_SIZE=128 python scripts/train.py
```

## Docker Workflow

### Build and Run
```bash
# Build image
docker build -t can-lss-mamba .

# Run preprocessing
docker-compose run preprocess

# Run training
docker-compose run train

# Run evaluation
docker-compose run evaluate

# Run Jupyter (optional)
docker-compose up jupyter
# Access at http://localhost:8888
```

### Custom Docker Run
```bash
# Run with specific GPU
CUDA_VISIBLE_DEVICES=1 docker-compose run train

# Run with custom hyperparameters
BATCH_SIZE=64 EPOCHS=100 docker-compose run train
```

## Environment Setup Examples

### vast.ai Setup
```bash
# In vast.ai terminal (SSH or web terminal)
cd /workspace
git clone https://github.com/jhoshcinco/can-lss-mamba.git
cd can-lss-mamba
bash setup.sh

# Copy environment template and customize
cp .env.example .env
nano .env  # Edit paths, WandB key, etc.

# Run pipeline
python preprocessing/CAN_preprocess.py
python train.py
python evaluate.py
```

### Local/Codespaces Setup
```bash
# Clone repository
git clone https://github.com/jhoshcinco/can-lss-mamba.git
cd can-lss-mamba

# Run setup
bash setup.sh

# Use Codespaces config
CONFIG_PATH=configs/codespaces.yaml python scripts/train.py
```

## Verification Examples

### Check Environment
```bash
# Verify all dependencies installed
python verify_environment.py

# Run setup tests
python tests/test_setup.py

# Validate implementation completeness
python validate_implementation.py
```

### Check GPU
```bash
# Check GPU availability
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Get GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Data Management Examples

### Dataset Organization
```bash
# Expected structure for vast.ai
/workspace/data/can-train-and-test-v1.5/set_01/
├── train_02_with_attacks/
│   ├── file1.csv
│   ├── file2.csv
│   └── ...
├── test_dos/
│   └── *.csv
├── test_fuzzing/
│   └── *.csv
└── test_rpm/
    └── *.csv

# Upload data to vast.ai
# Option 1: Use vast.ai web interface
# Option 2: Use scp
scp -r ./dataset/* root@vastai-ip:/workspace/data/
```

### Preprocessing with Custom Paths
```bash
# Preprocess from custom location
DATASET_ROOT=/custom/path/data OUTPUT_DIR=/custom/output python preprocessing/CAN_preprocess.py
```

## Training Examples

### Resume Training
```bash
# Training automatically resumes from last checkpoint
python train.py
# Will continue from last saved epoch
```

### Train on Specific GPU
```bash
# Use GPU 1
CUDA_VISIBLE_DEVICES=1 python train.py

# Use multiple GPUs (if supported)
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

### Custom Model Paths
```bash
# Save checkpoints to custom location
OUT_DIR=/custom/checkpoints python train.py
```

## Evaluation Examples

### Evaluate All Test Sets
```bash
# Default evaluation (all test_* folders)
python evaluate.py
```

### View Results
```bash
# View CSV results
cat /workspace/final_thesis_results_02.csv

# Or with pandas
python -c "import pandas as pd; df = pd.read_csv('/workspace/final_thesis_results_02.csv'); print(df)"
```

## WandB Examples

### Setup WandB
```bash
# Login to WandB
wandb login
# Paste your API key from https://wandb.ai/authorize

# Or set via environment variable
export WANDB_API_KEY=your_key_here
```

### Train with WandB
```bash
# Enable WandB
WANDB_ENABLED=true python train.py

# With custom project name
WANDB_ENABLED=true WANDB_PROJECT=my-project python train.py

# With tags
WANDB_ENABLED=true python train.py
# (tags set in config file)
```

### View WandB Results
```bash
# Open in browser
https://wandb.ai/YOUR_USERNAME/can-lss-mamba
```

## Debugging Examples

### Check Data Loading
```bash
# Test if preprocessed data exists
ls -lh /workspace/data/processed_data/set_01_run_02/

# Check data shapes
python -c "
import numpy as np
data = np.load('/workspace/data/processed_data/set_01_run_02/train_data.npz')
print('Train data shape:', data['ids'].shape)
"
```

### Test Model Loading
```bash
# Test if model can be loaded
python -c "
import torch
from model import LSS_CAN_Mamba
model = LSS_CAN_Mamba(num_unique_ids=100)
print('Model loaded successfully')
"
```

### Monitor Training
```bash
# Run training and save output
python train.py 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

# In another terminal, monitor
tail -f training_*.log
```

## Cleanup Examples

### Remove Generated Files
```bash
# Remove checkpoints (be careful!)
rm -rf /workspace/checkpoints/*

# Remove processed data
rm -rf /workspace/data/processed_data/*

# Remove results
rm -rf /workspace/results/*
```

### Start Fresh
```bash
# Re-run setup
bash setup.sh

# Preprocess again
python preprocessing/CAN_preprocess.py
```

## Advanced Examples

### Experiment Tracking
```bash
# Run multiple experiments with different hyperparameters
for bs in 32 64 128; do
    for lr in 0.0001 0.001 0.01; do
        echo "Training with batch_size=$bs, lr=$lr"
        BATCH_SIZE=$bs LR=$lr WANDB_ENABLED=true python train.py
    done
done
```

### Parallel Runs (if you have multiple GPUs)
```bash
# Run on GPU 0
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=32 python train.py &

# Run on GPU 1
CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=64 python train.py &
```

## Troubleshooting Examples

### Fix Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Install specific package
pip install torch --upgrade
```

### Fix CUDA Issues
```bash
# Check CUDA version
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Fix Out of Memory
```bash
# Reduce batch size
BATCH_SIZE=16 python train.py

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

---

For more details, see:
- [TERMINAL_QUICKSTART.md](TERMINAL_QUICKSTART.md) - Terminal workflow guide
- [README.md](README.md) - Main documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
