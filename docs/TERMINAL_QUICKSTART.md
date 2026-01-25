# CAN-LSS-Mamba Terminal Quick Start Guide

This guide is for running CAN-LSS-Mamba entirely from the terminal (no Jupyter notebook required).

## 🚀 Quick Start on vast.ai (Terminal Only)

### Step 1: Clone Repository
```bash
cd /workspace
git clone https://github.com/jhoshcinco/can-lss-mamba.git
cd can-lss-mamba
```

### Step 2: Run Setup
```bash
bash setup.sh
```

This creates all directories and installs dependencies automatically.

### Step 3: Upload Your Dataset
Upload your CAN dataset to:
```
/workspace/data/can-train-and-test-v1.5/set_01/
```

The dataset should contain:
- `train_02_with_attacks/*.csv` - Training data files
- `test_*/*.csv` - Test scenario folders

### Step 4: Preprocess Data
```bash
python preprocessing/CAN_preprocess.py
```

This will create processed data in:
```
/workspace/data/processed_data/set_01_run_02/
```

### Step 5: Train Model
```bash
# Basic training (20 epochs, batch size 32)
python train.py

# Custom hyperparameters
BATCH_SIZE=64 EPOCHS=50 LR=0.001 python train.py

# With WandB tracking
WANDB_ENABLED=true WANDB_API_KEY=your_key python train.py
```

### Step 6: Evaluate Model
```bash
python evaluate.py
```

Results will be saved to:
```
/workspace/final_thesis_results_02.csv
```

---

## 🔧 Advanced Terminal Usage

### Using Configuration Files

Instead of environment variables, use config files:

```bash
# Use vast.ai config (default paths)
CONFIG_PATH=configs/vastai.yaml python scripts/train.py

# Use custom config
CONFIG_PATH=configs/codespaces.yaml python scripts/train.py
```

### Override Specific Settings

Mix config files with environment variables:

```bash
# Use vastai config but change batch size
CONFIG_PATH=configs/vastai.yaml BATCH_SIZE=64 python scripts/train.py
```

### Resume Training

Training automatically resumes from the last checkpoint if it exists:

```bash
# Training will resume from last epoch
python train.py
```

### Monitor Training Progress

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Tail training logs (if redirected)
tail -f training.log
```

### Background Training

Run training in background:

```bash
# With nohup
nohup python train.py > training.log 2>&1 &

# Check if running
ps aux | grep train.py

# View logs
tail -f training.log
```

### Docker Terminal Workflow

```bash
# Build image
docker build -t can-lss-mamba .

# Run preprocessing
docker-compose run preprocess

# Run training
docker-compose run train

# Run evaluation
docker-compose run evaluate
```

---

## 📊 WandB Setup (Optional)

To enable experiment tracking:

### 1. Install WandB (already in requirements.txt)
```bash
pip install wandb
```

### 2. Login
```bash
wandb login
# Paste your API key from: https://wandb.ai/authorize
```

### 3. Enable in Training
```bash
WANDB_ENABLED=true python train.py
```

### 4. View Results
Open: https://wandb.ai/YOUR_USERNAME/can-lss-mamba

---

## 🔍 Verify Setup

### Check Environment
```bash
python verify_environment.py
```

### Run Tests
```bash
python tests/test_setup.py
```

### Check GPU
```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📂 Directory Structure After Setup

```
/workspace/
├── data/
│   ├── can-train-and-test-v1.5/
│   │   └── set_01/
│   │       ├── train_02_with_attacks/  # Your training CSV files
│   │       ├── test_dos/               # Test scenario 1
│   │       ├── test_fuzzing/           # Test scenario 2
│   │       └── test_rpm/               # Test scenario 3
│   └── processed_data/
│       └── set_01_run_02/
│           ├── train_data.npz
│           ├── val_data.npz
│           └── id_map.npy
├── checkpoints/
│   └── set_01/
│       ├── lss_can_mamba_best.pth
│       └── lss_can_mamba_last.pth
├── results/
│   └── final_thesis_results_02.csv
└── can-lss-mamba/                      # This repository
```

---

## 🐛 Troubleshooting

### Dependencies Not Installed
```bash
# Reinstall
pip install -r requirements.txt --force-reinstall
```

### GPU Not Found
```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Set GPU
CUDA_VISIBLE_DEVICES=0 python train.py
```

### Out of Memory
```bash
# Reduce batch size
BATCH_SIZE=16 python train.py
```

### Dataset Not Found
```bash
# Check dataset location
ls -la /workspace/data/can-train-and-test-v1.5/set_01/

# Or update path
DATA_DIR=/custom/path python preprocessing/CAN_preprocess.py
```

---

## 📝 Common Commands Cheat Sheet

```bash
# Full pipeline from scratch
bash setup.sh
python preprocessing/CAN_preprocess.py
python train.py
python evaluate.py

# Quick restart training
python train.py  # Resumes automatically

# Custom training run
BATCH_SIZE=64 EPOCHS=100 LR=0.0001 python train.py

# Train with WandB
WANDB_ENABLED=true python train.py

# Evaluate specific model
MODEL_PATH=/path/to/model.pth python evaluate.py

# Background training with logging
nohup python train.py > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Check training progress
tail -f logs/train_*.log

# Kill background training
pkill -f train.py  # Use with caution!
```

---

## 🎯 Typical vast.ai Workflow

1. **Start vast.ai instance** with PyTorch + CUDA

2. **Open terminal** (not Jupyter)

3. **Run these commands:**
```bash
cd /workspace
git clone https://github.com/jhoshcinco/can-lss-mamba.git
cd can-lss-mamba
bash setup.sh
# Upload dataset via vast.ai web interface
python preprocessing/CAN_preprocess.py
python train.py
python evaluate.py
```

4. **Download results:**
```bash
# Results are in:
# - /workspace/checkpoints/set_01/lss_can_mamba_best.pth
# - /workspace/final_thesis_results_02.csv

# Download via vast.ai web interface or scp
```

---

## 🔄 Update Code from GitHub

```bash
cd /workspace/can-lss-mamba
git pull origin main
# Re-run training with new code
python train.py
```

---

For more details, see the main [README.md](README.md).
