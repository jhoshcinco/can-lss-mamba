# CAN-LSS-Mamba

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning model for Controller Area Network (CAN) bus intrusion detection using Local State Space (LSS) and Mamba architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Workflow: GitHub Codespaces → vast.ai](#workflow-github-codespaces--vastai)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Advanced Features](#-advanced-features)
  - [Cross-Dataset Evaluation](#cross-dataset-evaluation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Combined Training](#combined-training)
- [Complete Research Workflow](#-complete-research-workflow)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

CAN-LSS-Mamba is a state-of-the-art intrusion detection system for automotive CAN bus networks. It leverages:

- **Mamba State Space Models** for efficient sequence modeling
- **Local-Global Feature Extraction** combining CNN and SSM
- **Efficient Channel Attention (ECA)** for feature refinement
- **Focal Loss** for handling class imbalance

## ✨ Features

### Core Features
- 🔧 **Configuration Management** - YAML-based configs for different environments
- 📊 **Experiment Tracking** - Integrated Weights & Biases (WandB) support
- 🐳 **Docker Support** - Containerized environment for reproducibility
- 📓 **Jupyter Workflow** - Interactive notebook for vast.ai
- 🔄 **Backwards Compatible** - Works with existing scripts and workflows
- ⚙️ **Easy Setup** - One-command setup script with smart dependency checking

### Advanced Features (New!)
- 🔬 **Cross-Dataset Evaluation** - Test generalization across vehicle datasets
- 🎯 **Hyperparameter Tuning** - Grid search and Bayesian optimization
- 📈 **Experiment Comparison** - Compare runs with validation metrics
- 🔗 **Combined Training** - Train on multiple datasets simultaneously
- 🚫 **Data Leakage Prevention** - Three-bucket strategy for ML research
- 📊 **Multi-Dataset Support** - Easy switching between datasets

## 🚀 Workflow: GitHub Codespaces → vast.ai

This project supports a modern ML research workflow optimized for **terminal usage**:

### 1. Edit Code (GitHub Codespaces or Local)
- Make changes in Codespaces or locally
- Commit and push to GitHub
- No GPU required for development

### 2. Run on vast.ai Terminal
```bash
# In vast.ai terminal (not Jupyter)
cd /workspace
git clone https://github.com/jhoshcinco/can-lss-mamba.git
cd can-lss-mamba
bash setup.sh

# Run the complete pipeline
python preprocessing/CAN_preprocess.py
python train.py
python evaluate.py
```

### 3. Track Results
- View training in real-time: https://wandb.ai/YOUR_USERNAME/can-lss-mamba
- Checkpoints auto-saved to WandB (won't lose them when instance terminates!)

> 📘 **For detailed terminal instructions, see [TERMINAL_QUICKSTART.md](TERMINAL_QUICKSTART.md)**

## 🏁 Quick Start

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (for training)
- 16GB+ RAM recommended

### 1. Clone Repository
```bash
git clone https://github.com/jhoshcinco/can-lss-mamba.git
cd can-lss-mamba
```

### 2. Run Setup
```bash
bash setup.sh
```

This will:
- ✅ Create all required directories
- ✅ Install dependencies from `requirements.txt`
- ✅ Verify your environment (GPU, packages)

### 3. Configure Environment (Optional)
```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Download Dataset
Place your CAN dataset in:
- vast.ai: `/workspace/data/can-train-and-test-v1.5/set_01/`
- Local: `./data/can-train-and-test-v1.5/set_01/`

### 5. Run Training
```bash
# Preprocess data
python preprocessing/CAN_preprocess.py

# Train model
python train.py

# Evaluate model
python evaluate.py
```

## 📦 Installation

### Method 1: pip (Recommended)
```bash
pip install -r requirements.txt
```

### Method 2: Docker
```bash
# Build image
docker build -t can-lss-mamba .

# Run with docker-compose
docker-compose up train
```

### Method 3: Conda
```bash
conda create -n can-mamba python=3.11
conda activate can-mamba
pip install -r requirements.txt
```

## ⚙️ Configuration

### Using Configuration Files

The project uses YAML configuration files for different environments:

- `configs/default.yaml` - Base configuration
- `configs/vastai.yaml` - vast.ai specific (uses `/workspace` paths)
- `configs/codespaces.yaml` - Codespaces specific (uses relative paths)

**Select a config:**
```bash
CONFIG_PATH=configs/vastai.yaml python train.py
```

### Configuration Structure

```yaml
data:
  root: /workspace/data
  raw: ${data.root}/can-train-and-test-v1.5/set_01
  processed: ${data.root}/processed_data/set_01_run_02

model:
  checkpoints_dir: /workspace/checkpoints/set_01
  name: lss_can_mamba

training:
  batch_size: 32
  epochs: 20
  learning_rate: 0.0001
  early_stop_patience: 10

wandb:
  enabled: true
  project: can-lss-mamba
```

### Environment Variables Override

Environment variables take precedence over config files:

```bash
BATCH_SIZE=64 EPOCHS=50 python train.py
```

## 🎮 Usage

### Preprocessing
```bash
# Using default config
python preprocessing/CAN_preprocess.py

# Using custom config
CONFIG_PATH=configs/vastai.yaml python scripts/preprocess.py

# With environment overrides
WINDOW_SIZE=128 STRIDE=64 python scripts/preprocess.py
```

### Training
```bash
# Basic training (terminal)
python train.py

# With WandB tracking
WANDB_ENABLED=true WANDB_API_KEY=your_key python train.py

# Custom hyperparameters
BATCH_SIZE=64 EPOCHS=50 LR=0.001 python train.py

# Using wrapper script with config
python scripts/train.py

# Background training (terminal)
nohup python train.py > training.log 2>&1 &
tail -f training.log  # Monitor progress
```

### Evaluation
```bash
# Evaluate on test sets (terminal)
python evaluate.py

# Using wrapper script
python scripts/evaluate.py

# View results
cat /workspace/final_thesis_results_02.csv
```

### Optional: Jupyter Notebook
If you prefer Jupyter (optional):
```bash
# Start Jupyter Lab
jupyter lab

# Open notebooks/vastai_workflow.ipynb
```

## 🎓 Advanced Features

### Cross-Dataset Evaluation

Test model generalization across different vehicle datasets:

```bash
# Train on one dataset, test on all others
python scripts/cross_dataset_eval.py --train-dataset set_01

# Full cross-dataset evaluation matrix
python scripts/cross_dataset_eval.py --all \
  --batch-size 64 --lr 0.0005 --epochs 30

# View results
cat /workspace/results/cross_dataset_matrix_*.csv
```

**Why this matters**: Tests whether your model truly generalizes or just memorizes specific dataset characteristics. Critical for ML research validity.

📖 [Read the full guide](docs/cross_dataset_evaluation.md)

### Hyperparameter Tuning

Systematically find the best hyperparameters using validation metrics only (no data leakage):

**⚠️ CRITICAL: Always use unique checkpoint directories for each hyperparameter configuration!**

```bash
# Automated Grid Search (Recommended - handles checkpoint isolation automatically)
python scripts/grid_search.py --dataset set_01 \
  --batch-sizes 32,64,128 \
  --learning-rates 0.0001,0.0005,0.001 \
  --epochs 20,30,50

# Quick Test (3 learning rates, automatic checkpoint isolation)
bash scripts/quick_test.sh

# Manual Testing (IMPORTANT: Always use unique OUT_DIR per config)
OUT_DIR=/workspace/checkpoints/config1 LR=0.0001 python train.py
OUT_DIR=/workspace/checkpoints/config2 LR=0.0005 python train.py

# Compare results
python scripts/compare_runs.py --tag hyperparameter_search

# WandB Bayesian optimization
wandb sweep configs/sweep.yaml
wandb agent <sweep-id>
```

**Why unique checkpoint directories matter**: Different hyperparameter configurations must start from scratch with fresh model weights. Sharing checkpoint directories causes configs to resume from each other's checkpoints, making comparisons invalid.

📖 [Read the full guide](docs/hyperparameter_tuning.md)

### Combined Training

Train on multiple datasets simultaneously for maximum performance:

```bash
python scripts/train_combined.py \
  --datasets set_01,set_02,set_03,set_04 \
  --batch-size 64 --lr 0.0005 --epochs 30
```

### Multi-Dataset Preprocessing

Preprocess all datasets at once:

```bash
# Preprocess all configured datasets
bash scripts/preprocess_all.sh

# Individual dataset
DATASET=set_02 python preprocessing/CAN_preprocess.py
```

## 📚 Complete Research Workflow

Here's a complete workflow from hyperparameter tuning to final evaluation:

```bash
# ============================================================================
# COMPLETE RESEARCH WORKFLOW
# ============================================================================

# Step 1: Setup environment
bash setup.sh

# Step 2: Preprocess all datasets
bash scripts/preprocess_all.sh

# Step 3: Hyperparameter tuning (uses validation metrics - Bucket 2)
python scripts/grid_search.py --dataset set_01
# → Finds best config: batch=64, lr=0.0005, epochs=30

# Step 4: Compare hyperparameter results
python scripts/compare_runs.py --tag hyperparameter_search
# → Shows validation F1 for all configs

# Step 5: Cross-dataset evaluation with best config (uses test metrics - Bucket 3)
python scripts/cross_dataset_eval.py --all \
  --batch-size 64 --lr 0.0005 --epochs 30
# → Trains on each dataset, tests on all others
# → Generates cross-dataset performance matrix

# Step 6: Train combined model (bonus)
python scripts/train_combined.py \
  --datasets set_01,set_02,set_03,set_04 \
  --batch-size 64 --lr 0.0005 --epochs 30

# Step 7: Export all results for thesis/paper
python scripts/compare_runs.py --tag cross_dataset_eval --output thesis_results.csv
```

### Three-Bucket Strategy (Avoiding Data Leakage)

Our implementation follows ML best practices:

- **Bucket 1 (Training)**: 80% of `train_02_with_attacks/` - Learn model parameters
- **Bucket 2 (Validation)**: 20% of `train_02_with_attacks/` - Tune hyperparameters
- **Bucket 3 (Test)**: `test_*/` folders - Final evaluation ONLY

⚠️ **Critical Rules**:
- ✅ Tune hyperparameters using **validation** metrics (Bucket 2)
- ❌ **NEVER** tune using test metrics (Bucket 3) - that's data leakage!
- ✅ Report test metrics in your thesis/paper

📖 [Read the detailed explanation](docs/three_bucket_strategy.md)

## 📁 Project Structure

```
can-lss-mamba/
├── configs/                    # Configuration files
│   ├── default.yaml           # Base config
│   ├── vastai.yaml            # vast.ai config
│   ├── codespaces.yaml        # Codespaces config
│   ├── datasets.yaml          # Multi-dataset configurations
│   └── sweep.yaml             # WandB sweep config (Bayesian optimization)
│
├── docs/                      # Documentation
│   ├── three_bucket_strategy.md       # Avoiding data leakage
│   ├── hyperparameter_tuning.md       # Tuning guide
│   └── cross_dataset_evaluation.md    # Cross-dataset guide
│
├── src/                       # Source code (modular)
│   ├── data/                  # Data processing
│   │   └── preprocessing.py
│   ├── models/                # Model definitions
│   │   └── lss_mamba.py
│   ├── training/              # Training utilities
│   │   └── wandb_logger.py
│   └── config.py              # Config loader
│
├── scripts/                   # Advanced workflow scripts
│   ├── preprocess.py          # Config-based preprocessing
│   ├── train.py               # Config-based training
│   ├── evaluate.py            # Config-based evaluation
│   ├── cross_dataset_eval.py  # Cross-dataset evaluation
│   ├── train_combined.py      # Combined dataset training
│   ├── grid_search.py         # Hyperparameter grid search
│   ├── compare_runs.py        # WandB experiment comparison
│   ├── quick_test.sh          # Quick hyperparameter test
│   └── preprocess_all.sh      # Batch preprocessing
│
├── notebooks/                 # Jupyter notebooks
│   └── vastai_workflow.ipynb
│
├── preprocessing/             # Original preprocessing (backwards compat)
│   └── CAN_preprocess.py
│
├── tests/                     # Tests
│   └── test_setup.py
│
├── train.py                   # Training script (original, with WandB)
├── evaluate.py                # Evaluation script (original)
├── model.py                   # Model definition (original)
├── requirements.txt           # Python dependencies
├── setup.sh                   # Auto-setup script (smart dependency checking)
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker services
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🔧 Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify NVIDIA driver
nvidia-smi
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.11+
```

### WandB Login Issues
```bash
# Login to WandB
wandb login

# Or set API key
export WANDB_API_KEY=your_key_here
```

### Out of Memory (OOM)
```bash
# Reduce batch size
BATCH_SIZE=16 python train.py

# Enable gradient checkpointing (if implemented)
GRADIENT_CHECKPOINTING=true python train.py
```

### Directory Not Found
```bash
# Re-run setup
bash setup.sh

# Manually create directories
mkdir -p data checkpoints results
```

## 🧪 Testing

```bash
# Run setup tests
python tests/test_setup.py

# Run with pytest (if installed)
pytest tests/
```

## 📊 Experiment Tracking with WandB

### Setup
```bash
# Install WandB
pip install wandb

# Login
wandb login

# Or set API key
export WANDB_API_KEY=your_key_here
```

### Enable Tracking
```bash
# Method 1: Environment variable
WANDB_ENABLED=true python train.py

# Method 2: Config file
# Edit configs/vastai.yaml:
# wandb:
#   enabled: true
```

### View Results
Visit: https://wandb.ai/YOUR_USERNAME/can-lss-mamba

## 🐳 Docker Usage

### Build Image
```bash
docker build -t can-lss-mamba .
```

### Run Services
```bash
# Preprocessing
docker-compose run preprocess

# Training
docker-compose run train

# Evaluation
docker-compose run evaluate

# Jupyter notebook
docker-compose up jupyter
# Open http://localhost:8888
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Mamba-SSM: https://github.com/state-spaces/mamba
- CAN Dataset: [Source]
- Weights & Biases: https://wandb.ai

## 📧 Contact

- GitHub: [@jhoshcinco](https://github.com/jhoshcinco)
- Project Link: https://github.com/jhoshcinco/can-lss-mamba

---

**⭐ If you find this project helpful, please consider giving it a star!**
