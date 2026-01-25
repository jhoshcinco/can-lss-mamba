# CAN-LSS-Mamba

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning model for Controller Area Network (CAN) bus intrusion detection using Local State Space (LSS) and Mamba architecture.

## 📋 Table of Contents

- [Overview](docs/README.md#overview)
- [Features](docs/README.md#features)
- [Workflow: GitHub Codespaces → vast.ai](docs/README.md#workflow-github-codespaces--vastai)
- [Quick Start](docs/README.md#quick-start)
- [Installation](docs/README.md#installation)
- [Configuration](docs/README.md#configuration)
- [Usage](docs/README.md#usage)
- [Advanced Features](docs/README.md#advanced-features)
  - [Cross-Dataset Evaluation](docs/CROSS_DATASET_IMPLEMENTATION_SUMMARY.md)
  - [Hyperparameter Tuning](docs/hyperparameter_tuning.md)
  - [Baseline Comparisons](docs/baseline_comparisons.md)
  - [Combined Training](docs/combined_training.md)
- [Complete Research Workflow](docs/complete_workflow.md)
- [Project Structure](docs/project_structure.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Contributing](docs/contributing.md)

## 🎯 Overview

CAN-LSS-Mamba is a state-of-the-art intrusion detection system for automotive CAN bus networks. It leverages:

- **Mamba State Space Models** for efficient sequence modeling
- **Local-Global Feature Extraction** combining CNN and SSM
- **Efficient Channel Attention (ECA)** for feature refinement
- **Focal Loss** for handling class imbalance

> 📘 **For detailed documentation, see the [docs/](docs/) folder.**

## 🏁 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/jhoshcinco/can-lss-mamba.git
cd can-lss-mamba
```

### 2. Run Setup
```bash
bash setup.sh
```

### 3. Run Training (New Package Structure)
```bash
# Using the new package structure
python -m can_lss_mamba.train \
  --data-dir /workspace/data/processed_data/set_01 \
  --out-dir /workspace/checkpoints/set_01
```

## 📁 Project Structure

```
can-lss-mamba/
├── configs/                    # Configuration files
├── docs/                       # ALL Documentation (guides, summaries, quickstarts)
├── src/                        # Source code
│   └── can_lss_mamba/          # Main package
│       ├── models/             # Model definitions
│       │   └── mamba.py
│       ├── train.py            # Training logic
│       └── evaluate.py         # Evaluation logic
├── scripts/                    # Helper scripts
├── notebooks/                  # Jupyter notebooks
├── preprocessing/              # Data preprocessing scripts
├── tests/                      # Tests
├── requirements.txt            # Python dependencies
├── setup.sh                    # Auto-setup script
└── README.md                   # This file
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
