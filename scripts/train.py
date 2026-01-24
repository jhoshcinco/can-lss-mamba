#!/usr/bin/env python
"""
Training script for CAN-LSS-Mamba with configuration file support.

This script wraps the existing training logic with:
- Configuration file support (YAML)
- WandB integration for experiment tracking
- Backwards compatibility with original train.py

Usage:
    # Use default config
    python scripts/train.py
    
    # Use specific config
    CONFIG_PATH=configs/vastai.yaml python scripts/train.py
    
    # Override with environment variables
    BATCH_SIZE=64 EPOCHS=50 python scripts/train.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, get_from_config_or_env
from src.training.wandb_logger import setup_wandb_from_config


def main():
    print("="*60)
    print("CAN-LSS-Mamba Training")
    print("="*60)
    
    # Load configuration
    cfg = load_config()
    
    # Get configuration values (with environment variable override support)
    data_dir = get_from_config_or_env(
        cfg, "data.processed", "DATA_DIR",
        default="/workspace/data/processed_data/set_01_run_02"
    )
    out_dir = get_from_config_or_env(
        cfg, "model.checkpoints_dir", "OUT_DIR",
        default="/workspace/checkpoints/set_01"
    )
    model_name = get_from_config_or_env(
        cfg, "model.name", "MODEL_NAME",
        default="lss_can_mamba"
    )
    batch_size = get_from_config_or_env(
        cfg, "training.batch_size", "BATCH_SIZE",
        default=32, cast_type=int
    )
    epochs = get_from_config_or_env(
        cfg, "training.epochs", "EPOCHS",
        default=20, cast_type=int
    )
    lr = get_from_config_or_env(
        cfg, "training.learning_rate", "LR",
        default=1e-4, cast_type=float
    )
    early_stop_patience = get_from_config_or_env(
        cfg, "training.early_stop_patience", "EARLY_STOP_PATIENCE",
        default=10, cast_type=int
    )
    id_dropout_prob = get_from_config_or_env(
        cfg, "training.id_dropout_prob", "ID_DROPOUT_PROB",
        default=0.0, cast_type=float
    )
    
    print(f"\n📋 Configuration:")
    print(f"  Data Dir: {data_dir}")
    print(f"  Output Dir: {out_dir}")
    print(f"  Model Name: {model_name}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Early Stop Patience: {early_stop_patience}")
    print(f"  ID Dropout: {id_dropout_prob}")
    print()
    
    # Setup WandB
    wandb_logger = setup_wandb_from_config(cfg, additional_config={
        "data_dir": data_dir,
        "model_name": model_name,
    })
    
    # Set environment variables for the original script
    os.environ["DATA_DIR"] = data_dir
    os.environ["OUT_DIR"] = out_dir
    os.environ["MODEL_NAME"] = model_name
    os.environ["BATCH_SIZE"] = str(batch_size)
    os.environ["EPOCHS"] = str(epochs)
    os.environ["LR"] = str(lr)
    os.environ["EARLY_STOP_PATIENCE"] = str(early_stop_patience)
    os.environ["ID_DROPOUT_PROB"] = str(id_dropout_prob)
    
    # Store wandb logger for the original script to use
    os.environ["_WANDB_LOGGER_INITIALIZED"] = "1"
    
    # Import the original training script components
    print("Starting training...")
    import train as train_module
    
    # Inject WandB logger into the training module
    train_module.wandb_logger = wandb_logger
    
    # The train.py module will execute its training loop on import
    # since it has the training code at module level
    
    print("\n✅ Training complete!")
    
    # Finish WandB
    wandb_logger.finish()


if __name__ == "__main__":
    main()
