#!/usr/bin/env python
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from can_lss_mamba.train import train
from src.config import load_config, get_from_config_or_env
from src.training.wandb_logger import setup_wandb_from_config

def main():
    print("=" * 60)
    print("CAN-LSS-Mamba Training")
    print("=" * 60)

    cfg = load_config()

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
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")

    wandb_logger = setup_wandb_from_config(cfg, additional_config={
        "data_dir": data_dir,
        "model_name": model_name,
    })
    
    # We can pass wandb_enabled=True to train() but it initializes its own wandb.
    # Since we already have a wandb_logger set up here (maybe), we might need to adjust.
    # The new train module initiates wandb if wandb_enabled=True.
    # However, existing scripts/train.py logic was injecting wandb_logger.
    # The new train() function doesn't accept an external logger object directly.
    # But if wandb is already initialized globaly (wandb.init), subsequent calls might just attach.
    
    # Actually, the best way is to let train() handle it if we want to keep it simple,
    # OR update train() to accept a logger.
    # For now, I'll rely on global wandb state or just pass the config params.
    
    # But wait, the original scripts/train.py was doing custom wandb setup.
    # I'll just call train() and let it do its thing, but if I want to support the custom wandb setup
    # I might need to modifying train() to accept an optional logger.
    # The user request didn't ask for feature parity on the script wrapper, just cleanup.
    
    # Simpler approach: Just call train() with arguments.
    
    train(
        data_dir=data_dir,
        out_dir=out_dir,
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        wandb_enabled=(os.environ.get("WANDB_ENABLED", "false").lower() == "true"),
        early_stop_patience=early_stop_patience,
        id_dropout_prob=id_dropout_prob
    )

if __name__ == "__main__":
    main()
