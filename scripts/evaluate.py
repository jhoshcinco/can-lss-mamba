#!/usr/bin/env python
"""
Evaluation script for CAN-LSS-Mamba with configuration file support.

This script wraps the existing evaluation logic with configuration file support.
It maintains backwards compatibility with the original evaluate.py.

Usage:
    # Use default config
    python scripts/evaluate.py
    
    # Use specific config
    CONFIG_PATH=configs/vastai.yaml python scripts/evaluate.py
    
    # Override with environment variables
    DATASET_ROOT=/custom/path python scripts/evaluate.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, get_from_config_or_env


def main():
    print("="*60)
    print("CAN-LSS-Mamba Evaluation")
    print("="*60)
    
    # Load configuration
    cfg = load_config()
    
    # Get paths from config or environment variables
    dataset_root = get_from_config_or_env(
        cfg, "data.raw", "DATASET_ROOT",
        default="/workspace/data/can-train-and-test-v1.5/set_01"
    )
    model_path = get_from_config_or_env(
        cfg, "model.checkpoints_dir", "CHECKPOINT_ROOT",
        default="/workspace/checkpoints/set_01"
    )
    
    # Construct full model path
    model_name = get_from_config_or_env(
        cfg, "model.name", "MODEL_NAME",
        default="lss_can_mamba"
    )
    full_model_path = os.path.join(model_path, f"{model_name}_best.pth")
    checkpoint_path = os.path.join(model_path, f"{model_name}_last.pth")
    
    # Get processed data path for ID map
    processed_dir = get_from_config_or_env(
        cfg, "data.processed", "DATA_DIR",
        default="/workspace/data/processed_data/set_01_run_02"
    )
    id_map_path = os.path.join(processed_dir, "id_map.npy")
    
    # Results output
    results_root = os.getenv("RESULTS_ROOT", "/workspace")
    output_csv = os.path.join(results_root, "final_thesis_results_02.csv")
    
    # Evaluation parameters
    batch_size = get_from_config_or_env(
        cfg, "evaluation.batch_size", "BATCH_SIZE",
        default=128, cast_type=int
    )
    chunk_windows = get_from_config_or_env(
        cfg, "evaluation.chunk_windows", "CHUNK_WINDOWS",
        default=20000, cast_type=int
    )
    
    print(f"\n📋 Configuration:")
    print(f"  Dataset Root: {dataset_root}")
    print(f"  Model Path: {full_model_path}")
    print(f"  Checkpoint Path: {checkpoint_path}")
    print(f"  ID Map Path: {id_map_path}")
    print(f"  Output CSV: {output_csv}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Chunk Windows: {chunk_windows}")
    print()
    
    # Set environment variables for the original script
    # The evaluate.py uses hardcoded paths, so we need to modify them
    import evaluate as eval_module
    
    # Monkey-patch the configuration into the module
    eval_module.DATASET_ROOT = dataset_root
    eval_module.MODEL_PATH = full_model_path
    eval_module.CHECKPOINT_PATH = checkpoint_path
    eval_module.ID_MAP_PATH = id_map_path
    eval_module.OUTPUT_CSV = output_csv
    eval_module.BATCH_SIZE = batch_size
    eval_module.CHUNK_WINDOWS = chunk_windows
    
    print("Starting evaluation...")
    eval_module.evaluate_all()
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
