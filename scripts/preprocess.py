#!/usr/bin/env python
"""
Preprocessing script for CAN-LSS-Mamba.

This script wraps the existing preprocessing logic with configuration file support.
It maintains backwards compatibility with the original preprocessing/CAN_preprocess.py.

Usage:
    # Use default config
    python scripts/preprocess.py
    
    # Use specific config
    CONFIG_PATH=configs/vastai.yaml python scripts/preprocess.py
    
    # Override with environment variables
    DATA_ROOT=/custom/path python scripts/preprocess.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, get_from_config_or_env, print_config


def main():
    print("="*60)
    print("CAN-LSS-Mamba Preprocessing")
    print("="*60)
    
    # Load configuration
    cfg = load_config()
    
    # Get paths from config or environment variables
    dataset_root = get_from_config_or_env(
        cfg, "data.raw", "DATASET_ROOT",
        default="/workspace/data/can-train-and-test-v1.5/set_01"
    )
    output_dir = get_from_config_or_env(
        cfg, "data.processed", "OUTPUT_DIR",
        default="/workspace/data/processed_data/set_01_run_02"
    )
    window_size = get_from_config_or_env(
        cfg, "preprocessing.window_size", "WINDOW_SIZE",
        default=64, cast_type=int
    )
    stride = get_from_config_or_env(
        cfg, "preprocessing.stride", "STRIDE",
        default=64, cast_type=int
    )
    
    print(f"\n📋 Configuration:")
    print(f"  Dataset Root: {dataset_root}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Window Size: {window_size}")
    print(f"  Stride: {stride}")
    print()
    
    # Set environment variables for the original script
    os.environ["DATASET_ROOT"] = dataset_root
    os.environ["OUTPUT_DIR"] = output_dir
    os.environ["WINDOW_SIZE"] = str(window_size)
    os.environ["STRIDE"] = str(stride)
    
    # Import and run the original preprocessing
    print("Running preprocessing...")
    from src.data.preprocessing import run_pipeline
    
    # Monkey-patch the configuration into the module
    import src.data.preprocessing as preprocess_module
    preprocess_module.DATASET_ROOT = dataset_root
    preprocess_module.OUTPUT_DIR = output_dir
    preprocess_module.WINDOW_SIZE = window_size
    preprocess_module.STRIDE = stride
    
    # Run the pipeline
    run_pipeline()
    
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
