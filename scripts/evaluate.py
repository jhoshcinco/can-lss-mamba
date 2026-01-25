#!/usr/bin/env python
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from can_lss_mamba.evaluate import evaluate
from src.config import load_config, get_from_config_or_env


def main():
    print("=" * 60)
    print("CAN-LSS-Mamba Evaluation")
    print("=" * 60)

    cfg = load_config()

    dataset_root = get_from_config_or_env(
        cfg, "data.raw", "DATASET_ROOT",
        default="/workspace/data/can-train-and-test-v1.5/set_01"
    )
    model_path = get_from_config_or_env(
        cfg, "model.checkpoints_dir", "CHECKPOINT_ROOT",
        default="/workspace/checkpoints/set_01"
    )
    model_name = get_from_config_or_env(
        cfg, "model.name", "MODEL_NAME",
        default="lss_can_mamba"
    )
    full_model_path = os.path.join(model_path, f"{model_name}_best.pth")
    checkpoint_path = os.path.join(model_path, f"{model_name}_last.pth")

    processed_dir = get_from_config_or_env(
        cfg, "data.processed", "DATA_DIR",
        default="/workspace/data/processed_data/set_01_run_02"
    )
    id_map_path = os.path.join(processed_dir, "id_map.npy")

    results_root = os.getenv("RESULTS_ROOT", "/workspace")
    output_csv = os.path.join(results_root, "final_thesis_results_02.csv")

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

    print("Starting evaluation...")
    evaluate(
        dataset_root=dataset_root,
        model_path=full_model_path,
        checkpoint_path=checkpoint_path,
        id_map_path=id_map_path,
        output_csv=output_csv,
        batch_size=batch_size,
        chunk_windows=chunk_windows
    )

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
