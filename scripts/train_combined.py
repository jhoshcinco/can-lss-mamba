#!/usr/bin/env python3
"""
Train on Combined Datasets

Trains a single model on ALL datasets combined.
This shows maximum achievable performance with all available data.

Usage:
    python scripts/train_combined.py --datasets set_01,set_02,set_03,set_04
    
    # With custom hyperparameters
    python scripts/train_combined.py --datasets set_01,set_02 \
        --batch-size 64 --lr 0.0005 --epochs 30
"""

import argparse
import os
import sys
import subprocess
import numpy as np
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_datasets_config():
    """Load dataset configurations from configs/datasets.yaml"""
    config_path = Path(__file__).parent.parent / "configs" / "datasets.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def combine_datasets(dataset_names, datasets_config, output_dir):
    """
    Combine multiple datasets into a single training set.
    
    Args:
        dataset_names: List of dataset names to combine
        datasets_config: Full datasets configuration
        output_dir: Output directory for combined data
    
    Returns:
        str: Path to combined dataset directory
    """
    print(f"\n{'='*60}")
    print(f"Combining Datasets: {', '.join(dataset_names)}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data from all datasets
    combined_train_data = {
        'ids': [],
        'payloads': [],
        'deltas': [],
        'labels': []
    }
    
    combined_val_data = {
        'ids': [],
        'payloads': [],
        'deltas': [],
        'labels': []
    }
    
    # Track all unique IDs across datasets
    all_ids = set()
    
    for dataset_name in dataset_names:
        if dataset_name not in datasets_config['datasets']:
            print(f"⚠️  Dataset {dataset_name} not found. Skipping.")
            continue
        
        dataset_config = datasets_config['datasets'][dataset_name]
        processed_path = dataset_config['processed']
        
        if not os.path.exists(processed_path):
            print(f"⚠️  Processed data not found for {dataset_name}: {processed_path}")
            print(f"   Please run preprocessing first.")
            continue
        
        print(f"Loading {dataset_name}...")
        
        # Load train data
        train_npz = np.load(os.path.join(processed_path, "train_data.npz"))
        combined_train_data['ids'].append(train_npz['ids'])
        combined_train_data['payloads'].append(train_npz['payloads'])
        combined_train_data['deltas'].append(train_npz['deltas'])
        combined_train_data['labels'].append(train_npz['labels'])
        
        # Load val data
        val_npz = np.load(os.path.join(processed_path, "val_data.npz"))
        combined_val_data['ids'].append(val_npz['ids'])
        combined_val_data['payloads'].append(val_npz['payloads'])
        combined_val_data['deltas'].append(val_npz['deltas'])
        combined_val_data['labels'].append(val_npz['labels'])
        
        # Load ID map
        id_map = np.load(os.path.join(processed_path, "id_map.npy"), allow_pickle=True).item()
        all_ids.update(id_map.keys())
        
        print(f"  ✓ Loaded {len(train_npz['ids'])} train + {len(val_npz['ids'])} val samples")
    
    if not combined_train_data['ids']:
        print("❌ No datasets loaded. Exiting.")
        return None
    
    # Concatenate all arrays
    print("\nCombining arrays...")
    train_data = {
        'ids': np.concatenate(combined_train_data['ids']),
        'payloads': np.concatenate(combined_train_data['payloads']),
        'deltas': np.concatenate(combined_train_data['deltas']),
        'labels': np.concatenate(combined_train_data['labels'])
    }
    
    val_data = {
        'ids': np.concatenate(combined_val_data['ids']),
        'payloads': np.concatenate(combined_val_data['payloads']),
        'deltas': np.concatenate(combined_val_data['deltas']),
        'labels': np.concatenate(combined_val_data['labels'])
    }
    
    # Create unified ID map
    unified_id_map = {id_val: idx for idx, id_val in enumerate(sorted(all_ids))}
    
    # Save combined data
    print(f"\nSaving combined dataset to {output_dir}...")
    np.savez(
        os.path.join(output_dir, "train_data.npz"),
        ids=train_data['ids'],
        payloads=train_data['payloads'],
        deltas=train_data['deltas'],
        labels=train_data['labels']
    )
    
    np.savez(
        os.path.join(output_dir, "val_data.npz"),
        ids=val_data['ids'],
        payloads=val_data['payloads'],
        deltas=val_data['deltas'],
        labels=val_data['labels']
    )
    
    np.save(os.path.join(output_dir, "id_map.npy"), unified_id_map)
    
    print(f"✓ Combined dataset created:")
    print(f"  - Train samples: {len(train_data['ids'])}")
    print(f"  - Val samples: {len(val_data['ids'])}")
    print(f"  - Unique IDs: {len(unified_id_map)}")
    
    return output_dir


def train_combined(dataset_names, datasets_config, batch_size=32, lr=0.0001, 
                   epochs=20, early_stop_patience=10, id_dropout_prob=0.0):
    """
    Train model on combined datasets.
    
    Args:
        dataset_names: List of dataset names to combine
        datasets_config: Full datasets configuration
        batch_size: Training batch size
        lr: Learning rate
        epochs: Number of epochs
        early_stop_patience: Early stopping patience
        id_dropout_prob: ID dropout probability
    
    Returns:
        bool: True if training succeeded
    """
    # Combine datasets
    combined_name = "_".join(dataset_names)
    output_dir = f"/workspace/data/processed_data/combined_{combined_name}"
    
    combined_path = combine_datasets(dataset_names, datasets_config, output_dir)
    
    if combined_path is None:
        return False
    
    # Train model
    print(f"\n{'='*60}")
    print(f"Training Combined Model: {combined_name}")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    env['DATA_DIR'] = combined_path
    env['OUT_DIR'] = f"/workspace/checkpoints/combined_{combined_name}"
    env['MODEL_NAME'] = f"lss_can_mamba_combined"
    env['BATCH_SIZE'] = str(batch_size)
    env['EPOCHS'] = str(epochs)
    env['LR'] = str(lr)
    env['EARLY_STOP_PATIENCE'] = str(early_stop_patience)
    env['ID_DROPOUT_PROB'] = str(id_dropout_prob)
    
    # Enable WandB
    env['WANDB_ENABLED'] = 'true'
    env['WANDB_PROJECT'] = 'can-lss-mamba'
    env['WANDB_ENTITY'] = 'jhoshcinco-ca-western-university'
    env['WANDB_TAGS'] = f"combined_training,datasets_{combined_name}"
    env['WANDB_NAME'] = f"combined_{combined_name}"
    
    # Run training
    result = subprocess.run(
        ['python', 'train.py'],
        env=env,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode == 0:
        print(f"\n✅ Combined model training complete!")
        print(f"   Model saved to: {env['OUT_DIR']}")
        return True
    else:
        print(f"\n❌ Training failed.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train CAN-LSS-Mamba on Combined Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Combine all datasets
    python scripts/train_combined.py --datasets set_01,set_02,set_03,set_04
    
    # Combine specific datasets
    python scripts/train_combined.py --datasets set_01,set_02 --batch-size 64
        """
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        required=True,
        help='Comma-separated list of datasets to combine (e.g., set_01,set_02,set_03)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    
    parser.add_argument(
        '--id-dropout-prob',
        type=float,
        default=0.0,
        help='ID dropout probability (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Parse dataset names
    dataset_names = [name.strip() for name in args.datasets.split(',')]
    
    if len(dataset_names) < 2:
        parser.error("Must specify at least 2 datasets to combine")
    
    # Load datasets configuration
    datasets_config = load_datasets_config()
    
    # Train combined model
    success = train_combined(
        dataset_names,
        datasets_config,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        id_dropout_prob=args.id_dropout_prob
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
