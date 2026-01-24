#!/usr/bin/env python3
"""
Cross-Dataset Evaluation Script

Trains model on one dataset and evaluates on ALL other datasets.
This tests true generalization capability.

Three-Bucket Strategy (NO DATA LEAKAGE):
- Bucket 1: Training data (80% of train_02_with_attacks)
- Bucket 2: Validation data (20% of train_02_with_attacks) - for hyperparameter tuning
- Bucket 3: Test data (test_*/ folders) - for final evaluation ONLY

⚠️  IMPORTANT: This script uses TEST metrics (Bucket 3) for evaluation ONLY.
    NO tuning should be done based on these results - that's data leakage!
    Use grid_search.py with validation metrics for hyperparameter tuning.

Usage:
    # Train on set_01, test on all others
    python scripts/cross_dataset_eval.py --train-dataset set_01
    
    # Run full cross-dataset matrix
    python scripts/cross_dataset_eval.py --all
    
    # Use specific hyperparameters
    python scripts/cross_dataset_eval.py --all --batch-size 64 --lr 0.0005 --epochs 30
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_datasets_config():
    """Load dataset configurations from configs/datasets.yaml"""
    config_path = Path(__file__).parent.parent / "configs" / "datasets.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_model(dataset_name, config, batch_size=32, lr=0.0001, epochs=20, 
                early_stop_patience=10, id_dropout_prob=0.0):
    """
    Train model on specified dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'set_01')
        config: Dataset configuration dict
        batch_size: Training batch size
        lr: Learning rate
        epochs: Number of training epochs
        early_stop_patience: Early stopping patience
        id_dropout_prob: ID dropout probability
    
    Returns:
        bool: True if training succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name}")
    print(f"{'='*60}")
    
    # Set environment variables for training
    env = os.environ.copy()
    env['DATA_DIR'] = config['processed']
    # Unique checkpoint directory per train dataset
    env['OUT_DIR'] = f"/workspace/checkpoints/cross_eval_train_{dataset_name}"
    env['MODEL_NAME'] = f"lss_can_mamba_{dataset_name}"
    env['BATCH_SIZE'] = str(batch_size)
    env['EPOCHS'] = str(epochs)
    env['LR'] = str(lr)
    env['EARLY_STOP_PATIENCE'] = str(early_stop_patience)
    env['ID_DROPOUT_PROB'] = str(id_dropout_prob)
    
    # Enable WandB with proper tags
    env['WANDB_ENABLED'] = 'true'
    env['WANDB_PROJECT'] = 'can-lss-mamba'
    env['WANDB_ENTITY'] = 'jhoshcinco-ca-western-university'
    env['WANDB_TAGS'] = f"cross_dataset_eval,train_{dataset_name}"
    env['WANDB_NAME'] = f"cross_eval_train_{dataset_name}"
    
    # Run training
    result = subprocess.run(
        ['python', 'train.py'],
        env=env,
        cwd=Path(__file__).parent.parent
    )
    
    return result.returncode == 0


def evaluate_model(train_dataset, test_dataset, train_config, test_config):
    """
    Evaluate model trained on train_dataset on test_dataset.
    
    Args:
        train_dataset: Name of training dataset
        test_dataset: Name of test dataset
        train_config: Training dataset configuration
        test_config: Test dataset configuration
    
    Returns:
        dict: Evaluation results with metrics for each test folder
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {train_dataset} → {test_dataset}")
    print(f"{'='*60}")
    
    results = {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'scenarios': []
    }
    
    # Check if test dataset has test folders
    if not test_config.get('test_folders'):
        print(f"⚠️  {test_dataset} has no test folders. Skipping.")
        return results
    
    # Set environment variables for evaluation
    env = os.environ.copy()
    env['DATASET_ROOT'] = test_config['raw']
    env['MODEL_PATH'] = f"/workspace/checkpoints/cross_eval_train_{train_dataset}/lss_can_mamba_{train_dataset}_best.pth"
    env['CHECKPOINT_PATH'] = f"/workspace/checkpoints/cross_eval_train_{train_dataset}/lss_can_mamba_{train_dataset}_last.pth"
    env['ID_MAP_PATH'] = f"{train_config['processed']}/id_map.npy"
    output_csv = f"/workspace/results/cross_eval_{train_dataset}_on_{test_dataset}.csv"
    env['OUTPUT_CSV'] = output_csv
    
    # Run evaluation
    result = subprocess.run(
        ['python', 'evaluate.py'],
        env=env,
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 and os.path.exists(output_csv):
        # Load results
        df = pd.read_csv(output_csv)
        results['scenarios'] = df.to_dict('records')
        
        # Calculate average metrics
        if len(df) > 0:
            results['avg_f1'] = df['F1_Score'].mean()
            results['avg_accuracy'] = df['Accuracy'].mean()
            results['avg_precision'] = df['Precision'].mean()
            results['avg_recall'] = df['Recall'].mean()
    else:
        print(f"❌ Evaluation failed for {train_dataset} → {test_dataset}")
        print(result.stderr)
    
    return results


def run_cross_dataset_eval(datasets_to_train, datasets_config, 
                          batch_size=32, lr=0.0001, epochs=20,
                          early_stop_patience=10, id_dropout_prob=0.0):
    """
    Run complete cross-dataset evaluation.
    
    Args:
        datasets_to_train: List of dataset names to train on
        datasets_config: Full datasets configuration
        batch_size: Training batch size
        lr: Learning rate
        epochs: Number of epochs
        early_stop_patience: Early stopping patience
        id_dropout_prob: ID dropout probability
    
    Returns:
        pd.DataFrame: Cross-dataset performance matrix
    """
    all_datasets = datasets_config['cross_eval_datasets']
    results = []
    
    for train_dataset in datasets_to_train:
        if train_dataset not in datasets_config['datasets']:
            print(f"⚠️  Dataset {train_dataset} not found in config. Skipping.")
            continue
        
        train_config = datasets_config['datasets'][train_dataset]
        
        # Check if processed data exists
        if not os.path.exists(train_config['processed']):
            print(f"⚠️  Processed data not found for {train_dataset}: {train_config['processed']}")
            print(f"   Please run preprocessing first: DATASET={train_dataset} python preprocessing/CAN_preprocess.py")
            continue
        
        # Train model
        success = train_model(
            train_dataset, 
            train_config,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            id_dropout_prob=id_dropout_prob
        )
        
        if not success:
            print(f"❌ Training failed for {train_dataset}. Skipping evaluation.")
            continue
        
        # Evaluate on all datasets (including self)
        for test_dataset in all_datasets:
            if test_dataset not in datasets_config['datasets']:
                continue
            
            test_config = datasets_config['datasets'][test_dataset]
            
            # Check if raw test data exists
            if not os.path.exists(test_config['raw']):
                print(f"⚠️  Test data not found for {test_dataset}: {test_config['raw']}")
                continue
            
            eval_results = evaluate_model(
                train_dataset,
                test_dataset,
                train_config,
                test_config
            )
            
            if eval_results.get('avg_f1') is not None:
                results.append({
                    'train_dataset': train_dataset,
                    'test_dataset': test_dataset,
                    'avg_f1': eval_results['avg_f1'],
                    'avg_accuracy': eval_results['avg_accuracy'],
                    'avg_precision': eval_results['avg_precision'],
                    'avg_recall': eval_results['avg_recall'],
                    'num_scenarios': len(eval_results['scenarios'])
                })
    
    # Create results DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/workspace/results/cross_dataset_matrix_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Cross-Dataset Evaluation Complete!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {output_file}")
        
        # Print matrix
        print("\n📊 Cross-Dataset Performance Matrix (F1 Score):")
        print("=" * 60)
        pivot = df.pivot(index='train_dataset', columns='test_dataset', values='avg_f1')
        print(pivot.round(4))
        
        print("\n📊 Cross-Dataset Performance Matrix (Accuracy):")
        print("=" * 60)
        pivot_acc = df.pivot(index='train_dataset', columns='test_dataset', values='avg_accuracy')
        print(pivot_acc.round(4))
        
        return df
    else:
        print("❌ No results generated. Check for errors above.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Dataset Evaluation for CAN-LSS-Mamba",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on set_01, test on all others
    python scripts/cross_dataset_eval.py --train-dataset set_01
    
    # Run full cross-dataset matrix
    python scripts/cross_dataset_eval.py --all
    
    # Use specific hyperparameters
    python scripts/cross_dataset_eval.py --all --batch-size 64 --lr 0.0005 --epochs 30
        """
    )
    
    parser.add_argument(
        '--train-dataset',
        type=str,
        help='Single dataset to train on (e.g., set_01)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train on all datasets and create full cross-dataset matrix'
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
    
    # Validate arguments
    if not args.train_dataset and not args.all:
        parser.error("Must specify either --train-dataset or --all")
    
    # Load datasets configuration
    datasets_config = load_datasets_config()
    
    # Determine which datasets to train on
    if args.all:
        datasets_to_train = datasets_config['cross_eval_datasets']
    else:
        datasets_to_train = [args.train_dataset]
    
    # Run evaluation
    run_cross_dataset_eval(
        datasets_to_train,
        datasets_config,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        id_dropout_prob=args.id_dropout_prob
    )


if __name__ == '__main__':
    main()
