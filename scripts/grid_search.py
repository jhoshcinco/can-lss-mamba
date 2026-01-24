#!/usr/bin/env python3
"""
Automated Grid Search for Hyperparameter Tuning

⚠️  CRITICAL: Uses ONLY validation metrics (Bucket 2) for optimization.
    Does NOT use test results (Bucket 3) to prevent data leakage.

Three-Bucket Strategy:
- Bucket 1: Training data (80% of train_02_with_attacks) - Model learning
- Bucket 2: Validation data (20% of train_02_with_attacks) - Hyperparameter tuning
- Bucket 3: Test data (test_*/ folders) - Final evaluation (evaluate.py)

This script ONLY uses Bucket 2 (validation) metrics for optimization.
Never call evaluate.py from this script - that would be data leakage!

Usage:
    # Default grid
    python scripts/grid_search.py
    
    # Custom grid
    python scripts/grid_search.py \
      --batch-sizes 32,64,128 \
      --learning-rates 0.0001,0.0005,0.001 \
      --epochs 20,30,50 \
      --dataset set_01
    
    # Specific parameter sweep
    python scripts/grid_search.py --param learning_rate --values 0.0001,0.0005,0.001
    
    # With WandB sweep
    python scripts/grid_search.py --use-wandb-sweep
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import itertools
import yaml
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_datasets_config():
    """Load dataset configurations"""
    config_path = Path(__file__).parent.parent / "configs" / "datasets.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_checkpoint_dir(dataset_name, hyperparams):
    """
    Generate unique checkpoint directory for hyperparameter configuration.
    
    Args:
        dataset_name: Name of dataset
        hyperparams: Dictionary of hyperparameters
    
    Returns:
        str: Unique checkpoint directory path
    """
    batch_size = hyperparams.get('batch_size', 32)
    lr = hyperparams.get('learning_rate', 0.0001)
    epochs = hyperparams.get('epochs', 20)
    dropout = hyperparams.get('id_dropout_prob', 0.0)
    
    # Create unique dir name based on hyperparameters
    dir_name = f"grid_bs{batch_size}_lr{lr}_ep{epochs}_drop{dropout}"
    return os.path.join("/workspace/checkpoints", dataset_name, dir_name)


def run_experiment(dataset_name, dataset_config, hyperparams, exp_num, total_exp):
    """
    Run single training experiment with given hyperparameters.
    
    Args:
        dataset_name: Name of dataset
        dataset_config: Dataset configuration
        hyperparams: Dictionary of hyperparameters
        exp_num: Experiment number
        total_exp: Total number of experiments
    
    Returns:
        dict: Results including validation metrics
    """
    print(f"\n{'='*60}")
    print(f"Experiment {exp_num}/{total_exp}")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Generate unique checkpoint directory for this config
    checkpoint_dir = get_checkpoint_dir(dataset_name, hyperparams)
    
    # Set environment variables
    env = os.environ.copy()
    env['DATA_DIR'] = dataset_config['processed']
    env['OUT_DIR'] = checkpoint_dir  # Use unique checkpoint directory
    env['MODEL_NAME'] = f"lss_can_mamba"
    
    # Set hyperparameters
    env['BATCH_SIZE'] = str(hyperparams['batch_size'])
    env['EPOCHS'] = str(hyperparams['epochs'])
    env['LR'] = str(hyperparams['learning_rate'])
    env['EARLY_STOP_PATIENCE'] = str(hyperparams['early_stop_patience'])
    env['ID_DROPOUT_PROB'] = str(hyperparams.get('id_dropout_prob', 0.0))
    
    # Create descriptive run name for WandB
    batch_size = hyperparams['batch_size']
    lr = hyperparams['learning_rate']
    epochs = hyperparams['epochs']
    run_name = f"grid_bs{batch_size}_lr{lr}_ep{epochs}"
    
    # Enable WandB with proper tags
    env['WANDB_ENABLED'] = 'true'
    env['WANDB_PROJECT'] = 'can-lss-mamba'
    env['WANDB_ENTITY'] = 'jhoshcinco-ca-western-university'
    env['WANDB_TAGS'] = f"hyperparameter_search,grid_search_{dataset_name}"
    env['WANDB_NAME'] = run_name
    
    # Run training
    result = subprocess.run(
        ['python', 'train.py'],
        env=env,
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True
    )
    
    # Parse validation metrics from output
    val_f1 = None
    val_acc = None
    val_threshold = None
    
    if result.returncode == 0:
        # Try to extract best validation F1 from output
        for line in result.stdout.split('\n'):
            if 'Best F1:' in line:
                try:
                    parts = line.split('Best F1:')[1].split('|')
                    val_f1 = float(parts[0].split()[0])
                    if 'Best Threshold:' in line:
                        val_threshold = float(line.split('Best Threshold:')[1].split()[0])
                except:
                    pass
        
        # If we couldn't parse, set default values
        if val_f1 is None:
            print("⚠️  Could not parse validation metrics from output")
            val_f1 = 0.0
            val_acc = 0.0
            val_threshold = 0.5
    else:
        print(f"❌ Experiment failed!")
        print(result.stderr)
        val_f1 = 0.0
        val_acc = 0.0
        val_threshold = 0.5
    
    print(f"\nResults: Val F1 = {val_f1:.4f}")
    
    return {
        'experiment': exp_num,
        'dataset': dataset_name,
        'val_f1': val_f1,
        'val_accuracy': val_acc,
        'val_threshold': val_threshold,
        'success': result.returncode == 0,
        **hyperparams
    }


def grid_search(dataset_name, datasets_config, param_grid):
    """
    Run grid search over hyperparameter combinations.
    
    Args:
        dataset_name: Name of dataset
        datasets_config: Full datasets configuration
        param_grid: Dictionary of hyperparameters to search
    
    Returns:
        pd.DataFrame: Results of all experiments
    """
    if dataset_name not in datasets_config['datasets']:
        print(f"❌ Dataset {dataset_name} not found in config")
        return None
    
    dataset_config = datasets_config['datasets'][dataset_name]
    
    # Check if processed data exists
    if not os.path.exists(dataset_config['processed']):
        print(f"❌ Processed data not found for {dataset_name}: {dataset_config['processed']}")
        print(f"   Please run preprocessing first.")
        return None
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"\n{'='*60}")
    print(f"Grid Search: {dataset_name}")
    print(f"{'='*60}")
    print(f"Parameters: {keys}")
    print(f"Total combinations: {len(combinations)}")
    print(f"{'='*60}\n")
    
    results = []
    
    # Run experiments with progress bar
    for i, combo in enumerate(tqdm(combinations, desc="Grid Search Progress"), 1):
        hyperparams = dict(zip(keys, combo))
        
        result = run_experiment(
            dataset_name,
            dataset_config,
            hyperparams,
            i,
            len(combinations)
        )
        
        results.append(result)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    os.makedirs("/workspace/results/grid_search", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/workspace/results/grid_search/grid_search_{dataset_name}_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Grid Search Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_file}")
    
    # Find and display best configuration
    if len(df[df['success']]) > 0:
        best_idx = df[df['success']]['val_f1'].idxmax()
        best_config = df.loc[best_idx]
        
        print(f"\n🏆 Best Configuration (Validation F1 = {best_config['val_f1']:.4f}):")
        print("=" * 60)
        print(f"  Batch Size: {int(best_config['batch_size'])}")
        print(f"  Learning Rate: {best_config['learning_rate']}")
        print(f"  Epochs: {int(best_config['epochs'])}")
        print(f"  Early Stop Patience: {int(best_config['early_stop_patience'])}")
        if 'id_dropout_prob' in best_config:
            print(f"  ID Dropout Prob: {best_config['id_dropout_prob']}")
        print(f"  Validation Threshold: {best_config['val_threshold']:.4f}")
        
        # Save best config
        best_config_dict = {
            'batch_size': int(best_config['batch_size']),
            'learning_rate': float(best_config['learning_rate']),
            'epochs': int(best_config['epochs']),
            'early_stop_patience': int(best_config['early_stop_patience']),
            'val_f1': float(best_config['val_f1']),
            'val_threshold': float(best_config['val_threshold'])
        }
        
        if 'id_dropout_prob' in best_config:
            best_config_dict['id_dropout_prob'] = float(best_config['id_dropout_prob'])
        
        best_config_file = f"/workspace/results/grid_search/best_config_{dataset_name}_{timestamp}.yaml"
        with open(best_config_file, 'w') as f:
            yaml.dump(best_config_dict, f)
        
        print(f"\n✅ Best config saved to: {best_config_file}")
        
        # Print comparison table (top 10)
        print(f"\n📊 Top 10 Configurations (sorted by Validation F1):")
        print("=" * 60)
        top_10 = df[df['success']].nlargest(10, 'val_f1')[
            ['experiment', 'learning_rate', 'batch_size', 'epochs', 'val_f1']
        ]
        print(top_10.to_string(index=False))
    else:
        print("\n❌ No successful experiments to analyze.")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Grid Search for Hyperparameter Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  IMPORTANT: This script uses VALIDATION metrics (Bucket 2) ONLY.
    Test metrics (Bucket 3) are never used to prevent data leakage!

Examples:
    # Default grid
    python scripts/grid_search.py
    
    # Custom grid
    python scripts/grid_search.py \\
      --batch-sizes 32,64,128 \\
      --learning-rates 0.0001,0.0005,0.001 \\
      --epochs 20,30,50
    
    # Specific parameter sweep
    python scripts/grid_search.py --param learning_rate --values 0.0001,0.0005,0.001
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='set_01',
        help='Dataset to use for grid search (default: set_01)'
    )
    
    parser.add_argument(
        '--batch-sizes',
        type=str,
        default='32,64,128',
        help='Comma-separated batch sizes to test (default: 32,64,128)'
    )
    
    parser.add_argument(
        '--learning-rates',
        type=str,
        default='0.0001,0.0005,0.001',
        help='Comma-separated learning rates to test (default: 0.0001,0.0005,0.001)'
    )
    
    parser.add_argument(
        '--epochs',
        type=str,
        default='20,30,50',
        help='Comma-separated epoch counts to test (default: 20,30,50)'
    )
    
    parser.add_argument(
        '--early-stop-patience',
        type=str,
        default='5,10,15',
        help='Comma-separated patience values (default: 5,10,15)'
    )
    
    parser.add_argument(
        '--param',
        type=str,
        help='Single parameter to sweep (learning_rate, batch_size, epochs, early_stop_patience)'
    )
    
    parser.add_argument(
        '--values',
        type=str,
        help='Comma-separated values for single parameter sweep'
    )
    
    parser.add_argument(
        '--use-wandb-sweep',
        action='store_true',
        help='Use WandB sweep instead of local grid search'
    )
    
    args = parser.parse_args()
    
    # Load datasets configuration
    datasets_config = load_datasets_config()
    
    # WandB sweep mode
    if args.use_wandb_sweep:
        print("⚠️  WandB sweep mode not yet implemented.")
        print("   Please use: wandb sweep configs/sweep.yaml")
        print("   Then: wandb agent <sweep_id>")
        sys.exit(1)
    
    # Build parameter grid
    if args.param and args.values:
        # Single parameter sweep
        param_grid = {
            'batch_size': [32],
            'learning_rate': [0.0001],
            'epochs': [20],
            'early_stop_patience': [10]
        }
        
        # Parse values
        if args.param in ['batch_size', 'epochs', 'early_stop_patience']:
            values = [int(v) for v in args.values.split(',')]
        else:
            values = [float(v) for v in args.values.split(',')]
        
        param_grid[args.param] = values
    else:
        # Full grid search
        param_grid = {
            'batch_size': [int(x) for x in args.batch_sizes.split(',')],
            'learning_rate': [float(x) for x in args.learning_rates.split(',')],
            'epochs': [int(x) for x in args.epochs.split(',')],
            'early_stop_patience': [int(x) for x in args.early_stop_patience.split(',')]
        }
    
    # Run grid search
    grid_search(args.dataset, datasets_config, param_grid)


if __name__ == '__main__':
    main()
