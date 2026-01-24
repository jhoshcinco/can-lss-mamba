#!/usr/bin/env python3
"""
Compare Experiment Results

Fetches all runs from WandB and compares validation metrics.
Shows best configurations based on validation F1 (Bucket 2).

⚠️  Uses validation metrics ONLY - not test metrics!

Usage:
    # Compare all hyperparameter search runs
    python scripts/compare_runs.py
    
    # Compare specific tag
    python scripts/compare_runs.py --tag hyperparameter_search
    
    # Export to CSV
    python scripts/compare_runs.py --output results.csv
    
    # Compare cross-dataset runs
    python scripts/compare_runs.py --tag cross_dataset_eval
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WandB not installed. Install with: pip install wandb")


def fetch_runs(entity, project, tag=None, limit=1000):
    """
    Fetch runs from WandB.
    
    Args:
        entity: WandB entity name
        project: WandB project name
        tag: Optional tag to filter runs
        limit: Maximum number of runs to fetch
    
    Returns:
        list: List of run data dictionaries
    """
    if not WANDB_AVAILABLE:
        print("❌ WandB is not available. Cannot fetch runs.")
        return []
    
    print(f"Fetching runs from {entity}/{project}...")
    if tag:
        print(f"Filtering by tag: {tag}")
    
    # Initialize WandB API
    api = wandb.Api()
    
    # Build filter
    filters = {}
    if tag:
        filters["tags"] = tag
    
    # Fetch runs
    runs = api.runs(
        f"{entity}/{project}",
        filters=filters,
        per_page=100
    )
    
    run_data = []
    
    for run in runs[:limit]:
        try:
            # Extract config
            config = run.config
            
            # Extract summary metrics (final values)
            summary = run.summary
            
            # Build run data
            data = {
                'run_id': run.id,
                'run_name': run.name,
                'state': run.state,
                'created_at': run.created_at,
                'runtime': run.summary.get('_runtime', 0),
                
                # Hyperparameters
                'batch_size': config.get('batch_size'),
                'learning_rate': config.get('learning_rate'),
                'epochs': config.get('epochs'),
                'early_stop_patience': config.get('early_stop_patience'),
                'id_dropout_prob': config.get('id_dropout_prob'),
                
                # Validation metrics (Bucket 2)
                'val_f1': summary.get('val/f1'),
                'val_accuracy': summary.get('val/accuracy'),
                'val_threshold': summary.get('val/threshold'),
                'val_separation': summary.get('val/separation'),
                
                # Training metrics
                'train_loss': summary.get('train/loss'),
                'train_lr': summary.get('train/lr'),
                
                # Tags
                'tags': ','.join(run.tags) if run.tags else '',
            }
            
            run_data.append(data)
        
        except Exception as e:
            print(f"⚠️  Error processing run {run.id}: {e}")
            continue
    
    print(f"✓ Fetched {len(run_data)} runs")
    
    return run_data


def display_comparison(df, top_n=20):
    """
    Display comparison table of runs.
    
    Args:
        df: DataFrame of run data
        top_n: Number of top runs to display
    """
    if len(df) == 0:
        print("No runs to display.")
        return
    
    print(f"\n{'='*80}")
    print(f"Experiment Comparison (sorted by val_f1)")
    print(f"{'='*80}\n")
    
    # Filter to successful runs with metrics
    df_valid = df[df['val_f1'].notna()].copy()
    
    if len(df_valid) == 0:
        print("No runs with validation metrics found.")
        return
    
    # Sort by validation F1
    df_sorted = df_valid.sort_values('val_f1', ascending=False)
    
    # Select columns to display
    display_cols = [
        'run_name', 'learning_rate', 'batch_size', 'epochs',
        'val_f1', 'val_accuracy', 'val_threshold'
    ]
    
    # Filter to available columns
    display_cols = [col for col in display_cols if col in df_sorted.columns]
    
    # Display top N
    print(f"Top {min(top_n, len(df_sorted))} Configurations:\n")
    print(df_sorted[display_cols].head(top_n).to_string(index=False))
    
    # Display best configuration
    if len(df_sorted) > 0:
        best = df_sorted.iloc[0]
        
        print(f"\n{'='*80}")
        print(f"🏆 Best Configuration (Validation F1 = {best['val_f1']:.4f})")
        print(f"{'='*80}")
        print(f"Run ID: {best['run_id']}")
        print(f"Run Name: {best['run_name']}")
        print(f"\nHyperparameters:")
        if best['learning_rate']:
            print(f"  Learning Rate: {best['learning_rate']}")
        if best['batch_size']:
            print(f"  Batch Size: {int(best['batch_size'])}")
        if best['epochs']:
            print(f"  Epochs: {int(best['epochs'])}")
        if best['early_stop_patience']:
            print(f"  Early Stop Patience: {int(best['early_stop_patience'])}")
        if best['id_dropout_prob']:
            print(f"  ID Dropout Prob: {best['id_dropout_prob']}")
        
        print(f"\nValidation Metrics:")
        print(f"  F1 Score: {best['val_f1']:.4f}")
        if best['val_accuracy']:
            print(f"  Accuracy: {best['val_accuracy']:.4f}")
        if best['val_threshold']:
            print(f"  Threshold: {best['val_threshold']:.4f}")
        if best['val_separation']:
            print(f"  Separation: {best['val_separation']:.4f}")
        
        print(f"\nTags: {best['tags']}")
        print(f"State: {best['state']}")
        print(f"Runtime: {best['runtime']:.1f}s")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare WandB Experiment Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare all runs
    python scripts/compare_runs.py
    
    # Compare hyperparameter search runs
    python scripts/compare_runs.py --tag hyperparameter_search
    
    # Compare cross-dataset runs
    python scripts/compare_runs.py --tag cross_dataset_eval
    
    # Export to CSV
    python scripts/compare_runs.py --output thesis_results.csv
        """
    )
    
    parser.add_argument(
        '--entity',
        type=str,
        default='jhoshcinco-ca-western-university',
        help='WandB entity name (default: jhoshcinco-ca-western-university)'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='can-lss-mamba',
        help='WandB project name (default: can-lss-mamba)'
    )
    
    parser.add_argument(
        '--tag',
        type=str,
        help='Filter runs by tag (e.g., hyperparameter_search, cross_dataset_eval)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top runs to display (default: 20)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='Maximum number of runs to fetch (default: 1000)'
    )
    
    args = parser.parse_args()
    
    if not WANDB_AVAILABLE:
        print("❌ WandB is required for this script.")
        print("   Install with: pip install wandb")
        sys.exit(1)
    
    # Fetch runs
    run_data = fetch_runs(
        args.entity,
        args.project,
        tag=args.tag,
        limit=args.limit
    )
    
    if not run_data:
        print("No runs found.")
        sys.exit(0)
    
    # Create DataFrame
    df = pd.DataFrame(run_data)
    
    # Display comparison
    display_comparison(df, top_n=args.top_n)
    
    # Export to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"✅ Results exported to: {args.output}")


if __name__ == '__main__':
    main()
