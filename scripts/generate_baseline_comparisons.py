#!/usr/bin/env python3
"""
Generate Baseline Comparisons

Trains the main LSS-CAN-Mamba model and all baseline models, then compares their
performance on validation and test sets. Generates comprehensive comparison tables
and visualizations.

This script provides a complete baseline comparison for research papers and thesis work.

Usage:
    # Compare on single dataset
    python scripts/generate_baseline_comparisons.py --dataset set_01
    
    # Compare on all datasets
    python scripts/generate_baseline_comparisons.py --all
    
    # Export results to CSV
    python scripts/generate_baseline_comparisons.py --dataset set_01 --output results.csv
    
    # Skip training (use existing models)
    python scripts/generate_baseline_comparisons.py --dataset set_01 --skip-training
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.baselines import get_baseline_model
from models.lss_mamba import LSS_CAN_Mamba


def evaluate_model(model, data_loader, device, threshold=0.5):
    """
    Evaluate a model on the given data loader.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with test data
        device: Device to run evaluation on
        threshold: Classification threshold
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for ids_batch, feats_batch, labels_batch in data_loader:
            ids_batch = ids_batch.to(device)
            feats_batch = feats_batch.to(device)
            
            logits = model(ids_batch, feats_batch)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= threshold).long()
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels_batch.numpy())
            all_preds.append(preds.cpu().numpy())
    
    inference_time = time.time() - start_time
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Not enough classes or samples for AUC calculation
        auc = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    # Calculate separation
    normal_probs = all_probs[all_labels == 0]
    attack_probs = all_probs[all_labels == 1]
    separation = float(np.mean(attack_probs) - np.mean(normal_probs)) if len(attack_probs) > 0 else 0.0
    
    return {
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'separation': separation,
        'inference_time': inference_time,
        'num_samples': len(all_labels),
    }


def load_data(data_dir, batch_size=128):
    """Load train, validation, and test data."""
    print(f"Loading data from {data_dir}...")
    
    train_npz = np.load(os.path.join(data_dir, "train_data.npz"))
    val_npz = np.load(os.path.join(data_dir, "val_data.npz"))
    test_npz = np.load(os.path.join(data_dir, "test_data.npz"))
    id_map = np.load(
        os.path.join(data_dir, "id_map.npy"), allow_pickle=True
    ).item()
    
    vocab_size = len(id_map)
    seq_len = train_npz["ids"].shape[1]
    
    def get_loader(npz, shuffle=False):
        feats = np.concatenate([npz["payloads"], npz["deltas"]], axis=-1)
        dataset = TensorDataset(
            torch.LongTensor(npz["ids"]),
            torch.FloatTensor(feats),
            torch.LongTensor(npz["labels"]),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return {
        'train_loader': get_loader(train_npz),
        'val_loader': get_loader(val_npz),
        'test_loader': get_loader(test_npz),
        'vocab_size': vocab_size,
        'seq_len': seq_len,
    }


def train_main_model(data_dir, out_dir, batch_size, lr, epochs):
    """Train the main LSS-CAN-Mamba model."""
    print(f"\n{'='*80}")
    print("Training Main LSS-CAN-Mamba Model")
    print(f"{'='*80}")
    
    # Set environment variables
    env = os.environ.copy()
    env['DATA_DIR'] = data_dir
    env['OUT_DIR'] = out_dir
    env['MODEL_NAME'] = 'lss_can_mamba'
    env['BATCH_SIZE'] = str(batch_size)
    env['EPOCHS'] = str(epochs)
    env['LR'] = str(lr)
    env['EARLY_STOP_PATIENCE'] = '10'
    env['WANDB_ENABLED'] = 'false'
    
    # Run training
    result = subprocess.run(
        ['python3', '-m', 'can_lss_mamba.train'],
        env=env,
        cwd=Path(__file__).parent.parent / "src",
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Training failed:")
        print(result.stderr)
        return False
    
    print("✅ Main model training complete")
    return True


def generate_comparison(dataset, data_dir, checkpoint_dir, batch_size=128, output_file=None, skip_training=False):
    """
    Generate comprehensive baseline comparison.
    
    Args:
        dataset: Dataset name (e.g., 'set_01')
        data_dir: Directory containing processed data
        checkpoint_dir: Directory containing model checkpoints
        batch_size: Batch size for evaluation
        output_file: Optional CSV file to save results
        skip_training: Skip training and use existing models
    """
    print(f"\n{'='*80}")
    print(f"GENERATING BASELINE COMPARISONS FOR {dataset.upper()}")
    print(f"{'='*80}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    data = load_data(data_dir, batch_size=batch_size)
    
    # Define models to compare
    models_to_compare = {
        'LSS-CAN-Mamba': {
            'checkpoint': os.path.join(checkpoint_dir, 'main', 'lss_can_mamba_best.pth'),
            'model_class': 'main',
        },
        'MLP': {
            'checkpoint': os.path.join(checkpoint_dir, 'baselines', dataset, 'mlp', 'baseline_mlp_best.pth'),
            'model_class': 'mlp',
        },
        'LSTM': {
            'checkpoint': os.path.join(checkpoint_dir, 'baselines', dataset, 'lstm', 'baseline_lstm_best.pth'),
            'model_class': 'lstm',
        },
        'CNN': {
            'checkpoint': os.path.join(checkpoint_dir, 'baselines', dataset, 'cnn', 'baseline_cnn_best.pth'),
            'model_class': 'cnn',
        },
        'GRU': {
            'checkpoint': os.path.join(checkpoint_dir, 'baselines', dataset, 'gru', 'baseline_gru_best.pth'),
            'model_class': 'gru',
        },
    }
    
    # Results storage
    results = []
    
   models_to_compare = {
        'LSS-CAN-Mamba': {
            # CHANGED: Pointing to 'balanced_model/best_model.pth'
            'checkpoint': os.path.join(checkpoint_dir, 'main', 'lss_can_mamba_best.pth'), 
            'model_class': 'main',
        },
        'MLP': {
            'checkpoint': os.path.join(checkpoint_dir, 'baselines', dataset, 'mlp', 'baseline_mlp_best.pth'),
            'model_class': 'mlp',
        },
        'LSTM': {
            'checkpoint': os.path.join(checkpoint_dir, 'baselines', dataset, 'lstm', 'baseline_lstm_best.pth'),
            'model_class': 'lstm',
        },
        'CNN': {
            'checkpoint': os.path.join(checkpoint_dir, 'baselines', dataset, 'cnn', 'baseline_cnn_best.pth'),
            'model_class': 'cnn',
        },
        'GRU': {
            'checkpoint': os.path.join(checkpoint_dir, 'baselines', dataset, 'gru', 'baseline_gru_best.pth'),
            'model_class': 'gru',
        },
    }
    
    # Results storage
    results = []
    
    # Evaluate each model
    for model_name, config in models_to_compare.items():
        checkpoint_path = config['checkpoint']
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found for {model_name}: {checkpoint_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Initialize model
            if config['model_class'] == 'main':
                model = LSS_CAN_Mamba(
                    num_unique_ids=data['vocab_size']
                ).to(device)
            else:
                model = get_baseline_model(
                    config['model_class'],
                    num_unique_ids=data['vocab_size'],
                    num_continuous_feats=9,
                    d_model=256,
                    seq_len=data['seq_len']
                ).to(device)
            
            # 2. FIX: Robust Weight Loading (Fixes KeyError: 'model')
            # This handles different checkpoint formats automatically
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint  # Assume file is just weights
                
            # Handle 'module.' prefix if trained on multi-GPU
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
                    
            model.load_state_dict(new_state_dict)
            
            # Get best threshold from checkpoint
            best_threshold = checkpoint.get('best_threshold', 0.5)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Parameters: {num_params:,} (Trainable: {num_trainable_params:,})")
            print(f"Best threshold: {best_threshold:.4f}")
            
            # Evaluate on validation set
            print("\nEvaluating on validation set...")
            val_metrics = evaluate_model(model, data['val_loader'], device, threshold=best_threshold)
            
            # Evaluate on test set
            print("Evaluating on test set...")
            test_metrics = evaluate_model(model, data['test_loader'], device, threshold=best_threshold)
            
            # Store results
            result = {
                'Model': model_name,
                'Dataset': dataset,
                'Parameters': num_params,
                'Trainable Parameters': num_trainable_params,
                'Best Threshold': best_threshold,
                # Validation metrics
                'Val F1': val_metrics['f1'],
                'Val Accuracy': val_metrics['accuracy'],
                'Val Precision': val_metrics['precision'],
                'Val Recall': val_metrics['recall'],
                'Val AUC': val_metrics['auc'],
                'Val Separation': val_metrics['separation'],
                # Test metrics
                'Test F1': test_metrics['f1'],
                'Test Accuracy': test_metrics['accuracy'],
                'Test Precision': test_metrics['precision'],
                'Test Recall': test_metrics['recall'],
                'Test AUC': test_metrics['auc'],
                'Test TP': test_metrics['tp'],
                'Test FP': test_metrics['fp'],
                'Test TN': test_metrics['tn'],
                'Test FN': test_metrics['fn'],
                'Test Separation': test_metrics['separation'],
                'Inference Time (s)': test_metrics['inference_time'],
            }
            
            results.append(result)
            
            # Print results
            print(f"\nValidation Metrics:")
            print(f"  F1 Score:   {val_metrics['f1']:.4f}")
            print(f"  Accuracy:   {val_metrics['accuracy']:.4f}")
            print(f"  Precision:  {val_metrics['precision']:.4f}")
            print(f"  Recall:     {val_metrics['recall']:.4f}")
            print(f"  AUC:        {val_metrics['auc']:.4f}")
            
            print(f"\nTest Metrics:")
            print(f"  F1 Score:   {test_metrics['f1']:.4f}")
            print(f"  Accuracy:   {test_metrics['accuracy']:.4f}")
            print(f"  Precision:  {test_metrics['precision']:.4f}")
            print(f"  Recall:     {test_metrics['recall']:.4f}")
            print(f"  AUC:        {test_metrics['auc']:.4f}")
            print(f"  TP/FP/TN/FN: {test_metrics['tp']}/{test_metrics['fp']}/{test_metrics['tn']}/{test_metrics['fn']}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison DataFrame
    if not results:
        print("\n❌ No results to compare!")
        return None
    
    df = pd.DataFrame(results)
    
    # Sort by Test F1 Score
    df = df.sort_values('Test F1', ascending=False)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(f"BASELINE COMPARISON RESULTS - {dataset.upper()}")
    print(f"{'='*80}\n")
    
    # Print compact table
    display_cols = ['Model', 'Parameters', 'Val F1', 'Val Accuracy', 'Test F1', 'Test Accuracy', 'Test AUC']
    print(df[display_cols].to_string(index=False))
    
    # Print detailed metrics for best model
    best_model = df.iloc[0]
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model['Model']}")
    print(f"{'='*80}")
    print(f"Parameters: {best_model['Parameters']:,}")
    print(f"Best Threshold: {best_model['Best Threshold']:.4f}")
    print(f"\nValidation Metrics:")
    print(f"  F1:         {best_model['Val F1']:.4f}")
    print(f"  Accuracy:   {best_model['Val Accuracy']:.4f}")
    print(f"  Precision:  {best_model['Val Precision']:.4f}")
    print(f"  Recall:     {best_model['Val Recall']:.4f}")
    print(f"  AUC:        {best_model['Val AUC']:.4f}")
    print(f"\nTest Metrics:")
    print(f"  F1:         {best_model['Test F1']:.4f}")
    print(f"  Accuracy:   {best_model['Test Accuracy']:.4f}")
    print(f"  Precision:  {best_model['Test Precision']:.4f}")
    print(f"  Recall:     {best_model['Test Recall']:.4f}")
    print(f"  AUC:        {best_model['Test AUC']:.4f}")
    print(f"{'='*80}\n")
    
    # Calculate improvement over baselines
    if len(df) > 1:
        best_f1 = df.iloc[0]['Test F1']
        baseline_avg_f1 = df.iloc[1:]['Test F1'].mean()
        improvement = ((best_f1 - baseline_avg_f1) / baseline_avg_f1) * 100
        
        print(f"Improvement over average baseline: {improvement:.2f}%\n")
    
    # Save to CSV if requested
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Baseline Comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='set_01',
        help='Dataset to compare on (default: set_01)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Compare on all datasets'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=128,
        help='Batch size for evaluation (default: 128)'
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
        help='Number of epochs (default: 20)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for results'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and use existing models'
    )
    
    args = parser.parse_args()
    
    # Determine datasets
    if args.all:
        datasets = ['set_01', 'set_02', 'set_03', 'set_04']
    else:
        datasets = [args.dataset]
    
    all_results = []
    
    # Get base directories from environment or use defaults
    data_root = os.getenv('DATA_ROOT', '/workspace/data/processed_data')
    checkpoint_root = os.getenv('CHECKPOINT_ROOT', '/workspace/checkpoints')
    
    for dataset in datasets:
        data_dir = os.path.join(data_root, dataset)
        
        # Check if data exists
        if not os.path.exists(data_dir):
            print(f"⚠️  Dataset {dataset} not found at {data_dir}. Skipping...")
            continue
        
        # Train models if not skipping
        if not args.skip_training:
            # Train main model
            main_out_dir = os.path.join(checkpoint_root, 'main')
            train_main_model(data_dir, main_out_dir, args.batch_size, args.lr, args.epochs)
            
            # Train baselines
            baseline_script = Path(__file__).parent / 'train_baselines.py'
            subprocess.run([
                'python3', str(baseline_script),
                '--dataset', dataset,
                '--batch-size', str(args.batch_size),
                '--lr', str(args.lr),
                '--epochs', str(args.epochs),
            ])
        
        # Generate comparison
        output_file = args.output if len(datasets) == 1 else None
        if output_file and len(datasets) > 1:
            output_file = output_file.replace('.csv', f'_{dataset}.csv')
        
        df = generate_comparison(
            dataset=dataset,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_root,
            batch_size=args.eval_batch_size,
            output_file=output_file,
            skip_training=args.skip_training
        )
        
        if df is not None:
            all_results.append(df)
    
    # Print overall summary if multiple datasets
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY ACROSS DATASETS")
        print(f"{'='*80}\n")
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Group by model and show average performance
        summary = combined_df.groupby('Model').agg({
            'Val F1': 'mean',
            'Test F1': 'mean',
            'Test Accuracy': 'mean',
            'Test AUC': 'mean',
        }).round(4)
        
        summary = summary.sort_values('Test F1', ascending=False)
        print(summary.to_string())
        print()


if __name__ == '__main__':
    main()
