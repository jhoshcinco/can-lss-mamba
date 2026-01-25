#!/usr/bin/env python3
"""
Train Baseline Models for Comparison

Trains simple baseline models (MLP, LSTM, CNN, GRU) to compare against LSS-CAN-Mamba.
Uses the same training pipeline and data as the main model for fair comparison.

Usage:
    # Train all baselines on set_01
    python scripts/train_baselines.py --dataset set_01
    
    # Train specific baseline
    python scripts/train_baselines.py --dataset set_01 --model lstm
    
    # With custom hyperparameters
    python scripts/train_baselines.py --dataset set_01 --batch-size 64 --lr 0.0005 --epochs 30
    
    # Train on all datasets
    python scripts/train_baselines.py --all
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import time
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.baselines import get_baseline_model


def train_baseline(
    model_name,
    data_dir,
    out_dir,
    batch_size=32,
    epochs=20,
    lr=1e-4,
    wandb_enabled=False,
    wandb_project="can-lss-mamba",
    early_stop_patience=10,
):
    """
    Train a baseline model.
    
    Args:
        model_name: Name of baseline model ('mlp', 'lstm', 'cnn', 'gru')
        data_dir: Directory containing processed data
        out_dir: Output directory for checkpoints
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        wandb_enabled: Whether to use WandB logging
        wandb_project: WandB project name
        early_stop_patience: Early stopping patience
    """
    # Setup directories
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"baseline_{model_name}_best.pth")
    last_path = os.path.join(out_dir, f"baseline_{model_name}_last.pth")
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training Baseline: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Running on: {device}")
    print(f"DATA_DIR: {data_dir}")
    print(f"OUT_DIR : {out_dir}")
    print(f"BATCH_SIZE={batch_size} EPOCHS={epochs} LR={lr}")
    
    # WandB Setup
    wandb_logger = None
    if wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=f"baseline_{model_name}",
                config={
                    "model": f"baseline_{model_name}",
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "learning_rate": lr,
                },
                tags=["baseline", model_name]
            )
            wandb_logger = wandb
            print("✅ WandB initialized")
        except ImportError:
            print("⚠️  WandB enabled but not installed. Skipping.")
    
    # 1. LOAD DATA
    print("\nLoading data...")
    try:
        train_npz = np.load(os.path.join(data_dir, "train_data.npz"))
        val_npz = np.load(os.path.join(data_dir, "val_data.npz"))
        id_map = np.load(
            os.path.join(data_dir, "id_map.npy"), allow_pickle=True
        ).item()
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    vocab_size = len(id_map)
    
    # Get sequence length from data
    seq_len = train_npz["ids"].shape[1]
    
    def get_loader(npz, shuffle=True):
        # Combine Payload (8) and Delta (1) -> 9 Features
        feats = np.concatenate([npz["payloads"], npz["deltas"]], axis=-1)
        
        dataset = TensorDataset(
            torch.LongTensor(npz["ids"]),
            torch.FloatTensor(feats),
            torch.LongTensor(npz["labels"]),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    train_loader = get_loader(train_npz, shuffle=True)
    val_loader = get_loader(val_npz, shuffle=False)
    
    # 2. INIT MODEL
    print(f"\nInitializing {model_name} model...")
    
    # Calculate Class Weights
    y_train_indices = train_npz["labels"]
    num_normal = len(y_train_indices) - np.sum(y_train_indices)
    num_attacks = np.sum(y_train_indices)
    pos_weight_val = num_normal / (num_attacks + 1e-6)
    pos_weight_val = min(pos_weight_val, 10.0)
    print(f"Calculated Class Weight (Pos_Weight): {pos_weight_val:.2f}")
    
    class_weights = torch.tensor(
        [1.0, pos_weight_val], device=device, dtype=torch.float32
    )
    
    # Get baseline model
    model = get_baseline_model(
        model_name,
        num_unique_ids=vocab_size,
        num_continuous_feats=9,
        d_model=256,
        seq_len=seq_len
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Focal Loss
    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, logits, targets):
            ce_loss = nn.functional.cross_entropy(
                logits, targets, reduction="none", weight=self.alpha
            )
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()
    
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    from torch.optim.lr_scheduler import OneCycleLR
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    
    # Resume Logic
    start_epoch = 0
    best_f1 = 0.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    
    if os.path.exists(last_path):
        print("Found last checkpoint. Resuming...")
        ckpt = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_f1 = float(ckpt.get("best_f1", 0.0))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        print(
            f"Resumed at epoch {start_epoch} with best_f1={best_f1:.4f}, threshold={best_threshold:.4f}"
        )
    
    # 3. TRAINING LOOP
    print(f"\nStarting training from epoch {start_epoch}...")
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # --- TRAINING ---
        model.train()
        train_loss_accum = 0.0
        
        for ids_batch, feats_batch, labels_batch in train_loader:
            ids_batch = ids_batch.to(device)
            feats_batch = feats_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(ids_batch, feats_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss_accum += loss.item()
        
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_probs_list = []
        val_labels_list = []
        
        with torch.no_grad():
            for ids_batch, feats_batch, labels_batch in val_loader:
                ids_batch = ids_batch.to(device)
                feats_batch = feats_batch.to(device)
                
                logits = model(ids_batch, feats_batch)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                
                val_probs_list.append(probs.cpu().numpy())
                val_labels_list.append(labels_batch.numpy())
        
        val_probs = np.concatenate(val_probs_list)
        val_labels = np.concatenate(val_labels_list)
        
        # Find optimal threshold
        thresholds = np.linspace(0.1, 0.9, 50)
        f1_scores = []
        for thresh in thresholds:
            preds = (val_probs >= thresh).astype(int)
            f1 = f1_score(val_labels, preds, zero_division=0)
            f1_scores.append(f1)
        
        best_idx = np.argmax(f1_scores)
        current_threshold = thresholds[best_idx]
        current_f1 = f1_scores[best_idx]
        
        val_preds = (val_probs >= current_threshold).astype(int)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Calculate AUC
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            # Not enough classes or samples
            val_auc = 0.0
        
        # Calculate separation metric
        normal_probs = val_probs[val_labels == 0]
        attack_probs = val_probs[val_labels == 1]
        separation = float(np.mean(attack_probs) - np.mean(normal_probs)) if len(attack_probs) > 0 else 0.0
        
        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val F1: {current_f1:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Thresh: {current_threshold:.3f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Log to WandB
        if wandb_logger:
            wandb_logger.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "train/lr": current_lr,
                "val/f1": current_f1,
                "val/accuracy": val_acc,
                "val/auc": val_auc,
                "val/threshold": current_threshold,
                "val/separation": separation,
            })
        
        # Save best model
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = current_threshold
            epochs_without_improvement = 0
            
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "best_threshold": best_threshold,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                },
                model_path
            )
            print(f"  💾 Saved new best model (F1={best_f1:.4f})")
        else:
            epochs_without_improvement += 1
        
        # Save last checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
                "best_threshold": best_threshold,
                "epochs_without_improvement": epochs_without_improvement,
            },
            last_path
        )
        
        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            print(f"\n⚠️  Early stopping triggered after {early_stop_patience} epochs without improvement")
            break
    
    print(f"\n✅ Training complete!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    
    if wandb_logger:
        wandb_logger.finish()
    
    return {
        "model_name": model_name,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "val_acc": val_acc,
        "val_auc": val_auc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Baseline Models for Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='set_01',
        help='Dataset to train on (default: set_01)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['mlp', 'lstm', 'cnn', 'gru', 'all'],
        default='all',
        help='Baseline model to train (default: all)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train on all datasets (set_01, set_02, set_03, set_04)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
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
        '--early-stop-patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable WandB logging'
    )
    
    args = parser.parse_args()
    
    # Determine datasets to train on
    if args.all:
        #datasets = ['set_01', 'set_02', 'set_03', 'set_04']
        datasets = ['set_01_run_02']
    else:
        datasets = [args.dataset]
    
    # Determine models to train
    if args.model == 'all':
        models = ['mlp', 'lstm', 'cnn', 'gru']
    else:
        models = [args.model]
    
    # Train all combinations
    results = []
    
    # Get base directories from environment or use defaults
    data_root = os.getenv('DATA_ROOT', '/workspace/data/processed_data')
    checkpoint_root = os.getenv('CHECKPOINT_ROOT', '/workspace/checkpoints')
    
    for dataset in datasets:
        data_dir = os.path.join(data_root, dataset)
        
        # Check if data exists
        if not os.path.exists(data_dir):
            print(f"⚠️  Dataset {dataset} not found at {data_dir}. Skipping...")
            continue
        
        for model_name in models:
            out_dir = os.path.join(checkpoint_root, 'baselines', dataset, model_name)
            
            result = train_baseline(
                model_name=model_name,
                data_dir=data_dir,
                out_dir=out_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                wandb_enabled=args.wandb,
                early_stop_patience=args.early_stop_patience,
            )
            
            if result:
                result['dataset'] = dataset
                results.append(result)
    
    # Print summary
    if results:
        print(f"\n{'='*80}")
        print("BASELINE TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"{'Dataset':<15} {'Model':<10} {'Val F1':<10} {'Val Acc':<10} {'Val AUC':<10}")
        print(f"{'-'*80}")
        for r in results:
            print(f"{r['dataset']:<15} {r['model_name']:<10} {r['best_f1']:<10.4f} {r.get('val_acc', 0):<10.4f} {r.get('val_auc', 0):<10.4f}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
