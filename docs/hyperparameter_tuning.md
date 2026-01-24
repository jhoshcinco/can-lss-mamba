# Hyperparameter Tuning Guide

## ⚠️ CRITICAL: Checkpoint Isolation

**When testing different hyperparameter configurations, each config MUST train from scratch with fresh model initialization.**

### Common Mistake
```bash
# ❌ WRONG - All configs share same checkpoint directory
BATCH_SIZE=32 LR=0.0001 python train.py
BATCH_SIZE=64 LR=0.0005 python train.py  # Resumes from previous config!
```

**Problem**: The second configuration resumes training from the first configuration's checkpoint instead of starting fresh. This makes hyperparameter comparisons invalid because you're comparing:
- Config 1: Fresh training for 20 epochs
- Config 2: Continued training for epochs 21-40 with different hyperparameters

### Correct Approach
```bash
# ✅ CORRECT - Each config has unique checkpoint directory
OUT_DIR=/workspace/checkpoints/bs32_lr0.0001 BATCH_SIZE=32 LR=0.0001 python train.py
OUT_DIR=/workspace/checkpoints/bs64_lr0.0005 BATCH_SIZE=64 LR=0.0005 python train.py
```

### Best Practice
**Use the provided scripts** which handle checkpoint isolation automatically:
- `scripts/grid_search.py` - Automatic unique directories per config
- `scripts/quick_test.sh` - Automatic unique directories per learning rate
- Manual training - Always specify unique `OUT_DIR` for each config

### Validation
Test checkpoint isolation is working:
```bash
python scripts/validate_hyperparameter_isolation.py
```

---

## Overview

This guide explains how to systematically tune hyperparameters for CAN-LSS-Mamba using **validation metrics only** to avoid data leakage.

## Why Hyperparameter Tuning Matters

Hyperparameters significantly impact model performance:
- **Learning Rate**: Too high → unstable training; too low → slow convergence
- **Batch Size**: Affects gradient noise and memory usage
- **Epochs**: Too few → underfitting; too many → overfitting
- **Early Stop Patience**: Controls when to stop training

## Three Approaches

### 1. Quick Test (Fastest)

**When to use**: Quick feedback on hyperparameter sensitivity

**Time**: ~1 hour (3 experiments)

```bash
bash scripts/quick_test.sh
```

Tests 3 learning rates with fixed settings:
- LR: 0.0001, 0.0005, 0.001
- Batch size: 32
- Epochs: 20

**Output**: Identifies which learning rate range works best.

### 2. Grid Search (Systematic)

**When to use**: Thorough hyperparameter optimization

**Time**: ~6-27 hours (depending on grid size)

```bash
# Default grid (27 combinations)
python scripts/grid_search.py --dataset set_01

# Custom grid (9 combinations)
python scripts/grid_search.py \
  --batch-sizes 32,64 \
  --learning-rates 0.0001,0.0005,0.001 \
  --epochs 20,30,50
```

**Default grid**:
- Batch sizes: 32, 64, 128
- Learning rates: 0.0001, 0.0005, 0.001
- Epochs: 20, 30, 50
- Early stop patience: 5, 10, 15

**Total combinations**: 3 × 3 × 3 × 3 = 81 (default) or custom

**Output**:
- CSV with all results
- Best configuration based on validation F1
- Comparison table showing top 10 configs

### 3. WandB Sweep (Bayesian Optimization)

**When to use**: Smart hyperparameter search with limited budget

**Time**: Configurable (default: 50 runs max)

```bash
# Initialize sweep
wandb sweep configs/sweep.yaml

# Run agent
wandb agent <sweep-id>
```

**Advantage**: Bayesian optimization is smarter than grid search, exploring promising regions more.

**Parameters optimized**:
- Learning rate (log-uniform: 0.00001 - 0.001)
- Batch size (32, 64, 128)
- Epochs (20, 30, 50)
- Early stop patience (5, 10, 15)
- ID dropout prob (uniform: 0.0 - 0.2)

## Step-by-Step Workflow

### Step 1: Prepare Data

```bash
# Preprocess your dataset
DATASET=set_01 python preprocessing/CAN_preprocess.py
```

### Step 2: Quick Test (Optional but Recommended)

```bash
# Get quick feedback
bash scripts/quick_test.sh
```

**Look for**:
- Which learning rate gives best validation F1?
- Is the model learning at all? (validation F1 > 0.5)
- Are there any errors?

### Step 3: Grid Search

```bash
# Based on quick test results, run grid search
python scripts/grid_search.py \
  --dataset set_01 \
  --batch-sizes 32,64,128 \
  --learning-rates 0.0001,0.0005,0.001 \
  --epochs 20,30,50
```

**Monitor progress**: Check WandB dashboard for real-time results.

### Step 4: Compare Results

```bash
# View comparison table
python scripts/compare_runs.py --tag hyperparameter_search

# Export to CSV for analysis
python scripts/compare_runs.py --tag hyperparameter_search --output results.csv
```

**Look for**:
- Best configuration (highest validation F1)
- Is there a clear winner or are multiple configs similar?
- How sensitive is performance to each hyperparameter?

### Step 5: Final Evaluation

```bash
# Use best config for cross-dataset evaluation
python scripts/cross_dataset_eval.py --all \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30
```

## Understanding Results

### Validation Metrics (Bucket 2)

These are used for hyperparameter selection:
- **val/f1**: F1 score on validation set (primary metric)
- **val/accuracy**: Classification accuracy
- **val/threshold**: Optimal classification threshold
- **val/separation**: How well the model separates classes

**Goal**: Maximize validation F1 score.

### How to Interpret

```
Experiment Comparison (sorted by val_f1):
┌────────────┬────────┬──────────┬────────┬─────────┬──────────┐
│ Run        │ LR     │ Batch    │ Epochs │ Val F1  │ Val Acc  │
├────────────┼────────┼──────────┼────────┼─────────┼──────────┤
│ exp_42     │ 0.0005 │   64     │   30   │ 0.8892 ⭐│  0.9123  │
│ exp_23     │ 0.0001 │   32     │   20   │ 0.8543  │  0.8956  │
│ exp_17     │ 0.0010 │   32     │   20   │ 0.8621  │  0.9001  │
└────────────┴────────┴──────────┴────────┴─────────┴──────────┘
```

**Interpretation**:
- Best: LR=0.0005, Batch=64, Epochs=30 (Val F1=0.8892)
- Higher LR (0.001) performs worse than 0.0005
- Larger batch size (64) helps performance
- More epochs (30) better than 20

### Common Patterns

**Pattern 1: Learning Rate Too High**
```
LR=0.001 → Val F1=0.65, Loss=unstable
LR=0.0005 → Val F1=0.85, Loss=stable ✓
LR=0.0001 → Val F1=0.80, Loss=very stable
```
→ Use 0.0005 (best performance) or 0.0001 (more stable)

**Pattern 2: Underfitting**
```
Epochs=10 → Val F1=0.70
Epochs=20 → Val F1=0.82
Epochs=30 → Val F1=0.85 ✓
Epochs=50 → Val F1=0.85 (early stopping at epoch 35)
```
→ Use 30-50 epochs with early stopping

**Pattern 3: Batch Size Effect**
```
Batch=32 → Val F1=0.80, slower training
Batch=64 → Val F1=0.85 ✓, faster training
Batch=128 → Val F1=0.83, fastest but slightly worse
```
→ Use 64 (best trade-off)

## Tips and Best Practices

### 1. Start Small
Don't run a huge grid immediately. Start with:
```bash
# Quick test first
bash scripts/quick_test.sh

# Then small grid
python scripts/grid_search.py --epochs 10,20 --learning-rates 0.0001,0.0005
```

### 2. Use Early Stopping
Always use early stopping to avoid wasting time:
```bash
--early-stop-patience 10
```

### 3. Monitor WandB
Check WandB dashboard during grid search:
- Are experiments finishing quickly? → Might need more epochs
- Are all F1 scores < 0.5? → Check data preprocessing
- Is loss exploding? → Learning rate too high

### 4. Log Everything
Grid search automatically logs to WandB with tags:
- `hyperparameter_search` - All tuning experiments
- `grid_search_set_01` - Dataset-specific

### 5. Document Your Choices
Keep notes on why you chose certain hyperparameters:
```
Best Config:
- LR=0.0005: Best validation F1, stable training
- Batch=64: Good trade-off between speed and performance
- Epochs=30: Early stopping usually triggers around epoch 25
- Rationale: This config consistently achieves val_f1 > 0.88
```

## Common Mistakes

### ❌ Mistake 1: Tuning on Test Data
```bash
# WRONG: Don't do this!
python evaluate.py  # See test results
# "Hmm, F1 is low. Let me try more epochs."
python train.py --epochs 50
python evaluate.py  # "Better! Let's use this."
```

**Why wrong**: You've tuned on test data → data leakage!

**Correct approach**: Use validation F1 from `train.py` output or grid search.

### ❌ Mistake 2: Not Using Validation Splits
```python
# WRONG
train_model(all_data)  # No validation set!
evaluate_on_test()  # Only see test results
```

**Why wrong**: No way to tune hyperparameters without data leakage.

**Correct approach**: Always use train/val/test split (three buckets).

### ❌ Mistake 3: Random Search Without Analysis
```bash
# Trying random configs without learning from results
python train.py --lr 0.0001
python train.py --lr 0.005  # Why this jump?
python train.py --lr 0.00003  # Random exploration
```

**Why wrong**: Inefficient, no systematic understanding.

**Correct approach**: Use grid search or Bayesian optimization.

## Summary

```
┌────────────────────────────────────────────────────────────┐
│           Hyperparameter Tuning Workflow                   │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Quick Test (1 hour)                                    │
│     └─> Find promising learning rate range                 │
│                                                             │
│  2. Grid Search (6-27 hours)                               │
│     └─> Systematically test combinations                   │
│                                                             │
│  3. Compare Results                                        │
│     └─> Select best config based on VALIDATION F1          │
│                                                             │
│  4. Final Evaluation (once!)                               │
│     └─> Test on held-out data for thesis/paper            │
│                                                             │
└────────────────────────────────────────────────────────────┘

Key Rule: Tune on validation (Bucket 2), evaluate on test (Bucket 3)
```

## Further Reading

- [Three-Bucket Strategy](three_bucket_strategy.md) - Avoiding data leakage
- [Cross-Dataset Evaluation](cross_dataset_evaluation.md) - Testing generalization
- WandB Hyperparameter Tuning Guide: https://docs.wandb.ai/guides/sweeps
