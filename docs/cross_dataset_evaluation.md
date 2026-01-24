# Cross-Dataset Evaluation Guide

## Overview

Cross-dataset evaluation tests whether your model truly generalizes or just memorizes specific dataset characteristics. This is critical for ML research validity.

## Why Cross-Dataset Evaluation?

### The Problem
Training and testing on the same dataset can lead to **dataset-specific overfitting**:
- Model memorizes quirks of specific vehicles
- Model learns dataset artifacts, not general patterns
- Performance drops dramatically on new vehicles/datasets

### The Solution
**Cross-dataset evaluation**: Train on one dataset, test on completely different datasets.

```
Train on set_01 (Vehicles 1-2) → Test on set_03 (Vehicles 3-4)
Train on set_03 (Vehicles 3-4) → Test on set_01 (Vehicles 1-2)
```

This reveals:
- ✅ True generalization capability
- ✅ Which features are universal vs dataset-specific
- ✅ Whether the model learns real attack patterns or just memorizes

## Dataset Structure

```
data/can-train-and-test-v1.5/
├── set_01/
│   ├── train_02_with_attacks/  (Vehicles 1-2)
│   └── test_*/ folders          (Vehicles 3-4)
├── set_02/
│   ├── train_02_with_attacks/  (Vehicles 1-2, different attacks)
│   └── test_*/ folders          (Vehicles 3-4, different attacks)
├── set_03/
│   ├── train_02_with_attacks/  (Vehicles 3-4)
│   └── test_*/ folders          (Vehicles 1-2, reversed)
├── set_04/
│   ├── train_02_with_attacks/  (Vehicles 3-4)
│   └── test_*/ folders          (Vehicles 1-2, reversed)
```

**Key insight**: 
- set_01/set_02: Train on vehicles 1-2, test on vehicles 3-4
- set_03/set_04: Train on vehicles 3-4, test on vehicles 1-2
- Perfect for testing vehicle-to-vehicle generalization!

## Usage

### Option 1: Train on One Dataset, Test on All Others

```bash
# Train on set_01, evaluate on set_02, set_03, set_04
python scripts/cross_dataset_eval.py --train-dataset set_01

# Train on set_03 (vehicles 3-4), test on all others
python scripts/cross_dataset_eval.py --train-dataset set_03
```

**Use case**: Quick check of how well set_01 model generalizes.

**Time**: ~2-4 hours per training dataset

### Option 2: Full Cross-Dataset Matrix

```bash
# Train on ALL datasets, test on ALL datasets
python scripts/cross_dataset_eval.py --all

# With custom hyperparameters (from grid search)
python scripts/cross_dataset_eval.py --all \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30
```

**Use case**: Complete cross-dataset analysis for thesis/paper.

**Time**: ~8-16 hours (4 training datasets × 2-4 hours each)

**Output**: Full cross-dataset performance matrix

### Option 3: Combined Training

```bash
# Train on ALL datasets combined
python scripts/train_combined.py \
  --datasets set_01,set_02,set_03,set_04 \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30
```

**Use case**: Maximum performance by training on all available data.

**Time**: ~4-6 hours (larger dataset, longer training)

## Understanding Results

### Cross-Dataset Performance Matrix

Example output from `--all`:

```
Cross-Dataset Performance Matrix (F1 Score):
┌────────────┬─────────┬─────────┬─────────┬─────────┐
│ Train ↓    │ set_01  │ set_02  │ set_03  │ set_04  │
│ Test →     │         │         │         │         │
├────────────┼─────────┼─────────┼─────────┼─────────┤
│ set_01     │  0.8892 │  0.8654 │  0.7123 │  0.7089 │
│ set_02     │  0.8756 │  0.8934 │  0.7234 │  0.7156 │
│ set_03     │  0.6823 │  0.6712 │  0.8567 │  0.8623 │
│ set_04     │  0.6934 │  0.6845 │  0.8645 │  0.8712 │
└────────────┴─────────┴─────────┴─────────┴─────────┘
```

### Interpreting the Matrix

**Diagonal (Same-Dataset Performance)**:
- set_01→set_01: 0.8892
- set_02→set_02: 0.8934
- set_03→set_03: 0.8567
- set_04→set_04: 0.8712

→ In-dataset performance is high (~0.85-0.89)

**Off-Diagonal (Cross-Dataset Performance)**:
- set_01→set_03: 0.7123 (drop of ~0.18)
- set_03→set_01: 0.6823 (drop of ~0.19)

→ Cross-dataset performance drops significantly!

**Key Questions**:

1. **How much does performance drop?**
   - Small drop (<0.05): ✅ Good generalization
   - Medium drop (0.10-0.20): ⚠️ Some overfitting
   - Large drop (>0.20): ❌ Poor generalization

2. **Which direction generalizes better?**
   - set_01→set_03 vs set_03→set_01
   - May reveal which vehicles are "easier" to learn from

3. **Are there vehicle clusters?**
   - set_01↔set_02 (same vehicles): High performance
   - set_03↔set_04 (same vehicles): High performance
   - set_01→set_03 (different vehicles): Lower performance

### Example Analysis

```
┌────────────────────────────────────────────────────────┐
│  Analysis of Cross-Dataset Results                     │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Within-Vehicle-Set Performance:                       │
│  • set_01 → set_02: 0.8654 (high)                     │
│  • set_03 → set_04: 0.8645 (high)                     │
│  → Model generalizes well within same vehicle set     │
│                                                         │
│  Cross-Vehicle-Set Performance:                        │
│  • set_01 → set_03: 0.7123 (medium)                   │
│  • set_03 → set_01: 0.6823 (medium)                   │
│  → Performance drops when switching vehicle sets       │
│                                                         │
│  Conclusion:                                            │
│  Model learns some vehicle-specific patterns but       │
│  also captures general attack characteristics          │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Common Patterns

### Pattern 1: Good Generalization
```
Train    Test     F1
set_01 → set_01: 0.88
set_01 → set_02: 0.86  (drop: 0.02) ✓
set_01 → set_03: 0.84  (drop: 0.04) ✓
```
→ Model learned general features, not dataset-specific quirks

### Pattern 2: Poor Generalization (Overfitting)
```
Train    Test     F1
set_01 → set_01: 0.92
set_01 → set_02: 0.75  (drop: 0.17) ⚠️
set_01 → set_03: 0.58  (drop: 0.34) ❌
```
→ Model memorized set_01 specifics, doesn't generalize

### Pattern 3: Dataset Difficulty Asymmetry
```
Train    Test     F1
set_01 → set_03: 0.75
set_03 → set_01: 0.68
```
→ set_03 is harder to train on OR set_01 is harder to test on

### Pattern 4: Combined Model Benefit
```
Model              Avg F1
Single dataset:    0.75
Combined (all):    0.82  (+0.07) ✓
```
→ Training on more diverse data improves generalization

## Best Practices

### 1. Always Report Cross-Dataset Results

**In your thesis/paper**:
```
Results:
- In-dataset F1: 0.89 (set_01 → set_01)
- Cross-dataset F1: 0.72 (set_01 → set_03)
- Generalization gap: 0.17

Conclusion: Model shows some overfitting to training vehicles
but still detects attacks on unseen vehicles with F1=0.72.
```

### 2. Use Best Hyperparameters from Validation

```bash
# Step 1: Find best hyperparameters on set_01 validation
python scripts/grid_search.py --dataset set_01

# Step 2: Use those hyperparameters for cross-dataset eval
python scripts/cross_dataset_eval.py --all \
  --batch-size 64 --lr 0.0005 --epochs 30
```

**Why**: Hyperparameters tuned on one dataset's validation may not be optimal for others, but it's the only unbiased approach.

### 3. Analyze Failure Cases

```bash
# After cross-dataset eval, examine specific scenarios
python evaluate.py  # Set environment variables for specific train/test combo
```

Look for:
- Which test scenarios fail most?
- Are certain attack types not generalizing?
- Do specific CAN IDs cause problems?

### 4. Consider Combined Training

If cross-dataset results are poor:
```bash
# Train on all datasets together
python scripts/train_combined.py --datasets set_01,set_02,set_03,set_04
```

This can improve generalization but:
- ✅ Better cross-dataset performance
- ⚠️ May reduce in-dataset performance slightly
- ⚠️ Requires more training time

## Complete Workflow

### Step 1: Hyperparameter Tuning (Validation)
```bash
# Tune on set_01 using validation metrics (Bucket 2)
python scripts/grid_search.py --dataset set_01
```

### Step 2: Cross-Dataset Evaluation (Test)
```bash
# Use best hyperparameters for full cross-dataset eval
python scripts/cross_dataset_eval.py --all \
  --batch-size 64 --lr 0.0005 --epochs 30
```

### Step 3: Compare Results
```bash
# Export results for analysis
python scripts/compare_runs.py --tag cross_dataset_eval --output thesis_results.csv
```

### Step 4: (Optional) Combined Training
```bash
# If cross-dataset results are poor, try combined training
python scripts/train_combined.py \
  --datasets set_01,set_02,set_03,set_04 \
  --batch-size 64 --lr 0.0005 --epochs 30
```

### Step 5: Report Results
```
Table 1: Cross-Dataset Evaluation Results

Train    Test     F1 Score  Precision  Recall   Notes
────────────────────────────────────────────────────────
set_01 → set_01   0.8892    0.8945     0.8840   Baseline
set_01 → set_02   0.8654    0.8723     0.8586   Same vehicles
set_01 → set_03   0.7123    0.7456     0.6812   Different vehicles
set_01 → set_04   0.7089    0.7398     0.6801   Different vehicles
────────────────────────────────────────────────────────
Combined → set_01 0.8756    0.8812     0.8701   +0.03 vs. single
Combined → set_03 0.7845    0.8123     0.7589   +0.07 vs. single
```

## Troubleshooting

### Issue 1: Very Low Cross-Dataset Performance (<0.50)

**Possible causes**:
1. ID vocabulary mismatch between datasets
2. Different data distributions (check attack rates)
3. Model is completely overfitting to training data

**Solutions**:
- Use ID dropout during training (`--id-dropout-prob 0.1`)
- Check data preprocessing consistency
- Try combined training on multiple datasets

### Issue 2: Symmetric Performance Drop

```
set_01 → set_03: 0.65
set_03 → set_01: 0.63
```

**Interpretation**: Both datasets are equally hard to generalize from/to.

**Likely cause**: Genuinely different vehicle characteristics.

**Solution**: Combined training or transfer learning.

### Issue 3: Asymmetric Performance Drop

```
set_01 → set_03: 0.75  (good)
set_03 → set_01: 0.58  (poor)
```

**Interpretation**: set_01 is easier to test on OR set_03 is harder to train on.

**Investigation**:
1. Check data quality of both datasets
2. Compare attack type distributions
3. Analyze CAN ID coverage

## Summary

```
┌─────────────────────────────────────────────────────────┐
│       Cross-Dataset Evaluation Workflow                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Tune hyperparameters on single dataset              │
│     (using validation metrics)                          │
│                                                          │
│  2. Run cross-dataset evaluation                        │
│     python scripts/cross_dataset_eval.py --all          │
│                                                          │
│  3. Analyze generalization gap                          │
│     In-dataset F1 vs Cross-dataset F1                   │
│                                                          │
│  4. (Optional) Combined training if needed              │
│     python scripts/train_combined.py                    │
│                                                          │
│  5. Report both in-dataset and cross-dataset results    │
│                                                          │
└─────────────────────────────────────────────────────────┘

Key Metric: Generalization Gap = In-Dataset F1 - Cross-Dataset F1
Goal: Minimize this gap (< 0.10 is excellent)
```

## Further Reading

- [Three-Bucket Strategy](three_bucket_strategy.md) - Avoiding data leakage
- [Hyperparameter Tuning](hyperparameter_tuning.md) - Finding best config
- Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. CVPR 2011.
