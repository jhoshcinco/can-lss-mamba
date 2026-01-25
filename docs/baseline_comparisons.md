# Baseline Comparisons Guide

## Overview

This guide explains how to generate baseline comparisons for the CAN-LSS-Mamba model. Baseline comparisons are essential for demonstrating the effectiveness of the proposed model by comparing it against simpler, well-established architectures.

## Why Baseline Comparisons?

In machine learning research, it's crucial to demonstrate that your proposed model provides a meaningful improvement over simpler alternatives. This helps to:

- **Validate the complexity**: Show that the advanced architecture (LSS-CAN-Mamba with Mamba SSM, ECA attention, etc.) is justified
- **Establish benchmarks**: Provide standard baselines that other researchers can compare against
- **Identify strengths**: Understand where the proposed model excels compared to baselines
- **Support thesis/paper claims**: Provide quantitative evidence of model superiority

## Available Baseline Models

The repository includes four standard baseline models:

### 1. MLP (Multi-Layer Perceptron)
- **Description**: Simplest baseline - flattens the sequence and processes with fully connected layers
- **Strengths**: Fast training, minimal parameters
- **Weaknesses**: Cannot capture sequential patterns, limited capacity
- **Use case**: Lower bound baseline to show minimum expected performance

### 2. LSTM (Long Short-Term Memory)
- **Description**: Bidirectional LSTM for sequence processing
- **Strengths**: Good at capturing temporal dependencies, proven architecture
- **Weaknesses**: Slower than CNNs, can struggle with very long sequences
- **Use case**: Strong baseline for sequence tasks

### 3. CNN (Convolutional Neural Network)
- **Description**: 1D convolutions with batch normalization
- **Strengths**: Fast, good at capturing local patterns
- **Weaknesses**: Limited long-range context, fixed receptive field
- **Use case**: Baseline for local feature extraction

### 4. GRU (Gated Recurrent Unit)
- **Description**: Simpler variant of LSTM
- **Strengths**: Faster than LSTM, fewer parameters
- **Weaknesses**: Similar limitations to LSTM
- **Use case**: Alternative RNN baseline

## Quick Start

### Option 1: Generate Complete Comparison (Recommended)

This trains all models and generates a comprehensive comparison:

```bash
# Generate comparison for set_01
python scripts/generate_baseline_comparisons.py --dataset set_01

# Generate comparisons for all datasets
python scripts/generate_baseline_comparisons.py --all

# Export results to CSV
python scripts/generate_baseline_comparisons.py --dataset set_01 --output comparison_results.csv
```

**What it does**:
1. Trains the main LSS-CAN-Mamba model
2. Trains all baseline models (MLP, LSTM, CNN, GRU)
3. Evaluates all models on validation and test sets
4. Generates comparison tables and metrics
5. Exports results to CSV (optional)

**Output**:
```
BASELINE COMPARISON RESULTS - SET_01
================================================================================

Model           Parameters  Val F1   Val Accuracy  Test F1  Test Accuracy  Test AUC
LSS-CAN-Mamba   2,456,789   0.8892   0.9123        0.8856   0.9087         0.9456
LSTM            1,234,567   0.8543   0.8956        0.8501   0.8912         0.9234
GRU             1,156,432   0.8489   0.8901        0.8445   0.8867         0.9178
CNN               987,654   0.8321   0.8756        0.8289   0.8723         0.9045
MLP               856,321   0.7892   0.8423        0.7845   0.8389         0.8756

================================================================================
BEST MODEL: LSS-CAN-Mamba
================================================================================
Parameters: 2,456,789
Best Threshold: 0.5234

Validation Metrics:
  F1:         0.8892
  Accuracy:   0.9123
  Precision:  0.8756
  Recall:     0.9034
  AUC:        0.9456

Test Metrics:
  F1:         0.8856
  Accuracy:   0.9087
  Precision:  0.8723
  Recall:     0.8991
  AUC:        0.9456
================================================================================

Improvement over average baseline: 6.24%
```

### Option 2: Train Baselines Only

If you already have a trained LSS-CAN-Mamba model:

```bash
# Train all baselines on set_01
python scripts/train_baselines.py --dataset set_01

# Train specific baseline
python scripts/train_baselines.py --dataset set_01 --model lstm

# Train on all datasets
python scripts/train_baselines.py --all
```

### Option 3: Use Existing Models (Skip Training)

If you have already trained models and just want to generate the comparison:

```bash
python scripts/generate_baseline_comparisons.py --dataset set_01 --skip-training
```

## Customizing Training

### Hyperparameters

You can customize training hyperparameters for fair comparison:

```bash
# Use hyperparameters from grid search
python scripts/generate_baseline_comparisons.py \
  --dataset set_01 \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30
```

### Individual Baseline Training

```bash
# Train LSTM baseline with custom parameters
python scripts/train_baselines.py \
  --dataset set_01 \
  --model lstm \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30 \
  --early-stop-patience 10
```

### WandB Integration

Enable WandB logging for experiment tracking:

```bash
python scripts/train_baselines.py --dataset set_01 --wandb
```

This will log all baseline experiments to WandB with the tag `baseline`.

## Understanding Results

### Key Metrics

The comparison script reports several metrics for each model:

- **F1 Score**: Harmonic mean of precision and recall (primary metric)
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **AUC**: Area Under the ROC Curve
- **Separation**: Mean difference between attack and normal probabilities
- **Parameters**: Total number of model parameters

### Validation vs Test Metrics

- **Validation metrics**: Used during training for early stopping and threshold tuning
- **Test metrics**: Final evaluation on held-out test set (reported in papers)

⚠️ **Important**: Test metrics should only be used once, after all training and hyperparameter tuning is complete.

### Interpreting Improvements

When comparing LSS-CAN-Mamba to baselines:

1. **Look at F1 Score**: Primary metric for imbalanced classification
2. **Check consistency**: Model should perform well on both validation and test sets
3. **Consider complexity**: Is the improvement worth the increased model complexity?
4. **Analyze parameter efficiency**: Performance per parameter

Example interpretation:
```
LSS-CAN-Mamba: F1=0.8856, Parameters=2.4M
LSTM:          F1=0.8501, Parameters=1.2M

Improvement: 4.2% F1 score with 2x parameters
→ Significant improvement, complexity justified
```

## Advanced Usage

### Cross-Dataset Comparison

Compare baseline performance across different datasets:

```bash
# Generate comparisons for all datasets
python scripts/generate_baseline_comparisons.py --all --output results.csv

# Compare results
cat results_set_01.csv results_set_02.csv results_set_03.csv results_set_04.csv
```

### Checkpoint Management

All models are saved to `/workspace/checkpoints`:

```
/workspace/checkpoints/
├── main/
│   └── lss_can_mamba_best.pth         # Main model
└── baselines/
    ├── set_01/
    │   ├── mlp/
    │   │   └── baseline_mlp_best.pth
    │   ├── lstm/
    │   │   └── baseline_lstm_best.pth
    │   ├── cnn/
    │   │   └── baseline_cnn_best.pth
    │   └── gru/
    │       └── baseline_gru_best.pth
    └── set_02/
        └── ...
```

Each checkpoint contains:
- Model weights
- Optimizer state
- Best validation F1 score
- Best classification threshold
- Training epoch

### Resuming Training

If training is interrupted, it will automatically resume from the last checkpoint:

```bash
# Resume training baselines (automatically detects checkpoints)
python scripts/train_baselines.py --dataset set_01
```

## Best Practices

### 1. Use Same Hyperparameters

For fair comparison, use the same hyperparameters across all models:

```bash
# Find best hyperparameters with grid search first
python scripts/grid_search.py --dataset set_01

# Then use those hyperparameters for baselines
python scripts/generate_baseline_comparisons.py \
  --dataset set_01 \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30
```

### 2. Multiple Runs

For robust comparisons, train each model multiple times with different random seeds:

```bash
for seed in 42 123 456; do
  RANDOM_SEED=$seed python scripts/train_baselines.py --dataset set_01 --model lstm
done
```

Then average the results across runs.

### 3. Report Complete Metrics

When writing papers/thesis, report:
- Mean and standard deviation across multiple runs
- All key metrics (F1, Accuracy, Precision, Recall, AUC)
- Model complexity (parameters, inference time)
- Confusion matrix for test set

### 4. Document Choices

Keep notes on why certain baselines were chosen:

```
Baselines Selected:
- MLP: Lower bound, simplest possible model
- LSTM: State-of-the-art for sequences until Transformers
- CNN: Fast baseline for local patterns
- GRU: Simpler RNN variant

Rationale: These represent standard baselines in the intrusion detection literature.
```

## Common Issues

### Issue 1: Out of Memory

If you run out of GPU memory:

```bash
# Reduce batch size
python scripts/train_baselines.py --dataset set_01 --batch-size 16

# Or train baselines one at a time
python scripts/train_baselines.py --dataset set_01 --model mlp
python scripts/train_baselines.py --dataset set_01 --model lstm
# etc.
```

### Issue 2: Poor Baseline Performance

If baselines perform very poorly (F1 < 0.5):

1. **Check data preprocessing**: Ensure baselines use the same preprocessed data
2. **Verify class weights**: Make sure focal loss is properly configured
3. **Increase training time**: Baselines might need more epochs to converge
4. **Adjust learning rate**: Try different learning rates

### Issue 3: Baselines Outperform Main Model

If baselines outperform LSS-CAN-Mamba:

1. **Check main model training**: Did it converge properly?
2. **Verify hyperparameters**: Are you using optimal hyperparameters from grid search?
3. **Compare on test set**: Sometimes validation performance differs from test
4. **Consider model complexity**: Simpler models can work well on simpler datasets

## Integration with Existing Workflow

### Step-by-Step Research Workflow

```bash
# 1. Preprocess data
python preprocessing/CAN_preprocess.py

# 2. Hyperparameter tuning (main model only)
python scripts/grid_search.py --dataset set_01

# 3. Train main model with best hyperparameters
BATCH_SIZE=64 LR=0.0005 EPOCHS=30 python -m can_lss_mamba.train

# 4. Generate baseline comparisons
python scripts/generate_baseline_comparisons.py \
  --dataset set_01 \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30 \
  --output baseline_comparison.csv

# 5. Cross-dataset evaluation (optional)
python scripts/cross_dataset_eval.py --all

# 6. Compare all experiments
python scripts/compare_runs.py --tag baseline
```

## Example Output for Paper

Here's an example table for your paper:

| Model          | Parameters | Val F1  | Test F1 | Test Acc | Test AUC | Inference Time |
|----------------|------------|---------|---------|----------|----------|----------------|
| LSS-CAN-Mamba  | 2.4M       | 0.8892  | 0.8856  | 0.9087   | 0.9456   | 1.2s           |
| LSTM           | 1.2M       | 0.8543  | 0.8501  | 0.8912   | 0.9234   | 0.8s           |
| GRU            | 1.1M       | 0.8489  | 0.8445  | 0.8867   | 0.9178   | 0.7s           |
| CNN            | 987K       | 0.8321  | 0.8289  | 0.8723   | 0.9045   | 0.3s           |
| MLP            | 856K       | 0.7892  | 0.7845  | 0.8389   | 0.8756   | 0.2s           |

**Key findings**:
- LSS-CAN-Mamba achieves 4.2% higher F1 score than the best baseline (LSTM)
- The improvement is consistent across both validation and test sets
- Model complexity (2.4M parameters) is justified by the performance gain

## Summary

```
┌────────────────────────────────────────────────────────────┐
│         Baseline Comparison Workflow                       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Tune Main Model (grid search)                          │
│     └─> Find best hyperparameters on validation set        │
│                                                             │
│  2. Generate Baseline Comparisons                          │
│     └─> Train all baselines with same hyperparameters      │
│                                                             │
│  3. Evaluate on Test Set (once!)                           │
│     └─> Report final metrics for paper/thesis              │
│                                                             │
│  4. Document Results                                       │
│     └─> Create comparison tables and analysis              │
│                                                             │
└────────────────────────────────────────────────────────────┘

Key Command:
  python scripts/generate_baseline_comparisons.py --dataset set_01
```

## Further Reading

- [Hyperparameter Tuning Guide](hyperparameter_tuning.md) - For tuning the main model
- [Three-Bucket Strategy](three_bucket_strategy.md) - For avoiding data leakage
- [Cross-Dataset Evaluation](cross_dataset_evaluation.md) - For testing generalization
