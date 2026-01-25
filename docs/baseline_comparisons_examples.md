# Baseline Comparisons - Usage Examples

## Example 1: Quick Comparison on Single Dataset

This is the simplest and most common use case.

```bash
# Generate complete baseline comparison for set_01
python scripts/generate_baseline_comparisons.py --dataset set_01
```

**What happens:**
1. Trains LSS-CAN-Mamba on set_01
2. Trains all 4 baseline models (MLP, LSTM, CNN, GRU)
3. Evaluates all models on validation and test sets
4. Prints comparison table

**Expected output:**
```
BASELINE COMPARISON RESULTS - SET_01
================================================================================

Model           Parameters  Val F1   Val Accuracy  Test F1  Test Accuracy  Test AUC
LSS-CAN-Mamba   2,456,789   0.8892   0.9123        0.8856   0.9087         0.9456
LSTM            1,234,567   0.8543   0.8956        0.8501   0.8912         0.9234
GRU             1,156,432   0.8489   0.8901        0.8445   0.8867         0.9178
CNN               987,654   0.8321   0.8756        0.8289   0.8723         0.9045
MLP               856,321   0.7892   0.8423        0.7845   0.8389         0.8756
```

**Time:** ~2-3 hours (with early stopping)

---

## Example 2: Export Results to CSV for Paper

```bash
# Generate comparison and export to CSV
python scripts/generate_baseline_comparisons.py \
  --dataset set_01 \
  --output baseline_comparison_results.csv
```

**Output file:** `baseline_comparison_results.csv`

CSV contains all metrics for each model, which you can:
- Import into Excel/Google Sheets
- Use in LaTeX tables
- Analyze with pandas
- Include in your thesis/paper appendix

---

## Example 3: Use Best Hyperparameters from Grid Search

First, find the best hyperparameters:

```bash
# Step 1: Run grid search on main model
python scripts/grid_search.py --dataset set_01

# Check results
python scripts/compare_runs.py --tag hyperparameter_search
```

Suppose the best hyperparameters are:
- Batch size: 64
- Learning rate: 0.0005
- Epochs: 30

Then use them for baseline comparison:

```bash
# Step 2: Generate baseline comparisons with best hyperparameters
python scripts/generate_baseline_comparisons.py \
  --dataset set_01 \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30
```

This ensures a **fair comparison** - all models trained with the same hyperparameters.

---

## Example 4: Train Only Specific Baseline

If you only want to train one baseline (e.g., LSTM):

```bash
# Train only LSTM baseline
python scripts/train_baselines.py --dataset set_01 --model lstm
```

**Use case:** 
- Quick experimentation
- Re-training a single model
- Comparing against a specific baseline

---

## Example 5: Compare Across All Datasets

For comprehensive evaluation:

```bash
# Compare on all 4 datasets
python scripts/generate_baseline_comparisons.py --all
```

**What happens:**
1. Processes set_01, set_02, set_03, set_04
2. Trains all models on each dataset
3. Shows comparison for each dataset
4. Prints overall summary

**Output includes:**
```
OVERALL SUMMARY ACROSS DATASETS
================================================================================

Model           Val F1   Test F1  Test Accuracy  Test AUC
LSS-CAN-Mamba   0.8856   0.8812   0.9045         0.9423
LSTM            0.8501   0.8467   0.8901         0.9201
GRU             0.8445   0.8411   0.8856         0.9156
CNN             0.8289   0.8256   0.8712         0.9034
MLP             0.7845   0.7812   0.8378         0.8745
```

**Time:** ~8-12 hours (4 datasets × 2-3 hours each)

---

## Example 6: Skip Training (Use Existing Models)

If you've already trained models and just want to regenerate the comparison:

```bash
# Use existing model checkpoints
python scripts/generate_baseline_comparisons.py \
  --dataset set_01 \
  --skip-training
```

**Use case:**
- Models already trained
- Want to regenerate comparison table
- Experimenting with different output formats

---

## Example 7: Custom Paths for Different Environments

If you're not using the default `/workspace` directory:

```bash
# Set custom paths
export DATA_ROOT=/home/user/data
export CHECKPOINT_ROOT=/home/user/checkpoints

# Run comparison
python scripts/generate_baseline_comparisons.py --dataset set_01
```

**Use case:**
- Running on local machine (not vast.ai)
- Different directory structure
- Multiple experiments in different locations

---

## Example 8: Complete Research Workflow

Full workflow from data to results:

```bash
# Step 1: Preprocess data
python preprocessing/CAN_preprocess.py

# Step 2: Tune hyperparameters on main model
python scripts/grid_search.py --dataset set_01

# Step 3: Check best hyperparameters
python scripts/compare_runs.py --tag hyperparameter_search

# Step 4: Generate baseline comparisons with best hyperparameters
python scripts/generate_baseline_comparisons.py \
  --dataset set_01 \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30 \
  --output thesis_baseline_comparison.csv

# Step 5: Cross-dataset evaluation (optional)
python scripts/cross_dataset_eval.py --all \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30
```

This gives you:
- ✅ Baseline comparison on set_01
- ✅ Cross-dataset generalization results
- ✅ All results exported to CSV
- ✅ Ready for thesis/paper

---

## Example 9: Train All Baselines Separately

For more control, train each baseline individually:

```bash
# Train each baseline model separately
python scripts/train_baselines.py --dataset set_01 --model mlp
python scripts/train_baselines.py --dataset set_01 --model lstm
python scripts/train_baselines.py --dataset set_01 --model cnn
python scripts/train_baselines.py --dataset set_01 --model gru

# Then generate comparison (skip training)
python scripts/generate_baseline_comparisons.py \
  --dataset set_01 \
  --skip-training \
  --output comparison.csv
```

**Use case:**
- Want to monitor each model separately
- Training on different GPUs
- Need to stop/resume individual models

---

## Example 10: WandB Integration

Track all experiments in WandB:

```bash
# Enable WandB logging
python scripts/train_baselines.py --dataset set_01 --wandb
```

Then check WandB dashboard to see:
- Training curves for each baseline
- Real-time metrics comparison
- Hyperparameter configurations
- Model artifacts

All baseline experiments tagged with `baseline` tag.

---

## Tips for Paper/Thesis

### Generating Tables

After running the comparison, you get metrics for LaTeX tables:

```latex
\begin{table}[h]
\centering
\caption{Baseline Model Comparison on CAN Dataset (Set 01)}
\begin{tabular}{lcccc}
\hline
Model & Parameters & Val F1 & Test F1 & Test AUC \\
\hline
LSS-CAN-Mamba & 2.4M & 0.8892 & 0.8856 & 0.9456 \\
LSTM & 1.2M & 0.8543 & 0.8501 & 0.9234 \\
GRU & 1.1M & 0.8489 & 0.8445 & 0.9178 \\
CNN & 987K & 0.8321 & 0.8289 & 0.9045 \\
MLP & 856K & 0.7892 & 0.7845 & 0.8756 \\
\hline
\end{tabular}
\end{table}
```

### Key Points to Mention

In your paper/thesis, highlight:

1. **Improvement**: "LSS-CAN-Mamba achieves 4.2% higher F1 score than the best baseline (LSTM)"
2. **Consistency**: "Performance is consistent across validation and test sets"
3. **Complexity justification**: "The model complexity (2.4M params) is justified by the performance gain"
4. **Baseline selection**: "We compare against standard baselines (MLP, LSTM, CNN, GRU) commonly used in intrusion detection"

---

## Troubleshooting Examples

### Out of Memory Error

```bash
# Reduce batch size
python scripts/generate_baseline_comparisons.py \
  --dataset set_01 \
  --batch-size 16 \
  --eval-batch-size 64
```

### Dataset Not Found

```bash
# Check data location
ls /workspace/data/processed_data/set_01/

# If in different location, set environment variable
export DATA_ROOT=/path/to/your/data
python scripts/generate_baseline_comparisons.py --dataset set_01
```

### Model Already Exists (Want to Retrain)

```bash
# Delete existing checkpoints
rm -rf /workspace/checkpoints/baselines/set_01/

# Then run comparison again
python scripts/generate_baseline_comparisons.py --dataset set_01
```

---

For more information, see:
- [Full Documentation](baseline_comparisons.md)
- [Quick Reference](BASELINE_COMPARISONS_QUICKREF.md)
