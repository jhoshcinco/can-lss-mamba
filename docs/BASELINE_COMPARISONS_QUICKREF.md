# Baseline Comparisons - Quick Reference

## One-Line Commands

```bash
# Generate complete baseline comparison for set_01
python scripts/generate_baseline_comparisons.py --dataset set_01

# Train all baselines only
python scripts/train_baselines.py --dataset set_01

# Train specific baseline (LSTM)
python scripts/train_baselines.py --dataset set_01 --model lstm

# Compare across all datasets
python scripts/generate_baseline_comparisons.py --all

# Export results to CSV
python scripts/generate_baseline_comparisons.py --dataset set_01 --output results.csv

# Use existing models (skip training)
python scripts/generate_baseline_comparisons.py --dataset set_01 --skip-training
```

## Available Baseline Models

| Model | Description | Typical Use Case |
|-------|-------------|------------------|
| **MLP** | Multi-layer perceptron (flattens sequence) | Lower bound baseline |
| **LSTM** | Bidirectional LSTM | Strong RNN baseline |
| **CNN** | 1D Convolutional network | Fast local pattern baseline |
| **GRU** | Gated Recurrent Unit | Efficient RNN baseline |

## Typical Workflow

```bash
# 1. Preprocess data (if not done)
python preprocessing/CAN_preprocess.py

# 2. Tune main model hyperparameters
python scripts/grid_search.py --dataset set_01

# 3. Generate baseline comparisons with best hyperparameters
python scripts/generate_baseline_comparisons.py \
  --dataset set_01 \
  --batch-size 64 \
  --lr 0.0005 \
  --epochs 30 \
  --output comparison_results.csv
```

## Output Location

```
/workspace/checkpoints/
├── main/
│   └── lss_can_mamba_best.pth         # Main model
└── baselines/
    └── set_01/
        ├── mlp/baseline_mlp_best.pth
        ├── lstm/baseline_lstm_best.pth
        ├── cnn/baseline_cnn_best.pth
        └── gru/baseline_gru_best.pth
```

## Key Metrics

- **F1 Score**: Primary metric (harmonic mean of precision/recall)
- **Accuracy**: Overall classification accuracy
- **AUC**: Area under ROC curve
- **Parameters**: Model complexity

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset to use (set_01, set_02, etc.) | set_01 |
| `--batch-size` | Training batch size | 32 |
| `--lr` | Learning rate | 0.0001 |
| `--epochs` | Number of epochs | 20 |
| `--output` | CSV file for results | None |
| `--skip-training` | Use existing models | False |
| `--wandb` | Enable WandB logging | False |

## Example Output

```
BASELINE COMPARISON RESULTS - SET_01
================================================================================

Model           Parameters  Val F1   Test F1  Test Accuracy  Test AUC
LSS-CAN-Mamba   2,456,789   0.8892   0.8856   0.9087         0.9456
LSTM            1,234,567   0.8543   0.8501   0.8912         0.9234
GRU             1,156,432   0.8489   0.8445   0.8867         0.9178
CNN               987,654   0.8321   0.8289   0.8723         0.9045
MLP               856,321   0.7892   0.7845   0.8389         0.8756

Improvement over average baseline: 6.24%
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of GPU memory | Reduce `--batch-size` to 16 or 8 |
| Checkpoint not found | Run training first without `--skip-training` |
| Poor baseline performance | Increase `--epochs` or adjust `--lr` |
| Missing data | Run preprocessing first |

## See Full Documentation

For detailed information, see: [docs/baseline_comparisons.md](baseline_comparisons.md)
