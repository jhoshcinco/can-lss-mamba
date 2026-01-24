# CAN Data Preprocessing Guide

## Overview

The CAN-LSS-Mamba preprocessing pipeline converts raw CAN bus data from CSV files into windowed sequences suitable for deep learning models.

## Quick Start

```bash
# Preprocess default dataset (set_01)
python preprocessing/CAN_preprocess.py

# Preprocess specific dataset
DATASET=set_02 python preprocessing/CAN_preprocess.py

# With custom paths
python preprocessing/CAN_preprocess.py \
  --dataset set_02 \
  --dataset-root /path/to/raw/data \
  --output-dir /path/to/processed/data
```

## Data Format

### Input CSV Format

The preprocessing script expects CSV files with the following columns:

- `Timestamp`: Float value representing time in seconds
- `Arbitration_ID`: Hexadecimal CAN ID (e.g., '0x123', '199')
- `Data_Field`: Hexadecimal data payload (16 characters representing 8 bytes)
- `Attack`: Binary label (0 = normal, 1 = attack)

### Example:

```csv
Timestamp,Arbitration_ID,Data_Field,Attack
1234567.890,0x123,0011223344556677,0
1234567.891,0x456,8899aabbccddeeff,0
1234567.892,na,na,1
```

## Handling Invalid Data

### Problem

Some datasets (notably set_02) contain 'na' values in CAN ID and data fields instead of valid hexadecimal values. This causes the preprocessing to fail with errors like:

```
Error parsing file.csv: invalid literal for int() with base 16: 'na'
```

### Solution

The preprocessing script now includes robust error handling for invalid values:

#### 1. Replace with Defaults (Default Behavior)

Invalid values are automatically replaced with defaults:
- Invalid CAN IDs → 0
- Invalid data bytes → 0
- Invalid timestamps → 0.0

```bash
# This is the default behavior
DATASET=set_02 python preprocessing/CAN_preprocess.py
```

#### 2. Skip Invalid Rows

Alternatively, you can skip rows with invalid data entirely:

```bash
python preprocessing/CAN_preprocess.py --skip-invalid-rows
```

## Data Quality Tracking

### Logging

The preprocessing script logs data quality issues:

```
WARNING: fuzzing-1.csv: Found 'na' values in columns: arbitration_id(45), data_field(12)
INFO: Data quality report for fuzzing-1.csv:
  Total rows: 1000
  Valid rows: 943
  Rows with 'na' values: 57
  Invalid CAN IDs: 45
  Invalid data bytes: 12
```

### Quality Report

After preprocessing, a JSON report is saved:

```bash
cat /workspace/data/processed_data/set_02_run_02/data_quality_report.json
```

Example output:
```json
{
  "dataset": "set_02",
  "total_files": 4,
  "preprocessing_completed": true,
  "skip_invalid_rows": false
}
```

## Configuration

### Environment Variables

```bash
# Dataset name
DATASET=set_02

# Raw data path
DATASET_ROOT=/workspace/data/can-train-and-test-v1.5/set_02

# Output path
OUTPUT_DIR=/workspace/data/processed_data/set_02_run_02

# Window size (number of messages per sequence)
WINDOW_SIZE=64

# Stride (overlap between windows)
STRIDE=64
```

### Command-Line Arguments

```bash
python preprocessing/CAN_preprocess.py \
  --dataset set_02 \
  --dataset-root /path/to/data \
  --output-dir /path/to/output \
  --skip-invalid-rows
```

## Output Format

### Files Generated

```
/workspace/data/processed_data/set_02_run_02/
├── train_data.npz          # Training windows
├── val_data.npz            # Validation windows
├── id_map.npy              # CAN ID vocabulary mapping
└── data_quality_report.json # Quality statistics
```

### NPZ File Structure

Each `.npz` file contains:

- `ids`: CAN ID indices [N_windows, window_size]
- `payloads`: Normalized data bytes [N_windows, window_size, 8]
- `deltas`: Log-normalized time deltas [N_windows, window_size, 1]
- `labels`: Binary attack labels [N_windows]

### Window Labeling

A window is labeled as attack (1) if ANY message in the window is an attack. Otherwise, it's labeled as normal (0).

## Advanced Usage

### Custom Window Configuration

```bash
# Larger windows with overlap
WINDOW_SIZE=128 STRIDE=64 python preprocessing/CAN_preprocess.py

# No overlap
WINDOW_SIZE=64 STRIDE=64 python preprocessing/CAN_preprocess.py

# High overlap (for more training samples)
WINDOW_SIZE=64 STRIDE=32 python preprocessing/CAN_preprocess.py
```

### Batch Preprocessing

Process multiple datasets:

```bash
for dataset in set_01 set_02 set_03 set_04; do
    echo "Processing $dataset..."
    DATASET=$dataset python preprocessing/CAN_preprocess.py
done
```

### Using with Config Files

```bash
# Use YAML configuration
CONFIG_PATH=configs/vastai.yaml python scripts/preprocess.py
```

## Data Quality Best Practices

1. **Always check the data quality report** after preprocessing
2. **Verify the vocabulary size** - too many unique IDs might indicate data issues
3. **Check for missing 'na' handling** in older preprocessing versions
4. **Compare statistics across datasets** to identify anomalies
5. **Test with both skip and replace strategies** to see which gives better model performance

## Troubleshooting

### Error: "invalid literal for int() with base 16: 'na'"

**Cause:** Old preprocessing script without 'na' handling  
**Solution:** Use the updated preprocessing script (this version)

### Error: "No CSV files found"

**Cause:** Incorrect dataset path  
**Solution:** Check `DATASET_ROOT` environment variable or `--dataset-root` argument

### Warning: "No training windows created"

**Cause:** Window size larger than available data  
**Solution:** Reduce `WINDOW_SIZE` or provide more data

### Error: "Missing columns"

**Cause:** CSV format doesn't match expected structure  
**Solution:** Verify CSV has required columns (timestamp, arbitration_id, data_field, attack)

## Technical Details

### Safe Conversion Functions

The preprocessing script uses safe conversion functions:

```python
def safe_hex_to_int(value, default=0):
    """Convert hex string to int, handling 'na' values."""
    if value in ['na', 'NA', 'nan', 'NaN', '']:
        return default
    try:
        return int(value, 16)
    except (ValueError, TypeError):
        return default
```

### Validation Logic

Before processing, each CSV is validated:

```python
def validate_dataframe(df, filename):
    """Check for required columns and 'na' values."""
    required_columns = ['timestamp', 'arbitration_id', 'data_field', 'attack']
    # Check columns and log warnings for 'na' values
```

### Data Flow

1. **Read CSV** → Load raw data
2. **Validate** → Check columns and data quality
3. **Parse IDs** → Convert hex strings, handle 'na'
4. **Parse Timestamps** → Convert to float, handle 'na'
5. **Parse Payloads** → Split hex strings, handle 'na'
6. **Parse Labels** → Convert to binary
7. **Create Windows** → Sliding window extraction
8. **Save NPZ** → Store processed data
9. **Save Report** → Data quality statistics

## Integration with Training

The preprocessed data is automatically loaded by the training script:

```python
# In train.py
train_data = np.load('train_data.npz')
X_ids = train_data['ids']
X_payloads = train_data['payloads']
X_deltas = train_data['deltas']
y = train_data['labels']
```

## Performance Considerations

- **Window Size**: Larger windows capture more context but increase memory usage
- **Stride**: Smaller strides create more training samples but increase disk usage
- **Parallel Processing**: The script uses ProcessPoolExecutor for parallel CSV parsing
- **Memory**: Large datasets may require chunked processing

## Future Improvements

- [ ] Support for additional CSV formats
- [ ] Automatic data quality visualization
- [ ] Online/streaming preprocessing for real-time detection
- [ ] Data augmentation strategies
- [ ] Multi-file parallel processing optimization

## References

- [Three Bucket Strategy](three_bucket_strategy.md) - Train/Val/Test split strategy
- [Cross-Dataset Evaluation](cross_dataset_evaluation.md) - Multi-dataset preprocessing
- Main README - General usage and setup
