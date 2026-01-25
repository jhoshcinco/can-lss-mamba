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

### Understanding 'na' Values

**Important Discovery:** Investigation shows that **'na' values appear ONLY in attack files** (fuzzing-*.csv, systematic-*.csv), not in normal traffic files.

**This means 'na' is likely part of the attack signature!**

In attack scenarios (fuzzing, systematic), 'na' often indicates:
- Malformed CAN frames (attack behavior)
- Bus flooding causing capture failures
- Intentionally corrupted messages

### Three Strategies for Handling 'na'

The preprocessing script supports three different strategies for handling 'na' values, controlled by the `--treat-na-as` argument:

#### 1. Special Token (Default, Recommended)

**Default Behavior:** Preserves 'na' as a special token (-1) that the model can learn as an attack indicator.

```bash
# Default: Treat 'na' as attack signature indicator
DATASET=set_02 python preprocessing/CAN_preprocess.py

# Explicit:
python preprocessing/CAN_preprocess.py --treat-na-as special_token
```

**Why this is better:**
- 'na' is mapped to -1, which gets vocab index 0
- Model learns that vocab[0] is strongly associated with attacks
- Normal CAN ID 0x000 gets a different vocab index, preventing conflation
- Preserves attack signature information

**Example mapping:**
```
Attack with 'na':  can_id='na' → -1 → vocab[0] = "ATTACK_INDICATOR"
Normal CAN ID 0:   can_id=0x0  → 0  → vocab[1] = "normal_id_0"
Normal CAN ID 100: can_id=0x100 → 256 → vocab[2] = "normal_id_256"
```

#### 2. Replace with Zero (Not Recommended)

**Old Behavior:** Replaces 'na' with 0, losing attack information.

```bash
python preprocessing/CAN_preprocess.py --treat-na-as zero
```

**Why this is bad:**
- Replacing 'na' with 0 destroys attack signatures
- CAN ID 0x000 is a valid ID, so we conflate attacks with normal traffic
- Model cannot learn that 'na' indicates malicious behavior

**Not recommended for attack detection tasks.**

#### 3. Skip Rows (If 'na' is Measurement Error)

**Alternative:** Remove rows with 'na' values entirely.

```bash
python preprocessing/CAN_preprocess.py --treat-na-as skip
```

**Use only if:**
- Investigation shows 'na' is measurement error, not attack
- You want to train only on clean data
- You're willing to lose some attack examples

### Data Quality Tracking

The preprocessing script tracks 'na' value statistics:

```
NA Statistics for fuzzing-1.csv:
  Total rows: 1000
  'na' in CAN ID: 45 (4.50%)
  'na' in DLC: 12 (1.20%)
  'na' in data bytes: 23 rows
  Rows with any 'na': 57
```

### Quality Report

After preprocessing, a JSON report is saved with the strategy used:

```bash
cat /workspace/data/processed_data/set_02_run_02/data_quality_report.json
```

Example output:
```json
{
  "dataset": "set_02",
  "total_files": 4,
  "preprocessing_completed": true,
  "skip_invalid_rows": false,
  "treat_na_as": "special_token"
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
  --treat-na-as special_token \
  --skip-invalid-rows
```

**Arguments:**
- `--dataset`: Dataset name (e.g., set_01, set_02)
- `--dataset-root`: Path to raw dataset directory
- `--output-dir`: Path to save processed data
- `--treat-na-as`: How to handle 'na' values (special_token, zero, skip) - default: special_token
- `--skip-invalid-rows`: Skip rows with invalid data (deprecated, use --treat-na-as skip instead)

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

The preprocessing script uses safe conversion functions with special token support:

```python
def safe_hex_to_int(value, default=0, allow_na_token=False):
    """
    Convert hex string to int, handling 'na' values.
    
    Args:
        value: String value to convert (e.g., '0x123', '123', 'na')
        default: Default value if conversion fails
        allow_na_token: If True, 'na' returns -1 (special token)
                       If False, 'na' returns default
    
    Returns:
        Integer value, -1 for 'na' (if allow_na_token=True), 
        or default for errors
    """
    if pd.isna(value):
        return -1 if allow_na_token else default
    
    value_str = str(value).strip().lower()
    
    # Handle 'na' as special token (ATTACK SIGNATURE)
    if value_str in ['na', 'nan', 'none', '']:
        return -1 if allow_na_token else default
    
    try:
        if value_str.startswith('0x'):
            return int(value_str, 16)
        else:
            return int(value_str, 16)
    except (ValueError, TypeError):
        return default
```

### Statistics Tracking

The `NAStatistics` class tracks 'na' values throughout processing:

```python
class NAStatistics:
    """Track statistics about 'na' values in the dataset."""
    def __init__(self):
        self.total_rows = 0
        self.na_in_can_id = 0
        self.na_in_dlc = 0
        self.na_in_data_bytes = 0
        self.rows_with_any_na = 0
        self.skipped_rows = 0
```

### Vocabulary Building with Special Tokens

When `--treat-na-as special_token` is used, the vocabulary reserves index 0 for the 'na' token:

```python
# If 'na' token (-1) exists in data
id_map = {}
id_map[-1] = 0  # Reserve index 0 for 'na' token

# Map other CAN IDs starting from index 1
idx = 1
for can_id in sorted(unique_ids):
    if can_id != -1:
        id_map[can_id] = idx
        idx += 1

id_map['<UNK>'] = len(id_map)  # Unknown token
```

This ensures:
- Vocab index 0 = 'na' token (attack indicator)
- Vocab index 1+ = normal CAN IDs
- Model can learn that index 0 is strongly associated with attacks

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
