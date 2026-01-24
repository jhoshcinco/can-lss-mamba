# Testing Guide for 'na' Value Fix

## Quick Test Commands

### Basic Test (Set 02 - with 'na' values)
```bash
DATASET=set_02 python preprocessing/CAN_preprocess.py
```

### With Skip Invalid Rows
```bash
DATASET=set_02 python preprocessing/CAN_preprocess.py --skip-invalid-rows
```

### Check Data Quality Report
```bash
cat /workspace/data/processed_data/set_02_run_02/data_quality_report.json
```

### View Logs
The script will log:
- Warnings about 'na' values found
- Data quality statistics per file
- Total rows processed
- Invalid field counts

## Expected Output

### Success Indicators

1. **No Errors**: Script should complete without errors like:
   ```
   Error: invalid literal for int() with base 16: 'na'
   ```

2. **Data Quality Warnings**: You should see warnings like:
   ```
   WARNING: fuzzing-1.csv: Found 'na' values in columns: arbitration_id(45), data_field(12)
   INFO: Data quality report for fuzzing-1.csv:
     Total rows: 1000
     Valid rows: 943
     Rows with 'na' values: 57
     Invalid CAN IDs: 45
     Invalid data bytes: 12
   ```

3. **Files Created**:
   ```
   /workspace/data/processed_data/set_02_run_02/
   ├── train_data.npz          ✓
   ├── val_data.npz            ✓
   ├── id_map.npy              ✓
   └── data_quality_report.json ✓
   ```

4. **Data Quality Report**: Should contain:
   ```json
   {
     "dataset": "set_02",
     "total_files": 4,
     "preprocessing_completed": true,
     "skip_invalid_rows": false
   }
   ```

## Test Other Datasets

Verify backwards compatibility with datasets that don't have 'na' values:

```bash
# Set 01
DATASET=set_01 python preprocessing/CAN_preprocess.py

# Set 03
DATASET=set_03 python preprocessing/CAN_preprocess.py

# Set 04
DATASET=set_04 python preprocessing/CAN_preprocess.py
```

All should complete successfully with no errors.

## Verify Data Integrity

After preprocessing, verify the data:

```python
import numpy as np

# Load processed data
data = np.load('/workspace/data/processed_data/set_02_run_02/train_data.npz')

# Check shapes
print(f"IDs shape: {data['ids'].shape}")
print(f"Payloads shape: {data['payloads'].shape}")
print(f"Deltas shape: {data['deltas'].shape}")
print(f"Labels shape: {data['labels'].shape}")

# Check for NaN values (should be none)
print(f"NaN in IDs: {np.isnan(data['ids']).sum()}")
print(f"NaN in payloads: {np.isnan(data['payloads']).sum()}")
print(f"NaN in deltas: {np.isnan(data['deltas']).sum()}")

# Check value ranges
print(f"ID range: {data['ids'].min()} - {data['ids'].max()}")
print(f"Payload range: {data['payloads'].min():.3f} - {data['payloads'].max():.3f}")
print(f"Label distribution: {np.bincount(data['labels'])}")
```

## Test Training Integration

After preprocessing, test that training works:

```bash
# Train on preprocessed set_02 data
DATASET=set_02 python train.py
```

Should start training without errors.

## Performance Comparison

Compare preprocessing with and without skip-invalid-rows:

```bash
# With replacement (default)
time DATASET=set_02 python preprocessing/CAN_preprocess.py

# With skipping
time DATASET=set_02 python preprocessing/CAN_preprocess.py --skip-invalid-rows
```

Compare:
1. Processing time
2. Number of windows created
3. Data quality reports

## Troubleshooting

### If you get "No CSV files found"
```bash
# Check dataset path
ls /workspace/data/can-train-and-test-v1.5/set_02/train_02_with_attacks/

# Or specify custom path
python preprocessing/CAN_preprocess.py \
  --dataset set_02 \
  --dataset-root /path/to/your/data
```

### If you get "Missing columns"
The dataset format might be different. Check the CSV structure:
```bash
head -n 2 /workspace/data/can-train-and-test-v1.5/set_02/train_02_with_attacks/fuzzing-1.csv
```

Expected columns: `Timestamp`, `Arbitration_ID`, `Data_Field`, `Attack`

### If preprocessing is too slow
Try reducing window size or increasing stride:
```bash
WINDOW_SIZE=32 STRIDE=32 DATASET=set_02 python preprocessing/CAN_preprocess.py
```

## Success Criteria

✅ set_02 preprocessing completes without errors  
✅ Logs show 'na' values were encountered and handled  
✅ Data quality report is generated  
✅ NPZ files are created with valid data  
✅ Other datasets (set_01, set_03, set_04) still work  
✅ Training can use preprocessed set_02 data  
✅ No breaking changes to existing functionality  

## Unit Tests

Run unit tests to verify logic:

```bash
# Basic structure tests
python tests/test_preprocessing_basic.py

# Comprehensive tests (requires pandas)
python tests/test_preprocessing.py
```

All tests should pass.
