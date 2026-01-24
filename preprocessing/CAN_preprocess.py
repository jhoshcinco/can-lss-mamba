import pandas as pd
import numpy as np
import os
import glob
import logging
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Supports both hardcoded paths (backwards compatible) and environment variable override

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='CAN Data Preprocessing with NA value handling')
    parser.add_argument('--skip-invalid-rows', action='store_true',
                        help='Skip rows with invalid data instead of replacing with defaults')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (e.g., set_01, set_02)')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Path to raw dataset directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Path to save processed data')
    return parser.parse_args()


def get_config():
    """Get configuration from environment variables and command-line args."""
    # Get dataset name from environment variable (e.g., set_01, set_02, etc.)
    dataset = os.environ.get("DATASET", "set_01")
    
    # DATASET_ROOT: Path to raw dataset
    # Can be overridden with DATASET_ROOT environment variable
    dataset_root = os.environ.get(
        "DATASET_ROOT",
        f"/workspace/data/can-train-and-test-v1.5/{dataset}"
    )
    
    # OUTPUT_DIR: Path to save processed data
    # Can be overridden with OUTPUT_DIR environment variable
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        f"/workspace/data/processed_data/{dataset}_run_02"
    )
    
    window_size = int(os.environ.get("WINDOW_SIZE", 64))
    stride = int(os.environ.get("STRIDE", 64))
    
    return dataset, dataset_root, output_dir, window_size, stride


# Initialize with default config (can be overridden in main)
DATASET, DATASET_ROOT, OUTPUT_DIR, WINDOW_SIZE, STRIDE = get_config()

print(f"\n{'='*60}")
print(f"CAN Preprocessing Configuration")
print(f"{'='*60}")
print(f"Dataset: {DATASET}")
print(f"Dataset Root: {DATASET_ROOT}")
print(f"Output Dir: {OUTPUT_DIR}")
print(f"Window Size: {WINDOW_SIZE}")
print(f"Stride: {STRIDE}")
print(f"{'='*60}\n")

# --- HELPER FUNCTIONS ---

def is_na_value(value):
    """
    Check if a value is considered 'na' or invalid.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is 'na', False otherwise
    """
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.lower() in ['na', 'nan', '']:
        return True
    return False


def safe_hex_to_int(value, default=0):
    """
    Safely convert hex string to integer, handling invalid values.
    
    Args:
        value: String value to convert (e.g., '0x123', '123', 'na')
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    if is_na_value(value):
        return default
    
    try:
        # Handle both '0x123' and '123' formats
        if isinstance(value, str):
            value = value.strip().lower()
            if value.startswith('0x'):
                return int(value, 16)
            else:
                return int(value, 16)
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """
    Safely convert to float, handling invalid values.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    if is_na_value(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def split_payload(hex_str):
    """
    Split hex payload string into 8 bytes.
    
    Args:
        hex_str: Hexadecimal payload string (e.g., '0011223344556677')
        
    Returns:
        List of 8 integers representing bytes
    """
    hex_str = str(hex_str).strip().lower()
    
    # Handle 'na' values
    if hex_str in ['na', 'nan', '']:
        return [0] * 8  # Return default values
    
    # Ensure it is exactly 16 chars (8 bytes); pad with zeros if shorter
    hex_str = hex_str.ljust(16, '0')
    
    # Split into 8 integers with safe conversion
    result = []
    for i in range(0, 16, 2):
        byte_val = safe_hex_to_int(hex_str[i:i + 2], default=0)
        result.append(byte_val)
    return result


def validate_dataframe(df, filename):
    """Validate DataFrame has expected columns and data quality."""
    required_columns = ['timestamp', 'arbitration_id', 'data_field', 'attack']
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.warning(f"{filename}: Missing columns: {missing_cols}")
        return False
    
    # Check for 'na' values
    na_columns = []
    for col in required_columns:
        if col in df.columns:
            na_count = df[col].isin(['na', 'NA', 'nan', 'NaN']).sum()
            if na_count > 0:
                na_columns.append(f"{col}({na_count})")
    
    if na_columns:
        logger.warning(f"{filename}: Found 'na' values in columns: {', '.join(na_columns)}")
        logger.warning(f"  These will be replaced with default values (0)")
    
    return True


def parse_csv(file_path, id_map=None, skip_invalid_rows=False):
    """
    Parses CSV, extracts IDs, Payloads, and Time Deltas.
    Returns: ids, payloads, delta_feat, labels
    """
    # Initialize statistics
    stats = {
        'total_rows': 0,
        'invalid_can_id': 0,
        'invalid_timestamp': 0,
        'invalid_data_bytes': 0,
        'rows_with_na': 0,
        'valid_rows': 0,
        'skipped_rows': 0
    }
    
    try:
        # 1. Read Data
        df = pd.read_csv(file_path)
        stats['total_rows'] = len(df)

        # Standardize headers (strip spaces, lowercase) just in case
        df.columns = df.columns.str.strip().str.lower()

        # Validate DataFrame
        filename = os.path.basename(file_path)
        if not validate_dataframe(df, filename):
            logger.error(f"Validation failed for {filename}, attempting to continue with available columns")

        # Column Names (Adjusted to your screenshot)
        id_col = 'arbitration_id'
        label_col = 'attack'
        time_col = 'timestamp'
        data_col = 'data_field'

        # 2. Process Time Delta (CRITICAL FOR MAMBA)
        # Convert timestamp to float with safe conversion
        df[time_col] = df[time_col].apply(lambda x: safe_float(x, default=0.0))
        stats['invalid_timestamp'] = (df[time_col] == 0.0).sum()

        # Calculate Delta (Time since last message)
        delta = df[time_col].diff().fillna(0).values

        # Log-Normalize to squash spikes (microseconds vs seconds)
        # log1p(x) = log(x + 1) -> Handles 0 safely
        delta_norm = np.log1p(delta + 1e-6)

        # Reshape for concatenation [N, 1]
        delta_feat = delta_norm.reshape(-1, 1)

        # 3. Process IDs with safe conversion
        # Handle Hex Strings (e.g., '199' or '0x199') and 'na' values
        def safe_parse_id(x):
            if isinstance(x, str):
                # Check if it's a hex string (contains letters)
                if any(c.isalpha() for c in str(x).lower().replace('x', '')):
                    return safe_hex_to_int(x, default=0)
                else:
                    # Try to parse as integer
                    try:
                        return int(x)
                    except (ValueError, TypeError):
                        return safe_hex_to_int(x, default=0)
            else:
                try:
                    return int(x)
                except (ValueError, TypeError):
                    return 0
        
        df[id_col] = df[id_col].apply(safe_parse_id)
        stats['invalid_can_id'] = (df[id_col] == 0).sum()

        if id_map:
            # Map IDs to learned indices (Training Phase)
            df['id_idx'] = df[id_col].apply(lambda x: id_map.get(x, id_map.get('<UNK>')))
        else:
            # Return unique IDs for Vocabulary Building (filter out 0s if they're invalid)
            unique_ids = df[id_col].unique()
            # Optionally filter out invalid IDs (0) if skip_invalid_rows is True
            if skip_invalid_rows:
                unique_ids = unique_ids[unique_ids != 0]
            return unique_ids

        # 4. Process Payload (Split Hex String -> 8 Bytes)
        payload_list = df[data_col].apply(split_payload).tolist()
        
        # Count invalid data bytes
        for payload in payload_list:
            if payload == [0] * 8:
                stats['invalid_data_bytes'] += 1
        
        payloads = np.array(payload_list) / 255.0  # Normalize bytes to [0, 1]

        # 5. Process Labels
        # Ensure labels are integers (0 or 1)
        # If your CSV uses 'T' for attack, convert it. If it's already 0/1, this is safe.
        labels = df[label_col].apply(lambda x: 1 if str(x).upper() in ['1', 'T', 'ATTACK'] else 0).values

        # 6. Calculate statistics
        # Count unique rows with at least one invalid field
        has_invalid_id = (df[id_col] == 0)
        has_invalid_time = (df[time_col] == 0.0)
        # Note: Invalid data bytes are tracked globally, not per-row due to implementation constraints
        stats['rows_with_na'] = (has_invalid_id | has_invalid_time).sum()
        stats['valid_rows'] = stats['total_rows'] - stats['rows_with_na']
        
        # 7. Filter out invalid rows if requested
        if skip_invalid_rows:
            valid_mask = (df[id_col] != 0) & (df[time_col] != 0.0)
            if not valid_mask.all():
                df = df[valid_mask].reset_index(drop=True)
                stats['skipped_rows'] = stats['total_rows'] - len(df)
                logger.info(f"Skipped {stats['skipped_rows']} invalid rows from {filename}")
                
                # Recalculate features for valid rows only
                delta = df[time_col].diff().fillna(0).values
                delta_norm = np.log1p(delta + 1e-6)
                delta_feat = delta_norm.reshape(-1, 1)
                
                df['id_idx'] = df[id_col].apply(lambda x: id_map.get(x, id_map.get('<UNK>')))
                payload_list = df[data_col].apply(split_payload).tolist()
                payloads = np.array(payload_list) / 255.0
                labels = df[label_col].apply(lambda x: 1 if str(x).upper() in ['1', 'T', 'ATTACK'] else 0).values
        
        # 8. Log data quality report
        if stats['rows_with_na'] > 0:
            logger.info(f"Data quality report for {filename}:")
            logger.info(f"  Total rows: {stats['total_rows']}")
            logger.info(f"  Valid rows: {stats['valid_rows']}")
            logger.info(f"  Rows with 'na' values: {stats['rows_with_na']}")
            if stats['invalid_can_id'] > 0:
                logger.info(f"  Invalid CAN IDs: {stats['invalid_can_id']}")
            if stats['invalid_timestamp'] > 0:
                logger.info(f"  Invalid timestamps: {stats['invalid_timestamp']}")
            if stats['invalid_data_bytes'] > 0:
                logger.info(f"  Invalid data bytes: {stats['invalid_data_bytes']}")
            if stats['skipped_rows'] > 0:
                logger.info(f"  Skipped rows: {stats['skipped_rows']}")

        return df['id_idx'].values, payloads, delta_feat, labels

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        logger.error(f"  This file may contain 'na' values or other data quality issues")
        logger.error(f"  Try running with --skip-invalid-rows flag")
        return None


def create_windows(ids, payloads, deltas, labels):
    """Slices arrays into sliding windows."""
    # Calculate number of valid windows
    num_windows = (len(ids) - WINDOW_SIZE) // STRIDE

    X_ids = []
    X_payloads = []
    X_deltas = []
    y = []

    for i in range(0, len(ids) - WINDOW_SIZE, STRIDE):
        # Slice Window
        w_id = ids[i: i + WINDOW_SIZE]
        w_pay = payloads[i: i + WINDOW_SIZE]  # Shape: [64, 8]
        w_time = deltas[i: i + WINDOW_SIZE]  # Shape: [64, 1]
        w_lbl = labels[i: i + WINDOW_SIZE]

        # Label Logic: If ANY packet in window is attack, window = 1
        label = 1 if np.any(w_lbl == 1) else 0

        X_ids.append(w_id)
        X_payloads.append(w_pay)
        X_deltas.append(w_time)
        y.append(label)

    return np.array(X_ids), np.array(X_payloads), np.array(X_deltas), np.array(y)


# --- MAIN PIPELINE ---

def run_pipeline(skip_invalid_rows=False, dataset=None, dataset_root=None, output_dir=None):
    # Use provided parameters or fall back to globals
    dataset = dataset or DATASET
    dataset_root = dataset_root or DATASET_ROOT
    output_dir = output_dir or OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)

    # Overall statistics
    overall_stats = {
        'dataset': dataset,
        'total_files': 0,
        'total_rows': 0,
        'valid_rows': 0,
        'invalid_rows': 0,
        'skipped_rows': 0
    }

    # 1. IDENTIFY FILES
    train_folder = os.path.join(dataset_root, "train_02_with_attacks")

    # Check if folder exists
    if not os.path.exists(train_folder):
        logger.error(f"Folder not found: {train_folder}")
        return

    all_files = sorted(glob.glob(os.path.join(train_folder, "*.csv")))

    if not all_files:
        logger.error(f"No CSV files found in {train_folder}")
        return

    overall_stats['total_files'] = len(all_files)

    # CHRONOLOGICAL SPLIT (Three Bucket Rule)
    # Bucket #1 (80%): Training data → learn model parameters
    # Bucket #2 (20%): Validation data → model selection + threshold tuning
    # Bucket #3 (separate test_* folders): Final evaluation (handled by evaluate.py)
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    logger.info(f"Found {len(all_files)} files. Train: {len(train_files)}, Val: {len(val_files)}")

    # 2. BUILD VOCABULARY (IDs)
    logger.info("Building ID Map...")
    unique_ids = set()
    for f in train_files:
        try:
            current_ids = parse_csv(f, skip_invalid_rows=skip_invalid_rows)  # Returns unique IDs list
            if current_ids is not None:
                unique_ids.update(current_ids)
        except Exception as e:
            logger.warning(f"Skipping {f} during vocab build: {e}")

    # Create Mapping
    id_map = {can_id: i for i, can_id in enumerate(sorted(unique_ids))}
    id_map['<UNK>'] = len(id_map)  # Handle unknown IDs
    logger.info(f"Vocab Size: {len(id_map)} IDs (saved to {output_dir}/id_map.npy)")

    # Save ID Map
    np.save(os.path.join(output_dir, "id_map.npy"), id_map)
    # to check if UNK is in the preprocessed file
    assert "<UNK>" in id_map, "ERROR: <UNK> not found in id_map"
    logger.info(f"UNK index: {id_map['<UNK>']}")

    # 3. PROCESS TRAIN DATA
    logger.info("Processing Training Data...")
    X_tr_ids, X_tr_pay, X_tr_time, y_tr = [], [], [], []

    for f in train_files:
        res = parse_csv(f, id_map, skip_invalid_rows=skip_invalid_rows)
        if res:
            ids, pay, time, lbl = res
            w_ids, w_pay, w_time, w_lbl = create_windows(ids, pay, time, lbl)

            if len(w_ids) > 0:
                X_tr_ids.append(w_ids)
                X_tr_pay.append(w_pay)
                X_tr_time.append(w_time)
                y_tr.append(w_lbl)

    # Concatenate and Save
    if X_tr_ids:
        np.savez(os.path.join(output_dir, "train_data.npz"),
                 ids=np.concatenate(X_tr_ids),
                 payloads=np.concatenate(X_tr_pay),
                 deltas=np.concatenate(X_tr_time),
                 labels=np.concatenate(y_tr))
        logger.info("Training Data Saved.")
    else:
        logger.warning("No training windows created (check window size vs file length)")

    # 4. PROCESS VAL DATA
    logger.info("Processing Validation Data...")
    X_val_ids, X_val_pay, X_val_time, y_val = [], [], [], []

    for f in val_files:
        res = parse_csv(f, id_map, skip_invalid_rows=skip_invalid_rows)
        if res:
            ids, pay, time, lbl = res
            w_ids, w_pay, w_time, w_lbl = create_windows(ids, pay, time, lbl)

            if len(w_ids) > 0:
                X_val_ids.append(w_ids)
                X_val_pay.append(w_pay)
                X_val_time.append(w_time)
                y_val.append(w_lbl)

    if X_val_ids:
        np.savez(os.path.join(output_dir, "val_data.npz"),
                 ids=np.concatenate(X_val_ids),
                 payloads=np.concatenate(X_val_pay),
                 deltas=np.concatenate(X_val_time),
                 labels=np.concatenate(y_val))
        logger.info("Validation Data Saved.")

    # 5. Save data quality report
    quality_report = {
        'dataset': dataset,
        'total_files': overall_stats['total_files'],
        'preprocessing_completed': True,
        'skip_invalid_rows': skip_invalid_rows
    }
    
    quality_report_path = os.path.join(output_dir, 'data_quality_report.json')
    with open(quality_report_path, 'w') as f:
        json.dump(quality_report, f, indent=2)
    logger.info(f"Data quality report saved to {quality_report_path}")

    logger.info("Done! Data ready for LSS-CAN-Mamba.")



if __name__ == "__main__":
    args = parse_args()
    
    # Get configuration, allowing command-line args to override
    dataset = args.dataset if args.dataset else DATASET
    dataset_root = args.dataset_root if args.dataset_root else DATASET_ROOT
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    
    # Update dataset_root and output_dir if dataset was changed
    if args.dataset and not args.dataset_root:
        dataset_root = f"/workspace/data/can-train-and-test-v1.5/{dataset}"
    if args.dataset and not args.output_dir:
        output_dir = f"/workspace/data/processed_data/{dataset}_run_02"
    
    # Print updated configuration if arguments were provided
    if args.dataset or args.dataset_root or args.output_dir:
        print(f"\n{'='*60}")
        print(f"Updated Configuration (from command-line args)")
        print(f"{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"Dataset Root: {dataset_root}")
        print(f"Output Dir: {output_dir}")
        print(f"{'='*60}\n")
    
    logger.info(f"Starting preprocessing with skip_invalid_rows={args.skip_invalid_rows}")
    run_pipeline(
        skip_invalid_rows=args.skip_invalid_rows,
        dataset=dataset,
        dataset_root=dataset_root,
        output_dir=output_dir
    )