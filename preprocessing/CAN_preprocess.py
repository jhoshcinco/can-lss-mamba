import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor

# --- CONFIGURATION ---
# DATASET_ROOT = r"C:\Users\Jhosh\Desktop\CAN-LSS-Mamba\dataset\can-train-and-test-v1.5\set_01"
# OUTPUT_DIR = "processed_data/set_01"

DATASET_ROOT = "/workspace/data/can-train-and-test-v1.5/set_01"
OUTPUT_DIR = "/workspace/data/processed_data/set_01"

WINDOW_SIZE = 64
STRIDE = 64


# --- HELPER FUNCTIONS ---

def parse_csv(file_path, id_map=None):
    """
    Parses CSV, extracts IDs, Payloads, and Time Deltas.
    Returns: ids, payloads, delta_feat, labels
    """
    try:
        # 1. Read Data
        df = pd.read_csv(file_path)

        # Standardize headers (strip spaces, lowercase) just in case
        df.columns = df.columns.str.strip().str.lower()

        # Column Names (Adjusted to your screenshot)
        id_col = 'arbitration_id'
        label_col = 'attack'
        time_col = 'timestamp'
        data_col = 'data_field'

        # 2. Process Time Delta (CRITICAL FOR MAMBA)
        # Convert timestamp to float
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')

        # Calculate Delta (Time since last message)
        delta = df[time_col].diff().fillna(0).values

        # Log-Normalize to squash spikes (microseconds vs seconds)
        # log1p(x) = log(x + 1) -> Handles 0 safely
        delta_norm = np.log1p(delta + 1e-6)

        # Reshape for concatenation [N, 1]
        delta_feat = delta_norm.reshape(-1, 1)

        # 3. Process IDs
        # Handle Hex Strings (e.g., '199' or '0x199')
        df[id_col] = df[id_col].apply(
            lambda x: int(str(x), 16) if isinstance(x, str) and any(c.isalpha() for c in str(x)) else int(x))

        if id_map:
            # Map IDs to learned indices (Training Phase)
            df['id_idx'] = df[id_col].apply(lambda x: id_map.get(x, id_map.get('<UNK>')))
        else:
            # Return unique IDs for Vocabulary Building
            return df[id_col].unique()

        # 4. Process Payload (Split Hex String -> 8 Bytes)
        def split_payload(hex_str):
            hex_str = str(hex_str).strip()
            # Ensure it is exactly 16 chars (8 bytes); pad with zeros if shorter
            hex_str = hex_str.ljust(16, '0')
            # Split into 8 integers
            return [int(hex_str[i:i + 2], 16) for i in range(0, 16, 2)]

        payload_list = df[data_col].apply(split_payload).tolist()
        payloads = np.array(payload_list) / 255.0  # Normalize bytes to [0, 1]

        # 5. Process Labels
        # Ensure labels are integers (0 or 1)
        # If your CSV uses 'T' for attack, convert it. If it's already 0/1, this is safe.
        labels = df[label_col].apply(lambda x: 1 if str(x).upper() in ['1', 'T', 'ATTACK'] else 0).values

        return df['id_idx'].values, payloads, delta_feat, labels

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
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

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. IDENTIFY FILES
    train_folder = os.path.join(DATASET_ROOT, "train_02_with_attacks")

    # Check if folder exists
    if not os.path.exists(train_folder):
        print(f"ERROR: Folder not found: {train_folder}")
        return

    all_files = sorted(glob.glob(os.path.join(train_folder, "*.csv")))

    if not all_files:
        print(f"ERROR: No CSV files found in {train_folder}")
        return

    # CHRONOLOGICAL SPLIT (Three Bucket Rule)
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Found {len(all_files)} files. Train: {len(train_files)}, Val: {len(val_files)}")

    # 2. BUILD VOCABULARY (IDs)
    print("Building ID Map...")
    unique_ids = set()
    for f in train_files:
        try:
            current_ids = parse_csv(f)  # Returns unique IDs list
            if current_ids is not None:
                unique_ids.update(current_ids)
        except Exception as e:
            print(f"Skipping {f} during vocab build: {e}")

    # Create Mapping
    id_map = {can_id: i for i, can_id in enumerate(sorted(unique_ids))}
    id_map['<UNK>'] = len(id_map)  # Handle unknown IDs
    print(f"Vocab Size: {len(id_map)} IDs (saved to {OUTPUT_DIR}/id_map.npy)")

    # Save ID Map
    np.save(os.path.join(OUTPUT_DIR, "id_map.npy"), id_map)

    # 3. PROCESS TRAIN DATA
    print("Processing Training Data...")
    X_tr_ids, X_tr_pay, X_tr_time, y_tr = [], [], [], []

    for f in train_files:
        res = parse_csv(f, id_map)
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
        np.savez(os.path.join(OUTPUT_DIR, "train_data.npz"),
                 ids=np.concatenate(X_tr_ids),
                 payloads=np.concatenate(X_tr_pay),
                 deltas=np.concatenate(X_tr_time),
                 labels=np.concatenate(y_tr))
        print("Training Data Saved.")
    else:
        print("WARNING: No training windows created (check window size vs file length)")

    # 4. PROCESS VAL DATA
    print("Processing Validation Data...")
    X_val_ids, X_val_pay, X_val_time, y_val = [], [], [], []

    for f in val_files:
        res = parse_csv(f, id_map)
        if res:
            ids, pay, time, lbl = res
            w_ids, w_pay, w_time, w_lbl = create_windows(ids, pay, time, lbl)

            if len(w_ids) > 0:
                X_val_ids.append(w_ids)
                X_val_pay.append(w_pay)
                X_val_time.append(w_time)
                y_val.append(w_lbl)

    if X_val_ids:
        np.savez(os.path.join(OUTPUT_DIR, "val_data.npz"),
                 ids=np.concatenate(X_val_ids),
                 payloads=np.concatenate(X_val_pay),
                 deltas=np.concatenate(X_val_time),
                 labels=np.concatenate(y_val))
        print("Validation Data Saved.")

    print("Done! Data ready for LSS-CAN-Mamba.")


if __name__ == "__main__":
    run_pipeline()