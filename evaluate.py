import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import glob
import gc  # Added Garbage Collection
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from model import LSS_CAN_Mamba

# --- MANUAL CONFIGURATION ---
# UNCOMMENT THE FOLDER YOU WANT TO TEST RIGHT NOW:

TARGET_SCENARIO = "test_01_known_vehicle_known_attack"
# TARGET_SCENARIO = "test_02_unknown_vehicle_known_attack"
# TARGET_SCENARIO = "test_03_known_vehicle_unknown_attack"
# TARGET_SCENARIO = "test_04_unknown_vehicle_unknown_attack"
# TARGET_SCENARIO = "test_05_suppress"
# TARGET_SCENARIO = "test_06_masquerade"

# ---------------------------
DATASET_ROOT = r"/workspace/data/can-train-and-test-v1.5/set_01"
MODEL_PATH = "/workspace/checkpoints/set_01/lss_can_mamba_best.pth"
ID_MAP_PATH = "/workspace/data/processed_data/set_01/id_map.npy"
BATCH_SIZE = 64
WINDOW_SIZE = 64
STRIDE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_csv_for_inference(file_path, id_map):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()

        # 1. Time Delta (With NaN Safety Fix)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        delta = df['timestamp'].diff().fillna(0).values
        delta[delta < 0] = 0  # Fix negative time crashing Mamba
        delta_norm = np.log1p(delta + 1e-6).reshape(-1, 1)

        # 2. IDs
        id_col = 'arbitration_id'
        df[id_col] = df[id_col].apply(
            lambda x: int(str(x), 16) if isinstance(x, str) and any(c.isalpha() for c in str(x)) else int(x))

        unk_idx = id_map['<UNK>']
        ids = df[id_col].apply(lambda x: id_map.get(x, unk_idx)).values

        # 3. Payloads
        def split_payload(hex_str):
            hex_str = str(hex_str).strip().ljust(16, '0')
            return [int(hex_str[i:i + 2], 16) for i in range(0, 16, 2)]

        payloads = np.array(df['data_field'].apply(split_payload).tolist()) / 255.0

        # 4. Labels
        label_col = 'attack'
        labels = df[label_col].apply(lambda x: 1 if str(x).upper() in ['1', 'T', 'ATTACK'] else 0).values

        return ids, payloads, delta_norm, labels
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def create_windows(ids, payloads, deltas, labels):
    X_ids, X_feats, y = [], [], []
    for i in range(0, len(ids) - WINDOW_SIZE, STRIDE):
        w_id = ids[i:i + WINDOW_SIZE]
        w_pay = payloads[i:i + WINDOW_SIZE]
        w_time = deltas[i:i + WINDOW_SIZE]
        w_lbl = labels[i:i + WINDOW_SIZE]

        w_feat = np.concatenate([w_pay, w_time], axis=-1)

        X_ids.append(w_id)
        X_feats.append(w_feat)
        y.append(1 if np.any(w_lbl == 1) else 0)

    return np.array(X_ids), np.array(X_feats), np.array(y)


def evaluate():
    print(f"--- RUNNING MANUAL EVAL ON: {TARGET_SCENARIO} ---")

    # 1. Load Map
    id_map = np.load(ID_MAP_PATH, allow_pickle=True).item()

    # CRITICAL FIX: Ensure Vocab Size covers <UNK> to prevent Index Error Crash
    max_id = max(id_map.values())
    vocab_size = max_id + 2

    # 2. Load Model
    model = LSS_CAN_Mamba(num_unique_ids=vocab_size).to(DEVICE)

    # Handle vocab mismatch safely
    state_dict = torch.load(MODEL_PATH)
    saved_vocab = state_dict['id_embedding.weight'].shape[0]
    if saved_vocab != vocab_size:
        print(f"Adjusting vocab from {vocab_size} to {saved_vocab}")
        model = LSS_CAN_Mamba(num_unique_ids=saved_vocab).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    # 3. Target Specific Folder
    target_path = os.path.join(DATASET_ROOT, TARGET_SCENARIO)
    csv_files = glob.glob(os.path.join(target_path, "*.csv"))

    if not csv_files:
        print(f"No files found in {target_path}")
        return

    # Process all files
    all_ids, all_feats, all_labels = [], [], []

    for f in csv_files:
        print(f"Processing {os.path.basename(f)}...")
        res = parse_csv_for_inference(f, id_map)
        if res:
            ids, pay, time, lbl = res
            w_ids, w_feats, w_lbl = create_windows(ids, pay, time, lbl)
            if len(w_ids) > 0:
                all_ids.append(w_ids)
                all_feats.append(w_feats)
                all_labels.append(w_lbl)

        # Free memory per file
        del ids, pay, time, lbl, w_ids, w_feats, w_lbl
        gc.collect()

    if not all_ids:
        print("NO DATA EXTRACTED.")
        return

    print("Converting to Tensor (This may take RAM)...")
    x_ids = torch.LongTensor(np.concatenate(all_ids)).to(DEVICE)
    x_feats = torch.FloatTensor(np.concatenate(all_feats)).to(DEVICE)
    y_true = np.concatenate(all_labels)

    # Clean RAM
    del all_ids, all_feats, all_labels
    gc.collect()

    print("Running Inference...")
    dataset = TensorDataset(x_ids, x_feats)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_pred = []
    with torch.no_grad():
        for batch_ids, batch_feats in loader:
            logits = model(batch_ids, batch_feats)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)

    # Metrics
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)

    print(f"\nRESULT: {TARGET_SCENARIO}")
    print(f"F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}")

    # SAVE TO CSV (APPEND MODE)
    results = [{'Scenario': TARGET_SCENARIO, 'F1': f1, 'Acc': acc, 'Recall': rec, 'Precision': prec}]
    df_res = pd.DataFrame(results)

    # Append if file exists, write header if not
    use_header = not os.path.exists("thesis_results_manual.csv")
    df_res.to_csv("thesis_results_manual.csv", mode='a', header=use_header, index=False)
    print(">>> Saved to thesis_results_manual.csv")


if __name__ == "__main__":
    evaluate()