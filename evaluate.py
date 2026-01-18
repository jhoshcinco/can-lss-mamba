import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import glob
import gc  # Garbage Collector
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from model import LSS_CAN_Mamba

# --- CONFIG ---
DATASET_ROOT = r"/workspace/data/can-train-and-test-v1.5/set_01"
MODEL_PATH = "/workspace/checkpoints/set_01/lss_can_mamba_best.pth"
ID_MAP_PATH = "/workspace/data/processed_data/set_01/id_map.npy"
BATCH_SIZE = 128  # Increased batch size for speed (safe because we process 1 file at a time)
WINDOW_SIZE = 64
STRIDE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- HELPER FUNCTIONS ---
def parse_csv_stream(file_path, id_map):
    """ Reads CSV and returns processed arrays immediately """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()

        # 1. Time
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        delta = df['timestamp'].diff().fillna(0).values
        delta_norm = np.log1p(delta + 1e-6).reshape(-1, 1)

        # 2. IDs
        id_col = 'arbitration_id'
        unk_idx = id_map['<UNK>']

        # Safe Hex Conversion
        df[id_col] = df[id_col].apply(
            lambda x: int(str(x), 16) if isinstance(x, str) and any(c.isalpha() for c in str(x)) else int(x))
        ids = df[id_col].apply(lambda x: id_map.get(x, unk_idx)).values

        # 3. Payloads
        def split_payload(hex_str):
            hex_str = str(hex_str).strip().ljust(16, '0')
            return [int(hex_str[i:i + 2], 16) for i in range(0, 16, 2)]

        payloads = np.array(df['data_field'].apply(split_payload).tolist()) / 255.0

        # 4. Labels
        labels = df['attack'].apply(lambda x: 1 if str(x).upper() in ['1', 'T', 'ATTACK'] else 0).values

        return ids, payloads, delta_norm, labels
    except Exception as e:
        return None


def create_windows(ids, payloads, deltas, labels):
    X_ids, X_feats, y = [], [], []
    # Vectorized windowing would be faster, but loop is safer for memory
    for i in range(0, len(ids) - WINDOW_SIZE, STRIDE):
        X_ids.append(ids[i:i + WINDOW_SIZE])
        # Combine payload + delta
        feat = np.concatenate([payloads[i:i + WINDOW_SIZE], deltas[i:i + WINDOW_SIZE]], axis=-1)
        X_feats.append(feat)
        y.append(1 if np.any(labels[i:i + WINDOW_SIZE] == 1) else 0)

    return np.array(X_ids), np.array(X_feats), np.array(y)


# --- MAIN LOOP ---
def evaluate():
    print(f"Loading Resources on {DEVICE}...")

    # Load Map & Model
    id_map = np.load(ID_MAP_PATH, allow_pickle=True).item()
    vocab_size = len(id_map) + 1  # Ensure this matches training!

    model = LSS_CAN_Mamba(num_unique_ids=vocab_size).to(DEVICE)
    # Load weights safely
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    test_folders = sorted(glob.glob(os.path.join(DATASET_ROOT, "test_*")))

    print(f"\n{'=' * 75}")
    print(f"{'TEST SCENARIO':<40} | {'F1':<8} | {'ACC':<8} | {'PREC':<8} | {'REC':<8}")
    print(f"{'=' * 75}")

    results = []

    for folder in test_folders:
        folder_name = os.path.basename(folder)
        csv_files = glob.glob(os.path.join(folder, "*.csv"))

        if not csv_files:
            continue

        # Accumulators for the WHOLE FOLDER
        y_true_all = []
        y_pred_all = []

        for f in csv_files:
            # 1. Process Single File
            res = parse_csv_stream(f, id_map)
            if res is None: continue

            ids, pay, time, lbl = res
            w_ids, w_feats, w_lbl = create_windows(ids, pay, time, lbl)

            if len(w_ids) == 0: continue

            # 2. Run Inference Immediately
            x_ids_t = torch.LongTensor(w_ids).to(DEVICE)
            x_feats_t = torch.FloatTensor(w_feats).to(DEVICE)

            dataset = TensorDataset(x_ids_t, x_feats_t)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

            with torch.no_grad():
                for batch_ids, batch_feats in loader:
                    logits = model(batch_ids, batch_feats)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()

                    y_pred_all.extend(preds)

            # Save True Labels
            y_true_all.extend(w_lbl)

            # 3. MEMORY CLEANUP
            del w_ids, w_feats, w_lbl, x_ids_t, x_feats_t, dataset, loader
            gc.collect()  # Force RAM release

        # 4. Calculate Folder Metrics
        if len(y_true_all) > 0:
            f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
            acc = accuracy_score(y_true_all, y_pred_all)
            prec = precision_score(y_true_all, y_pred_all, zero_division=0)
            rec = recall_score(y_true_all, y_pred_all, zero_division=0)

            print(f"{folder_name:<40} | {f1:.4f}   | {acc:.4f}   | {prec:.4f}   | {rec:.4f}")
            results.append({'Scenario': folder_name, 'F1': f1, 'Acc': acc, 'Precision': prec, 'Recall': rec})
        else:
            print(f"{folder_name:<40} | NO DATA (Check Window Size)")

    print(f"{'=' * 75}")
    pd.DataFrame(results).to_csv("thesis_final_results.csv", index=False)
    print("Results saved to thesis_final_results.csv")


if __name__ == "__main__":
    evaluate()