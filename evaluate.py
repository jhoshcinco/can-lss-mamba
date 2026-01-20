import os
import glob
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from model import LSS_CAN_Mamba

# =========================
# THREE-BUCKET EVALUATION STRATEGY
# =========================
# 1. TRAINING SET (train_02_with_attacks - 80%): Model parameter learning
# 2. VALIDATION SET (train_02_with_attacks - 20%): Model selection + Threshold tuning
# 3. TEST SETS (test_* folders): Final unbiased evaluation on unseen attack scenarios
#
# This script evaluates on bucket #3 (test sets) using the threshold learned from bucket #2.
# This ensures NO DATA LEAKAGE and proper generalization assessment.
# =========================

# --- CONFIGURATION ---
DATASET_ROOT = r"/workspace/data/can-train-and-test-v1.5/set_01"
MODEL_PATH = "/workspace/checkpoints/set_01/lss_can_mamba_best.pth"
CHECKPOINT_PATH = "/workspace/checkpoints/set_01/lss_can_mamba_last.pth"  # For threshold
ID_MAP_PATH = "/workspace/data/processed_data/set_01/id_map.npy"
OUTPUT_CSV = "/workspace/final_thesis_results.csv"

BATCH_SIZE = 128
WINDOW_SIZE = 64
STRIDE = 64
CHUNK_WINDOWS = 20000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = 0.5  # Fallback if no checkpoint found


def parse_csv_exact_match(file_path, id_map):
    """
    EXACT REPLICATION of your training preprocessing logic.
    Includes safety for Fuzzing/Garbage data in test sets.
    """
    # 1. Read Data (Optimized for speed/memory)
    cols = ["timestamp", "arbitration_id", "data_field", "attack"]
    dtypes = {"timestamp": "float64", "arbitration_id": "string", "data_field": "string", "attack": "string"}

    try:
        df = pd.read_csv(file_path, usecols=cols, dtype=dtypes, on_bad_lines="skip", engine="c")
    except:
        df = pd.read_csv(file_path, usecols=cols, dtype=dtypes, on_bad_lines="skip", engine="python")

    df.columns = df.columns.str.strip().str.lower()
    unk_idx = id_map["<UNK>"]

    # 2. Process Time Delta (Matches: log1p(delta + 1e-6))
    ts = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0).astype(np.float32).values
    delta = np.diff(ts, prepend=ts[0]).astype(np.float32)
    delta[delta < 0] = 0  # Safety for bad timestamps
    delta_norm = np.log1p(delta + 1e-6).astype(np.float32).reshape(-1, 1)

    # 3. Process IDs (Matches: Hybrid Hex/Int logic)
    def safe_id_convert(x):
        try:
            s = str(x).strip()
            # Fuzzing Protection
            if s.lower() in ["na", "nan", "null", ""]: return -1

            # --- EXACT LOGIC FROM YOUR PREPROCESS.PY ---
            # "int(str(x), 16) if ... any(c.isalpha()...) else int(x)"
            if any(c.isalpha() for c in s):
                return int(s, 16)
            else:
                return int(float(s))
                # -------------------------------------------
        except:
            return -1

    arb = df["arbitration_id"].apply(safe_id_convert).to_numpy()
    ids = np.array([id_map.get(v, unk_idx) if v != -1 else unk_idx for v in arb], dtype=np.int64)

    # 4. Process Payloads (Matches: split_payload / 255.0)
    def split_payload(hex_str):
        try:
            s = str(hex_str).strip()
            if s.lower() in ["na", "nan", ""]: s = "00" * 8
            s = s.ljust(16, "0")[:16]
            return [int(s[i:i + 2], 16) for i in range(0, 16, 2)]
        except:
            return [0] * 8

    # Fast Stack
    payloads = np.stack(df["data_field"].apply(split_payload).to_numpy()).astype(np.uint8)
    payloads = (payloads.astype(np.float32) / 255.0)

    # 5. Process Labels
    labels = df["attack"].astype(str).str.upper().isin(["1", "T", "ATTACK"]).astype(np.int64).values

    return ids, payloads, delta_norm, labels


def iter_windows_chunked(ids, payloads, deltas, labels, window=64, stride=64, chunk_windows=20000):
    n = len(ids)
    start = 0
    while start + window <= n:
        max_w = ((n - window - start) // stride) + 1
        if max_w <= 0: break

        w_count = min(chunk_windows, max_w)

        base = start + np.arange(w_count, dtype=np.int64) * stride
        idx = base[:, None] + np.arange(window, dtype=np.int64)[None, :]

        w_feats = np.concatenate([payloads[idx], deltas[idx]], axis=-1).astype(np.float32)
        y = np.any(labels[idx] == 1, axis=1).astype(np.int64)

        yield ids[idx], w_feats, y

        start += w_count * stride


def evaluate_all():
    print(f"--- STARTING FULL EVALUATION ---")
    print(f"Device: {DEVICE}")

    # 1. Load Resources
    id_map = np.load(ID_MAP_PATH, allow_pickle=True).item()
    max_id = max(id_map.values())
    vocab_size = max_id + 2

    # 2. Load Model and Optimal Threshold
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    saved_vocab = state_dict["id_embedding.weight"].shape[0]
    if saved_vocab != vocab_size: vocab_size = saved_vocab

    model = LSS_CAN_Mamba(num_unique_ids=vocab_size).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Load optimal threshold from checkpoint
    optimal_threshold = DEFAULT_THRESHOLD
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        optimal_threshold = float(ckpt.get("best_threshold", DEFAULT_THRESHOLD))
        print(f"Loaded optimal threshold: {optimal_threshold:.4f}")
    else:
        print(f"No checkpoint found. Using default threshold: {optimal_threshold:.4f}")

    # 3. Find All Test Folders
    test_folders = sorted(glob.glob(os.path.join(DATASET_ROOT, "test_*")))
    print(f"Found {len(test_folders)} test scenarios.")

    results_list = []

    # 4. Main Loop
    for folder in test_folders:
        scenario_name = os.path.basename(folder)
        print(f"\n>>> PROCESSING: {scenario_name}")

        csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        tp = fp = fn = tn = 0

        for f in csv_files:
            print(f"   Reading {os.path.basename(f)}...", end="\r")
            try:
                ids, pay, time_d, lbl = parse_csv_exact_match(f, id_map)
            except Exception as e:
                print(f"   [Skipping corrupt file: {e}]")
                continue

            # Process in Chunks (RAM Safe)
            for w_ids, w_feats, y_true in iter_windows_chunked(ids, pay, time_d, lbl, WINDOW_SIZE, STRIDE,
                                                               CHUNK_WINDOWS):
                dataset = TensorDataset(torch.from_numpy(w_ids), torch.from_numpy(w_feats))
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

                with torch.no_grad():
                    offset = 0
                    for batch_ids, batch_feats in loader:
                        batch_ids, batch_feats = batch_ids.to(DEVICE), batch_feats.to(DEVICE)
                        logits = model(batch_ids, batch_feats)
                        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Get attack probabilities
                        preds = (probs >= optimal_threshold).astype(int)  # Apply optimal threshold

                        y_batch = y_true[offset:offset + len(preds)]
                        offset += len(preds)

                        tp += np.sum((preds == 1) & (y_batch == 1))
                        tn += np.sum((preds == 0) & (y_batch == 0))
                        fp += np.sum((preds == 1) & (y_batch == 0))
                        fn += np.sum((preds == 0) & (y_batch == 1))

                del dataset, loader
                gc.collect()

            del ids, pay, time_d, lbl
            gc.collect()

        # Calculate Metrics for this Scenario
        total = tp + tn + fp + fn
        if total > 0:
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            acc = (tp + tn) / (total + 1e-6)

            print(f"\n   [RESULT] F1: {f1:.4f} | Acc: {acc:.4f} | Rec: {recall:.4f}")

            results_list.append({
                "Scenario": scenario_name,
                "F1_Score": f1,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn
            })

            # Save Intermediate (Safety measure)
            pd.DataFrame(results_list).to_csv(OUTPUT_CSV, index=False)
        else:
            print("\n   [WARNING] No valid data processed for this folder.")

    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete. Results saved to: {OUTPUT_CSV}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    evaluate_all()