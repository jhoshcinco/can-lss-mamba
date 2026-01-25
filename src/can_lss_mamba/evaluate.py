import os
import glob
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import argparse
from .models.mamba import LSS_CAN_Mamba


def evaluate(
    dataset_root,
    model_path,
    checkpoint_path,
    id_map_path,
    output_csv,
    batch_size=128,
    window_size=64,
    stride=64,
    chunk_windows=20000,
    device=None,
    default_threshold=0.5,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- STARTING EVALUATION ---")
    print(f"Device: {device}")
    print(f"Dataset Root: {dataset_root}")
    print(f"Model Path: {model_path}")

    # Helper Functions
    def parse_csv_exact_match(file_path, id_map):
        cols = ["timestamp", "arbitration_id", "data_field", "attack"]
        dtypes = {
            "timestamp": "float64",
            "arbitration_id": "string",
            "data_field": "string",
            "attack": "string",
        }

        try:
            df = pd.read_csv(
                file_path, usecols=cols, dtype=dtypes, on_bad_lines="skip", engine="c"
            )
        except:
            df = pd.read_csv(
                file_path,
                usecols=cols,
                dtype=dtypes,
                on_bad_lines="skip",
                engine="python",
            )

        df.columns = df.columns.str.strip().str.lower()
        unk_idx = id_map["<UNK>"]

        ts = (
            pd.to_numeric(df["timestamp"], errors="coerce")
            .fillna(0.0)
            .astype(np.float32)
            .values
        )
        delta = np.diff(ts, prepend=ts[0]).astype(np.float32)
        delta[delta < 0] = 0
        delta_norm = np.log1p(delta + 1e-6).astype(np.float32).reshape(-1, 1)

        def safe_id_convert(x):
            try:
                s = str(x).strip()
                if s.lower() in ["na", "nan", "null", ""]:
                    return -1
                if any(c.isalpha() for c in s):
                    return int(s, 16)
                else:
                    return int(float(s))
            except:
                return -1

        arb = df["arbitration_id"].apply(safe_id_convert).to_numpy()
        ids = np.array(
            [id_map.get(v, unk_idx) if v != -1 else unk_idx for v in arb],
            dtype=np.int64,
        )

        def split_payload(hex_str):
            try:
                s = str(hex_str).strip()
                if s.lower() in ["na", "nan", ""]:
                    s = "00" * 8
                s = s.ljust(16, "0")[:16]
                return [int(s[i : i + 2], 16) for i in range(0, 16, 2)]
            except:
                return [0] * 8

        payloads = np.stack(
            df["data_field"].apply(split_payload).to_numpy()
        ).astype(np.uint8)
        payloads = payloads.astype(np.float32) / 255.0

        labels = (
            df["attack"]
            .astype(str)
            .str.upper()
            .isin(["1", "T", "ATTACK"])
            .astype(np.int64)
            .values
        )

        return ids, payloads, delta_norm, labels

    def iter_windows_chunked(
        ids, payloads, deltas, labels, window=64, stride=64, chunk_windows=20000
    ):
        n = len(ids)
        start = 0
        while start + window <= n:
            max_w = ((n - window - start) // stride) + 1
            if max_w <= 0:
                break

            w_count = min(chunk_windows, max_w)

            base = start + np.arange(w_count, dtype=np.int64) * stride
            idx = base[:, None] + np.arange(window, dtype=np.int64)[None, :]

            w_feats = np.concatenate([payloads[idx], deltas[idx]], axis=-1).astype(
                np.float32
            )
            y = np.any(labels[idx] == 1, axis=1).astype(np.int64)

            yield ids[idx], w_feats, y

            start += w_count * stride

    # 1. Load Resources
    try:
        id_map = np.load(id_map_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"❌ Error: ID map not found at {id_map_path}")
        return

    max_id = max(id_map.values())
    vocab_size = max_id + 2

    # 2. Load Model
    try:
        state_dict = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"❌ Error: Model not found at {model_path}")
        return

    saved_vocab = state_dict["id_embedding.weight"].shape[0]
    if saved_vocab != vocab_size:
        vocab_size = saved_vocab

    model = LSS_CAN_Mamba(num_unique_ids=vocab_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load optimal threshold
    optimal_threshold = default_threshold
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        optimal_threshold = float(ckpt.get("best_threshold", default_threshold))
        print(f"Loaded optimal threshold: {optimal_threshold:.4f}")
    else:
        print(f"No checkpoint found. Using default threshold: {optimal_threshold:.4f}")

    # 3. Find All Test Folders
    test_folders = sorted(glob.glob(os.path.join(dataset_root, "test_*")))
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

            # Process in Chunks
            for w_ids, w_feats, y_true in iter_windows_chunked(
                ids, pay, time_d, lbl, window_size, stride, chunk_windows
            ):
                dataset = TensorDataset(
                    torch.from_numpy(w_ids), torch.from_numpy(w_feats)
                )
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                with torch.no_grad():
                    offset = 0
                    for batch_ids, batch_feats in loader:
                        batch_ids, batch_feats = batch_ids.to(device), batch_feats.to(
                            device
                        )
                        logits = model(batch_ids, batch_feats)
                        probs = (
                            torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        )  # Get attack probabilities
                        preds = (probs >= optimal_threshold).astype(
                            int
                        )  # Apply optimal threshold

                        y_batch = y_true[offset : offset + len(preds)]
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

            results_list.append(
                {
                    "Scenario": scenario_name,
                    "F1_Score": f1,
                    "Accuracy": acc,
                    "Precision": precision,
                    "Recall": recall,
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                }
            )

            # Save Intermediate
            pd.DataFrame(results_list).to_csv(output_csv, index=False)
        else:
            print("\n   [WARNING] No valid data processed for this folder.")

    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete. Results saved to: {output_csv}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CAN LSS Mamba Model")
    parser.add_argument(
        "--dataset-root",
        default=r"/workspace/data/can-train-and-test-v1.5/set_01",
    )
    parser.add_argument(
        "--model-path", default="/workspace/checkpoints/set_01/lss_can_mamba_best.pth"
    )
    parser.add_argument(
        "--checkpoint-path",
        default="/workspace/checkpoints/set_01/lss_can_mamba_last.pth",
    )
    parser.add_argument(
        "--id-map-path", default="/workspace/data/processed_data/set_01/id_map.npy"
    )
    parser.add_argument(
        "--output-csv", default="/workspace/final_thesis_results_02.csv"
    )
    parser.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()

    evaluate(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        id_map_path=args.id_map_path,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
    )
