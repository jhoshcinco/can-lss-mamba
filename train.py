import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import time
import os
from model import LSS_CAN_Mamba  # DO NOT CHANGE MODEL

# =========================
# --- CONFIG (Colab/Drive + Resume Friendly) ---
# You can override these via environment variables when running:
#   DATA_DIR, OUT_DIR, MODEL_NAME, BATCH_SIZE, EPOCHS, LR
# Example (Colab):
# !DATA_DIR="/content/drive/MyDrive/mamba_thesis/data/set_01" \
#  OUT_DIR="/content/drive/MyDrive/mamba_thesis/checkpoints/set_01" \
#  EPOCHS=50 LR=1e-4 BATCH_SIZE=64 \
#  python train.py
# =========================

# change to the location of the preprocessed files 
# DATA_DIR = os.environ.get("DATA_DIR", "processed_data/set_01")
# OUT_DIR = os.environ.get("OUT_DIR", "outputs")
DATA_DIR = os.environ.get("DATA_DIR", "/workspace/data/processed_data/set_01_run_02")
OUT_DIR  = os.environ.get("OUT_DIR",  "/workspace/checkpoints/set_01")
MODEL_NAME = os.environ.get("MODEL_NAME", "lss_can_mamba")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
EPOCHS = int(os.environ.get("EPOCHS", 20))
LR = float(os.environ.get("LR", 1e-4))

os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, f"{MODEL_NAME}_best.pth")
LAST_PATH = os.path.join(OUT_DIR, f"{MODEL_NAME}_last.pth")

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"OUT_DIR : {OUT_DIR}")
print(f"Best ckpt: {MODEL_PATH}")
print(f"Last ckpt: {LAST_PATH}")
print(f"BATCH_SIZE={BATCH_SIZE} EPOCHS={EPOCHS} LR={LR}")

# 1. LOAD DATA
print("Loading data...")
train_npz = np.load(os.path.join(DATA_DIR, "train_data.npz"))
val_npz = np.load(os.path.join(DATA_DIR, "val_data.npz"))
id_map = np.load(os.path.join(DATA_DIR, "id_map.npy"), allow_pickle=True).item()

# check if <UNK> exists in id map
vocab_size = len(id_map)
# preprocessing already includes <UNK>

# this will choose the best threshold based on F1 score
def best_f1_threshold(y_true, p_attack):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (p_attack >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def get_loader(npz, shuffle=True):
    # Combine Payload (8) and Delta (1) -> 9 Features
    feats = np.concatenate([npz["payloads"], npz["deltas"]], axis=-1)
    dataset = TensorDataset(
        torch.LongTensor(npz["ids"]),
        torch.FloatTensor(feats),
        torch.LongTensor(npz["labels"]),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = get_loader(train_npz, shuffle=True)
val_loader = get_loader(val_npz, shuffle=False)
print(f"Data Loaded. Vocab Size: {vocab_size}")

# === DIAGNOSTIC: Check class imbalance ===
train_attack_rate = train_npz["labels"].mean()
val_attack_rate = val_npz["labels"].mean()
print(f"\n{'='*60}")
print(f"DATASET STATISTICS:")
print(f"{'='*60}")
print(f"Training set:")
print(f"  - Total windows: {len(train_npz['labels'])}")
print(f"  - Attack windows: {train_npz['labels'].sum()}")
print(f"  - Attack rate: {train_attack_rate:.4f} ({train_attack_rate*100:.2f}%)")
print(f"  - Imbalance ratio: 1:{1/train_attack_rate:.1f} (attack:normal)")
print(f"\nValidation set:")
print(f"  - Total windows: {len(val_npz['labels'])}")
print(f"  - Attack windows: {val_npz['labels'].sum()}")
print(f"  - Attack rate: {val_attack_rate:.4f} ({val_attack_rate*100:.2f}%)")
print(f"  - Imbalance ratio: 1:{1/val_attack_rate:.1f} (attack:normal)")
print(f"{'='*60}\n")

# 2. INIT MODEL (UNCHANGED)
# 1. CALCULATE CLASS WEIGHTS
# Count positives (Attacks) and negatives (Normal) in training labels
y_train_indices = train_npz['labels'] # Assuming this is the label array
num_normal = len(y_train_indices) - np.sum(y_train_indices)
num_attacks = np.sum(y_train_indices)

# Weight = Number of Negatives / Number of Positives
# If normal packets are 10x more common, attacks get 10x weight
pos_weight_val = num_normal / (num_attacks + 1e-6) # +1e-6 avoids division by zero
pos_weight_val = min(pos_weight_val, 10.0)
print(f"Clipped Pos_Weight: {pos_weight_val:.2f}")
print(f"Calculated Class Weight (Pos_Weight): {pos_weight_val:.2f}")

# Convert to Tensor
# Note: For CrossEntropyLoss with 2 classes, we usually use the 'weight' argument tensor([1.0, pos_weight])
class_weights = torch.tensor([1.0, pos_weight_val], device=device, dtype=torch.float32)

# 2. INIT MODEL
model = LSS_CAN_Mamba(num_unique_ids=vocab_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05) # Increased weight_decay for regularization

# 3. UPDATE LOSS FUNCTION
# We pass the calculated weights here
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 2.5 RESUME (NEW)
start_epoch = 0
best_f1 = 0.0
best_threshold = 0.5  # Initialize best threshold

if os.path.exists(LAST_PATH):
    print("Found last checkpoint. Resuming...")
    ckpt = torch.load(LAST_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_f1 = float(ckpt.get("best_f1", 0.0))
    best_threshold = float(ckpt.get("best_threshold", 0.5))
    print(f"Resumed at epoch {start_epoch} with best_f1={best_f1:.4f}, threshold={best_threshold:.4f}")
else:
    print("No last checkpoint found. Starting fresh.")

# 3. TRAIN LOOP
print(f"Starting Training ({EPOCHS} Epochs)...")

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    model.train()
    train_loss = 0.0

    for ids, feats, labels in train_loader:
        ids, feats, labels = ids.to(device), feats.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(ids, feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ids, feats, labels in val_loader:
            ids, feats, labels = ids.to(device), feats.to(device), labels.to(device)
            logits = model(ids, feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())  # store probabilities
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    p_attack = np.array(all_preds)
    y_true = np.array(all_labels)

    # === DIAGNOSTIC: Probability distribution analysis ===
    # Check if model is learning to separate classes
    p_normal = p_attack[y_true == 0]  # Probabilities for normal samples
    p_attacks = p_attack[y_true == 1]  # Probabilities for attack samples

    # Find optimal threshold on validation set
    # Note: In three-bucket approach, this validation set is used for:
    # 1. Model selection (best epoch)
    # 2. Threshold tuning
    # Final testing is done on separate test_* folders (evaluate.py)
    best_t, val_f1 = best_f1_threshold(y_true, p_attack)
    val_acc = accuracy_score(y_true, (p_attack >= best_t).astype(int))

    # Additional diagnostics every 5 epochs or when saving best model
    show_detailed_stats = (epoch % 5 == 0) or (val_f1 > best_f1)

    if show_detailed_stats:
        print(f"\n--- Probability Distribution Analysis (Epoch {epoch + 1}) ---")
        print(f"Normal samples (label=0, n={len(p_normal)}):")
        print(f"  Mean prob: {p_normal.mean():.4f} | Std: {p_normal.std():.4f}")
        print(f"  Min: {p_normal.min():.4f} | Max: {p_normal.max():.4f}")
        print(f"Attack samples (label=1, n={len(p_attacks)}):")
        print(f"  Mean prob: {p_attacks.mean():.4f} | Std: {p_attacks.std():.4f}")
        print(f"  Min: {p_attacks.min():.4f} | Max: {p_attacks.max():.4f}")
        print(f"Separation: {p_attacks.mean() - p_normal.mean():.4f} (higher is better)")
        print(f"---")


    print(
        f"Epoch {epoch + 1} | "
        f"Loss: {train_loss / len(train_loader):.4f} | "
        f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | "
        f"Thr: {best_t:.2f} | "
        f"Time: {time.time() - start:.1f}s"
    )

    # === WARNING: Detect suspicious threshold behavior ===
    if best_t < 0.05:
        print(f"⚠️  WARNING: Threshold = {best_t:.4f} is extremely low!")
        print(f"    This means the model classifies almost everything as attack.")
        print(f"    Possible causes:")
        print(f"    1. Severe class imbalance (attack rate: {val_attack_rate:.4f})")
        print(f"    2. Poor probability calibration (check separation above)")
        print(f"    3. Model not learning meaningful patterns")
        print(f"    Consider: Fβ score (β=0.5), PR-AUC, or constrained threshold range")
    elif best_t > 0.95:
        print(f"⚠️  WARNING: Threshold = {best_t:.4f} is extremely high!")
        print(f"    This means the model rarely predicts attacks.")
        print(f"    The model may be biased toward the majority class.")

    # Save best
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_threshold = best_t  # Update best threshold
        torch.save(model.state_dict(), MODEL_PATH)
        print(f">>> Best model saved! (F1={best_f1:.4f}, Threshold={best_threshold:.4f})")

    # Always save "last" checkpoint for continuity (NEW)
    torch.save(
        {
            "epoch": epoch,
            "best_f1": best_f1,
            "best_threshold": best_threshold,  # Save threshold
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
        },
        LAST_PATH,
    )
    print(">>> Last checkpoint saved.")

print(f"Done. Best F1: {best_f1:.4f} | Best Threshold: {best_threshold:.4f}")
