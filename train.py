import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import time
import os
from model import LSS_CAN_Mamba  # DO NOT CHANGE MODEL

# Optional WandB integration (initialized by scripts/train.py if available)
wandb_logger = None
try:
    if os.getenv("_WANDB_LOGGER_INITIALIZED"):
        # Logger was initialized by wrapper script
        pass
    elif os.getenv("WANDB_ENABLED", "false").lower() == "true":
        # Direct WandB initialization
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "can-lss-mamba"),
            config={
                "batch_size": int(os.environ.get("BATCH_SIZE", 32)),
                "epochs": int(os.environ.get("EPOCHS", 20)),
                "learning_rate": float(os.environ.get("LR", 1e-4)),
            }
        )
        wandb_logger = wandb
        print("✅ WandB initialized")
except ImportError:
    pass  # WandB not installed, continue without it

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

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 20))
LR = float(os.environ.get("LR", 1e-4))
ID_DROPOUT_PROB = float(os.environ.get("ID_DROPOUT_PROB", 0.00))  # Disabled - baseline performs best
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", 10))  # Stop if no improvement for N epochs

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


def get_loader(npz, shuffle=True, id_dropout_prob=0.0):
    # Combine Payload (8) and Delta (1) -> 9 Features
    feats = np.concatenate([npz["payloads"], npz["deltas"]], axis=-1)

    # ID Dropout Augmentation: Replace random IDs with <UNK> during training
    # Forces model to learn from payload/timing instead of just memorizing IDs
    ids = npz["ids"].copy()
    if id_dropout_prob > 0 and shuffle:  # Only apply during training
        unk_token = id_map.get("<UNK>", vocab_size - 1)
        dropout_mask = np.random.rand(len(ids)) < id_dropout_prob
        ids[dropout_mask] = unk_token

    dataset = TensorDataset(
        torch.LongTensor(ids),
        torch.FloatTensor(feats),
        torch.LongTensor(npz["labels"]),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = get_loader(train_npz, shuffle=True, id_dropout_prob=ID_DROPOUT_PROB)
val_loader = get_loader(val_npz, shuffle=False, id_dropout_prob=0.0)  # No dropout for validation
print(f"Data Loaded. Vocab Size: {vocab_size}")
print(f"ID Dropout: {ID_DROPOUT_PROB*100:.1f}% (training only)")

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

# === CRITICAL WARNING: Check train/val distribution mismatch ===
imbalance_ratio = val_attack_rate / train_attack_rate
if imbalance_ratio > 2.0 or imbalance_ratio < 0.5:
    print(f"🔴 CRITICAL WARNING: Train/Val attack rate mismatch!")
    print(f"   Train: {train_attack_rate*100:.2f}% | Val: {val_attack_rate*100:.2f}%")
    print(f"   Ratio: {imbalance_ratio:.2f}x difference")
    print(f"   This violates the IID assumption and may cause poor generalization.")
    print(f"   Consider: Stratified split or check for temporal attack clustering.\n")

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
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)  # Reduced weight_decay (was causing collapse)

# 3. FOCAL LOSS for severe imbalance
class FocalLoss(nn.Module):
    """
    Focal Loss to focus learning on hard-to-classify examples.
    Helps when model outputs near-zero probabilities for minority class.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights [normal_weight, attack_weight]
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=class_weights, gamma=2.0)

# Learning rate scheduler with warmup
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(
    optimizer,
    max_lr=LR,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)

# 2.5 RESUME (NEW)
start_epoch = 0
best_f1 = 0.0
best_threshold = 0.5  # Initialize best threshold
epochs_without_improvement = 0  # Early stopping counter

if os.path.exists(LAST_PATH):
    print("Found last checkpoint. Resuming...")
    ckpt = torch.load(LAST_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_f1 = float(ckpt.get("best_f1", 0.0))
    best_threshold = float(ckpt.get("best_threshold", 0.5))
    epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
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

        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # Step learning rate scheduler
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
    
    # Log to WandB if enabled
    if wandb_logger:
        try:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss / len(train_loader),
                "train/lr": optimizer.param_groups[0]['lr'],
                "val/f1": val_f1,
                "val/accuracy": val_acc,
                "val/threshold": best_t,
                "val/separation": p_attacks.mean() - p_normal.mean(),
            }
            if hasattr(wandb_logger, 'log'):
                wandb_logger.log(log_dict)
            else:
                wandb_logger.log(log_dict, step=epoch + 1)
        except Exception as e:
            pass  # Silently fail WandB logging

    # === WARNING: Detect suspicious threshold behavior ===
    # Calculate expected threshold range based on attack rate (Bayes optimal)
    # threshold ≈ log(P(attack) / (1-P(attack))) for balanced cost
    import math
    expected_threshold = math.log(val_attack_rate / (1 - val_attack_rate)) if 0 < val_attack_rate < 1 else 0.5
    expected_threshold = 1.0 / (1.0 + math.exp(-expected_threshold))  # Convert to probability scale

    # Check if separation is poor (indicates non-learning)
    separation = p_attacks.mean() - p_normal.mean()

    # Only warn if BOTH low threshold AND poor separation
    if best_t < 0.01 and separation < 0.05:
        print(f"🔴 CRITICAL: Threshold = {best_t:.4f} AND Separation = {separation:.4f} (very low)")
        print(f"    Model is NOT learning - attack/normal probabilities overlap almost completely")
        print(f"    Actions:")
        print(f"    1. Increase model capacity (d_model=256 in model.py)")
        print(f"    2. Train for more epochs")
        print(f"    3. Check data quality and labels")
    elif best_t < 0.01 and separation > 0.08:
        print(f"✅ INFO: Threshold = {best_t:.4f} is low but EXPECTED")
        print(f"    Separation = {separation:.4f} indicates model learning well")
        print(f"    Low threshold is correct for attack rate = {val_attack_rate*100:.2f}%")
    elif best_t > 0.95:
        print(f"⚠️  WARNING: Threshold = {best_t:.4f} is extremely high!")
        print(f"    The model rarely predicts attacks (may be biased toward majority class)")

    # Save best
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_threshold = best_t  # Update best threshold
        epochs_without_improvement = 0  # Reset counter
        torch.save(model.state_dict(), MODEL_PATH)
        print(f">>> Best model saved! (F1={best_f1:.4f}, Threshold={best_threshold:.4f})")
        
        # Save best model to WandB
        if wandb_logger:
            try:
                if hasattr(wandb_logger, 'save'):
                    wandb_logger.save(MODEL_PATH)
                else:
                    wandb_logger.save(MODEL_PATH)
            except Exception:
                pass  # Silently fail WandB save
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping triggered! No improvement for {EARLY_STOP_PATIENCE} epochs.")
            print(f"   Best F1: {best_f1:.4f} (Threshold: {best_threshold:.4f})")
            break

    # Always save "last" checkpoint for continuity (NEW)
    torch.save(
        {
            "epoch": epoch,
            "best_f1": best_f1,
            "best_threshold": best_threshold,  # Save threshold
            "epochs_without_improvement": epochs_without_improvement,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
        },
        LAST_PATH,
    )
    print(">>> Last checkpoint saved.")

print(f"Done. Best F1: {best_f1:.4f} | Best Threshold: {best_threshold:.4f}")
