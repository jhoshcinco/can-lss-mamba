import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import time
import os
import argparse
from .models.mamba import LSS_CAN_Mamba


def train(
    data_dir,
    out_dir,
    model_name="lss_can_mamba",
    batch_size=32,
    epochs=20,
    lr=1e-4,
    wandb_enabled=False,
    wandb_project="can-lss-mamba",
    early_stop_patience=10,
    id_dropout_prob=0.0,
):
    # Setup directories
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{model_name}_best.pth")
    last_path = os.path.join(out_dir, f"{model_name}_last.pth")

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    print(f"DATA_DIR: {data_dir}")
    print(f"OUT_DIR : {out_dir}")
    print(f"BATCH_SIZE={batch_size} EPOCHS={epochs} LR={lr}")

    # WandB Setup
    wandb_logger = None
    if wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                config={
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "model_name": model_name,
                },
            )
            wandb_logger = wandb
            print("✅ WandB initialized")
        except ImportError:
            print("⚠️ WandB enabled but not installed. Skipping.")

    # 1. LOAD DATA
    print("Loading data...")
    try:
        train_npz = np.load(os.path.join(data_dir, "train_data.npz"))
        val_npz = np.load(os.path.join(data_dir, "val_data.npz"))
        id_map = np.load(
            os.path.join(data_dir, "id_map.npy"), allow_pickle=True
        ).item()
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return

    vocab_size = len(id_map)

    def get_loader(npz, shuffle=True, id_dropout_prob=0.0):
        # Combine Payload (8) and Delta (1) -> 9 Features
        feats = np.concatenate([npz["payloads"], npz["deltas"]], axis=-1)

        ids = npz["ids"].copy()
        if id_dropout_prob > 0 and shuffle:
            unk_token = id_map.get("<UNK>", vocab_size - 1)
            dropout_mask = np.random.rand(len(ids)) < id_dropout_prob
            ids[dropout_mask] = unk_token

        dataset = TensorDataset(
            torch.LongTensor(ids),
            torch.FloatTensor(feats),
            torch.LongTensor(npz["labels"]),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = get_loader(train_npz, shuffle=True, id_dropout_prob=id_dropout_prob)
    val_loader = get_loader(val_npz, shuffle=False, id_dropout_prob=0.0)

    # 2. INIT MODEL
    # Calculate Class Weights
    y_train_indices = train_npz["labels"]
    num_normal = len(y_train_indices) - np.sum(y_train_indices)
    num_attacks = np.sum(y_train_indices)
    pos_weight_val = num_normal / (num_attacks + 1e-6)
    pos_weight_val = min(pos_weight_val, 10.0)
    print(f"Calculated Class Weight (Pos_Weight): {pos_weight_val:.2f}")

    class_weights = torch.tensor(
        [1.0, pos_weight_val], device=device, dtype=torch.float32
    )

    model = LSS_CAN_Mamba(num_unique_ids=vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, logits, targets):
            ce_loss = nn.functional.cross_entropy(
                logits, targets, reduction="none", weight=self.alpha
            )
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    from torch.optim.lr_scheduler import OneCycleLR

    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Resume Logic
    start_epoch = 0
    best_f1 = 0.0
    best_threshold = 0.5
    epochs_without_improvement = 0

    if os.path.exists(last_path):
        print("Found last checkpoint. Resuming...")
        ckpt = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_f1 = float(ckpt.get("best_f1", 0.0))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        print(
            f"Resumed at epoch {start_epoch} with best_f1={best_f1:.4f}, threshold={best_threshold:.4f}"
        )
    else:
        print("No last checkpoint found. Starting fresh.")

    # Threshold Helper
    def best_f1_threshold(y_true, p_attack):
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            y_pred = (p_attack >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return best_t, best_f1

    # 3. TRAIN LOOP
    print(f"Starting Training ({epochs} Epochs)...")

    for epoch in range(start_epoch, epochs):
        start = time.time()
        model.train()
        train_loss = 0.0

        for ids, feats, labels in train_loader:
            ids, feats, labels = ids.to(device), feats.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(ids, feats)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for ids, feats, labels in val_loader:
                ids, feats, labels = ids.to(device), feats.to(device), labels.to(device)
                logits = model(ids, feats)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Metrics
        p_attack = np.array(all_preds)
        y_true = np.array(all_labels)

        best_t, val_f1 = best_f1_threshold(y_true, p_attack)
        val_acc = accuracy_score(y_true, (p_attack >= best_t).astype(int))

        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {train_loss / len(train_loader):.4f} | "
            f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | "
            f"Thr: {best_t:.2f} | "
            f"Time: {time.time() - start:.1f}s"
        )

        if wandb_logger:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss / len(train_loader),
                "train/lr": optimizer.param_groups[0]["lr"],
                "val/f1": val_f1,
                "val/accuracy": val_acc,
                "val/threshold": best_t,
            }
            wandb_logger.log(log_dict)

        # Save Best
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_threshold = best_t
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print(f">>> Best model saved! (F1={best_f1:.4f})")
            if wandb_logger:
                wandb_logger.save(model_path, base_path=os.path.dirname(model_path))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(
                    f"\n🛑 Early stopping triggered! No improvement for {early_stop_patience} epochs."
                )
                break

        # Save Last
        torch.save(
            {
                "epoch": epoch,
                "best_f1": best_f1,
                "best_threshold": best_threshold,
                "epochs_without_improvement": epochs_without_improvement,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
            },
            last_path,
        )
        print(">>> Last checkpoint saved.")

    print(f"Done. Best F1: {best_f1:.4f} | Best Threshold: {best_threshold:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CAN LSS Mamba Model")
    parser.add_argument(
        "--data-dir", default=os.environ.get("DATA_DIR", "/workspace/data/processed_data/set_01_run_02")
    )
    parser.add_argument(
        "--out-dir", default=os.environ.get("OUT_DIR", "/workspace/checkpoints/set_01")
    )
    parser.add_argument("--batch-size", type=int, default=os.environ.get("BATCH_SIZE", 32))
    parser.add_argument("--epochs", type=int, default=os.environ.get("EPOCHS", 20))
    parser.add_argument("--lr", type=float, default=os.environ.get("LR", 1e-4))
    parser.add_argument("--wandb", action="store_true", default=os.environ.get("WANDB_ENABLED", "false").lower() == "true")

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        wandb_enabled=args.wandb,
    )
