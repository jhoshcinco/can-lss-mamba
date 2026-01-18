import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import time
import os
from model import LSS_CAN_Mamba  # Imports from the file above

# --- CONFIG ---
DATA_DIR = "processed_data/set_01"
MODEL_PATH = "lss_can_mamba_best.pth"
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# 1. LOAD DATA
print("Loading data...")
train_npz = np.load(os.path.join(DATA_DIR, "train_data.npz"))
val_npz = np.load(os.path.join(DATA_DIR, "val_data.npz"))
id_map = np.load(os.path.join(DATA_DIR, "id_map.npy"), allow_pickle=True).item()
vocab_size = len(id_map) + 1


def get_loader(npz, shuffle=True):
    # Combine Payload (8) and Delta (1) -> 9 Features
    feats = np.concatenate([npz['payloads'], npz['deltas']], axis=-1)
    dataset = TensorDataset(
        torch.LongTensor(npz['ids']),
        torch.FloatTensor(feats),
        torch.LongTensor(npz['labels'])
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = get_loader(train_npz, shuffle=True)
val_loader = get_loader(val_npz, shuffle=False)
print(f"Data Loaded. Vocab Size: {vocab_size}")

# 2. INIT MODEL
model = LSS_CAN_Mamba(num_unique_ids=vocab_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# 3. TRAIN LOOP
best_f1 = 0
print(f"Starting Training ({EPOCHS} Epochs)...")

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    train_loss = 0

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
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    val_acc = accuracy_score(all_labels, all_preds)

    print(
        f"Epoch {epoch + 1} | Loss: {train_loss / len(train_loader):.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time() - start:.1f}s")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), MODEL_PATH)
        print(">>> Model Saved!")

print(f"Done. Best F1: {best_f1:.4f}")