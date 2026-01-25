# Testing Documentation: Checkpoint Isolation Fixes

This document demonstrates that the checkpoint reuse bugs have been fixed.

## Test 1: Grid Search Checkpoint Isolation

The `scripts/grid_search.py` now generates unique checkpoint directories for each hyperparameter configuration:

### Example Configuration Mappings:

```python
Config 1: batch_size=32, lr=0.0001, epochs=20, dropout=0.0
→ /workspace/checkpoints/set_01/grid_bs32_lr0.0001_ep20_drop0.0

Config 2: batch_size=64, lr=0.0005, epochs=30, dropout=0.0
→ /workspace/checkpoints/set_01/grid_bs64_lr0.0005_ep30_drop0.0

Config 3: batch_size=32, lr=0.0001, epochs=20, dropout=0.1
→ /workspace/checkpoints/set_01/grid_bs32_lr0.0001_ep20_drop0.1
```

### Key Changes:
1. Added `get_checkpoint_dir()` function that generates unique directory names based on all hyperparameters
2. Each configuration uses its own unique `OUT_DIR` environment variable
3. WandB run names now include hyperparameters for easy identification

### Verification:
```bash
python scripts/grid_search.py --batch-sizes 32,64 --learning-rates 0.0001,0.0005 --epochs 20
```

Expected behavior: Each of the 4 configurations trains from scratch in separate directories.

## Test 2: Quick Test Checkpoint Isolation

The `scripts/quick_test.sh` now generates unique checkpoint directories for each learning rate:

### Example Directory Mappings:

```bash
LR=0.0001 → /workspace/checkpoints/quick_test_lr0.0001
LR=0.0005 → /workspace/checkpoints/quick_test_lr0.0005
LR=0.001  → /workspace/checkpoints/quick_test_lr0.001
```

### Key Changes:
1. Added unique `OUT_DIR` environment variable per learning rate
2. Each learning rate test starts from epoch 1 with fresh model weights

### Verification:
```bash
bash scripts/quick_test.sh
```

Expected behavior: Each learning rate trains from scratch in a separate directory.

## Test 3: Cross-Dataset Evaluation Checkpoint Isolation

The `scripts/cross_dataset_eval.py` now uses explicit unique checkpoint directories:

### Example Directory Mappings:

```bash
Train on set_01 → /workspace/checkpoints/cross_eval_train_set_01
Train on set_02 → /workspace/checkpoints/cross_eval_train_set_02
Train on set_03 → /workspace/checkpoints/cross_eval_train_set_03
```

### Key Changes:
1. Changed from `/workspace/checkpoints/{dataset_name}` to `/workspace/checkpoints/cross_eval_train_{dataset_name}`
2. More explicit naming to prevent conflicts with other training runs

### Verification:
```bash
python scripts/cross_dataset_eval.py --train-dataset set_01
```

Expected behavior: Training uses the unique cross_eval directory.

## Test 4: WandB File Saving Without Warnings

The `train.py` and `wandb_logger.py` now properly use `base_path` when saving checkpoints to WandB.

### Key Changes:

**In train.py:**
```python
# When saving best model
checkpoint_base = os.path.dirname(MODEL_PATH)
wandb_logger.save(MODEL_PATH, base_path=checkpoint_base)

# When saving last checkpoint
checkpoint_base = os.path.dirname(LAST_PATH)
wandb_logger.save(LAST_PATH, base_path=checkpoint_base)
```

**In wandb_logger.py:**
```python
def save(self, path: str, base_path: Optional[str] = None):
    if self.enabled and self.wandb is not None:
        try:
            self.wandb.save(path, base_path=base_path)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save to WandB: {e}")
```

### Verification:
```bash
OUT_DIR=/workspace/checkpoints/wandb_test \
WANDB_ENABLED=true EPOCHS=1 python train.py 2>&1 | grep -i "wandb warning"
```

Expected behavior: No warnings about "Saving files without folders"

## Test 5: Validation Script

Created `scripts/validate_hyperparameter_isolation.py` to automatically test checkpoint isolation:

### What it tests:
1. Different configs start from epoch 1 (not resumed)
2. Checkpoints saved to correct unique directories
3. No cross-contamination between configs

### Usage:
```bash
python scripts/validate_hyperparameter_isolation.py
```

Expected output:
```
Testing checkpoint isolation...
============================================================

Config 1: LR=0.0001, BS=32
Checkpoint directory: /tmp/test_config1
  ✅ Config 1 started from epoch 1 (correct)
  ✅ Checkpoint saved to: /tmp/test_config1

Config 2: LR=0.0005, BS=32
Checkpoint directory: /tmp/test_config2
  ✅ Config 2 started from epoch 1 (correct)
  ✅ Checkpoint saved to: /tmp/test_config2

============================================================
✅ All tests passed! Checkpoint isolation is working correctly.
```

## Test 6: Documentation Updates

Updated documentation to warn users about checkpoint isolation:

### docs/hyperparameter_tuning.md
- Added prominent "⚠️ CRITICAL: Checkpoint Isolation" section at the top
- Clear examples of wrong vs. correct approaches
- Validation instructions

### README.md
- Updated hyperparameter tuning section with checkpoint isolation warnings
- Added examples showing unique OUT_DIR usage
- Emphasized using provided scripts for automatic handling

## Summary

All critical bugs have been fixed:

✅ **Bug 1 Fixed**: Each hyperparameter configuration now uses a unique checkpoint directory
  - grid_search.py: Generates unique dirs based on all hyperparameters
  - quick_test.sh: Unique dir per learning rate
  - cross_dataset_eval.py: Unique dir per dataset

✅ **Bug 2 Fixed**: WandB file saving now uses base_path parameter
  - train.py: Passes base_path when saving best and last checkpoints
  - wandb_logger.py: Accepts and uses base_path parameter

✅ **Documentation Updated**: Clear warnings and examples added
  - Hyperparameter tuning guide has critical warning section
  - README has updated examples
  - Validation script provided

✅ **Backwards Compatible**: All existing workflows continue to work
  - Default behavior unchanged for single training runs
  - Scripts handle checkpoint isolation automatically
