#!/usr/bin/env python3
"""
Test script for baseline comparisons functionality.

This script performs basic validation of the baseline models implementation
without requiring actual training data or GPU resources.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 80)
print("BASELINE COMPARISONS - VALIDATION TEST")
print("=" * 80)

# Test 1: Check if baseline models can be imported
print("\n[Test 1] Checking baseline models import...")
try:
    from models.baselines import get_baseline_model, BASELINE_MODELS
    print(f"✅ Successfully imported baseline models")
    print(f"   Available models: {list(BASELINE_MODELS.keys())}")
except Exception as e:
    print(f"❌ Failed to import baseline models: {e}")
    sys.exit(1)

# Test 2: Check if LSS-CAN-Mamba can be imported
print("\n[Test 2] Checking LSS-CAN-Mamba import...")
try:
    from models.lss_mamba import LSS_CAN_Mamba
    print(f"✅ Successfully imported LSS-CAN-Mamba")
except Exception as e:
    print(f"❌ Failed to import LSS-CAN-Mamba: {e}")
    sys.exit(1)

# Test 3: Check if scripts exist
print("\n[Test 3] Checking if scripts exist...")
scripts_to_check = [
    "scripts/train_baselines.py",
    "scripts/generate_baseline_comparisons.py",
]

all_exist = True
for script in scripts_to_check:
    script_path = Path(__file__).parent.parent / script
    if script_path.exists():
        print(f"✅ {script} exists")
    else:
        print(f"❌ {script} not found")
        all_exist = False

if not all_exist:
    sys.exit(1)

# Test 4: Check documentation
print("\n[Test 4] Checking documentation...")
docs_to_check = [
    "docs/baseline_comparisons.md",
    "docs/BASELINE_COMPARISONS_QUICKREF.md",
]

all_docs_exist = True
for doc in docs_to_check:
    doc_path = Path(__file__).parent.parent / doc
    if doc_path.exists():
        print(f"✅ {doc} exists")
    else:
        print(f"❌ {doc} not found")
        all_docs_exist = False

if not all_docs_exist:
    sys.exit(1)

# Test 5: Verify model classes can be instantiated (with mock parameters)
print("\n[Test 5] Testing model instantiation...")
try:
    import torch
    import torch.nn as nn
    
    # Test parameters
    num_unique_ids = 100
    seq_len = 50
    
    for model_name in BASELINE_MODELS.keys():
        try:
            model = get_baseline_model(
                model_name,
                num_unique_ids=num_unique_ids,
                num_continuous_feats=9,
                d_model=256,
                seq_len=seq_len
            )
            
            # Test forward pass with dummy data
            dummy_ids = torch.randint(0, num_unique_ids, (2, seq_len))
            dummy_feats = torch.randn(2, seq_len, 9)
            
            with torch.no_grad():
                output = model(dummy_ids, dummy_feats)
            
            assert output.shape == (2, 2), f"Expected output shape (2, 2), got {output.shape}"
            
            print(f"✅ {model_name}: Successfully instantiated and tested")
        except Exception as e:
            print(f"❌ {model_name}: Failed - {e}")
            raise

except ImportError:
    print("⚠️  PyTorch not installed - skipping instantiation test")
    print("   This test requires PyTorch to be installed")
except Exception as e:
    print(f"❌ Model instantiation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL VALIDATION TESTS PASSED!")
print("=" * 80)
print("\nBaseline comparisons implementation is ready to use.")
print("\nNext steps:")
print("  1. Ensure PyTorch and dependencies are installed")
print("  2. Preprocess your data")
print("  3. Run: python scripts/generate_baseline_comparisons.py --dataset set_01")
print()
