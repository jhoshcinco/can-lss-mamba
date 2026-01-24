"""Basic tests for CAN-LSS-Mamba setup and environment."""

import os
import sys
from pathlib import Path


def test_directories_exist():
    """Test that required directories can be created."""
    # Test local directories
    test_dirs = ["./data", "./checkpoints", "./results"]
    for dir_path in test_dirs:
        os.makedirs(dir_path, exist_ok=True)
        assert os.path.exists(dir_path), f"Failed to create directory: {dir_path}"
    print("✅ Directory creation test passed")


def test_config_files_exist():
    """Test that configuration files exist."""
    config_files = [
        "configs/default.yaml",
        "configs/vastai.yaml",
        "configs/codespaces.yaml"
    ]
    for config_file in config_files:
        assert os.path.exists(config_file), f"Config file not found: {config_file}"
    print("✅ Config files test passed")


def test_imports():
    """Test that core dependencies can be imported."""
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
    except ImportError as e:
        assert False, f"Failed to import torch: {e}"
    
    try:
        import pandas
        print(f"  Pandas: {pandas.__version__}")
    except ImportError as e:
        assert False, f"Failed to import pandas: {e}"
    
    try:
        import sklearn
        print(f"  Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        assert False, f"Failed to import sklearn: {e}"
    
    try:
        import numpy
        print(f"  NumPy: {numpy.__version__}")
    except ImportError as e:
        assert False, f"Failed to import numpy: {e}"
    
    print("✅ Core imports test passed")


def test_mamba_import():
    """Test that mamba-ssm can be imported (optional)."""
    try:
        import mamba_ssm
        print(f"  Mamba-SSM: Installed ✓")
        print("✅ Mamba-SSM import test passed")
    except ImportError:
        print("⚠️  Mamba-SSM not installed (optional for testing)")


def test_config_loader():
    """Test that configuration loader works."""
    try:
        from src.config import load_config
        cfg = load_config("configs/default.yaml")
        assert cfg is not None, "Failed to load config"
        print("✅ Config loader test passed")
    except Exception as e:
        assert False, f"Config loader test failed: {e}"


def test_model_import():
    """Test that model can be imported."""
    try:
        from model import LSS_CAN_Mamba
        print("  LSS_CAN_Mamba model import successful")
        print("✅ Model import test passed")
    except Exception as e:
        assert False, f"Model import test failed: {e}"


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running CAN-LSS-Mamba Setup Tests")
    print("="*60)
    print()
    
    tests = [
        ("Directories", test_directories_exist),
        ("Config Files", test_config_files_exist),
        ("Core Imports", test_imports),
        ("Mamba Import", test_mamba_import),
        ("Config Loader", test_config_loader),
        ("Model Import", test_model_import),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        print("-" * 40)
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_name} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            failed += 1
    
    print()
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
