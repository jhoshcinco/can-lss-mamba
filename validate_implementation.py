#!/usr/bin/env python
"""
Validate that the implementation is complete and backwards compatible.
This script checks file structure and configuration without requiring dependencies.
"""

import os
import sys
from pathlib import Path


def check_file_exists(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} missing: {path}")
        return False


def check_dir_exists(path, description):
    """Check if a directory exists."""
    if os.path.isdir(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} missing: {path}")
        return False


def validate_implementation():
    """Validate the complete implementation."""
    print("="*60)
    print("CAN-LSS-Mamba Implementation Validation")
    print("="*60)
    print()
    
    all_good = True
    
    # Core files
    print("1. Core Files:")
    files = [
        ("requirements.txt", "Dependencies file"),
        ("README.md", "Main documentation"),
        ("TERMINAL_QUICKSTART.md", "Terminal quick start guide"),
        (".env.example", "Environment template"),
        (".gitignore", "Git ignore rules"),
        ("setup.sh", "Setup script"),
        ("Dockerfile", "Docker configuration"),
        ("docker-compose.yml", "Docker compose"),
    ]
    for file, desc in files:
        all_good &= check_file_exists(file, desc)
    print()
    
    # Configuration files
    print("2. Configuration Files:")
    configs = [
        ("configs/default.yaml", "Default config"),
        ("configs/vastai.yaml", "vast.ai config"),
        ("configs/codespaces.yaml", "Codespaces config"),
    ]
    for config, desc in configs:
        all_good &= check_file_exists(config, desc)
    print()
    
    # Original scripts (backwards compatibility)
    print("3. Original Scripts (Backwards Compatibility):")
    scripts = [
        ("train.py", "Training script"),
        ("evaluate.py", "Evaluation script"),
        ("model.py", "Model definition"),
        ("preprocessing/CAN_preprocess.py", "Preprocessing script"),
        ("verify_environment.py", "Environment check"),
    ]
    for script, desc in scripts:
        all_good &= check_file_exists(script, desc)
    print()
    
    # New modular structure
    print("4. Modular Structure:")
    dirs = [
        ("src", "Source package"),
        ("src/data", "Data module"),
        ("src/models", "Models module"),
        ("src/training", "Training module"),
        ("scripts", "Wrapper scripts"),
        ("notebooks", "Jupyter notebooks"),
        ("tests", "Test suite"),
    ]
    for dir_path, desc in dirs:
        all_good &= check_dir_exists(dir_path, desc)
    print()
    
    # New wrapper scripts
    print("5. Wrapper Scripts (Config Support):")
    wrapper_scripts = [
        ("scripts/preprocess.py", "Preprocessing wrapper"),
        ("scripts/train.py", "Training wrapper"),
        ("scripts/evaluate.py", "Evaluation wrapper"),
    ]
    for script, desc in wrapper_scripts:
        all_good &= check_file_exists(script, desc)
    print()
    
    # Utilities
    print("6. Utility Modules:")
    utils = [
        ("src/config.py", "Config loader"),
        ("src/training/wandb_logger.py", "WandB logger"),
    ]
    for util, desc in utils:
        all_good &= check_file_exists(util, desc)
    print()
    
    # Tests
    print("7. Tests:")
    tests = [
        ("tests/test_setup.py", "Setup tests"),
    ]
    for test, desc in tests:
        all_good &= check_file_exists(test, desc)
    print()
    
    # Check that original scripts have WandB support
    print("8. Feature Integration:")
    try:
        with open("train.py", "r") as f:
            train_content = f.read()
            # Check for specific WandB integration patterns
            has_wandb_import = "import wandb" in train_content or "wandb_logger" in train_content
            has_wandb_log = "wandb.log(" in train_content or "wandb_logger.log(" in train_content
            
            if has_wandb_import and has_wandb_log:
                print("✓ WandB integration added to train.py")
            else:
                print("✗ WandB integration missing or incomplete in train.py")
                all_good = False
    except Exception as e:
        print(f"✗ Could not verify train.py: {e}")
        all_good = False
    print()
    
    # Summary
    print("="*60)
    if all_good:
        print("✅ All implementation requirements met!")
        print()
        print("Next steps:")
        print("  1. Run: bash setup.sh")
        print("  2. Test: python verify_environment.py")
        print("  3. Review: cat TERMINAL_QUICKSTART.md")
    else:
        print("❌ Some files are missing. Please complete the implementation.")
    print("="*60)
    
    return all_good


if __name__ == "__main__":
    success = validate_implementation()
    sys.exit(0 if success else 1)
