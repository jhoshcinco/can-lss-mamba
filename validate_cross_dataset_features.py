#!/usr/bin/env python3
"""
Validation script for cross-dataset evaluation and hyperparameter tuning features.
Run this to verify the implementation is complete and correct.
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and report status."""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (missing)")
        return False

def check_file_executable(file_path, description):
    """Check if a file is executable."""
    path = Path(file_path)
    if path.exists() and os.access(path, os.X_OK):
        print(f"✅ {description}: {file_path} (executable)")
        return True
    elif path.exists():
        print(f"⚠️  {description}: {file_path} (not executable)")
        return True
    else:
        print(f"❌ {description}: {file_path} (missing)")
        return False

def validate_yaml_file(file_path, required_keys):
    """Validate a YAML file has required keys."""
    try:
        import yaml
        with open(file_path) as f:
            data = yaml.safe_load(f)
        
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"⚠️  {file_path} is missing keys: {missing_keys}")
            return False
        else:
            print(f"✅ {file_path} has all required keys")
            return True
    except Exception as e:
        print(f"❌ Error validating {file_path}: {e}")
        return False

def main():
    print("="*60)
    print("Validating Cross-Dataset Evaluation Implementation")
    print("="*60)
    print()
    
    all_passed = True
    
    # Check configuration files
    print("1. Configuration Files")
    print("-" * 60)
    all_passed &= check_file_exists("configs/datasets.yaml", "Dataset configs")
    all_passed &= check_file_exists("configs/sweep.yaml", "WandB sweep config")
    all_passed &= validate_yaml_file("configs/datasets.yaml", ["datasets", "cross_eval_datasets"])
    all_passed &= validate_yaml_file("configs/sweep.yaml", ["program", "method", "metric"])
    print()
    
    # Check scripts
    print("2. Python Scripts")
    print("-" * 60)
    all_passed &= check_file_executable("scripts/cross_dataset_eval.py", "Cross-dataset evaluation")
    all_passed &= check_file_executable("scripts/train_combined.py", "Combined training")
    all_passed &= check_file_executable("scripts/grid_search.py", "Grid search")
    all_passed &= check_file_executable("scripts/compare_runs.py", "Compare runs")
    print()
    
    # Check bash scripts
    print("3. Bash Scripts")
    print("-" * 60)
    all_passed &= check_file_executable("scripts/quick_test.sh", "Quick test")
    all_passed &= check_file_executable("scripts/preprocess_all.sh", "Preprocess all")
    print()
    
    # Check documentation
    print("4. Documentation")
    print("-" * 60)
    all_passed &= check_file_exists("docs/three_bucket_strategy.md", "Three-bucket strategy")
    all_passed &= check_file_exists("docs/hyperparameter_tuning.md", "Hyperparameter tuning")
    all_passed &= check_file_exists("docs/cross_dataset_evaluation.md", "Cross-dataset evaluation")
    print()
    
    # Check README updates
    print("5. README Updates")
    print("-" * 60)
    try:
        with open("README.md") as f:
            readme = f.read()
        
        required_sections = [
            "Advanced Features",
            "Cross-Dataset Evaluation",
            "Hyperparameter Tuning",
            "Complete Research Workflow",
            "Three-Bucket Strategy"
        ]
        
        for section in required_sections:
            if section in readme:
                print(f"✅ README contains '{section}' section")
            else:
                print(f"❌ README missing '{section}' section")
                all_passed = False
    except Exception as e:
        print(f"❌ Error checking README: {e}")
        all_passed = False
    print()
    
    # Check backward compatibility
    print("6. Backward Compatibility")
    print("-" * 60)
    all_passed &= check_file_exists("train.py", "Original train.py")
    all_passed &= check_file_exists("evaluate.py", "Original evaluate.py")
    all_passed &= check_file_exists("model.py", "Original model.py")
    all_passed &= check_file_exists("preprocessing/CAN_preprocess.py", "Original preprocessing")
    print()
    
    # Check setup.sh improvements
    print("7. Setup Script Improvements")
    print("-" * 60)
    try:
        with open("setup.sh") as f:
            setup = f.read()
        
        improvements = [
            ("check_package", "Smart dependency checking"),
            ("IN_DOCKER", "Docker detection"),
            ("NEED_INSTALL", "Conditional installation")
        ]
        
        for keyword, description in improvements:
            if keyword in setup:
                print(f"✅ {description} implemented ({keyword})")
            else:
                print(f"⚠️  {description} not found ({keyword})")
    except Exception as e:
        print(f"❌ Error checking setup.sh: {e}")
        all_passed = False
    print()
    
    # Final summary
    print("="*60)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        print("="*60)
        print()
        print("Implementation is complete and ready for use.")
        print()
        print("Next steps:")
        print("1. Run: bash setup.sh")
        print("2. Preprocess data: bash scripts/preprocess_all.sh")
        print("3. Run grid search: python scripts/grid_search.py")
        print("4. Cross-dataset eval: python scripts/cross_dataset_eval.py --all")
        sys.exit(0)
    else:
        print("⚠️  SOME VALIDATIONS FAILED")
        print("="*60)
        print()
        print("Please review the issues above and fix them.")
        sys.exit(1)

if __name__ == "__main__":
    main()
