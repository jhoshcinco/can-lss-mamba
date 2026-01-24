# Verify environment for CAN-LSS-Mamba
# This script checks that all dependencies are properly installed

import sys
import os


def check_environment():
    print(f"{'='*60}")
    print(f"CAN-LSS-Mamba Environment Verification")
    print(f"{'='*60}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    all_good = True

    # 1. Check PyTorch Installation
    try:
        import torch
        print(f"✓ PyTorch Version: {torch.__version__}")
    except ImportError:
        print("✗ CRITICAL: PyTorch is NOT installed.")
        print("  Fix: pip install torch")
        all_good = False

    # 2. Check CUDA (GPU) Availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA Available: Yes")
            device_count = torch.cuda.device_count()
            print(f"✓ GPU Count: {device_count}")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  - GPU {i}: {gpu_name}")
                # Simple memory check
                try:
                    t = torch.tensor([1.0, 2.0]).cuda(i)
                    print(f"    → Allocation test: PASSED")
                except Exception as e:
                    print(f"    → Allocation test: FAILED ({e})")
                    all_good = False
        else:
            print("⚠️  CUDA Available: No (CPU-only mode)")
            print("  Note: Training will be very slow on CPU")
            print("  Check: nvidia-smi to verify GPU drivers")
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        all_good = False

    # 3. Check other required packages
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
    ]
    
    for import_name, package_name in required_packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name}: {version}")
        except ImportError:
            print(f"✗ {package_name} is NOT installed")
            print(f"  Fix: pip install {package_name}")
            all_good = False

    # 4. Check Mamba-SSM (Specific to your project)
    try:
        import mamba_ssm
        print(f"✓ Mamba-SSM: Installed")
    except ImportError:
        print("✗ WARNING: 'mamba_ssm' is missing")
        print("  Fix: pip install mamba-ssm")
        print("  Note: This is required for training")
        all_good = False

    # 5. Check optional packages
    optional_packages = [
        ('omegaconf', 'omegaconf', 'for config file support'),
        ('wandb', 'wandb', 'for experiment tracking'),
        ('dotenv', 'python-dotenv', 'for .env file support'),
    ]
    
    print(f"\nOptional packages:")
    for import_name, package_name, purpose in optional_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} ({purpose})")
        except ImportError:
            print(f"○ {package_name} not installed ({purpose})")

    # 6. Check directory structure
    print(f"\nDirectory structure:")
    important_dirs = ['configs', 'src', 'scripts', 'preprocessing']
    for dir_name in important_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ exists")
        else:
            print(f"○ {dir_name}/ not found (may not be needed)")

    print(f"{'='*60}")
    if all_good:
        print("✅ Environment is properly configured!")
        print("You can now run: python train.py")
    else:
        print("❌ Some issues detected. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
    print(f"{'='*60}")
    
    return all_good


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)