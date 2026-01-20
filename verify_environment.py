# test the new environment to see if it works

#write docker files, requirements.txt so it's a one run thing. Also cross platform compatibility
# you need torch,pandas

import sys
import os


def check_environment():
    print(f"--- Environment Check ---")
    print(f"Python Executable: {sys.executable}")

    # 1. Check PyTorch Installation
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__} (SUCCESS)")
    except ImportError:
        print("CRITICAL: PyTorch is NOT installed in this interpreter.")
        return

    # 2. Check CUDA (GPU) Availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("CRITICAL: CUDA is not available. You are running on CPU only!")
        print("Possible fix: Switch PyCharm to 'System Interpreter' or verify NVIDIA drivers.")
        return

    # 3. Verify GPU Hardware
    device_count = torch.cuda.device_count()
    print(f"GPU Count: {device_count}")

    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
        # Simple memory check to ensure we can talk to the driver
        try:
            t = torch.tensor([1.0, 2.0]).cuda()
            print(f"  -> Tensor allocation test on GPU {i}: PASSED")
        except Exception as e:
            print(f"  -> Tensor allocation test on GPU {i}: FAILED ({e})")

    # 4. Check Mamba-SSM (Specific to your project)
    try:
        import mamba_ssm
        print(f"Mamba-SSM: Installed (SUCCESS)")
    except ImportError:
        print("WARNING: 'mamba_ssm' is missing. Run 'pip install mamba-ssm' or check your path.")

    print("--- Check Complete ---")


if __name__ == "__main__":
    check_environment()