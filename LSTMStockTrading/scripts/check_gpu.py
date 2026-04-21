#!/usr/bin/env python3
"""
GPU Detection and Verification Script
Checks if PyTorch can detect CUDA/ROCm GPU
"""

import torch
import sys


def check_gpu():
    """Check GPU availability and print detailed information"""
    print("=" * 70)
    print("PyTorch GPU Detection (CUDA/ROCm)")
    print("=" * 70)

    # PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")

    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA/ROCm Available: {cuda_available}")

    if cuda_available:
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        # Detailed info for each GPU
        for i in range(torch.cuda.device_count()):
            print(f"\n{'=' * 70}")
            print(f"GPU {i} Details:")
            print(f"{'=' * 70}")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")

            props = torch.cuda.get_device_properties(i)
            print(f"  Total Memory: {props.total_memory / 1024 ** 3:.2f} GB")
            print(f"  Multi Processor Count: {props.multi_processor_count}")

            # Current device
            if i == torch.cuda.current_device():
                print(f"  Status: CURRENT DEVICE")

            # Memory info
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                print(f"  Memory Allocated: {allocated:.2f} GB")
                print(f"  Memory Reserved: {reserved:.2f} GB")

        # ROCm specific info
        print(f"\n{'=' * 70}")
        print("Backend Information:")
        print(f"{'=' * 70}")
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            print(f"  Backend: ROCm")
            print(f"  ROCm Version: {torch.version.hip}")
        else:
            print(f"  Backend: CUDA")
            if torch.version.cuda:
                print(f"  CUDA Version: {torch.version.cuda}")

        # cuDNN
        if torch.backends.cudnn.is_available():
            print(f"  cuDNN Available: Yes")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")

        # Test tensor operation on GPU
        print(f"\n{'=' * 70}")
        print("GPU Functionality Test:")
        print(f"{'=' * 70}")
        try:
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print("  Matrix Multiplication Test: PASSED ✓")
            print(f"  Result shape: {z.shape}")
            print(f"  Device: {z.device}")
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Matrix Multiplication Test: FAILED ✗")
            print(f"  Error: {e}")
            return False
    else:
        print("\nNo GPU detected. Training will use CPU.")
        print("\nPossible reasons:")
        print("  1. ROCm/CUDA drivers not installed")
        print("  2. PyTorch not compiled with ROCm/CUDA support")
        print("  3. GPU not visible to the current environment")
        print("\nTo install PyTorch with ROCm support:")
        print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    if cuda_available:
        print(f"✓ GPU is available and functional for training")
        print(f"✓ {torch.cuda.device_count()} GPU(s) detected")
        print(f"✓ Primary GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print(f"✗ No GPU available - will use CPU")
        return False


if __name__ == "__main__":
    gpu_available = check_gpu()
    sys.exit(0 if gpu_available else 1)