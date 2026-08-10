#!/usr/bin/env python3
"""
Quick test to verify the environment setup is correct.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    errors = []

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"✗ PyTorch: {e}")

    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError as e:
        errors.append(f"✗ torchvision: {e}")

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"✗ NumPy: {e}")

    try:
        from autoattack import AutoAttack
        print(f"✓ AutoAttack")
    except ImportError as e:
        errors.append(f"✗ AutoAttack: {e}")

    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"✗ pandas: {e}")

    try:
        from utils import get_eps_min, find_smallest_eps_spgd
        print(f"✓ utils.py (get_eps_min, find_smallest_eps_spgd)")
    except ImportError as e:
        errors.append(f"✗ utils.py: {e}")

    try:
        from spgd import SparsePGD
        print(f"✓ spgd.py (SparsePGD)")
    except ImportError as e:
        errors.append(f"✗ spgd.py: {e}")

    return errors


def test_models():
    """Test that model files exist."""
    print("\nChecking models directory...")
    models_dir = Path("./models")

    if not models_dir.exists():
        print("✗ models/ directory not found")
        return False

    model_files = list(models_dir.glob("*.pt"))
    if not model_files:
        print("✗ No .pt model files found in models/")
        return False

    print(f"✓ Found {len(model_files)} model files:")
    for model_file in sorted(model_files):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name} ({size_mb:.1f} MB)")

    return True


def test_scripts():
    """Test that required scripts exist."""
    print("\nChecking scripts...")
    required_scripts = [
        "run_all_models.py",
        "run_single_model.py",
        "compare_results.py",
        "utils.py",
        "spgd.py",
        "run.sh"
    ]

    missing = []
    for script in required_scripts:
        if Path(script).exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} not found")
            missing.append(script)

    return len(missing) == 0


def test_cuda():
    """Test CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available (will use CPU)")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")


def main():
    """Run all tests."""
    print("="*70)
    print("Environment Setup Test")
    print("="*70 + "\n")

    # Test imports
    import_errors = test_imports()

    # Test models
    models_ok = test_models()

    # Test scripts
    scripts_ok = test_scripts()

    # Test CUDA
    test_cuda()

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    if import_errors:
        print("\n❌ FAILED: Some packages are missing:")
        for error in import_errors:
            print(f"  {error}")
        print("\nFix: Make sure you activated the virtual environment:")
        print("  source .venv/bin/activate")
        sys.exit(1)

    if not models_ok:
        print("\n⚠ WARNING: Models directory issue")
        print("Make sure you have downloaded the model files to ./models/")

    if not scripts_ok:
        print("\n⚠ WARNING: Some scripts are missing")

    if import_errors or not models_ok or not scripts_ok:
        print("\n❌ Setup is INCOMPLETE")
        sys.exit(1)
    else:
        print("\n✅ Setup is COMPLETE and ready to run experiments!")
        print("\nNext steps:")
        print("  1. ./run.sh list                    # List available models")
        print("  2. ./run.sh single cnn3_mnist.pt    # Test one model")
        print("  3. ./run.sh all                     # Run all experiments")
        print("  4. ./run.sh compare                 # Compare results")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
