#!/usr/bin/env python3
"""
Run XAI bounds experiment on a single specified model.
Usage: python run_single_model.py <model_name>
Example: python run_single_model.py cnn3_mnist.pt
"""

import sys
from pathlib import Path
from run_all_models import run_experiment


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_single_model.py <model_name> [k_sparse] [num_samples] [eps_fav]")
        print("\nArguments:")
        print("  model_name   : Name of the model file (required)")
        print("  k_sparse     : Sparsity budget for Sparse-PGD (default: 50)")
        print("  num_samples  : Number of test samples (default: 10)")
        print("  eps_fav      : FAVEX radius / upper bound for eps_min (default: 0.25)")
        print("\nExamples:")
        print("  python run_single_model.py cnn3_mnist.pt")
        print("  python run_single_model.py cnn3_mnist.pt 100")
        print("  python run_single_model.py cnn3_mnist.pt 100 20")
        print("  python run_single_model.py cnn3_cifar10.pt 50 10 0.031  # 8/255 for CIFAR-10")
        print("\nAvailable models:")
        models_dir = Path("./models")
        for model in sorted(models_dir.glob("*.pt")):
            print(f"  - {model.name}")
        sys.exit(1)

    model_name = sys.argv[1]
    model_path = Path("./models") / model_name

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Parse optional parameters
    k_sparse = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    eps_fav = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25

    print(f"\nRunning experiment on: {model_name}")
    print(f"Parameters: k={k_sparse}, num_samples={num_samples}, eps_fav={eps_fav}\n")

    run_experiment(
        model_path=str(model_path),
        k_sparse=k_sparse,
        eps_fav=eps_fav,
        num_samples=num_samples
    )


if __name__ == "__main__":
    main()
