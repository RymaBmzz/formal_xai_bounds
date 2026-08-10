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
        print("Usage: python run_single_model.py <model_name>")
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

    print(f"\nRunning experiment on: {model_name}\n")

    run_experiment(
        model_path=str(model_path),
        k_sparse=50,
        eps_fav=0.25,
        num_samples=10
    )


if __name__ == "__main__":
    main()
