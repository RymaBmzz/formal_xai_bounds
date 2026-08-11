#!/usr/bin/env python3
"""
Run XAI bounds experiments on all models in the models/ directory.
Automatically detects architecture and dataset from model filename.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import csv
import os
import sys
from pathlib import Path

sys.path.append(".")
from utils import get_eps_min, find_smallest_eps_spgd


# ============================================================================
# Model Architecture Definitions
# ============================================================================

def cnn3(in_ch=1, in_dim=28, width=64, num_class=10):
    """CNN3 architecture from FAVEX."""
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4) ** 2 * 2 * width, num_class),
    )
    return model


def fc_network(input_dim, hidden_dim, num_layers, num_class):
    """Fully connected network with specified architecture."""
    layers = []
    layers.append(nn.Flatten())
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(hidden_dim, num_class))
    return nn.Sequential(*layers)


# ============================================================================
# Dataset Configuration
# ============================================================================

def get_dataset_config(dataset_name):
    """Get configuration for a specific dataset."""
    configs = {
        'mnist': {
            'num_classes': 10,
            'in_channels': 1,
            'input_dim': 28,
            'input_size': 28 * 28,
            'loader': lambda: datasets.MNIST(
                root="./data", train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor()])
            )
        },
        'cifar10': {
            'num_classes': 10,
            'in_channels': 3,
            'input_dim': 32,
            'input_size': 32 * 32 * 3,
            'loader': lambda: datasets.CIFAR10(
                root="./data", train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor()])
            )
        },
        'gtsrb': {
            'num_classes': 43,  # GTSRB has 43 classes
            'in_channels': 3,
            'input_dim': 32,
            'input_size': 32 * 32 * 3,
            'loader': lambda: None  # GTSRB needs custom loading
        }
    }
    return configs.get(dataset_name.lower())


# ============================================================================
# Model Configuration Parser
# ============================================================================

def parse_model_filename(filename):
    """
    Parse model filename to extract architecture and dataset info.

    Examples:
        - cnn3_mnist.pt -> (arch='cnn3', dataset='mnist')
        - fc_50x2_cifar10.pt -> (arch='fc', hidden=50, layers=2, dataset='cifar10')
    """
    name = Path(filename).stem  # Remove .pt extension
    parts = name.split('_')

    if parts[0] == 'cnn3':
        return {
            'arch': 'cnn3',
            'dataset': parts[1],
        }
    elif parts[0] == 'fc':
        # Parse fc_HIDDENxLAYERS_dataset
        dims = parts[1].split('x')
        return {
            'arch': 'fc',
            'hidden_dim': int(dims[0]),
            'num_layers': int(dims[1]),
            'dataset': parts[2],
        }
    else:
        raise ValueError(f"Unknown architecture in filename: {filename}")


def create_model(model_info, dataset_config, device):
    """Create model based on parsed filename info and dataset config."""
    if model_info['arch'] == 'cnn3':
        model = cnn3(
            in_ch=dataset_config['in_channels'],
            in_dim=dataset_config['input_dim'],
            width=64,
            num_class=dataset_config['num_classes']
        )
    elif model_info['arch'] == 'fc':
        model = fc_network(
            input_dim=dataset_config['input_size'],
            hidden_dim=model_info['hidden_dim'],
            num_layers=model_info['num_layers'],
            num_class=dataset_config['num_classes']
        )
    else:
        raise ValueError(f"Unknown architecture: {model_info['arch']}")

    return model.to(device)


# ============================================================================
# Model and Data Loading
# ============================================================================

def load_model_and_data(model_path, num_samples=10):
    """
    Load model and corresponding test data based on model filename.

    Args:
        model_path: Path to the model file
        num_samples: Number of test samples to load

    Returns:
        model, images, labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse model filename
    model_info = parse_model_filename(model_path)
    dataset_name = model_info['dataset']

    print(f"\n{'='*70}")
    print(f"Loading model: {os.path.basename(model_path)}")
    print(f"Architecture: {model_info}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    if dataset_config is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create model
    model = create_model(model_info, dataset_config, device)

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)

    # Extract state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel prefix
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Load dataset
    if dataset_name.lower() == 'gtsrb':
        print(f"Warning: GTSRB dataset loading not implemented. Skipping {model_path}")
        return None, None, None

    test_dataset = dataset_config['loader']()

    # Extract test samples
    images = torch.stack([test_dataset[i][0] for i in range(num_samples)]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in range(num_samples)]).to(device)

    # Verify model predictions
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    print(f"Ground Truth Labels : {labels.tolist()}")
    print(f"Model Predictions   : {predictions.tolist()}")
    print(f"Accuracy on samples : {(predictions == labels).sum().item()}/{num_samples}")

    return model, images, labels


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(model_path, k_sparse=50, eps_fav=0.25, num_samples=10, results_dir=None):
    """
    Run XAI bounds experiment for a single model.

    Args:
        model_path: Path to model file
        k_sparse: Sparsity budget for Sparse-PGD
        eps_fav: FAVEX radius (upper bound for eps_min search)
        num_samples: Number of test samples to evaluate
        results_dir: Custom results directory (default: results/{num_samples}_samples_k_{k_sparse})
    """
    # Load model and data
    model, images, labels = load_model_and_data(model_path, num_samples)

    if model is None:
        return  # Skip if dataset not supported

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create results directory
    if results_dir is None:
        results_dir = f"results/{num_samples}_samples_k_{k_sparse}"

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    model_name = Path(model_path).stem
    csv_filename = results_path / f"eps_bounds_{model_name}.csv"

    print(f"\nImage stats: max={images.max():.4f}, min={images.min():.4f}, mean={images.mean():.4f}")
    print(f"Running experiment with k={k_sparse}, eps_fav={eps_fav}")
    print(f"Results will be saved to: {csv_filename}\n")

    results = []

    # Process each sample
    for index in range(num_samples):
        print(f"\n{'─'*70}")
        print(f"Processing sample {index+1}/{num_samples}")
        print(f"{'─'*70}")

        image = images[index:index+1].to(device)
        label = labels[index:index+1].to(device)

        # Compute eps_min (minimum perturbation to fool the model)
        print(f"\n[1/2] Computing eps_min (AutoAttack)...")
        eps_min = get_eps_min(model, image, label, eps_high=eps_fav)
        print(f"✓ eps_min = {eps_min:.4f}")

        # Compute eps_max (minimum perturbation for k-sparse attack)
        print(f"\n[2/2] Computing eps_max (Sparse-PGD with k={k_sparse})...")
        eps_max = find_smallest_eps_spgd(
            model, image, label,
            k=k_sparse,
            eps_low=eps_fav,
            eps_high=1.0
        )
        print(f"✓ eps_max = {eps_max:.4f}")

        true_label = label.item()
        gap = eps_max - eps_min

        results.append({
            "index": index,
            "true_label": true_label,
            "eps_min": round(eps_min, 4),
            "eps_max": round(eps_max, 4),
            "gap": round(gap, 4)
        })

        print(f"\n📊 Result: eps_min={eps_min:.4f} | eps_max={eps_max:.4f} | gap={gap:.4f}")

    # Save results to CSV
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "true_label", "eps_min", "eps_max", "gap"])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print(f"\n{'='*70}")
    print(f"✓ Experiment complete for {model_name}")
    print(f"✓ Results saved to: {csv_filename}")

    avg_gap = np.mean([r['gap'] for r in results])
    print(f"\n📈 Summary Statistics:")
    print(f"   - Average gap: {avg_gap:.4f}")
    print(f"   - Min gap: {min(r['gap'] for r in results):.4f}")
    print(f"   - Max gap: {max(r['gap'] for r in results):.4f}")
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run experiments on all models in the models/ directory."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run XAI bounds experiments on all models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run_all_models.py                    # Default: k=50, num_samples=10
  python run_all_models.py --k 100            # k=100, num_samples=10
  python run_all_models.py --samples 20       # k=50, num_samples=20
  python run_all_models.py --k 100 --samples 20  # k=100, num_samples=20

Results will be saved to: results/{num_samples}_samples_k_{k_sparse}/
        '''
    )

    parser.add_argument(
        '--k', '--k-sparse',
        type=int,
        default=50,
        dest='k_sparse',
        help='Sparsity budget for Sparse-PGD attack (default: 50)'
    )

    parser.add_argument(
        '--samples', '--num-samples',
        type=int,
        default=10,
        dest='num_samples',
        help='Number of test samples to evaluate (default: 10)'
    )

    parser.add_argument(
        '--eps-fav',
        type=float,
        default=0.25,
        help='FAVEX radius / upper bound for eps_min search (default: 0.25)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.k_sparse < 1:
        parser.error("k_sparse must be at least 1")
    if args.num_samples < 1:
        parser.error("num_samples must be at least 1")
    if args.eps_fav <= 0 or args.eps_fav > 1:
        parser.error("eps_fav must be between 0 and 1")

    models_dir = Path("./models")
    model_files = sorted(models_dir.glob("*.pt"))

    if not model_files:
        print("No model files found in ./models/")
        return

    print(f"\n{'#'*70}")
    print(f"# XAI Bounds Experiment Suite")
    print(f"# Found {len(model_files)} models to evaluate")
    print(f"# Parameters: k={args.k_sparse}, num_samples={args.num_samples}, eps_fav={args.eps_fav}")
    print(f"# Results directory: results/{args.num_samples}_samples_k_{args.k_sparse}/")
    print(f"{'#'*70}")

    for i, model_path in enumerate(model_files, 1):
        print(f"\n\n{'█'*70}")
        print(f"█ Model {i}/{len(model_files)}: {model_path.name}")
        print(f"{'█'*70}")

        try:
            run_experiment(
                model_path=str(model_path),
                k_sparse=args.k_sparse,
                eps_fav=args.eps_fav,
                num_samples=args.num_samples
            )
        except Exception as e:
            print(f"\n❌ Error processing {model_path.name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping to next model...\n")
            continue

    print(f"\n{'#'*70}")
    print(f"# All experiments complete!")
    print(f"# Results saved to: results/{args.num_samples}_samples_k_{args.k_sparse}/")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
