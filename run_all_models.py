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
import pickle
from pathlib import Path

sys.path.append(".")
from utils import get_eps_min, find_smallest_eps_spgd


# ============================================================================
# Normalization Wrapper
# ============================================================================

class NormalizedModel(nn.Module):
    """
    Wrapper that applies normalization before passing to the model.
    This makes the model work in [0,1] input space while internally
    using normalized features.
    """
    def __init__(self, model, mean=None, std=None):
        super().__init__()
        self.model = model

        # Default: no normalization (mean=0, std=1)
        if mean is None:
            mean = torch.zeros(1)
        if std is None:
            std = torch.ones(1)

        # Register as buffers (not parameters, but part of state_dict)
        self.register_buffer('mean', mean.view(1, -1, 1, 1))
        self.register_buffer('std', std.view(1, -1, 1, 1))

    def forward(self, x):
        # Normalize: (x - mean) / std
        x_normalized = (x - self.mean) / self.std
        return self.model(x_normalized)


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

def load_gtsrb_dataset():
    """
    Load GTSRB dataset from pickle file (10-class version from VeriX).

    Based on: https://github.com/NeuralNetworkVerification/VeriX/blob/main/gtsrb.py

    Returns:
        TensorDataset: Test dataset with images and labels
    """
    gtsrb_pickle_path = 'data/GTSRB/gtsrb.pickle'

    if not os.path.exists(gtsrb_pickle_path):
        # Try alternative locations
        alternative_paths = [
            'models/gtsrb.pickle',
            './gtsrb.pickle',
            'data/gtsrb.pickle'
        ]

        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                gtsrb_pickle_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"GTSRB pickle file not found. Tried:\n"
                f"  - data/GTSRB/gtsrb.pickle\n"
                f"  - models/gtsrb.pickle\n"
                f"  - ./gtsrb.pickle\n"
                f"  - data/gtsrb.pickle\n"
                f"\nPlease download from: https://github.com/NeuralNetworkVerification/VeriX"
            )

    print(f"Loading GTSRB from: {gtsrb_pickle_path}")

    with open(gtsrb_pickle_path, 'rb') as handle:
        gtsrb = pickle.load(handle)

    # Extract test data (VeriX format)
    # Images are in (N, H, W, C) format, need to transpose to (N, C, H, W)
    x_test = np.transpose(gtsrb['x_test'], (0, 3, 1, 2))
    y_test = gtsrb['y_test']

    # Create TensorDataset (normalize to [0, 1])
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_test, dtype=torch.float32) / 255.0,
        torch.tensor(y_test, dtype=torch.long)
    )

    print(f"✓ GTSRB loaded: {len(test_dataset)} test samples")
    print(f"  Image shape: {x_test.shape[1:]}")
    print(f"  Num classes: {len(np.unique(y_test))}")

    return test_dataset


def get_model_epsilon(model_path):
    """
    Get dataset-specific and architecture-specific epsilon value.

    Args:
        model_path: Path to model file (e.g., "models/cnn3_mnist.pt")

    Returns:
        float: Appropriate epsilon value for this model
    """
    model_info = parse_model_filename(model_path)
    dataset = model_info['dataset'].lower()
    arch = model_info['arch']

    # MNIST epsilon values
    if dataset == 'mnist':
        if arch == 'fc':
            hidden_dim = model_info.get('hidden_dim', 10)
            if hidden_dim == 10:
                return 0.1  # FC-10x2
            elif hidden_dim == 50:
                return 0.2  # FC-50x2
            else:
                return 0.25  # Default for other FC architectures
        elif arch == 'cnn3':
            return 0.25  # CNN-3
        elif arch == 'cnn7':
            return 0.25  # CNN-7
        else:
            return 0.25  # Default MNIST

    # GTSRB epsilon values
    elif dataset == 'gtsrb':
        if arch == 'fc':
            hidden_dim = model_info.get('hidden_dim', 10)
            if hidden_dim == 10:
                return 0.05  # FC-10x2
            elif hidden_dim == 50:
                return 0.1   # FC-50x2
            else:
                return 0.1   # Default for other FC architectures
        else:
            return 0.1  # Default GTSRB

    # CIFAR-10 epsilon values
    elif dataset == 'cifar10':
        if arch == 'fc':
            hidden_dim = model_info.get('hidden_dim', 50)
            if hidden_dim == 50:
                return 2/255  # FC-50x2: ~0.0078
            else:
                return 8/255  # Default for other FC
        elif arch == 'cnn3':
            return 8/255   # CNN-3: ~0.0314
        elif arch == 'cnn7':
            return 16/255  # CNN-7: ~0.0627
        else:
            return 8/255   # Default CIFAR-10

    # Fallback default
    return 0.25


def get_dataset_config(dataset_name):
    """
    Get configuration for a specific dataset.

    Note: Normalization is NOT applied in data loading.
    Images are in [0, 1] range after ToTensor().
    Normalization is handled by NormalizedModel wrapper.
    """
    configs = {
        'mnist': {
            'num_classes': 10,
            'in_channels': 1,
            'input_dim': 28,
            'input_size': 28 * 28,
            'mean': torch.tensor([0.0]),
            'std': torch.tensor([1.0]),
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
            'mean': torch.tensor([0.4914, 0.4822, 0.4465]),
            'std': torch.tensor([0.2023, 0.1994, 0.2010]),
            'loader': lambda: datasets.CIFAR10(
                root="./data", train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor()])  # No normalization here
            )
        },
        'gtsrb': {
            'num_classes': 43,  # GTSRB standard has 43 classes (but models may have 10)
            'in_channels': 3,
            'input_dim': 32,
            'input_size': 32 * 32 * 3,
            'mean': torch.tensor([0.0, 0.0, 0.0]),
            'std': torch.tensor([1.0, 1.0, 1.0]),
            'loader': load_gtsrb_dataset  # Custom GTSRB loader from pickle
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


def detect_num_classes_from_checkpoint(checkpoint, model_info):
    """
    Detect the number of output classes from checkpoint state_dict.

    Args:
        checkpoint: Loaded checkpoint (state_dict or dict containing state_dict)
        model_info: Parsed model information

    Returns:
        int: Number of output classes
    """
    # Extract state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel prefix
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Find the final layer based on architecture
    if model_info['arch'] == 'cnn3':
        # For CNN3, last layer is the Linear layer (key: '7.weight' or similar)
        # Find the largest numbered layer
        final_layer_key = None
        for key in state_dict.keys():
            if 'weight' in key and key.split('.')[0].isdigit():
                final_layer_key = key

        if final_layer_key:
            num_classes = state_dict[final_layer_key].shape[0]
            return num_classes

    elif model_info['arch'] == 'fc':
        # For FC networks, last layer weight has shape (num_classes, hidden_dim)
        # Find the largest numbered layer
        max_layer_num = -1
        for key in state_dict.keys():
            if 'weight' in key:
                parts = key.split('.')
                if parts[0].isdigit():
                    layer_num = int(parts[0])
                    if layer_num > max_layer_num:
                        max_layer_num = layer_num

        if max_layer_num >= 0:
            final_weight_key = f"{max_layer_num}.weight"
            if final_weight_key in state_dict:
                num_classes = state_dict[final_weight_key].shape[0]
                return num_classes

    # Fallback: return None if detection fails
    return None


def create_model(model_info, dataset_config, device, num_classes_override=None):
    """
    Create model based on parsed filename info and dataset config.

    The model is wrapped with NormalizedModel to handle dataset-specific
    normalization internally. Attacks work in [0,1] space, and the model
    applies normalization in its forward pass.

    Args:
        model_info: Parsed model information
        dataset_config: Dataset configuration
        device: Target device
        num_classes_override: Override number of classes (for checkpoint mismatch)
    """
    num_classes = num_classes_override if num_classes_override is not None else dataset_config['num_classes']

    if model_info['arch'] == 'cnn3':
        base_model = cnn3(
            in_ch=dataset_config['in_channels'],
            in_dim=dataset_config['input_dim'],
            width=64,
            num_class=num_classes
        )
    elif model_info['arch'] == 'fc':
        base_model = fc_network(
            input_dim=dataset_config['input_size'],
            hidden_dim=model_info['hidden_dim'],
            num_layers=model_info['num_layers'],
            num_class=num_classes
        )
    else:
        raise ValueError(f"Unknown architecture: {model_info['arch']}")

    # Wrap with normalization layer
    # Mean and std are moved to device inside NormalizedModel
    normalized_model = NormalizedModel(
        base_model,
        mean=dataset_config['mean'],
        std=dataset_config['std']
    )

    return normalized_model.to(device)


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
        model, images, labels, dataset_config
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

    # Load checkpoint first to detect number of classes
    checkpoint = torch.load(model_path, map_location=device)

    # Detect actual number of classes from checkpoint
    detected_num_classes = detect_num_classes_from_checkpoint(checkpoint, model_info)

    if detected_num_classes is not None:
        if detected_num_classes != dataset_config['num_classes']:
            print(f"⚠️  Warning: Checkpoint has {detected_num_classes} classes, "
                  f"but {dataset_name} config expects {dataset_config['num_classes']}")
            print(f"   Using {detected_num_classes} classes from checkpoint")

    # Create model with detected number of classes
    model = create_model(
        model_info,
        dataset_config,
        device,
        num_classes_override=detected_num_classes
    )

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
        try:
            # Try to load real GTSRB data
            test_dataset = dataset_config['loader']()

            # Extract samples from dataset
            actual_num_classes = detected_num_classes if detected_num_classes else dataset_config['num_classes']

            # Get indices for valid classes (if model has fewer classes than dataset)
            all_images = []
            all_labels = []

            for img, label in test_dataset:
                if label < actual_num_classes:  # Only use samples within model's class range
                    all_images.append(img)
                    all_labels.append(label)
                    if len(all_images) >= num_samples:
                        break

            if len(all_images) < num_samples:
                print(f"⚠️  Warning: Found only {len(all_images)} samples with labels < {actual_num_classes}")
                print(f"   Requested {num_samples} samples")

            # Stack tensors
            images = torch.stack(all_images[:num_samples]).to(device)
            labels = torch.tensor(all_labels[:num_samples], dtype=torch.long).to(device)

            print(f"✓ Loaded {len(images)} real GTSRB samples (classes 0-{actual_num_classes-1})")

        except FileNotFoundError as e:
            # Fallback to synthetic data if pickle not found
            print(f"⚠️  Warning: GTSRB pickle file not found.")
            print(f"   {str(e)}")
            print(f"   Falling back to synthetic test data for model evaluation only.")

            actual_num_classes = detected_num_classes if detected_num_classes else dataset_config['num_classes']
            images = torch.rand(num_samples, 3, 32, 32).to(device)
            labels = torch.randint(0, actual_num_classes, (num_samples,)).to(device)

            print(f"   Generated {num_samples} synthetic samples with {actual_num_classes} classes")

            with torch.no_grad():
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)

            print(f"Synthetic Labels    : {labels.tolist()}")
            print(f"Model Predictions   : {predictions.tolist()}")
            print(f"Model output shape  : {outputs.shape}")

            return model, images, labels, dataset_config

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

    return model, images, labels, dataset_config


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(model_path, k_sparse=50, eps_fav=None, num_samples=10, results_dir=None):
    """
    Run XAI bounds experiment for a single model.

    Args:
        model_path: Path to model file
        k_sparse: Sparsity budget for Sparse-PGD
        eps_fav: FAVEX radius (upper bound for eps_min search).
                 If None, automatically determined from model architecture and dataset.
        num_samples: Number of test samples to evaluate
        results_dir: Custom results directory (default: results/{num_samples}_samples_k_{k_sparse})
    """
    # Auto-determine epsilon if not provided
    if eps_fav is None:
        eps_fav = get_model_epsilon(model_path)
        print(f"Auto-detected eps_fav = {eps_fav:.6f} for {os.path.basename(model_path)}")

    # Load model and data
    model, images, labels, dataset_config = load_model_and_data(model_path, num_samples)

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
