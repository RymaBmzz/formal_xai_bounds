#!/usr/bin/env python3
"""
Compare and visualize results from all model experiments.
Handles multiple experiment configurations (different samples/k values).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re


def parse_config_from_path(file_path):
    """
    Extract configuration (samples, k) from file path.

    Args:
        file_path: Path to CSV file

    Returns:
        tuple: (num_samples, k_sparse) or (None, None)
    """
    # Look for pattern: <num>_samples_k_<num>
    path_str = str(file_path)
    match = re.search(r'(\d+)_samples_k_(\d+)', path_str)

    if match:
        num_samples = int(match.group(1))
        k_sparse = int(match.group(2))
        return num_samples, k_sparse

    return None, None


def load_all_results(results_dir=None):
    """
    Load all CSV result files with configuration tracking.

    Args:
        results_dir: Directory to search for results (default: search all results/)

    Returns:
        dict: {(model_name, num_samples, k_sparse): DataFrame} or None
    """
    results = {}

    if results_dir:
        # Search in specific directory
        search_paths = [Path(results_dir)]
    else:
        # Search in all results/ subdirectories
        search_paths = []
        results_root = Path("results")
        if results_root.exists():
            search_paths.extend(sorted(results_root.glob("*_samples_k_*")))

        # Also check current directory
        if not search_paths:
            search_paths = [Path(".")]

    csv_files = []
    for search_path in search_paths:
        csv_files.extend(search_path.glob("eps_bounds_*.csv"))

    if not csv_files:
        print("No result files found.")
        print("\nSearched in:")
        for path in search_paths:
            print(f"  - {path}")
        print("\nRun experiments first with: ./run.sh all")
        return None

    print(f"\nFound {len(csv_files)} result files:")

    for csv_file in sorted(csv_files):
        model_name = csv_file.stem.replace("eps_bounds_", "")
        num_samples, k_sparse = parse_config_from_path(csv_file)

        try:
            df = pd.read_csv(csv_file)

            # Use (model, samples, k) as key to preserve all configurations
            key = (model_name, num_samples, k_sparse)
            results[key] = df

            config_str = f"(samples={num_samples}, k={k_sparse})" if num_samples else ""
            print(f"  ✓ {model_name:<25} {config_str:<20} {csv_file.relative_to('.')}")

        except Exception as e:
            print(f"  ✗ Warning: Could not load {csv_file}: {e}")

    return results if results else None


def print_summary_table(results):
    """Print a summary comparison table with configuration columns."""
    print("\n" + "="*120)
    print("SUMMARY: XAI Bounds Across All Models and Configurations")
    print("="*120)
    print(f"{'Model':<25} {'Samples':<8} {'k':<6} {'Avg Gap':<10} {'Min Gap':<10} {'Max Gap':<10} {'Avg ε_min':<10} {'Avg ε_max':<10}")
    print("-"*120)

    summary_data = []

    # Sort by model name, then samples, then k
    sorted_results = sorted(results.items(), key=lambda x: (x[0][0], x[0][1] or 0, x[0][2] or 0))

    for (model_name, num_samples, k_sparse), df in sorted_results:
        n_samples = len(df)
        avg_gap = df['gap'].mean()
        min_gap = df['gap'].min()
        max_gap = df['gap'].max()
        avg_eps_min = df['eps_min'].mean()
        avg_eps_max = df['eps_max'].mean()

        samples_str = str(num_samples) if num_samples else "?"
        k_str = str(k_sparse) if k_sparse else "?"

        print(f"{model_name:<25} {samples_str:<8} {k_str:<6} {avg_gap:<10.4f} {min_gap:<10.4f} {max_gap:<10.4f} {avg_eps_min:<10.4f} {avg_eps_max:<10.4f}")

        summary_data.append({
            'model': model_name,
            'num_samples': num_samples,
            'k_sparse': k_sparse,
            'samples_in_csv': n_samples,
            'avg_gap': avg_gap,
            'min_gap': min_gap,
            'max_gap': max_gap,
            'avg_eps_min': avg_eps_min,
            'avg_eps_max': avg_eps_max
        })

    print("="*120)

    return pd.DataFrame(summary_data)


def print_detailed_comparison(results):
    """Print detailed per-sample comparison for each configuration."""

    # Group by configuration
    configs = {}
    for (model_name, num_samples, k_sparse), df in results.items():
        config_key = (num_samples, k_sparse)
        if config_key not in configs:
            configs[config_key] = {}
        configs[config_key][model_name] = df

    for (num_samples, k_sparse), models in sorted(configs.items()):
        print("\n" + "="*90)
        print(f"DETAILED: Configuration samples={num_samples}, k={k_sparse}")
        print("="*90)

        model_names = sorted(models.keys())
        max_samples = max(len(df) for df in models.values())

        # Print header
        header = f"{'Sample':<8}"
        for model_name in model_names:
            header += f"{model_name[:20]:<22}"
        print(header)
        print("-"*90)

        # Print each sample
        for i in range(max_samples):
            row = f"{i:<8}"
            for model_name in model_names:
                df = models[model_name]
                if i < len(df):
                    gap = df.iloc[i]['gap']
                    row += f"{gap:<22.4f}"
                else:
                    row += f"{'N/A':<22}"
            print(row)

        print("="*90)


def print_architecture_comparison(results):
    """Compare results by architecture type."""
    print("\n" + "="*90)
    print("COMPARISON BY ARCHITECTURE")
    print("="*90)

    # Group by architecture
    cnn_models = {}
    fc_models = {}

    for (model_name, num_samples, k_sparse), df in results.items():
        key = (model_name, num_samples, k_sparse)
        if model_name.startswith('cnn3'):
            cnn_models[key] = df
        elif model_name.startswith('fc'):
            fc_models[key] = df

    print(f"\n📊 CNN Models ({len(cnn_models)} configurations):")
    if cnn_models:
        for (model_name, num_samples, k_sparse), df in sorted(cnn_models.items()):
            avg_gap = df['gap'].mean()
            config_str = f"samples={num_samples}, k={k_sparse}"
            print(f"  {model_name:<25} ({config_str:<20}) Avg Gap: {avg_gap:.4f}")

        cnn_avg = np.mean([df['gap'].mean() for df in cnn_models.values()])
        print(f"\n  Overall CNN Average Gap: {cnn_avg:.4f}")

    print(f"\n📊 Fully Connected Models ({len(fc_models)} configurations):")
    if fc_models:
        for (model_name, num_samples, k_sparse), df in sorted(fc_models.items()):
            avg_gap = df['gap'].mean()
            config_str = f"samples={num_samples}, k={k_sparse}"
            print(f"  {model_name:<25} ({config_str:<20}) Avg Gap: {avg_gap:.4f}")

        fc_avg = np.mean([df['gap'].mean() for df in fc_models.values()])
        print(f"\n  Overall FC Average Gap: {fc_avg:.4f}")

    print("="*90)


def print_dataset_comparison(results):
    """Compare results by dataset."""
    print("\n" + "="*90)
    print("COMPARISON BY DATASET")
    print("="*90)

    # Group by dataset
    datasets = {}
    for (model_name, num_samples, k_sparse), df in results.items():
        # Extract dataset from model name
        if 'mnist' in model_name:
            dataset = 'mnist'
        elif 'cifar10' in model_name:
            dataset = 'cifar10'
        elif 'gtsrb' in model_name:
            dataset = 'gtsrb'
        else:
            dataset = 'unknown'

        if dataset not in datasets:
            datasets[dataset] = {}

        key = (model_name, num_samples, k_sparse)
        datasets[dataset][key] = df

    for dataset_name, models in sorted(datasets.items()):
        print(f"\n📊 {dataset_name.upper()} ({len(models)} configurations):")
        for (model_name, num_samples, k_sparse), df in sorted(models.items()):
            avg_gap = df['gap'].mean()
            config_str = f"samples={num_samples}, k={k_sparse}"
            print(f"  {model_name:<25} ({config_str:<20}) Avg Gap: {avg_gap:.4f}")

        dataset_avg = np.mean([df['gap'].mean() for df in models.values()])
        print(f"\n  Overall {dataset_name.upper()} Average Gap: {dataset_avg:.4f}")

    print("="*90)


def print_parameter_analysis(results):
    """Analyze effect of k and num_samples parameters."""
    print("\n" + "="*90)
    print("PARAMETER ANALYSIS")
    print("="*90)

    # Group by model to see effect of parameters
    models_data = {}
    for (model_name, num_samples, k_sparse), df in results.items():
        if model_name not in models_data:
            models_data[model_name] = []

        models_data[model_name].append({
            'num_samples': num_samples,
            'k_sparse': k_sparse,
            'avg_gap': df['gap'].mean(),
            'avg_eps_min': df['eps_min'].mean(),
            'avg_eps_max': df['eps_max'].mean()
        })

    for model_name, configs in sorted(models_data.items()):
        if len(configs) <= 1:
            continue  # Skip models with only one configuration

        print(f"\n📈 {model_name}:")
        print(f"{'Samples':<10} {'k':<6} {'Avg Gap':<12} {'Avg ε_min':<12} {'Avg ε_max':<12}")
        print("-" * 60)

        for config in sorted(configs, key=lambda x: (x['num_samples'] or 0, x['k_sparse'] or 0)):
            print(f"{config['num_samples'] or '?':<10} {config['k_sparse'] or '?':<6} "
                  f"{config['avg_gap']:<12.4f} {config['avg_eps_min']:<12.4f} {config['avg_eps_max']:<12.4f}")

    print("="*90)


def export_summary(summary_df, results_dir=None, output_file="summary_all_configs.csv"):
    """
    Export summary to CSV.

    Args:
        summary_df: Summary DataFrame
        results_dir: Results directory to save to (default: current directory)
        output_file: Output filename
    """
    if results_dir:
        output_path = Path(results_dir) / output_file
    else:
        output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"\n✓ Summary exported to: {output_path}")


def main():
    """Main comparison function."""
    # Parse command line arguments
    results_dir = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n" + "#"*120)
    print("# XAI Bounds - Multi-Configuration Results Comparison")
    print("#"*120)

    if results_dir:
        print(f"\nSearching in: {results_dir}")

    # Load all results
    results = load_all_results(results_dir)

    if not results:
        sys.exit(1)

    print(f"\nTotal configurations: {len(results)}")

    # Count unique models
    unique_models = set(model_name for model_name, _, _ in results.keys())
    print(f"Unique models: {len(unique_models)}")

    # Count unique configs
    unique_configs = set((samples, k) for _, samples, k in results.keys())
    print(f"Unique parameter configs: {len(unique_configs)}")

    # Print comparisons
    summary_df = print_summary_table(results)
    print_parameter_analysis(results)
    print_detailed_comparison(results)
    print_architecture_comparison(results)
    print_dataset_comparison(results)

    # Export summary
    export_summary(summary_df, results_dir)

    print("\n" + "#"*120)
    print("# Comparison Complete")
    print("#"*120 + "\n")


if __name__ == "__main__":
    main()
