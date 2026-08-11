#!/usr/bin/env python3
"""
Compare and visualize results from all model experiments.
Reads all eps_bounds_*.csv files and generates a summary.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_all_results(results_dir=None):
    """
    Load all CSV result files.

    Args:
        results_dir: Directory to search for results (default: search current dir and results/)

    Returns:
        dict: {model_name: DataFrame} or None if no results found
    """
    results = {}

    if results_dir:
        # Search in specific directory
        search_paths = [Path(results_dir)]
    else:
        # Search in current directory and results/ subdirectories
        search_paths = [Path(".")]
        results_root = Path("results")
        if results_root.exists():
            search_paths.extend(results_root.glob("*_samples_k_*"))

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
        try:
            df = pd.read_csv(csv_file)
            results[model_name] = df
            print(f"  ✓ {csv_file.relative_to('.')}")
        except Exception as e:
            print(f"  ✗ Warning: Could not load {csv_file}: {e}")

    return results if results else None


def print_summary_table(results):
    """Print a summary comparison table."""
    print("\n" + "="*90)
    print("SUMMARY: XAI Bounds Across All Models")
    print("="*90)
    print(f"{'Model':<25} {'Samples':<8} {'Avg Gap':<10} {'Min Gap':<10} {'Max Gap':<10} {'Avg ε_min':<10} {'Avg ε_max':<10}")
    print("-"*90)

    summary_data = []

    for model_name, df in sorted(results.items()):
        n_samples = len(df)
        avg_gap = df['gap'].mean()
        min_gap = df['gap'].min()
        max_gap = df['gap'].max()
        avg_eps_min = df['eps_min'].mean()
        avg_eps_max = df['eps_max'].mean()

        print(f"{model_name:<25} {n_samples:<8} {avg_gap:<10.4f} {min_gap:<10.4f} {max_gap:<10.4f} {avg_eps_min:<10.4f} {avg_eps_max:<10.4f}")

        summary_data.append({
            'model': model_name,
            'samples': n_samples,
            'avg_gap': avg_gap,
            'min_gap': min_gap,
            'max_gap': max_gap,
            'avg_eps_min': avg_eps_min,
            'avg_eps_max': avg_eps_max
        })

    print("="*90)

    return pd.DataFrame(summary_data)


def print_detailed_comparison(results):
    """Print detailed per-sample comparison."""
    print("\n" + "="*90)
    print("DETAILED: Gap Values for Each Sample")
    print("="*90)

    # Get all model names
    model_names = sorted(results.keys())

    # Find max number of samples
    max_samples = max(len(df) for df in results.values())

    # Print header
    header = f"{'Sample':<8}"
    for model_name in model_names:
        header += f"{model_name[:15]:<16}"
    print(header)
    print("-"*90)

    # Print each sample
    for i in range(max_samples):
        row = f"{i:<8}"
        for model_name in model_names:
            df = results[model_name]
            if i < len(df):
                gap = df.iloc[i]['gap']
                row += f"{gap:<16.4f}"
            else:
                row += f"{'N/A':<16}"
        print(row)

    print("="*90)


def print_architecture_comparison(results):
    """Compare results by architecture type."""
    print("\n" + "="*90)
    print("COMPARISON BY ARCHITECTURE")
    print("="*90)

    # Group by architecture
    cnn_models = {k: v for k, v in results.items() if k.startswith('cnn3')}
    fc_models = {k: v for k, v in results.items() if k.startswith('fc')}

    print(f"\n📊 CNN Models ({len(cnn_models)} total):")
    if cnn_models:
        for model_name, df in sorted(cnn_models.items()):
            avg_gap = df['gap'].mean()
            print(f"  {model_name:<25} Average Gap: {avg_gap:.4f}")

        cnn_avg = np.mean([df['gap'].mean() for df in cnn_models.values()])
        print(f"\n  Overall CNN Average Gap: {cnn_avg:.4f}")

    print(f"\n📊 Fully Connected Models ({len(fc_models)} total):")
    if fc_models:
        for model_name, df in sorted(fc_models.items()):
            avg_gap = df['gap'].mean()
            print(f"  {model_name:<25} Average Gap: {avg_gap:.4f}")

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
    for model_name, df in results.items():
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
        datasets[dataset][model_name] = df

    for dataset_name, models in sorted(datasets.items()):
        print(f"\n📊 {dataset_name.upper()} ({len(models)} models):")
        for model_name, df in sorted(models.items()):
            avg_gap = df['gap'].mean()
            print(f"  {model_name:<25} Average Gap: {avg_gap:.4f}")

        dataset_avg = np.mean([df['gap'].mean() for df in models.values()])
        print(f"\n  Overall {dataset_name.upper()} Average Gap: {dataset_avg:.4f}")

    print("="*90)


def export_summary(summary_df, results_dir=None, output_file="summary_all_models.csv"):
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

    print("\n" + "#"*90)
    print("# XAI Bounds - Results Comparison")
    print("#"*90)

    if results_dir:
        print(f"\nSearching in: {results_dir}")

    # Load all results
    results = load_all_results(results_dir)

    if not results:
        sys.exit(1)

    # Print comparisons
    summary_df = print_summary_table(results)
    print_detailed_comparison(results)
    print_architecture_comparison(results)
    print_dataset_comparison(results)

    # Export summary
    export_summary(summary_df, results_dir)

    print("\n" + "#"*90)
    print("# Comparison Complete")
    print("#"*90 + "\n")


if __name__ == "__main__":
    main()
