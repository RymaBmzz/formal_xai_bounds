#!/usr/bin/env python3
"""
Visualization script for XAI bounds experiment results.
Analyzes the impact of k (sparsity) and num_samples on avg_gap.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_summary_data(csv_path='summary_all_configs.csv'):
    """Load summary CSV file."""
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found!")
        print("Run 'python compare_results.py' first to generate it.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} configurations")
    print(f"Models: {df['model'].nunique()}")
    print(f"Samples values: {sorted(df['num_samples'].unique())}")
    print(f"k values: {sorted(df['k_sparse'].unique())}")
    return df


def plot_gap_vs_k(df, output_dir='plots'):
    """Plot avg_gap vs k for different num_samples."""
    Path(output_dir).mkdir(exist_ok=True)

    models = df['model'].unique()

    for model in models:
        model_data = df[df['model'] == model].copy()

        if len(model_data) < 2:
            continue  # Skip if only one config

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot for each num_samples value
        for samples in sorted(model_data['num_samples'].unique()):
            subset = model_data[model_data['num_samples'] == samples].sort_values('k_sparse')

            ax.plot(subset['k_sparse'], subset['avg_gap'],
                   marker='o', markersize=8, linewidth=2,
                   label=f'{int(samples)} samples')

        ax.set_xlabel('k (Sparsity Budget)', fontsize=12)
        ax.set_ylabel('Average Gap (ε_max - ε_min)', fontsize=12)
        ax.set_title(f'Impact of k on XAI Bounds Gap - {model}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate points with values
        for samples in sorted(model_data['num_samples'].unique()):
            subset = model_data[model_data['num_samples'] == samples].sort_values('k_sparse')
            for _, row in subset.iterrows():
                ax.annotate(f'{row["avg_gap"]:.3f}',
                          (row['k_sparse'], row['avg_gap']),
                          textcoords="offset points", xytext=(0, 10),
                          ha='center', fontsize=8, alpha=0.7)

        plt.tight_layout()
        filename = f'{output_dir}/gap_vs_k_{model}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()


def plot_gap_vs_samples(df, output_dir='plots'):
    """Plot avg_gap vs num_samples for different k values."""
    Path(output_dir).mkdir(exist_ok=True)

    models = df['model'].unique()

    for model in models:
        model_data = df[df['model'] == model].copy()

        if len(model_data) < 2:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot for each k value
        for k in sorted(model_data['k_sparse'].unique()):
            subset = model_data[model_data['k_sparse'] == k].sort_values('num_samples')

            ax.plot(subset['num_samples'], subset['avg_gap'],
                   marker='s', markersize=8, linewidth=2,
                   label=f'k={int(k)}')

        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_ylabel('Average Gap (ε_max - ε_min)', fontsize=12)
        ax.set_title(f'Impact of Sample Size on XAI Bounds Gap - {model}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate points
        for k in sorted(model_data['k_sparse'].unique()):
            subset = model_data[model_data['k_sparse'] == k].sort_values('num_samples')
            for _, row in subset.iterrows():
                ax.annotate(f'{row["avg_gap"]:.3f}',
                          (row['num_samples'], row['avg_gap']),
                          textcoords="offset points", xytext=(0, 10),
                          ha='center', fontsize=8, alpha=0.7)

        plt.tight_layout()
        filename = f'{output_dir}/gap_vs_samples_{model}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()


def plot_heatmap(df, output_dir='plots'):
    """Create heatmap showing gap for each (samples, k) combination."""
    Path(output_dir).mkdir(exist_ok=True)

    models = df['model'].unique()

    for model in models:
        model_data = df[df['model'] == model].copy()

        if len(model_data) < 4:  # Need at least 2x2 grid
            continue

        # Create pivot table
        pivot = model_data.pivot(index='num_samples', columns='k_sparse', values='avg_gap')

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd',
                   cbar_kws={'label': 'Average Gap'}, ax=ax,
                   linewidths=0.5, linecolor='gray')

        ax.set_xlabel('k (Sparsity Budget)', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title(f'Gap Heatmap: Samples × k - {model}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        filename = f'{output_dir}/heatmap_{model}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()


def plot_comparison_by_dataset(df, output_dir='plots'):
    """Compare models grouped by dataset."""
    Path(output_dir).mkdir(exist_ok=True)

    # Extract dataset from model name
    df['dataset'] = df['model'].apply(lambda x: 'mnist' if 'mnist' in x
                                      else 'cifar10' if 'cifar10' in x
                                      else 'gtsrb' if 'gtsrb' in x
                                      else 'unknown')

    datasets = df['dataset'].unique()

    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset].copy()

        # Group by configuration
        configs = dataset_data.groupby(['num_samples', 'k_sparse'])

        for (samples, k), group in configs:
            if len(group) < 2:
                continue

            fig, ax = plt.subplots(figsize=(12, 7))

            x = np.arange(len(group))
            bars = ax.bar(x, group['avg_gap'], alpha=0.7, edgecolor='black')

            # Color bars by value
            cmap = plt.cm.RdYlGn
            normalize = plt.Normalize(vmin=group['avg_gap'].min(), vmax=group['avg_gap'].max())
            for i, bar in enumerate(bars):
                bar.set_color(cmap(normalize(group['avg_gap'].iloc[i])))

            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('Average Gap', fontsize=12)
            ax.set_title(f'{dataset.upper()} Models - samples={int(samples)}, k={int(k)}',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(group['model'], rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, (idx, row) in enumerate(group.iterrows()):
                ax.text(i, row['avg_gap'], f'{row["avg_gap"]:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.tight_layout()
            filename = f'{output_dir}/comparison_{dataset}_s{int(samples)}_k{int(k)}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
            plt.close()


def plot_eps_components(df, output_dir='plots'):
    """Plot eps_min and eps_max separately to see their contributions to gap."""
    Path(output_dir).mkdir(exist_ok=True)

    models = df['model'].unique()

    for model in models:
        model_data = df[df['model'] == model].copy()

        if len(model_data) < 2:
            continue

        # Select a specific k value for clarity (middle one)
        k_values = sorted(model_data['k_sparse'].unique())
        selected_k = k_values[len(k_values) // 2] if len(k_values) > 1 else k_values[0]

        subset = model_data[model_data['k_sparse'] == selected_k].sort_values('num_samples')

        if len(subset) < 2:
            continue

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        x = subset['num_samples']

        # Plot 1: eps_min and eps_max
        ax1.plot(x, subset['avg_eps_min'], marker='o', linewidth=2, label='ε_min (AutoAttack)')
        ax1.plot(x, subset['avg_eps_max'], marker='s', linewidth=2, label='ε_max (Sparse-PGD)')
        ax1.fill_between(x, subset['avg_eps_min'], subset['avg_eps_max'], alpha=0.3, label='Gap')
        ax1.set_ylabel('Epsilon Value', fontsize=11)
        ax1.set_title(f'ε_min and ε_max Evolution - {model} (k={int(selected_k)})',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gap
        ax2.plot(x, subset['avg_gap'], marker='D', linewidth=2, color='green', label='Gap')
        ax2.fill_between(x, subset['avg_gap'], alpha=0.3, color='green')
        ax2.set_ylabel('Gap (ε_max - ε_min)', fontsize=11)
        ax2.set_title(f'Gap Evolution - {model} (k={int(selected_k)})', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Ratio
        ratio = subset['avg_eps_max'] / subset['avg_eps_min'].replace(0, np.nan)
        ax3.plot(x, ratio, marker='^', linewidth=2, color='purple', label='ε_max / ε_min')
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Ratio = 1')
        ax3.set_xlabel('Number of Samples', fontsize=11)
        ax3.set_ylabel('Ratio', fontsize=11)
        ax3.set_title(f'ε_max / ε_min Ratio - {model} (k={int(selected_k)})',
                     fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'{output_dir}/eps_components_{model}_k{int(selected_k)}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()


def plot_best_gap_by_dataset(df, output_dir='plots'):
    """Plot best gap achieved by each dataset with the configuration that achieved it."""
    Path(output_dir).mkdir(exist_ok=True)

    # Extract dataset from model name
    df['dataset'] = df['model'].apply(lambda x: 'MNIST' if 'mnist' in x
                                      else 'CIFAR-10' if 'cifar10' in x
                                      else 'GTSRB' if 'gtsrb' in x
                                      else 'Unknown')

    # Get best gap per dataset
    best_by_dataset = []
    for dataset in sorted(df['dataset'].unique()):
        dataset_data = df[df['dataset'] == dataset]
        best_row = dataset_data.loc[dataset_data['avg_gap'].idxmax()]
        best_by_dataset.append({
            'dataset': dataset,
            'best_gap': best_row['avg_gap'],
            'model': best_row['model'],
            'num_samples': int(best_row['num_samples']),
            'k_sparse': int(best_row['k_sparse']),
            'eps_min': best_row['avg_eps_min'],
            'eps_max': best_row['avg_eps_max']
        })

    best_df = pd.DataFrame(best_by_dataset)

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(best_df))
    bars = ax.bar(x, best_df['best_gap'], alpha=0.8, edgecolor='black', linewidth=1.5)

    # Color by dataset
    colors = {'MNIST': 'steelblue', 'CIFAR-10': 'darkorange', 'GTSRB': 'forestgreen', 'Unknown': 'gray'}
    for i, row in best_df.iterrows():
        bars[i].set_color(colors.get(row['dataset'], 'gray'))

    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Average Gap', fontsize=13, fontweight='bold')
    ax.set_title('Best XAI Bounds Gap by Dataset', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(best_df['dataset'], fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels with configuration
    for i, row in best_df.iterrows():
        # Main value
        ax.text(i, row['best_gap'], f"{row['best_gap']:.4f}",
               ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Configuration details below bar
        config_text = f"{row['model']}\ns={row['num_samples']}, k={row['k_sparse']}"
        ax.text(i, -0.01 * ax.get_ylim()[1], config_text,
               ha='center', va='top', fontsize=9, style='italic')

    # Extend y-axis to make room for labels
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 0.15 * (ylim[1] - ylim[0]), ylim[1] * 1.1)

    plt.tight_layout()
    filename = f'{output_dir}/best_gap_by_dataset.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

    return best_df


def plot_eps_summary_by_dataset(df, output_dir='plots'):
    """Plot eps_min and eps_max summary by dataset."""
    Path(output_dir).mkdir(exist_ok=True)

    # Extract dataset from model name
    df['dataset'] = df['model'].apply(lambda x: 'MNIST' if 'mnist' in x
                                      else 'CIFAR-10' if 'cifar10' in x
                                      else 'GTSRB' if 'gtsrb' in x
                                      else 'Unknown')

    # Compute statistics per dataset
    dataset_stats = df.groupby('dataset').agg({
        'avg_eps_min': ['mean', 'std', 'min', 'max'],
        'avg_eps_max': ['mean', 'std', 'min', 'max'],
        'avg_gap': ['mean', 'std', 'min', 'max']
    }).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    datasets = sorted(dataset_stats['dataset'].unique())
    x = np.arange(len(datasets))
    width = 0.35

    colors = {'MNIST': 'steelblue', 'CIFAR-10': 'darkorange', 'GTSRB': 'forestgreen', 'Unknown': 'gray'}
    bar_colors = [colors.get(ds, 'gray') for ds in datasets]

    # Plot 1: Mean eps_min and eps_max
    ax1 = axes[0, 0]
    eps_min_means = [dataset_stats[dataset_stats['dataset'] == ds]['avg_eps_min']['mean'].values[0]
                     for ds in datasets]
    eps_max_means = [dataset_stats[dataset_stats['dataset'] == ds]['avg_eps_max']['mean'].values[0]
                     for ds in datasets]

    ax1.bar(x - width/2, eps_min_means, width, label='ε_min (mean)', alpha=0.8,
            color=bar_colors, edgecolor='black')
    ax1.bar(x + width/2, eps_max_means, width, label='ε_max (mean)', alpha=0.8,
            color=bar_colors, edgecolor='black', hatch='//')

    ax1.set_ylabel('Epsilon Value', fontsize=12, fontweight='bold')
    ax1.set_title('Mean ε_min and ε_max by Dataset', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (emin, emax) in enumerate(zip(eps_min_means, eps_max_means)):
        ax1.text(i - width/2, emin, f'{emin:.4f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, emax, f'{emax:.4f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Range (min to max) of eps values
    ax2 = axes[0, 1]
    for i, ds in enumerate(datasets):
        ds_data = dataset_stats[dataset_stats['dataset'] == ds]
        eps_min_range = [ds_data['avg_eps_min']['min'].values[0],
                        ds_data['avg_eps_min']['max'].values[0]]
        eps_max_range = [ds_data['avg_eps_max']['min'].values[0],
                        ds_data['avg_eps_max']['max'].values[0]]

        ax2.plot([i - width/2, i - width/2], eps_min_range, 'o-',
                linewidth=3, markersize=8, color=bar_colors[i], label=f'{ds} ε_min range')
        ax2.plot([i + width/2, i + width/2], eps_max_range, 's-',
                linewidth=3, markersize=8, color=bar_colors[i], alpha=0.6, label=f'{ds} ε_max range')

    ax2.set_ylabel('Epsilon Value Range', fontsize=12, fontweight='bold')
    ax2.set_title('ε_min and ε_max Range by Dataset', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Mean gap by dataset
    ax3 = axes[1, 0]
    gap_means = [dataset_stats[dataset_stats['dataset'] == ds]['avg_gap']['mean'].values[0]
                 for ds in datasets]
    gap_stds = [dataset_stats[dataset_stats['dataset'] == ds]['avg_gap']['std'].values[0]
                for ds in datasets]

    bars = ax3.bar(x, gap_means, yerr=gap_stds, alpha=0.8, capsize=5,
                   color=bar_colors, edgecolor='black', linewidth=1.5)

    ax3.set_ylabel('Average Gap', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Gap by Dataset', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (gap, std) in enumerate(zip(gap_means, gap_stds)):
        ax3.text(i, gap, f'{gap:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 4: Statistical table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create table data
    table_data = []
    for ds in datasets:
        ds_data = dataset_stats[dataset_stats['dataset'] == ds]
        row = [
            ds,
            f"{ds_data['avg_eps_min']['mean'].values[0]:.4f}",
            f"{ds_data['avg_eps_max']['mean'].values[0]:.4f}",
            f"{ds_data['avg_gap']['mean'].values[0]:.4f}",
            f"[{ds_data['avg_gap']['min'].values[0]:.4f}, {ds_data['avg_gap']['max'].values[0]:.4f}]"
        ]
        table_data.append(row)

    table = ax4.table(cellText=table_data,
                     colLabels=['Dataset', 'Mean ε_min', 'Mean ε_max', 'Mean Gap', 'Gap Range'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows by dataset
    for i, ds in enumerate(datasets):
        for j in range(5):
            table[(i+1, j)].set_facecolor(colors.get(ds, 'white'))
            table[(i+1, j)].set_alpha(0.3)

    ax4.set_title('Summary Statistics by Dataset', fontsize=13, fontweight='bold', pad=20)

    plt.suptitle('Epsilon Components Analysis by Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/eps_summary_by_dataset.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_architecture_comparison(df, output_dir='plots'):
    """Compare CNN vs FC architectures."""
    Path(output_dir).mkdir(exist_ok=True)

    # Classify models
    df['arch'] = df['model'].apply(lambda x: 'CNN' if x.startswith('cnn3')
                                   else 'FC' if x.startswith('fc')
                                   else 'Other')

    # Group by architecture and configuration
    for (samples, k), group in df.groupby(['num_samples', 'k_sparse']):
        arch_stats = group.groupby('arch')['avg_gap'].agg(['mean', 'std', 'count'])

        if len(arch_stats) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(arch_stats))
        bars = ax.bar(x, arch_stats['mean'], yerr=arch_stats['std'],
                     alpha=0.7, capsize=5, edgecolor='black')

        # Color by architecture
        colors = {'CNN': 'steelblue', 'FC': 'darkorange', 'Other': 'gray'}
        for i, (arch, _) in enumerate(arch_stats.iterrows()):
            bars[i].set_color(colors.get(arch, 'gray'))

        ax.set_xlabel('Architecture', fontsize=12)
        ax.set_ylabel('Average Gap (mean ± std)', fontsize=12)
        ax.set_title(f'Architecture Comparison - samples={int(samples)}, k={int(k)}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{arch}\n(n={int(row["count"])})' for arch, row in arch_stats.iterrows()])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (arch, row) in enumerate(arch_stats.iterrows()):
            ax.text(i, row['mean'], f'{row["mean"]:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'{output_dir}/arch_comparison_s{int(samples)}_k{int(k)}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()


def plot_summary_dashboard(df, output_dir='plots'):
    """Create a summary dashboard with multiple subplots."""
    Path(output_dir).mkdir(exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Gap distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['avg_gap'], bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Average Gap')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Gap Distribution', fontweight='bold')
    ax1.axvline(df['avg_gap'].mean(), color='red', linestyle='--', label=f'Mean: {df["avg_gap"].mean():.4f}')
    ax1.legend()

    # 2. Gap vs k (all models)
    ax2 = fig.add_subplot(gs[0, 1])
    for model in df['model'].unique()[:5]:  # Limit to 5 models for clarity
        model_data = df[df['model'] == model]
        ax2.scatter(model_data['k_sparse'], model_data['avg_gap'], alpha=0.6, label=model, s=50)
    ax2.set_xlabel('k (Sparsity)')
    ax2.set_ylabel('Average Gap')
    ax2.set_title('Gap vs k (All Models)', fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)

    # 3. Gap vs samples (all models)
    ax3 = fig.add_subplot(gs[0, 2])
    for model in df['model'].unique()[:5]:
        model_data = df[df['model'] == model]
        ax3.scatter(model_data['num_samples'], model_data['avg_gap'], alpha=0.6, label=model, s=50)
    ax3.set_xlabel('Number of Samples')
    ax3.set_ylabel('Average Gap')
    ax3.set_title('Gap vs Samples (All Models)', fontweight='bold')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)

    # 4. Box plot by k
    ax4 = fig.add_subplot(gs[1, 0])
    df.boxplot(column='avg_gap', by='k_sparse', ax=ax4)
    ax4.set_xlabel('k (Sparsity)')
    ax4.set_ylabel('Average Gap')
    ax4.set_title('Gap Distribution by k', fontweight='bold')
    plt.sca(ax4)
    plt.xticks(rotation=0)

    # 5. Box plot by samples
    ax5 = fig.add_subplot(gs[1, 1])
    df.boxplot(column='avg_gap', by='num_samples', ax=ax5)
    ax5.set_xlabel('Number of Samples')
    ax5.set_ylabel('Average Gap')
    ax5.set_title('Gap Distribution by Samples', fontweight='bold')
    plt.sca(ax5)
    plt.xticks(rotation=45)

    # 6. Correlation heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    corr_data = df[['num_samples', 'k_sparse', 'avg_gap', 'avg_eps_min', 'avg_eps_max']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax6,
               square=True, linewidths=1)
    ax6.set_title('Parameter Correlation', fontweight='bold')

    # 7. Best configurations
    ax7 = fig.add_subplot(gs[2, :])
    top_gaps = df.nlargest(10, 'avg_gap')
    x = np.arange(len(top_gaps))
    bars = ax7.barh(x, top_gaps['avg_gap'], alpha=0.7, edgecolor='black')

    # Color gradient
    cmap = plt.cm.RdYlGn
    normalize = plt.Normalize(vmin=top_gaps['avg_gap'].min(), vmax=top_gaps['avg_gap'].max())
    for i, bar in enumerate(bars):
        bar.set_color(cmap(normalize(top_gaps['avg_gap'].iloc[i])))

    labels = [f"{row['model']} (s={int(row['num_samples'])}, k={int(row['k_sparse'])})"
             for _, row in top_gaps.iterrows()]
    ax7.set_yticks(x)
    ax7.set_yticklabels(labels, fontsize=9)
    ax7.set_xlabel('Average Gap')
    ax7.set_title('Top 10 Configurations by Gap', fontweight='bold', fontsize=12)
    ax7.grid(True, alpha=0.3, axis='x')

    # Add values on bars
    for i, (_, row) in enumerate(top_gaps.iterrows()):
        ax7.text(row['avg_gap'], i, f' {row["avg_gap"]:.4f}',
                va='center', fontsize=9, fontweight='bold')

    plt.suptitle('XAI Bounds Experiment Analysis - Summary Dashboard',
                fontsize=16, fontweight='bold', y=0.995)

    filename = f'{output_dir}/summary_dashboard.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def generate_all_plots(csv_path='summary_all_configs.csv', output_dir='plots',
                       detailed_comparisons=False):
    """
    Generate all analysis plots.

    Args:
        csv_path: Path to summary CSV file
        output_dir: Output directory for plots
        detailed_comparisons: If True, generate detailed comparison plots
                             (arch_comparison_*.png and comparison_*_s*_k*.png)
                             These can generate 50+ plots. Default: False
    """
    print("\n" + "="*70)
    print("XAI Bounds Results Visualization")
    print("="*70 + "\n")

    # Load data
    df = load_summary_data(csv_path)
    print()

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Generate plots
    print("Generating plots...\n")

    print("1. Gap vs k plots (per model)...")
    plot_gap_vs_k(df, output_dir)

    print("\n2. Gap vs samples plots (per model)...")
    plot_gap_vs_samples(df, output_dir)

    print("\n3. Heatmaps (per model)...")
    plot_heatmap(df, output_dir)

    print("\n4. Epsilon components (per model)...")
    plot_eps_components(df, output_dir)

    print("\n5. Best gap by dataset...")
    plot_best_gap_by_dataset(df, output_dir)

    print("\n6. Epsilon summary by dataset...")
    plot_eps_summary_by_dataset(df, output_dir)

    print("\n7. Summary dashboard...")
    plot_summary_dashboard(df, output_dir)

    if detailed_comparisons:
        print("\n8. Detailed dataset comparisons (per config)...")
        plot_comparison_by_dataset(df, output_dir)

        print("\n9. Detailed architecture comparisons (per config)...")
        plot_architecture_comparison(df, output_dir)
    else:
        print("\n⏭️  Skipping detailed comparisons (use --detailed to enable)")
        print("   This skips ~50+ plots: comparison_*_s*_k*.png and arch_comparison_s*_k*.png")

    print("\n" + "="*70)
    print(f"✓ All plots saved to: {output_dir}/")
    print("="*70)

    # List generated files
    plot_files = list(Path(output_dir).glob("*.png"))
    print(f"\nGenerated {len(plot_files)} plots:")
    for f in sorted(plot_files):
        print(f"  - {f.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate visualization plots for XAI bounds experiment results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default: generate main plots only
  python plot_results.py

  # Generate all plots including detailed comparisons
  python plot_results.py --detailed

  # Custom CSV and output directory
  python plot_results.py summary.csv my_plots/

  # Custom paths with detailed plots
  python plot_results.py summary.csv my_plots/ --detailed
        '''
    )

    parser.add_argument('csv_path', nargs='?', default='summary_all_configs.csv',
                       help='Path to summary CSV file (default: summary_all_configs.csv)')
    parser.add_argument('output_dir', nargs='?', default='plots',
                       help='Output directory for plots (default: plots/)')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed comparison plots (arch_comparison_*.png, comparison_*_s*_k*.png). '
                            'This can generate 50+ additional plots.')

    args = parser.parse_args()

    generate_all_plots(args.csv_path, args.output_dir, detailed_comparisons=args.detailed)
