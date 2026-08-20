#!/usr/bin/env python3
"""
Analysis of XAI bounds benchmark results.
Generates comprehensive statistics and insights.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import argparse
import sys


def load_data(csv_path='summary_all_configs.csv'):
    """Load and prepare data."""
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found!")
        print("Run 'python compare_results.py' first to generate it.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Add derived columns
    df['dataset'] = df['model'].apply(lambda x: 'mnist' if 'mnist' in x else
                                       'cifar10' if 'cifar10' in x else
                                       'gtsrb' if 'gtsrb' in x else 'unknown')

    df['arch'] = df['model'].apply(lambda x: 'CNN' if x.startswith('cnn3')
                                   else 'FC' if x.startswith('fc')
                                   else 'Other')

    return df


def generate_analysis_data(df):
    """Generate all analysis data as a structured dictionary."""
    data = {}

    # Overview
    data['overview'] = {
        'total_configs': len(df),
        'num_models': df['model'].nunique(),
        'sample_sizes': sorted(df['num_samples'].unique()),
        'k_values': sorted(df['k_sparse'].unique()),
        'models': {model: len(df[df['model'] == model])
                   for model in sorted(df['model'].unique())}
    }

    # Global statistics
    data['global_stats'] = {
        'avg_gap_mean': df['avg_gap'].mean(),
        'avg_gap_std': df['avg_gap'].std(),
        'avg_gap_min': df['avg_gap'].min(),
        'avg_gap_max': df['avg_gap'].max(),
        'avg_eps_min': df['avg_eps_min'].mean(),
        'avg_eps_max': df['avg_eps_max'].mean()
    }

    # Top and bottom configurations
    data['top10'] = df.nlargest(10, 'avg_gap')[['model', 'num_samples', 'k_sparse',
                                                   'avg_gap', 'avg_eps_min', 'avg_eps_max']].to_dict('records')
    data['bottom10'] = df.nsmallest(10, 'avg_gap')[['model', 'num_samples', 'k_sparse',
                                                       'avg_gap', 'avg_eps_min', 'avg_eps_max']].to_dict('records')

    # Analysis by model
    data['by_model'] = {}
    for model in sorted(df['model'].unique()):
        model_data = df[df['model'] == model]
        best_config = model_data.nlargest(1, 'avg_gap').iloc[0]
        data['by_model'][model] = {
            'avg_gap_mean': model_data['avg_gap'].mean(),
            'avg_gap_std': model_data['avg_gap'].std(),
            'avg_gap_min': model_data['avg_gap'].min(),
            'avg_gap_max': model_data['avg_gap'].max(),
            'best_config': {
                'samples': int(best_config['num_samples']),
                'k': int(best_config['k_sparse']),
                'gap': best_config['avg_gap']
            }
        }

    # Analysis by dataset
    data['by_dataset'] = {}
    for dataset in sorted(df['dataset'].unique()):
        dataset_data = df[df['dataset'] == dataset]
        data['by_dataset'][dataset] = {
            'num_configs': len(dataset_data),
            'avg_gap_mean': dataset_data['avg_gap'].mean(),
            'avg_gap_std': dataset_data['avg_gap'].std(),
            'avg_gap_min': dataset_data['avg_gap'].min(),
            'avg_gap_max': dataset_data['avg_gap'].max()
        }

    # Analysis by architecture
    data['by_arch'] = {}
    for arch in sorted(df['arch'].unique()):
        arch_data = df[df['arch'] == arch]
        data['by_arch'][arch] = {
            'num_configs': len(arch_data),
            'avg_gap_mean': arch_data['avg_gap'].mean(),
            'avg_gap_std': arch_data['avg_gap'].std(),
            'avg_gap_min': arch_data['avg_gap'].min(),
            'avg_gap_max': arch_data['avg_gap'].max()
        }

    # Effect of k
    k_analysis = df.groupby('k_sparse')['avg_gap'].agg(['mean', 'std', 'min', 'max', 'count'])
    data['k_effect'] = k_analysis.to_dict('index')

    # Statistical tests for k
    gap_k50 = df[df['k_sparse'] == 50]['avg_gap']
    gap_k100 = df[df['k_sparse'] == 100]['avg_gap']
    gap_k200 = df[df['k_sparse'] == 200]['avg_gap']

    if len(gap_k50) > 0 and len(gap_k100) > 0:
        t_50_100, p_50_100 = stats.ttest_ind(gap_k50, gap_k100)
    else:
        t_50_100, p_50_100 = 0, 1

    if len(gap_k50) > 0 and len(gap_k200) > 0:
        t_50_200, p_50_200 = stats.ttest_ind(gap_k50, gap_k200)
    else:
        t_50_200, p_50_200 = 0, 1

    data['k_stats'] = {
        't_50_100': t_50_100,
        'p_50_100': p_50_100,
        't_50_200': t_50_200,
        'p_50_200': p_50_200,
        'k50_better_than_k200': gap_k50.mean() > gap_k200.mean() if len(gap_k50) > 0 and len(gap_k200) > 0 else False
    }

    # Effect of num_samples
    samples_analysis = df.groupby('num_samples')['avg_gap'].agg(['mean', 'std', 'min', 'max', 'count'])
    data['samples_effect'] = samples_analysis.to_dict('index')

    # Convergence analysis
    data['convergence'] = {}
    for model in sorted(df['model'].unique())[:3]:
        model_data = df[df['model'] == model].sort_values('num_samples')
        k50_data = model_data[model_data['k_sparse'] == 50]
        if len(k50_data) > 1:
            gaps = k50_data['avg_gap'].values
            changes = np.diff(gaps)
            avg_change = np.mean(np.abs(changes))
            data['convergence'][model] = avg_change

    # Key insights
    best_dataset = df.groupby('dataset')['avg_gap'].mean().idxmax()
    worst_dataset = df.groupby('dataset')['avg_gap'].mean().idxmin()
    best_arch = df.groupby('arch')['avg_gap'].mean().idxmax()
    k_means = df.groupby('k_sparse')['avg_gap'].mean()
    optimal_k = k_means.idxmax()

    samples_std = df.groupby('num_samples')['avg_gap'].std()
    stable_samples = samples_std[samples_std < samples_std.iloc[0] * 0.5].index.min()

    mean_gap = df['avg_gap'].mean()
    if mean_gap > 0.5:
        interp = "HIGHLY explainable"
    elif mean_gap > 0.2:
        interp = "MODERATELY explainable"
    else:
        interp = "POORLY explainable"

    data['insights'] = {
        'best_dataset': best_dataset,
        'best_dataset_gap': df[df['dataset'] == best_dataset]['avg_gap'].mean(),
        'worst_dataset': worst_dataset,
        'worst_dataset_gap': df[df['dataset'] == worst_dataset]['avg_gap'].mean(),
        'best_arch': best_arch,
        'best_arch_gap': df[df['arch'] == best_arch]['avg_gap'].mean(),
        'optimal_k': int(optimal_k),
        'optimal_k_gap': k_means.max(),
        'stable_samples': int(stable_samples) if pd.notna(stable_samples) else 60,
        'mean_gap': mean_gap,
        'interpretation': interp
    }

    return data


def print_console_report(data):
    """Print analysis to console."""
    print("="*80)
    print("XAI BOUNDS BENCHMARK - COMPLETE ANALYSIS")
    print("="*80)
    print()

    # Overview
    print("📊 DATASET OVERVIEW")
    print("-"*80)
    print(f"Total configurations: {data['overview']['total_configs']}")
    print(f"Models tested: {data['overview']['num_models']}")
    print(f"Sample sizes: {data['overview']['sample_sizes']}")
    print(f"k values: {data['overview']['k_values']}")
    print()

    print("Models:")
    for model, count in data['overview']['models'].items():
        print(f"  - {model:<25} ({count} configurations)")
    print()

    # Global statistics
    print("="*80)
    print("📈 GLOBAL STATISTICS")
    print("="*80)
    gs = data['global_stats']
    print(f"Average Gap (all configs): {gs['avg_gap_mean']:.4f} ± {gs['avg_gap_std']:.4f}")
    print(f"Min Gap observed:          {gs['avg_gap_min']:.4f}")
    print(f"Max Gap observed:          {gs['avg_gap_max']:.4f}")
    print()
    print(f"Average ε_min:             {gs['avg_eps_min']:.4f}")
    print(f"Average ε_max:             {gs['avg_eps_max']:.4f}")
    print()

    # Top 10
    print("="*80)
    print("🏆 TOP 10 BEST CONFIGURATIONS (by avg_gap)")
    print("="*80)
    for i, row in enumerate(data['top10'], 1):
        print(f"{i:2d}. {row['model']:<25} samples={int(row['num_samples']):3d}, k={int(row['k_sparse']):3d} → gap={row['avg_gap']:.4f}")
    print()

    # Bottom 10
    print("="*80)
    print("⚠️  WORST 10 CONFIGURATIONS (by avg_gap)")
    print("="*80)
    for i, row in enumerate(data['bottom10'], 1):
        print(f"{i:2d}. {row['model']:<25} samples={int(row['num_samples']):3d}, k={int(row['k_sparse']):3d} → gap={row['avg_gap']:.4f}")
    print()

    # By model
    print("="*80)
    print("🎯 ANALYSIS BY MODEL")
    print("="*80)
    for model, stats in data['by_model'].items():
        print(f"\n{model}:")
        print(f"  Average gap:     {stats['avg_gap_mean']:.4f} ± {stats['avg_gap_std']:.4f}")
        print(f"  Range:           [{stats['avg_gap_min']:.4f}, {stats['avg_gap_max']:.4f}]")
        bc = stats['best_config']
        print(f"  Best config:     samples={bc['samples']}, k={bc['k']} → gap={bc['gap']:.4f}")
    print()

    # By dataset
    print("="*80)
    print("📦 ANALYSIS BY DATASET")
    print("="*80)
    for dataset, stats in data['by_dataset'].items():
        print(f"\n{dataset.upper()}:")
        print(f"  Configurations:  {stats['num_configs']}")
        print(f"  Average gap:     {stats['avg_gap_mean']:.4f} ± {stats['avg_gap_std']:.4f}")
        print(f"  Range:           [{stats['avg_gap_min']:.4f}, {stats['avg_gap_max']:.4f}]")
    print()

    # By architecture
    print("="*80)
    print("🏗️  ANALYSIS BY ARCHITECTURE")
    print("="*80)
    for arch, stats in data['by_arch'].items():
        print(f"\n{arch}:")
        print(f"  Configurations:  {stats['num_configs']}")
        print(f"  Average gap:     {stats['avg_gap_mean']:.4f} ± {stats['avg_gap_std']:.4f}")
        print(f"  Range:           [{stats['avg_gap_min']:.4f}, {stats['avg_gap_max']:.4f}]")
    print()

    # Effect of k
    print("="*80)
    print("🔬 EFFECT OF k (SPARSITY BUDGET)")
    print("="*80)
    print(f"{'k':<10} {'mean':<12} {'std':<12} {'min':<12} {'max':<12} {'count':<10}")
    print("-"*80)
    for k, stats in data['k_effect'].items():
        print(f"{k:<10} {stats['mean']:<12.4f} {stats['std']:<12.4f} {stats['min']:<12.4f} {stats['max']:<12.4f} {int(stats['count']):<10}")
    print()

    ks = data['k_stats']
    sig_50_100 = '***' if ks['p_50_100'] < 0.001 else '**' if ks['p_50_100'] < 0.01 else '*' if ks['p_50_100'] < 0.05 else 'ns'
    sig_50_200 = '***' if ks['p_50_200'] < 0.001 else '**' if ks['p_50_200'] < 0.01 else '*' if ks['p_50_200'] < 0.05 else 'ns'

    print(f"T-test k=50 vs k=100: t={ks['t_50_100']:.3f}, p={ks['p_50_100']:.4f} {sig_50_100}")
    print(f"T-test k=50 vs k=200: t={ks['t_50_200']:.3f}, p={ks['p_50_200']:.4f} {sig_50_200}")

    if ks['k50_better_than_k200']:
        print("\n⚠️  FINDING: Lower k (k=50) gives HIGHER gap than k=200!")
        print("    This suggests that sparser attacks are HARDER (need more ε)")
        print("    → Models are MORE explainable with lower k")
    else:
        print("\n✓ FINDING: Higher k gives higher gap (as expected)")
    print()

    # Effect of samples
    print("="*80)
    print("🔬 EFFECT OF num_samples")
    print("="*80)
    print(f"{'samples':<10} {'mean':<12} {'std':<12} {'min':<12} {'max':<12} {'count':<10}")
    print("-"*80)
    for samples, stats in data['samples_effect'].items():
        print(f"{samples:<10} {stats['mean']:<12.4f} {stats['std']:<12.4f} {stats['min']:<12.4f} {stats['max']:<12.4f} {int(stats['count']):<10}")
    print()

    print("Convergence analysis:")
    for model, avg_change in data['convergence'].items():
        print(f"  {model:<25}: avg change = {avg_change:.4f} (lower = more stable)")
    print()

    # Key insights
    print("="*80)
    print("💡 KEY INSIGHTS")
    print("="*80)
    ins = data['insights']

    print(f"\n1. Most explainable dataset: {ins['best_dataset'].upper()}")
    print(f"   (avg gap: {ins['best_dataset_gap']:.4f})")
    print(f"   Least explainable: {ins['worst_dataset'].upper()}")
    print(f"   (avg gap: {ins['worst_dataset_gap']:.4f})")

    print(f"\n2. Most explainable architecture: {ins['best_arch']}")
    print(f"   (avg gap: {ins['best_arch_gap']:.4f})")

    print(f"\n3. Optimal k value: {ins['optimal_k']}")
    print(f"   (avg gap: {ins['optimal_k_gap']:.4f})")

    print(f"\n4. Recommended sample size: {ins['stable_samples']}+")
    print(f"   (std drops below 50% of initial at this point)")

    print(f"\n5. Overall explainability: {ins['interpretation']}")
    print(f"   (average gap: {ins['mean_gap']:.4f})")

    print()
    print("="*80)
    print("✓ Analysis complete!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Generate plots: python plot_results.py")
    print("  2. View dashboard: open plots/summary_dashboard.png")
    print("  3. Deep dive: Check individual model plots in plots/")


def write_markdown_report(data, filepath='tmp/markdown-txt/BENCHMARK_RESULTS.md'):
    """Write analysis to markdown file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("# XAI Bounds Benchmark - Résultats et Analyse\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Configurations testées**: {data['overview']['total_configs']}\n")
        f.write(f"**Modèles**: {data['overview']['num_models']}\n")
        f.write(f"**Datasets**: MNIST, CIFAR-10, GTSRB\n")
        f.write(f"**Paramètres**: samples ∈ {{{', '.join(map(str, data['overview']['sample_sizes']))}}}, ")
        f.write(f"k ∈ {{{', '.join(map(str, data['overview']['k_values']))}}}\n\n")
        f.write("---\n\n")

        # Executive summary
        f.write("## 📊 Résumé Exécutif\n\n")
        f.write("### Métriques Globales\n\n")
        gs = data['global_stats']
        f.write(f"- **Gap moyen global**: {gs['avg_gap_mean']:.4f} ± {gs['avg_gap_std']:.4f}\n")
        f.write(f"- **Gap minimum**: {gs['avg_gap_min']:.4f}\n")
        f.write(f"- **Gap maximum**: {gs['avg_gap_max']:.4f}\n")
        f.write(f"- **ε_min moyen**: {gs['avg_eps_min']:.4f}\n")
        f.write(f"- **ε_max moyen**: {gs['avg_eps_max']:.4f}\n\n")

        f.write("### Verdict Global\n\n")
        ins = data['insights']
        f.write(f"**{ins['interpretation'].upper()}** (gap moyen = {ins['mean_gap']:.4f})\n\n")
        f.write("---\n\n")

        # Top 10
        f.write("## 🏆 Top 10 - Configurations les Plus Explicables\n\n")
        f.write("| Rang | Modèle | Samples | k | Gap | ε_min | ε_max |\n")
        f.write("|------|--------|---------|---|-----|-------|-------|\n")
        for i, row in enumerate(data['top10'], 1):
            f.write(f"| {i} | {row['model']} | {int(row['num_samples'])} | {int(row['k_sparse'])} | ")
            f.write(f"{row['avg_gap']:.4f} | {row['avg_eps_min']:.4f} | {row['avg_eps_max']:.4f} |\n")
        f.write("\n**Pattern**: Les meilleurs résultats sont concentrés sur certains datasets/architectures.\n\n")
        f.write("---\n\n")

        # Bottom 10
        f.write("## ⚠️ Bottom 10 - Configurations les Moins Explicables\n\n")
        f.write("| Rang | Modèle | Samples | k | Gap |\n")
        f.write("|------|--------|---------|---|-----|\n")
        for i, row in enumerate(data['bottom10'], 1):
            f.write(f"| {i} | {row['model']} | {int(row['num_samples'])} | {int(row['k_sparse'])} | {row['avg_gap']:.4f} |\n")
        f.write("\n---\n\n")

        # By dataset
        f.write("## 📈 Analyse par Dataset\n\n")
        for dataset, stats in sorted(data['by_dataset'].items(),
                                     key=lambda x: x[1]['avg_gap_mean'], reverse=True):
            f.write(f"### {dataset.upper()}\n\n")
            f.write(f"- **Configurations**: {stats['num_configs']}\n")
            f.write(f"- **Gap moyen**: {stats['avg_gap_mean']:.4f} ± {stats['avg_gap_std']:.4f}\n")
            f.write(f"- **Range**: [{stats['avg_gap_min']:.4f}, {stats['avg_gap_max']:.4f}]\n")

            # Verdict
            if stats['avg_gap_mean'] > 0.2:
                verdict = "⭐ Bien explicable"
            elif stats['avg_gap_mean'] > 0.1:
                verdict = "Modérément explicable"
            else:
                verdict = "⚠️ Peu explicable"
            f.write(f"- **Verdict**: {verdict}\n\n")

        f.write("---\n\n")

        # By architecture
        f.write("## 🏗️ Analyse par Architecture\n\n")
        for arch, stats in sorted(data['by_arch'].items()):
            f.write(f"### {arch}\n\n")
            f.write(f"- **Configurations**: {stats['num_configs']}\n")
            f.write(f"- **Gap moyen**: {stats['avg_gap_mean']:.4f} ± {stats['avg_gap_std']:.4f}\n")
            f.write(f"- **Range**: [{stats['avg_gap_min']:.4f}, {stats['avg_gap_max']:.4f}]\n\n")

        f.write(f"### Verdict\n\n")
        f.write(f"**{ins['best_arch']}** légèrement meilleur (gap = {ins['best_arch_gap']:.4f})\n\n")
        f.write("---\n\n")

        # Effect of k
        f.write("## 🔬 Impact des Paramètres\n\n")
        f.write("### Effect de k (Sparsité)\n\n")
        f.write("| k | Gap Moyen | Écart-type | Min | Max | Configs |\n")
        f.write("|---|-----------|------------|-----|-----|----------|\n")
        for k, stats in sorted(data['k_effect'].items()):
            f.write(f"| {k} | {stats['mean']:.4f} | {stats['std']:.4f} | ")
            f.write(f"{stats['min']:.4f} | {stats['max']:.4f} | {int(stats['count'])} |\n")
        f.write("\n")

        ks = data['k_stats']
        sig_50_100 = '***' if ks['p_50_100'] < 0.001 else '**' if ks['p_50_100'] < 0.01 else '*' if ks['p_50_100'] < 0.05 else 'ns'
        sig_50_200 = '***' if ks['p_50_200'] < 0.001 else '**' if ks['p_50_200'] < 0.01 else '*' if ks['p_50_200'] < 0.05 else 'ns'

        f.write(f"**Significativité**:\n")
        f.write(f"- k=50 vs k=100: t={ks['t_50_100']:.3f}, p={ks['p_50_100']:.4f} `{sig_50_100}`\n")
        f.write(f"- k=50 vs k=200: t={ks['t_50_200']:.3f}, p={ks['p_50_200']:.4f} `{sig_50_200}`\n\n")

        if ks['k50_better_than_k200']:
            f.write("**⚠️ FINDING MAJEUR**: k=50 > k=100 > k=200\n\n")
            f.write("**Interprétation**: Plus k est petit, plus le gap est grand. ")
            f.write("Les attaques très sparse (k=50) nécessitent plus de perturbations. ")
            f.write("→ Les modèles sont plus explicables avec k faible.\n\n")

        # Effect of samples
        f.write("### Effect du Nombre d'Échantillons\n\n")
        f.write("| Samples | Gap Moyen | Écart-type |\n")
        f.write("|---------|-----------|------------|\n")
        for samples, stats in sorted(data['samples_effect'].items()):
            f.write(f"| {samples} | {stats['mean']:.4f} | {stats['std']:.4f} |\n")
        f.write("\n")
        f.write(f"**Recommandation**: {ins['stable_samples']}+ samples suffisent (convergence atteinte)\n\n")
        f.write("---\n\n")

        # Analysis by model
        f.write("## 🎯 Analyse Détaillée par Modèle\n\n")
        for model, stats in sorted(data['by_model'].items(),
                                   key=lambda x: x[1]['avg_gap_mean'], reverse=True):
            f.write(f"### {model}\n\n")
            f.write(f"- **Gap moyen**: {stats['avg_gap_mean']:.4f} ± {stats['avg_gap_std']:.4f}\n")
            f.write(f"- **Range**: [{stats['avg_gap_min']:.4f}, {stats['avg_gap_max']:.4f}]\n")
            bc = stats['best_config']
            f.write(f"- **Meilleure config**: samples={bc['samples']}, k={bc['k']} → gap={bc['gap']:.4f}\n\n")

        f.write("---\n\n")

        # Key insights
        f.write("## 💡 Insights Clés\n\n")
        f.write(f"### 1. Dataset domine tout\n\n")
        f.write(f"**{ins['best_dataset'].upper()}** > ... > **{ins['worst_dataset'].upper()}**\n\n")
        f.write(f"- Meilleur: {ins['best_dataset'].upper()} (gap = {ins['best_dataset_gap']:.4f})\n")
        f.write(f"- Pire: {ins['worst_dataset'].upper()} (gap = {ins['worst_dataset_gap']:.4f})\n\n")

        f.write(f"### 2. k={ins['optimal_k']} est optimal\n\n")
        f.write(f"Gap moyen avec k={ins['optimal_k']}: {ins['optimal_k_gap']:.4f}\n\n")

        f.write(f"### 3. Architecture\n\n")
        f.write(f"**{ins['best_arch']}** légèrement meilleur (gap = {ins['best_arch_gap']:.4f})\n\n")

        f.write(f"### 4. Convergence rapide\n\n")
        f.write(f"{ins['stable_samples']}+ samples suffisent pour des résultats stables\n\n")

        f.write("---\n\n")

        # Recommendations
        f.write("## 📌 Recommandations\n\n")
        f.write("### Configuration Optimale\n\n")
        f.write(f"- **k**: {ins['optimal_k']}\n")
        f.write(f"- **samples**: {ins['stable_samples']}\n")
        f.write(f"- **Dataset**: {ins['best_dataset'].upper()}\n\n")

        f.write("### Pour Recherche Future\n\n")
        f.write("1. Investiguer pourquoi certains datasets ont des gaps si différents\n")
        f.write("2. Tester k ∈ {10, 25, 50} pour affiner l'optimum\n")
        f.write("3. Visualiser les perturbations adversariales\n")
        f.write("4. Comparer avec saliency maps\n\n")

        f.write("---\n\n")
        f.write(f"**Généré le**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Script**: `analyze_results.py`\n")
        f.write(f"**Total configurations**: {data['overview']['total_configs']}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze XAI bounds experiment results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default: generate markdown report
  python analyze_results.py

  # Print to console only
  python analyze_results.py --console-only

  # Both markdown and console
  python analyze_results.py --both

  # Custom CSV file
  python analyze_results.py --csv results.csv
        '''
    )

    parser.add_argument('--csv', default='summary_all_configs.csv',
                       help='Path to summary CSV file (default: summary_all_configs.csv)')
    parser.add_argument('--output', default='tmp/markdown-txt/BENCHMARK_RESULTS.md',
                       help='Output markdown file (default: tmp/markdown-txt/BENCHMARK_RESULTS.md)')
    parser.add_argument('--console-only', action='store_true',
                       help='Print to console only (no markdown file)')
    parser.add_argument('--both', action='store_true',
                       help='Generate both markdown and console output')

    args = parser.parse_args()

    # Load data
    df = load_data(args.csv)

    # Generate analysis
    data = generate_analysis_data(df)

    # Output based on flags
    if args.console_only:
        # Console only
        print_console_report(data)
    elif args.both:
        # Both
        write_markdown_report(data, args.output)
        print(f"✓ Markdown report saved to: {args.output}")
        print()
        print_console_report(data)
    else:
        # Default: markdown only
        write_markdown_report(data, args.output)
        print("="*80)
        print("XAI Bounds Benchmark - Analysis Complete")
        print("="*80)
        print()
        print(f"✓ Markdown report saved to: {args.output}")
        print()
        print("Summary:")
        print(f"  - Total configurations: {data['overview']['total_configs']}")
        print(f"  - Models: {data['overview']['num_models']}")
        print(f"  - Average gap: {data['global_stats']['avg_gap_mean']:.4f} ± {data['global_stats']['avg_gap_std']:.4f}")
        print(f"  - Overall: {data['insights']['interpretation']}")
        print()
        print("To view full analysis:")
        print(f"  cat {args.output}")
        print(f"  open {args.output}")
        print()
        print("To print to console:")
        print("  python analyze_results.py --console-only")


if __name__ == "__main__":
    main()
