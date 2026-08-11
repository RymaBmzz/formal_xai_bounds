# Results Directory Structure

This directory contains experiment results organized by parameters.

## Directory Naming Convention

```
results/
├── <num_samples>_samples_k_<k_sparse>/
│   ├── eps_bounds_<model1>.csv
│   ├── eps_bounds_<model2>.csv
│   ├── ...
│   └── summary_all_models.csv
└── ...
```

### Parameters in Directory Name

- `<num_samples>`: Number of test samples evaluated (e.g., 10, 20, 50)
- `<k_sparse>`: Sparsity budget for Sparse-PGD attack (e.g., 50, 100)

### Examples

```
results/10_samples_k_50/     # 10 samples, k=50
results/20_samples_k_50/     # 20 samples, k=50
results/10_samples_k_100/    # 10 samples, k=100
```

## File Contents

### Individual Results: `eps_bounds_<model_name>.csv`

Columns:
- `index`: Sample index (0 to num_samples-1)
- `true_label`: Ground truth class label
- `eps_min`: Minimum perturbation for successful attack (AutoAttack)
- `eps_max`: Minimum perturbation for k-sparse attack (Sparse-PGD)
- `gap`: eps_max - eps_min (explainability measure)

### Summary: `summary_all_models.csv`

Aggregated statistics across all models:
- Model name
- Number of samples
- Average/Min/Max gap
- Average eps_min and eps_max

## How Results Are Generated

### Automatic (via run_all_models.py)

When you run experiments, results are automatically saved to:
```
results/<num_samples>_samples_k_<k_sparse>/
```

Example:
```bash
# This creates results in: results/10_samples_k_50/
./run.sh all
```

### Manual (via run_single_model.py)

You can specify custom parameters:
```bash
# k=100, num_samples=20 → results/20_samples_k_100/
python run_single_model.py cnn3_mnist.pt 100 20
```

## Comparing Results

To compare results from a specific experiment:

```bash
# Compare results from default experiment (10 samples, k=50)
python compare_results.py results/10_samples_k_50

# Compare results from custom experiment
python compare_results.py results/20_samples_k_100

# Compare all results in current directory (searches all subdirectories)
python compare_results.py
```

## Git Tracking

- Individual result CSVs are **gitignored** (too large)
- Summary CSVs are **tracked** (summary_all_models.csv)
- Directory structure is **documented** here

## Interpreting Results

### Gap (eps_max - eps_min)

- **Large gap**: Model is easier to explain
  - Sparse explanations require much larger perturbations than dense attacks
  - k-sparse attacks are less effective
  
- **Small gap**: Model is harder to explain
  - Sparse explanations work nearly as well as dense attacks
  - k-sparse attacks are very effective

### Typical Values

- eps_min: Usually 0.05 - 0.25 (depends on model robustness)
- eps_max: Usually 0.2 - 0.8 (depends on k and model)
- gap: Usually 0.1 - 0.6 (higher is better for explainability)

## Notes

- GTSRB models are automatically skipped (dataset loader not implemented)
- Experiments can take 30-80 minutes per model
- CUDA acceleration is used if available
- Results are deterministic (AutoAttack uses random_seed internally)
