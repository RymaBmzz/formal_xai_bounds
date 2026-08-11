#!/bin/bash
# Script to activate the environment and run experiments

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

# Parse command line arguments
case "${1:-all}" in
    all)
        shift  # Remove 'all' from arguments
        echo "Running experiments on all models..."
        python run_all_models.py "$@"
        ;;
    single)
        shift  # Remove 'single' from arguments
        if [ -z "$1" ]; then
            echo "Usage: ./run.sh single <model_name> [k_sparse] [num_samples]"
            echo "Example: ./run.sh single cnn3_mnist.pt 100 20"
            exit 1
        fi
        python run_single_model.py "$@"
        ;;
    mnist)
        echo "Running original MNIST CNN3 experiment..."
        python mnist_cnn3.py
        ;;
    list)
        echo "Available models:"
        ls -1 models/*.pt
        ;;
    compare)
        shift  # Remove 'compare' from arguments
        echo "Comparing results from all experiments..."
        python compare_results.py "$@"
        ;;
    help|--help|-h)
        echo "Usage: ./run.sh [COMMAND] [OPTIONS]"
        echo ""
        echo "Commands:"
        echo "  all [--k K] [--samples N]        Run experiments on all models"
        echo "  single <model> [k] [samples]     Run experiment on specific model"
        echo "  mnist                            Run original MNIST CNN3 script"
        echo "  list                             List available models"
        echo "  compare [results_dir]            Compare results from experiments"
        echo "  help                             Show this help message"
        echo ""
        echo "Options for 'all' command:"
        echo "  --k, --k-sparse K                Sparsity budget (default: 50)"
        echo "  --samples, --num-samples N       Number of samples (default: 10)"
        echo "  --eps-fav EPS                    FAVEX radius (default: 0.25)"
        echo ""
        echo "Examples:"
        echo "  ./run.sh all                                  # Default: k=50, samples=10"
        echo "  ./run.sh all --k 100                          # k=100, samples=10"
        echo "  ./run.sh all --samples 20                     # k=50, samples=20"
        echo "  ./run.sh all --k 100 --samples 20             # k=100, samples=20"
        echo ""
        echo "  ./run.sh single cnn3_mnist.pt                 # Default parameters"
        echo "  ./run.sh single cnn3_mnist.pt 100             # k=100"
        echo "  ./run.sh single cnn3_mnist.pt 100 20          # k=100, samples=20"
        echo ""
        echo "  ./run.sh compare                              # Compare all results"
        echo "  ./run.sh compare results/10_samples_k_50      # Compare specific experiment"
        echo ""
        echo "Results are saved to: results/{num_samples}_samples_k_{k_sparse}/"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './run.sh help' for usage information"
        exit 1
        ;;
esac
