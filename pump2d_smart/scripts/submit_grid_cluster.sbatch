#!/bin/bash
#SBATCH --job-name=smart2d_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ==============================================================================
# SLURM Job Script: 9-Experiment Grid for ECOTWIN SMART 2D Pump Surrogate
# ==============================================================================

set -e
mkdir -p logs

echo "=== Job Started: $(date) on $(hostname) ==="
echo "CUDA Devices available:"
nvidia-smi || true

# Determine script root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "Project Directory: $PWD"

# Sync dependencies with uv
echo "Syncing uv environment..."
uv sync

# Run 9-experiment grid with verified stable learning rate 0.001
echo "Starting 9-experiment grid execution (lr=0.001)..."
uv run python scripts/run_9_experiments_grid.py --lr 0.001

echo "=== Job Completed: $(date) ==="
