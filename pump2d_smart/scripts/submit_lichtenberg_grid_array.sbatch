#!/bin/bash
# ==============================================================================
# SLURM Job Array for TU Darmstadt Lichtenberg II (HHLR) Cluster
# Runs all 9 experiments in parallel across 9 GPUs (Total runtime: ~6 minutes!)
# ==============================================================================

#SBATCH -J pump2d_smart_array
#SBATCH -A l0003280
#SBATCH -p acc_short
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/%u/%x.err.%A_%a
#SBATCH -o /work/scratch/%u/%x.out.%A_%a
#
# Array specification: runs all 12 experiments, limiting concurrency to at most 2 nodes (%2)
#SBATCH --array=0-11%2
#
# CPU specification (optimized for PyTorch DataLoader num_workers=2 & ~1.6GB RAM)
#SBATCH -n 1                  # 1 process per task
#SBATCH -c 2                  # 2 CPU cores per task
#SBATCH --mem-per-cpu=2G      # 2GB per core = 4GB total (optimized efficiency)
#SBATCH -t 00:30:00           # 30 minutes limit (maximum for acc_short, plenty for 300 eps + eval)

# GPU specification
#SBATCH --gres=gpu:1          # 1 GPU per array task

# ------------------------------------------------------------------------------
# Environment Modules (Lichtenberg official stack)
# ------------------------------------------------------------------------------
module purge
module load gcc cuda python/3.13

# Navigate to working directory
export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
if [ -d "pump2d_smart" ]; then
    cd pump2d_smart
fi

# Activate virtual environment
[ -f ".venv/bin/activate" ] && source .venv/bin/activate || source ../.venv/bin/activate 2>/dev/null || true

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi 1>&2
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE
export DAGSHUB_USER_TOKEN="${DAGSHUB_USER_TOKEN:-eb369861c147078fa8825e89d0174c938c1d323c}"
export ANTIGRAVITY_ARTIFACT_DIR="${ANTIGRAVITY_ARTIFACT_DIR:-/home/vp51huji/.gemini/antigravity-cli/brain/f04a45a2-b7e3-4505-93ac-31d2429541b3}"

# Array Mapping (Full 12-Experiment Grid: 5, 10, 15 pts x 4 formulations):
EXPERIMENT_NAMES=(
    "grid_5pts_mse_only"
    "grid_5pts_hard_bc_only"
    "grid_5pts_mse_flux"
    "grid_5pts_hard_bc_flux"
    "grid_10pts_mse_only"
    "grid_10pts_hard_bc_only"
    "grid_10pts_mse_flux"
    "grid_10pts_hard_bc_flux"
    "grid_15pts_mse_only"
    "grid_15pts_hard_bc_only"
    "grid_15pts_mse_flux"
    "grid_15pts_hard_bc_flux"
)

TARGET_EXP="${EXPERIMENT_NAMES[$SLURM_ARRAY_TASK_ID]}"
echo "=== Starting Array Task $SLURM_ARRAY_TASK_ID: $TARGET_EXP on $(hostname) at $(date) ==="

python scripts/run_9_experiments_grid.py --lr 0.001 --epochs 300 --filter "$TARGET_EXP"
EXITCODE=$?

echo "=== Array Task $SLURM_ARRAY_TASK_ID ($TARGET_EXP) completed with exit code $EXITCODE at: $(date) ==="
exit $EXITCODE
