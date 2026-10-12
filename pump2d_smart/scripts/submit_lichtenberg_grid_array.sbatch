#!/bin/bash
# ==============================================================================
# SLURM Job Array for TU Darmstadt Lichtenberg II (HHLR) Cluster
# Runs all 9 experiments in parallel across 9 GPUs (Total runtime: ~6 minutes!)
# ==============================================================================

#SBATCH -J pump2d_smart_array
#SBATCH -A <project_name>
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/%u/%x.err.%A_%a
#SBATCH -o /work/scratch/%u/%x.out.%A_%a
#
# Array specification: 9 independent jobs (index 0 to 8)
#SBATCH --array=0-8
#
# CPU specification
#SBATCH -n 1                  # 1 process per task
#SBATCH -c 4                  # 4 CPU cores per process ($SLURM_CPUS_PER_TASK)
#SBATCH --mem-per-cpu=2G      # Hauptspeicher in MByte pro Rechenkern
#SBATCH -t 00:20:00           # 20 minutes limit per task (runs take ~5-6 min)

# GPU specification
#SBATCH --gres=gpu:1          # 1 GPU per array task

# ------------------------------------------------------------------------------
# Environment Modules (Lichtenberg official stack)
# ------------------------------------------------------------------------------
module purge
module load gcc cuda python/3.13

# Navigate to working directory
cd "${SLURM_SUBMIT_DIR:-.}"

# Activate virtual environment
[ -f ".venv/bin/activate" ] && source .venv/bin/activate || source ../.venv/bin/activate 2>/dev/null || true

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi 1>&2
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE

# Array Mapping:
EXPERIMENT_NAMES=(
    "grid_5pts_mse_only"
    "grid_5pts_hard_bc_only"
    "grid_5pts_hard_bc_flux"
    "grid_10pts_mse_only"
    "grid_10pts_hard_bc_only"
    "grid_10pts_hard_bc_flux"
    "grid_15pts_mse_only"
    "grid_15pts_hard_bc_only"
    "grid_15pts_hard_bc_flux"
)

TARGET_EXP="${EXPERIMENT_NAMES[$SLURM_ARRAY_TASK_ID]}"
echo "=== Starting Array Task $SLURM_ARRAY_TASK_ID: $TARGET_EXP on $(hostname) at $(date) ==="

python scripts/run_9_experiments_grid.py --lr 0.001 --filter "$TARGET_EXP"
EXITCODE=$?

echo "=== Array Task $SLURM_ARRAY_TASK_ID ($TARGET_EXP) completed with exit code $EXITCODE at: $(date) ==="
exit $EXITCODE
