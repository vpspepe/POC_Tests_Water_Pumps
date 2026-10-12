#!/bin/bash
# ==============================================================================
# SLURM Submission Script for TU Darmstadt Lichtenberg II (HHLR) Cluster
# Follows the exact official project template from submit_lichtenberg.sh
# ==============================================================================

#SBATCH -J pump2d_smart_grid
#SBATCH -A <project_name>
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/%u/%x.err.%j
#SBATCH -o /work/scratch/%u/%x.out.%j
#
# CPU specification
#SBATCH -n 1                  # 1 process
#SBATCH -c 4                  # 4 CPU cores per process ($SLURM_CPUS_PER_TASK)
#SBATCH --mem-per-cpu=2G      # Hauptspeicher in MByte pro Rechenkern
#SBATCH -t 02:00:00           # in hours:minutes (approx 45-60 min for 9 runs)

# GPU specification
#SBATCH --gres=gpu:1          # 1 GPU (generic "gpu:1" matches any available NVIDIA GPU)

# ------------------------------------------------------------------------------
# Environment Modules (Lichtenberg official stack)
# ------------------------------------------------------------------------------
module purge
module load gcc cuda python/3.13

# Navigate to working directory
cd "${SLURM_SUBMIT_DIR:-.}"

# Activate virtual environment (local or parent folder)
[ -f ".venv/bin/activate" ] && source .venv/bin/activate || source ../.venv/bin/activate 2>/dev/null || true

# Thread specification
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# GPU verification output (appears in the .err file)
nvidia-smi 1>&2

export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE

# ------------------------------------------------------------------------------
# Payload: Run the 9-experiment grid with verified stable lr=0.001
# ------------------------------------------------------------------------------
echo "=== Starting 9-experiment grid execution on Lichtenberg II: $(date) ==="
python scripts/run_9_experiments_grid.py --lr 0.001
EXITCODE=$?

echo "=== Grid execution finished with exit code $EXITCODE at: $(date) ==="
exit $EXITCODE
