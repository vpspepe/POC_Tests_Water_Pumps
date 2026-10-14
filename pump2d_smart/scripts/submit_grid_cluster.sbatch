#!/bin/bash
# ==============================================================================
# SLURM Submission Script for TU Darmstadt Lichtenberg II (HHLR) Cluster
# Follows the exact official project template from submit_lichtenberg.sh
# ==============================================================================

#SBATCH -J pump2d_smart_grid
#SBATCH -A l0003280
#SBATCH -p acc
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/%u/%x.err.%j
#SBATCH -o /work/scratch/%u/%x.out.%j
#
# CPU specification (optimized for PyTorch DataLoader num_workers=2 & ~1.6GB RAM)
#SBATCH -n 1                  # 1 process
#SBATCH -c 2                  # 2 CPU cores (cuts core-walltime in half, doubling efficiency)
#SBATCH --mem-per-cpu=2G      # 2GB per core = 4GB total (40-50% efficiency for ~1.6GB usage)
#SBATCH -t 06:00:00           # in hours:minutes

# GPU specification
#SBATCH --gres=gpu:1          # 1 GPU (Tesla V100)

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

# Activate virtual environment (local or parent folder)
[ -f ".venv/bin/activate" ] && source .venv/bin/activate || source ../.venv/bin/activate 2>/dev/null || true

# Thread specification
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# GPU verification output (appears in the .err file)
nvidia-smi 1>&2

export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE
export DAGSHUB_USER_TOKEN="${DAGSHUB_USER_TOKEN:-eb369861c147078fa8825e89d0174c938c1d323c}"
export ANTIGRAVITY_ARTIFACT_DIR="${ANTIGRAVITY_ARTIFACT_DIR:-/home/vp51huji/.gemini/antigravity-cli/brain/f04a45a2-b7e3-4505-93ac-31d2429541b3}"

# ------------------------------------------------------------------------------
# Payload: Run the 9-experiment grid with verified stable lr=0.001 and 300 epochs
# ------------------------------------------------------------------------------
FILTER_ARG="${1:-}"
echo "=== Starting sequential grid execution on Lichtenberg II: $(date) ==="
echo "Filter: ${FILTER_ARG:-All 9 experiments} | Epochs: 300 | Learning Rate: 0.001"
python scripts/run_9_experiments_grid.py --lr 0.001 --epochs 300 ${FILTER_ARG:+--filter "$FILTER_ARG"}
EXITCODE=$?

echo "=== Grid execution finished with exit code $EXITCODE at: $(date) ==="
exit $EXITCODE
