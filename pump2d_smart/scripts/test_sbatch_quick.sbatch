#!/bin/bash
# ==============================================================================
# SLURM Quick Test Script for TU Darmstadt Lichtenberg II (HHLR) Cluster
# Verifies GPU allocation, modules, dataset slicing, DAGsHub tracking & 1 epoch
# ==============================================================================

#SBATCH -J test_quick_sbatch
#SBATCH -A l0003280
#SBATCH -p acc_short
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/%u/%x.err.%j
#SBATCH -o /work/scratch/%u/%x.out.%j
#
# CPU specification
#SBATCH -n 1                  # 1 process
#SBATCH -c 4                  # 4 CPU cores per process ($SLURM_CPUS_PER_TASK)
#SBATCH --mem-per-cpu=4G      # 4GB per core = 16GB total
#SBATCH -t 00:10:00           # 10 minutes maximum

# GPU specification
#SBATCH --gres=gpu:1          # 1 GPU

set -e

echo "=== Quick SLURM Test Started: $(date) ==="
echo "Node: $(hostname)"
echo "Current directory: $PWD"
echo "Physical PWD: $(pwd -P)"
echo "User: $USER"

# 1. Environment Modules
echo "Loading Lichtenberg official modules..."
module purge
module load gcc cuda python/3.13

# 2. Paths and environment variables
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE
export DAGSHUB_USER_TOKEN="${DAGSHUB_USER_TOKEN:-eb369861c147078fa8825e89d0174c938c1d323c}"
export ANTIGRAVITY_ARTIFACT_DIR="${ANTIGRAVITY_ARTIFACT_DIR:-/home/vp51huji/.gemini/antigravity-cli/brain/f04a45a2-b7e3-4505-93ac-31d2429541b3}"

# 3. Virtual Environment
echo "Activating virtual environment..."
cd "${SLURM_SUBMIT_DIR:-.}"
if [ -d "pump2d_smart" ]; then
    cd pump2d_smart
fi
echo "Working directory for training: $PWD"
[ -f ".venv/bin/activate" ] && source .venv/bin/activate || source ../.venv/bin/activate

# 4. Check GPU visibility
echo "Verifying GPU with nvidia-smi:"
nvidia-smi

echo "Verifying PyTorch CUDA:"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"

# 5. Check Dataset Loading & Boundary Slicing
echo "Verifying Dataset boundary slice:"
python -c "
from src.data.dataset import Pump2DDataset
ds = Pump2DDataset(
    main_vtu_path='data/raw/comsol/Pump2D_MRFStudy_finer_Main.vtu',
    outlet_vtu_path='data/raw/comsol/Pump2D_MRFStudy_finer_Outlet.vtu',
    cache_dir='./cache',
    boundary_only=True
)
print('Dataset slice check passed! Boundary points:', ds.samples[0]['volume_coords'].shape)
"

# 6. Run 1-Epoch Quick Training Test
echo "Running 1-epoch quick training test with DAGsHub tracking:"
python -m src.training.train \
    --config-name=local_train \
    exp_name=quick_sbatch_test \
    data.n_train=5 \
    training.epochs=1 \
    tracking.target=remote \
    tracking.experiment_name=Pump2D_SMART_Surrogate

echo "=== Quick SLURM Test Finished Successfully at $(date) ==="
