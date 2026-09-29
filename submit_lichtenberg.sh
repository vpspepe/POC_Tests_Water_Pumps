#!/bin/bash

#SBATCH -J pump2d_smart
#SBATCH -A <project_name>
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/%u/%x.err.%j
#SBATCH -o /work/scratch/%u/%x.out.%j
#
# CPU specification
#SBATCH -n 1                  # 1 process
#SBATCH -c 2                  # 2 CPU cores per process 
#                               can be referenced as $SLURM_CPUS_PER_TASK in the "payload" part
#SBATCH --mem-per-cpu=2G      # Hauptspeicher in MByte pro Rechenkern
#SBATCH -t 01:00:00           # in hours:minutes, or '#SBATCH -t 10' - just minutes

# GPU specification
#SBATCH --gres=gpu:1          # 1 GPU (generic "gpu:1" matches any available NVIDIA GPU)
#                               can be referenced down below as $SLURM_GPUS_ON_NODE

# -------------------------------
# your job's "payload" in form of commands to execute, eg.
module purge
module load gcc cuda python/3.13
# navigate to your working directory
cd "${SLURM_SUBMIT_DIR:-.}"

# activate your virtual environment (checks local or parent folder)
[ -f ".venv/bin/activate" ] && source .venv/bin/activate || source ../.venv/bin/activate 2>/dev/null || true

# specification from OMP_NUM_THREADS depends on your program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# for checking whether and which GPUs have been allocated
# (output appears in the "#SBATCH -e" file specified above):
nvidia-smi 1>&2

# if your program supports this way of getting told how many GPUs to use:
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE

python -m src.training.train --config-name=lichtenberg_boundary
EXITCODE=$?

# end this job script with precisely the exit status of your scientific program above:
exit $EXITCODE
