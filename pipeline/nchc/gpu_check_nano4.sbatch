#!/usr/bin/env bash
# GPU solver check on Nano 4: one staged head, 2 mm then 1 mm, GPU vs CPU Jacobi-CG (pipeline/gpu_solver_check.py).
#   sbatch ~/spicehead_setup/gpu_check_nano4.sbatch
#SBATCH --job-name=spicehead_gpu_check
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/work/<account>/spicehead/jobs/gpu_check_%j.log
set -euo pipefail
PREFIX=/work/<account>/spicehead
source $PREFIX/activate_gpu.sh
export SPICEHEAD_CHARM_ROOT=$PREFIX/data SPICEHEAD_COHORT_CSV=$PREFIX/data/cohort.csv
export SPICEHEAD_RESULTS_ROOT=$PREFIX/results
mkdir -p $PREFIX/results
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cd $PREFIX/code
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== 2 mm, 3 CPU reference solves ==="
python pipeline/gpu_solver_check.py h001 2.0 3
echo "=== 1 mm, 1 CPU reference solve ==="
python pipeline/gpu_solver_check.py h001 1.0 1
echo "done $(date)"
