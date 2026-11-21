#!/usr/bin/env bash
# GPU scoring check on h001 for the three models against the CPU score files already produced.
#SBATCH --job-name=spicehead_score_check
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/<account>/spicehead/jobs/score_check_%j.log
set -uo pipefail
P=/work/<account>/spicehead
source $P/activate_gpu.sh
export SPICEHEAD_CHARM_ROOT=$P/data SPICEHEAD_COHORT_CSV=$P/data/cohort.csv SPICEHEAD_FEM_BASIS_ROOT=$P/fem_basis
export SPICEHEAD_FULLSEARCH_CACHE=$P/cache SPICEHEAD_FULLSEARCH_OUT=$P/results/v22_fem_fullsearch
printf "CaseID\nh001\n" > $P/heads/h001.csv; export SPICEHEAD_FS_HEADS=$P/heads/h001.csv
cd $P/code
for m in sp2 sp1 fem; do echo "=== $m ==="; python pipeline/gpu_score_check.py h001 $m; done
echo "done $(date)"
