#!/usr/bin/env bash
# Forerunner 1: run a list of shell commands in parallel on 112-core nodes (Paper A v23 recompute, 26 Oct 2026).
# Array task k of n runs lines k, k+n, k+2n, ... of the command file, PAR at a time, each with one BLAS thread.
# Every line logs to its own file and appends its name to logs/FAILED on failure (written by f1_make_cmds.sh).
#   sbatch --array=0-2 ~/f1_run.sbatch <commands file> [PAR]
#SBATCH --job-name=f1_v23
#SBATCH --account=<project>
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=112
#SBATCH --time=12:00:00
#SBATCH --output=/work1/<account>/spicehead/logs/f1_run_%A_%a.log
set -uo pipefail
P=/work1/<account>/spicehead
RD=${RUNDIR:-$P}                        # run folder (data, results_v23, logs); code, env and lists stay in $P
source $P/activate_cpu.sh
export SPICEHEAD_RESULTS_DIR=$RD/results_v23 SPICEHEAD_CHARM_ROOT=$RD/data SPICEHEAD_COHORT_CSV=$P/jobs/heads300.csv \
       SPICEHEAD_T7_SUBJECTS=$P/jobs/t7_pinned.csv \
       SPICEHEAD_T3=$RD/results_v23/montage_optimization/cohort_gm300_dx5_maxdir/cohort_montage_optima.json
CMDS=$1; PAR=${2:-56}
N=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
echo "task $K of $N on $(hostname): $(awk -v n=$N -v k=$K '(NR-1)%n==k' $CMDS | wc -l) commands, $PAR at a time | $(date)"
echo "code: $(cd $P/code && sha256sum spicehead/homogenize.py spicehead/mri.py spicehead/config.py pipeline/run_montage_cohort.py | cut -c1-16 | tr '\n' ' ')"
awk -v n=$N -v k=$K '(NR-1)%n==k' $CMDS | {
  while IFS= read -r line; do
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done
    bash -c "$line" &
  done
  wait
}
echo "task $K finished $(date)"
