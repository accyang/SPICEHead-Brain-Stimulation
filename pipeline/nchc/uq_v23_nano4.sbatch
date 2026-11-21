#!/usr/bin/env bash
# v23 uncertainty analysis on Nano 4 (24 Oct 2026; plan: results_v23/v23_uq/ANALYSIS_PLAN.md). One head per GPU,
# 8 heads per node; heads from $U/uq_heads.txt sharded over the array tasks. Each head is unpacked from the /home
# tarball store into node scratch (SHA-256 checked against $P/jobs/code_sha.txt or ~/code_sha.txt) and runs all 75
# scenario runs of pipeline/nchc/uq_v23.py (resumable: finished runs are skipped).
# Outputs: $U/runs/<code>/*.npz (F/T/O and peaks per run), $U/logs/<code>.log. $U is outside the folders of the
# 24 Oct archive, so delete_nano4_results.sh never touches it.
#   sbatch --array=0-4 ~/spicehead_setup/uq_v23_nano4.sbatch
#SBATCH --job-name=uq_v23
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour; GPUs and memory are not billed. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/work/<account>/spicehead/uq_v23/job_%A_%a.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; U=$P/uq_v23
SHA=$P/jobs/code_sha.txt; [ -s $SHA ] || SHA=/home/<account>/code_sha.txt
NT=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
SCR=${TMPDIR:-/tmp}/uq_v23_${SLURM_JOB_ID:-$$}
mkdir -p $U/runs $U/logs $SCR/data $SCR/cache; trap 'rm -rf "$SCR"' EXIT
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu SPICEHEAD_SCORE=gpu PYTHONPATH=$P/code:$P/code/pipeline PYTHONHASHSEED=0
export SPICEHEAD_CHARM_ROOT=$SCR/data SPICEHEAD_FULLSEARCH_CACHE=$SCR/cache SPICEHEAD_UQ_OUT=$U/runs SPICEHEAD_UQ_FIXED=$U/uq_fixed_montages.json
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NGPU=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python -c "import torch; assert torch.cuda.is_available(); (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()" 2>/dev/null \
    || { echo "GPU $g FAILED on $(hostname)"; exit 3; }
done
mine=$(tr -d '\r' < $U/uq_heads.txt | grep . | awk -v n=$NT -v k=$K '(NR-1)%n==k')
echo "v23 uncertainty analysis, task $K of $NT on $(hostname): $(echo "$mine" | wc -w) heads | $(date)"
echo "code: $(cd $P/code && sha256sum pipeline/nchc/uq_v23.py spicehead/homogenize.py spicehead/mri.py spicehead/field.py spicehead/score_gpu.py | cut -c1-16 | tr '\n' ' ')"
one() {
  local c=$1 gpu=$2 log=$U/logs/$1.log
  [ -e $U/runs/$c/DONE ] && { echo "$c done before"; return; }
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $SHA); got=$(sha256sum $STORE/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; return; }
  tar -xzf $STORE/m2m_$c.tar.gz -C $SCR/data || { echo "$c: unpack failed" | tee -a $log; return; }
  ( cd $P/code && CUDA_VISIBLE_DEVICES=$gpu python pipeline/nchc/uq_v23.py --head $c ) >> $log 2>&1
  echo "$c: $(ls $U/runs/$c/*.npz 2>/dev/null | wc -l) runs, $( [ -e $U/runs/$c/DONE ] && echo complete || echo INCOMPLETE ) | $(grep -c FAIL $log) failures logged"
  rm -rf $SCR/data/m2m_$c $SCR/cache/${c}__*
}
source ~/spicehead_setup/gpu_queue.sh
run_on_gpus one "$NGPU" $mine
echo "task $K finished $(date): $(for c in $mine; do [ -e $U/runs/$c/DONE ] && echo $c; done | wc -l) of $(echo "$mine" | wc -w) heads complete"
