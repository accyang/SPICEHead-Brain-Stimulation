#!/usr/bin/env bash
# v23 cohort run of the full-space SPICEhead stages on Nano 4 (24 Oct 2026, v23 ported to the pipelines).
# head_pipeline.sh stays the v22 production pipeline (it sets SPICEHEAD_MODEL=v22 itself, and its score files and
# outboxes share names with v22). This job runs only the model-dependent stages of v22_fem_fullsearch.py under
# SPICEHEAD_MODEL=v23: --sp-basis 2.0 and 1.0, then --score sp2 and sp1. FEM is model-independent and is not rerun.
# One head per GPU, 8 at a time per node; heads from a list file in $E, sharded over the array tasks; each head is
# unpacked from the /home tarball store into node scratch (SHA-256 checked against $P/jobs/code_sha.txt).
# Outputs: score JSONs  -> $P/results/v23_fem_fullsearch/<code>__sp{2,1}.json (kept on the cluster);
#          bases (KEEP_BASES=1, default) -> $E/v23_cohort/<code>__sp{2,1}_v23_basis.npz for wsl_pull_v23_cohort.sh,
#          which copies them to L:, verifies them and deletes them here. /work guard: a head starts only when
#          MINFREE_GB (15) are free.
#   sbatch --array=0-1 ~/spicehead_setup/v23_cohort_nano4.sbatch <list file in $E>
#SBATCH --job-name=v23_cohort
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour; GPUs and memory are not billed. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/work/<account>/spicehead/experiments/v23_cohort_%A_%a.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; E=$P/experiments
LIST=$E/${1:?usage: sbatch --array=0-N v23_cohort_nano4.sbatch <list file in $E>}
NT=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}; MINFREE_GB=${MINFREE_GB:-15}; KEEP_BASES=${KEEP_BASES:-1}
OUT=$P/results/v23_fem_fullsearch; BOUT=$E/v23_cohort
SCR=${TMPDIR:-/tmp}/v23cohort_${SLURM_JOB_ID:-$$}
mkdir -p $OUT $BOUT $E/logs $SCR; trap 'rm -rf "$SCR"' EXIT
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu SPICEHEAD_SCORE=gpu PYTHONPATH=$P/code:$P/code/pipeline PYTHONHASHSEED=0
export SPICEHEAD_COHORT_CSV=$P/data/cohort.csv SPICEHEAD_FULLSEARCH_OUT=$OUT OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NGPU=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python -c "import torch; assert torch.cuda.is_available(); (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()" 2>/dev/null \
    || { echo "GPU $g FAILED on $(hostname)"; exit 3; }
done
mine=$(tr -d '\r' < $LIST | grep . | awk -v n=$NT -v k=$K '(NR-1)%n==k')
echo "v23 cohort run, task $K of $NT on $(hostname): $(echo "$mine" | wc -w) heads from $(basename $LIST) | $(date)"
echo "code: $(cd $P/code && sha256sum spicehead/homogenize.py spicehead/mri.py spicehead/field.py pipeline/v22_fem_fullsearch.py | cut -c1-16 | tr '\n' ' ')"
freegb() { df -B1G --output=avail /work/<account> | tail -1 | tr -d ' '; }
one() {
  local c=$1 gpu=$2 log=$E/logs/v23_cohort_$1.log d=$SCR/$1
  if [ -s $OUT/${c}__sp1.json ] && [ -s $OUT/${c}__sp2.json ]; then echo "$c done before"; return; fi
  [ -s $STORE/m2m_$c.tar.gz ] || { echo "$c: no input tarball in the store" | tee -a $log; return; }
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $P/jobs/code_sha.txt); got=$(sha256sum $STORE/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; return; }
  mkdir -p $d/data $d/cache; tar -xzf $STORE/m2m_$c.tar.gz -C $d/data || { echo "$c: unpack failed" | tee -a $log; return; }
  printf "CaseID\n%s\n" "$c" > $d/head.csv
  ( export CUDA_VISIBLE_DEVICES=$gpu SPICEHEAD_FS_HEADS=$d/head.csv SPICEHEAD_CHARM_ROOT=$d/data SPICEHEAD_FULLSEARCH_CACHE=$d/cache
    cd $P/code
    for dx in 2.0 1.0; do python pipeline/v22_fem_fullsearch.py --sp-basis $dx --heads $c || exit 4; done
    for m in sp2 sp1; do python pipeline/v22_fem_fullsearch.py --score $m --heads $c || exit 5; done ) >> $log 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ "$KEEP_BASES" = 1 ]; then
    for t in sp2 sp1; do cp $d/cache/${c}__${t}_v23_basis.npz $BOUT/${c}__${t}_v23_basis.npz.part && mv $BOUT/${c}__${t}_v23_basis.npz.part $BOUT/${c}__${t}_v23_basis.npz; done
  fi
  echo "$c: $( [ $rc -eq 0 ] && echo ok || echo "FAILED (stage exit $rc, see $log)" ) $(grep -h 'superposition' $log | tail -1)"
  rm -rf $d
}
one_when_disk_ok() {
  w=0; while [ "$(freegb)" -lt $MINFREE_GB ] && [ $w -lt 180 ]; do [ $w -eq 0 ] && echo "$(date +%T) waiting for /work space ($(freegb) GB free)"; sleep 60; w=$((w+1)); done
  one "$1" "$2"
}
source ~/spicehead_setup/gpu_queue.sh
run_on_gpus one_when_disk_ok "$NGPU" $mine
echo "task $K finished $(date): $(for c in $mine; do [ -s $OUT/${c}__sp1.json ] && [ -s $OUT/${c}__sp2.json ] && echo $c; done | wc -l) of $(echo "$mine" | wc -w) heads scored"
