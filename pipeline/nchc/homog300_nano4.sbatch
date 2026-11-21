#!/usr/bin/env bash
# 300-head FEM comparison with the homogenized coarse network (23 Oct 2026, PI: "re-run the 300-head FEM
# comparison with it on Nano 4"). Same code and parameters as the 30-head test (job 602359):
# pipeline/nchc/homog_experiment.py, MIN_FILL 0.5, central-difference field, 2 and 1 mm, v22 contacts.
# One head per GPU, 8 at a time per node. Heads are read from a list file and sharded over the array tasks.
# Each head is unpacked from the /home tarball store into node scratch (SHA-256 checked against
# $P/jobs/code_sha.txt). Outputs (<code>__sp{2,1}hom.json + _basis.npz) go to $E/homog300/.
# The desktop puller (wsl_pull_homog300.sh) copies them to L:, verifies them, then deletes the bases here
# and leaves <code>.pulled; the JSONs stay for a cluster-side collection.
# /work has a 100 GB quota: a head starts only when at least MINFREE_GB are free (waits up to 3 h).
#   sbatch --array=0-1 ~/spicehead_setup/homog300_nano4.sbatch homog300_heads_a.txt
#SBATCH --job-name=homog300
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour; GPUs and memory are not billed. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/work/<account>/spicehead/experiments/homog300_%A_%a.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; E=$P/experiments
LIST=$E/${1:?usage: sbatch --array=0-N homog300_nano4.sbatch <list file in $E>}
NT=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}; MINFREE_GB=${MINFREE_GB:-15}
SCR=${TMPDIR:-/tmp}/homog300_${SLURM_JOB_ID:-$$}
mkdir -p $E/homog300 $E/logs $SCR/data $SCR/cache; trap 'rm -rf "$SCR"' EXIT
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu PYTHONPATH=$P/code:$P/code/pipeline
export SPICEHEAD_CHARM_ROOT=$SCR/data SPICEHEAD_FULLSEARCH_CACHE=$SCR/cache SPICEHEAD_FULLSEARCH_OUT=$P/results/v22_fem_fullsearch
export SPICEHEAD_EXP_OUT=$E/homog300 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NGPU=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python -c "import torch; assert torch.cuda.is_available(); (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()" 2>/dev/null \
    || { echo "GPU $g FAILED on $(hostname)"; exit 3; }
done
mine=$(tr -d '\r' < $LIST | grep . | awk -v n=$NT -v k=$K '(NR-1)%n==k')
echo "homogenized 300-head run, task $K of $NT on $(hostname): $(echo "$mine" | wc -w) heads from $(basename $LIST) | $(date)"
echo "code: homog_experiment.py sha256 $(sha256sum $P/code/pipeline/nchc/homog_experiment.py | cut -c1-16)"
freegb() { df -B1G --output=avail /work/<account> | tail -1 | tr -d ' '; }
one() {
  local c=$1 gpu=$2 log=$E/logs/homog300_$1.log
  if [ -e $E/homog300/$c.pulled ] || { [ -s $E/homog300/${c}__sp1hom.json ] && [ -s $E/homog300/${c}__sp2hom.json ]; }; then echo "$c done before"; return; fi
  [ -s $STORE/m2m_$c.tar.gz ] || { echo "$c: no input tarball in the store" | tee -a $log; return; }
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $P/jobs/code_sha.txt); got=$(sha256sum $STORE/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; return; }
  tar -xzf $STORE/m2m_$c.tar.gz -C $SCR/data || { echo "$c: unpack failed" | tee -a $log; return; }
  CUDA_VISIBLE_DEVICES=$gpu python $P/code/pipeline/nchc/homog_experiment.py --heads $c --dx 2,1 >> $log 2>&1
  grep -E "^$c [12] mm:" $log | tail -2 | tr '\n' ' '; echo; rm -rf $SCR/data/m2m_$c $SCR/cache/${c}__*
}
one_when_disk_ok() {
  w=0; while [ "$(freegb)" -lt $MINFREE_GB ] && [ $w -lt 180 ]; do [ $w -eq 0 ] && echo "$(date +%T) waiting for /work space ($(freegb) GB free)"; sleep 60; w=$((w+1)); done
  one "$1" "$2"
}
source ~/spicehead_setup/gpu_queue.sh
run_on_gpus one_when_disk_ok "$NGPU" $mine
echo "task $K finished $(date): $(for c in $mine; do [ -e $E/homog300/$c.pulled ] || [ -s $E/homog300/${c}__sp1hom.json ] && echo $c; done | wc -l) of $(echo "$mine" | wc -w) heads have results"
