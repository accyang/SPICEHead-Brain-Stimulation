#!/usr/bin/env bash
# Experiment 2 on Nano 4 (23 Oct 2026): SPICEhead with SimNIBS-like gel pads, 29 heads, 2 and 1 mm.
# One head per GPU, 8 at a time; each head is unpacked from the /home tarball store into node scratch
# (SHA-256 checked), the grid geometry is rebuilt there, and the outputs (score JSON + variant basis npz)
# go to $P/experiments/pad/ for download. FEM score files for the decisions are in $P/results/v22_fem_fullsearch.
#   sbatch ~/spicehead_setup/pad_nano4.sbatch
#SBATCH --job-name=pad_exp
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour. Reduced from measured mean use; this job type has CPU stages and was not run at this size. Check speed on a small array first.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/work/<account>/spicehead/experiments/pad_%j.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; E=$P/experiments
SCR=${TMPDIR:-/tmp}/padexp_${SLURM_JOB_ID:-$$}
mkdir -p $E/pad $E/logs $SCR/data $SCR/cache; trap 'rm -rf "$SCR"' EXIT
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu PYTHONPATH=$P/code:$P/code/pipeline
export SPICEHEAD_CHARM_ROOT=$SCR/data SPICEHEAD_FULLSEARCH_CACHE=$SCR/cache SPICEHEAD_FULLSEARCH_OUT=$P/results/v22_fem_fullsearch
export SPICEHEAD_EXP_OUT=$E/pad OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
NGPU=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python -c "import torch; assert torch.cuda.is_available(); (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()" 2>/dev/null \
    || { echo "GPU $g FAILED on $(hostname)"; exit 3; }
done
echo "pad experiment on $(hostname): $(wc -l < $E/pad_heads.txt) heads | $(date)"
one() {
  local c=$1 gpu=$2 log=$E/logs/pad_$1.log
  if [ -s $E/pad/${c}__sp1pad.json ] && [ -s $E/pad/${c}__sp2pad.json ]; then echo "$c cached"; return; fi
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $P/jobs/code_sha.txt); got=$(sha256sum $STORE/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; return; }
  tar -xzf $STORE/m2m_$c.tar.gz -C $SCR/data || { echo "$c: unpack failed" | tee -a $log; return; }
  CUDA_VISIBLE_DEVICES=$gpu python $P/code/pipeline/nchc/pad_experiment.py --heads $c --dx 2,1 >> $log 2>&1
  tail -2 $log | grep -E "mm:" ; rm -rf $SCR/data/m2m_$c $SCR/cache/${c}__*
}
source ~/spicehead_setup/gpu_queue.sh
run_on_gpus one "$NGPU" $(tr -d '\r' < $E/pad_heads.txt)
echo "finished $(date): $(ls $E/pad/*__sp1pad.json 2>/dev/null | wc -l) heads at 1 mm, $(ls $E/pad/*__sp2pad.json 2>/dev/null | wc -l) at 2 mm"
