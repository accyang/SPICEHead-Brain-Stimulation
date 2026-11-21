#!/usr/bin/env bash
# Experiment 3b on Nano 4 (23 Oct 2026): the standard network at 0.5 mm (~30 M nodes) on the 8 outlier heads.
# One head per GPU, 8 at a time; each head is unpacked from the /home tarball store into node scratch
# (SHA-256 checked), the grid geometry is rebuilt there, and the outputs (score JSON + variant basis npz)
# go to $P/experiments/pad/ for download. FEM score files for the decisions are in $P/results/v22_fem_fullsearch.
#   sbatch ~/spicehead_setup/fine_nano4.sbatch
#SBATCH --job-name=fine_exp
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour; GPUs and memory are not billed. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/work/<account>/spicehead/experiments/fine_%j.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; E=$P/experiments
SCR=${TMPDIR:-/tmp}/padexp_${SLURM_JOB_ID:-$$}
mkdir -p $E/fine $E/logs $SCR/data $SCR/cache; trap 'rm -rf "$SCR"' EXIT
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu PYTHONPATH=$P/code:$P/code/pipeline
export SPICEHEAD_CHARM_ROOT=$SCR/data SPICEHEAD_FULLSEARCH_CACHE=$SCR/cache SPICEHEAD_FULLSEARCH_OUT=$P/results/v22_fem_fullsearch
export SPICEHEAD_EXP_OUT=$E/fine OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 SPICEHEAD_FULLSEARCH_CACHE=$SCR/cache
NGPU=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python -c "import torch; assert torch.cuda.is_available(); (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()" 2>/dev/null \
    || { echo "GPU $g FAILED on $(hostname)"; exit 3; }
done
echo "0.5 mm experiment on $(hostname): $(wc -l < $E/fine_heads.txt) heads | $(date)"
one() {
  local c=$1 gpu=$2 log=$E/logs/fine_$1.log
  if [ -s $E/fine/${c}__sp0.5.json ]; then echo "$c cached"; return; fi
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $P/jobs/code_sha.txt); got=$(sha256sum $STORE/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; return; }
  tar -xzf $STORE/m2m_$c.tar.gz -C $SCR/data || { echo "$c: unpack failed" | tee -a $log; return; }
  CUDA_VISIBLE_DEVICES=$gpu python $P/code/pipeline/nchc/fine_grid_experiment.py --heads $c --dx 0.5 >> $log 2>&1
  tail -2 $log | grep -E "mm:" ; rm -rf $SCR/data/m2m_$c $SCR/cache/${c}__*
}
source ~/spicehead_setup/gpu_queue.sh
run_on_gpus one "$NGPU" $(tr -d '\r' < $E/fine_heads.txt)
echo "finished $(date): $(ls $E/fine/*__sp0.5.json 2>/dev/null | wc -l) heads at 0.5 mm"
