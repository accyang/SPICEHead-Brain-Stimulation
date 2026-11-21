#!/usr/bin/env bash
# v23 uncertainty analysis, extension to all 300 heads (31 Oct 2026; PI: a 35-head analysis is "not acceptable",
# "agree all and use nano4/5 and forerunner"). Same 75 perturbation runs per head as the 35-head analysis (same seeds
# and code: uq_v23.py), scored for all 163 targets in one run (atlas set + thal set, --also-thal) from one basis, plus
# the two target-mask runs (mask_erode, mask_dilate; review D7). One head per GPU, 8 heads per node; heads from
# $U/uq_heads_ext.txt (265) sharded over the array tasks. Each head is unpacked from the /home tarball store into node
# scratch (SHA-256 checked against ~/code_sha.txt). Target files $U/targets/<code>__uqtargets2.npz (uploaded by the PI)
# are checked against $U/targets/SHA256SUMS2_all.txt. Resumable: finished runs are skipped.
# Outputs: $U/runs_atlas/<code>/*.npz, $U/runs_thal/<code>/*.npz, $U/logs_ext/<code>.log; the desktop driver fetches and
# deletes finished heads because /work has about 24 GB free.
#   sbatch --array=0-15 ~/spicehead_setup/uq_v23_ext_nano4.sbatch
#   HEADS=uq_heads.txt ONLY=nominal,mask_erode,mask_dilate sbatch --array=0-1 ...   (the 35 original heads: mask runs only)
#SBATCH --job-name=uq_v23x
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour; GPUs and memory are not billed. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=47:00:00
#SBATCH --output=/work/<account>/spicehead/uq_v23/job_ext_%A_%a.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; U=$P/uq_v23
SHA=$P/jobs/code_sha.txt; [ -s $SHA ] || SHA=/home/<account>/code_sha.txt
HEADS=${HEADS:-uq_heads_ext.txt}; ONLY=${ONLY:-}
NT=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
SCR=${TMPDIR:-/tmp}/uq_v23x_${SLURM_JOB_ID:-$$}
mkdir -p $U/runs_atlas $U/runs_thal $U/logs_ext $SCR/data $SCR/cache; trap 'rm -rf "$SCR"' EXIT
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu SPICEHEAD_SCORE=gpu PYTHONPATH=$P/code:$P/code/pipeline PYTHONHASHSEED=0
export SPICEHEAD_CHARM_ROOT=$SCR/data SPICEHEAD_FULLSEARCH_CACHE=$SCR/cache \
       SPICEHEAD_UQ_ATLAS_OUT=$U/runs_atlas SPICEHEAD_UQ_ATLAS_FIXED=$U/uq_atlas_fixed_montages.json \
       SPICEHEAD_UQ_THAL_OUT=$U/runs_thal SPICEHEAD_UQ_THAL_FIXED=$U/uq_atlas_fixed_montages_thal.json
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for f in $U/uq_atlas_fixed_montages.json $U/uq_atlas_fixed_montages_thal.json $U/$HEADS $U/targets/SHA256SUMS2_all.txt; do
  [ -s $f ] || { echo "missing $f"; exit 2; }
done
NGPU=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python -c "import torch; assert torch.cuda.is_available(); (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()" 2>/dev/null \
    || { echo "GPU $g FAILED on $(hostname)"; exit 3; }
done
mine=$(tr -d '\r' < $U/$HEADS | grep . | awk -v n=$NT -v k=$K '(NR-1)%n==k')
echo "v23 uncertainty extension, task $K of $NT on $(hostname): $(echo "$mine" | wc -w) heads, only='${ONLY}' | $(date)"
echo "code: $(cd $P/code && sha256sum pipeline/nchc/uq_v23_atlas.py pipeline/nchc/uq_v23.py pipeline/nchc/score_targets_v2.py pipeline/nchc/atlas_thal_score.py spicehead/homogenize.py spicehead/mri.py spicehead/field.py spicehead/score_gpu.py | cut -c1-16 | tr '\n' ' ')"
one() {
  local c=$1 gpu=$2 log=$U/logs_ext/$1.log tf=$U/targets/$1__uqtargets2.npz xf=$P/atlas_ext/targets/$1__exttargets.npz
  [ -z "$ONLY" ] && [ -e $U/runs_atlas/$c/DONE ] && [ -e $U/runs_thal/$c/DONE ] && { echo "$c done before"; return; }
  [ "$(sha256sum $tf 2>/dev/null | cut -d' ' -f1)" = "$(awk -v f=$(basename $tf) '$2==f{print $1}' $U/targets/SHA256SUMS2_all.txt)" ] || { echo "$c: target file missing or checksum mismatch" | tee -a $log; return; }
  [ -s $xf ] || { echo "$c: ext target file missing" | tee -a $log; return; }
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $SHA); got=$(sha256sum $STORE/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; return; }
  tar -xzf $STORE/m2m_$c.tar.gz -C $SCR/data || { echo "$c: unpack failed" | tee -a $log; return; }
  ( cd $P/code && CUDA_VISIBLE_DEVICES=$gpu python pipeline/nchc/uq_v23_atlas.py --head $c --targets $tf --also-thal $xf ${ONLY:+--only $ONLY} ) >> $log 2>&1
  echo "$c: $(ls $U/runs_atlas/$c/*.npz 2>/dev/null | wc -l) atlas + $(ls $U/runs_thal/$c/*.npz 2>/dev/null | wc -l) thal runs, $( [ -e $U/runs_atlas/$c/DONE ] && echo complete || echo INCOMPLETE ) | $(grep -c -i "error\|traceback" $log) errors logged"
  rm -rf $SCR/data/m2m_$c $SCR/cache/${c}__*
}
source ~/spicehead_setup/gpu_queue.sh
run_on_gpus one "$NGPU" $mine
echo "task $K finished $(date): $(for c in $mine; do [ -e $U/runs_atlas/$c/DONE ] && echo $c; done | wc -l) of $(echo "$mine" | wc -w) heads complete"
