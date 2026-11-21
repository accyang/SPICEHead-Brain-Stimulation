#!/usr/bin/env bash
# A2 of the Rev0.5 analysis request (3 Nov 2026; PI: "run the 28 solves on nano4"): perturbed SPICEhead scores at the 28 SPICEhead
# cohort montages that the uncertainty archive did not keep. One head per GPU, free-GPU queue, 75 basis solves per head, two montages
# scored per target (no full search). Resumable: finished runs are skipped. Checks as in uq_v23_ext_nano4.sbatch.
#   sbatch --array=0-15 ~/spicehead_setup/uq_v23_sp_protocol_nano4.sbatch
#   HEADS_FILE=$U/sp_heads_rest.txt sbatch --array=0-13 ...   (a chosen set of heads)
#   Pilot: LIST="h002 h150" ONLY=nominal,cond_00,jit_00,rot_00 sbatch --partition=dev --gres=gpu:1 --cpus-per-task=2 --mem=60G --time=1:00:00 ...
#SBATCH --job-name=uq_v23sp
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs: billing is 2.5 SU per allocated CPU core-hour. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/work/<account>/spicehead/uq_v23/job_sp_%A_%a.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; U=$P/uq_v23
SHA=/home/<account>/code_sha.txt; PROTO=$U/uq_sp_protocols_28.json
LIST=${LIST:-}; ONLY=${ONLY:-}
NT=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
SCR=${TMPDIR:-/tmp}/uq_v23sp_${SLURM_JOB_ID:-$$}
mkdir -p $U/runs_sp_protocol $U/logs_sp $SCR/data $SCR/cache; trap 'rm -rf "$SCR"' EXIT
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu SPICEHEAD_SCORE=gpu PYTHONPATH=$P/code:$P/code/pipeline PYTHONHASHSEED=0
export SPICEHEAD_CHARM_ROOT=$SCR/data SPICEHEAD_FULLSEARCH_CACHE=$SCR/cache SPICEHEAD_UQ_SP_OUT=$U/runs_sp_protocol
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for f in $PROTO $U/uq_heads.txt $U/uq_heads_ext.txt $U/targets/SHA256SUMS2_all.txt $SHA; do
  [ -s $f ] || { echo "missing $f"; exit 2; }
done
NGPU=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python -c "import torch; assert torch.cuda.is_available(); (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()" 2>/dev/null \
    || { echo "GPU $g FAILED on $(hostname)"; exit 3; }
done
if [ -n "$LIST" ]; then mine=$LIST; else mine=$(cat ${HEADS_FILE:-$U/uq_heads.txt $U/uq_heads_ext.txt} | tr -d '\r' | grep . | sort | awk -v n=$NT -v k=$K '(NR-1)%n==k'); fi
echo "SPICEHead protocol solves, task $K of $NT on $(hostname): $(echo "$mine" | wc -w) heads, only='${ONLY}' | $(date)"
echo "code: $(cd $P/code && sha256sum pipeline/nchc/uq_v23_sp_protocol.py pipeline/nchc/uq_v23.py pipeline/nchc/score_targets_v2.py pipeline/nchc/atlas_thal_score.py spicehead/homogenize.py spicehead/mri.py spicehead/field.py spicehead/score_gpu.py | cut -c1-16 | tr '\n' ' ')"
one() {
  local c=$1 gpu=$2 log=$U/logs_sp/$1.log tf=$U/targets/$1__uqtargets2.npz xf=$P/atlas_ext/targets/$1__exttargets.npz
  [ -z "$ONLY" ] && [ -e $U/runs_sp_protocol/$c/DONE ] && { echo "$c done before"; return; }
  [ "$(sha256sum $tf 2>/dev/null | cut -d' ' -f1)" = "$(awk -v f=$(basename $tf) '$2==f{print $1}' $U/targets/SHA256SUMS2_all.txt)" ] || { echo "$c: target file missing or checksum mismatch" | tee -a $log; return; }
  [ -s $xf ] || { echo "$c: ext target file missing" | tee -a $log; return; }
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $SHA); got=$(sha256sum $STORE/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; return; }
  tar -xzf $STORE/m2m_$c.tar.gz -C $SCR/data || { echo "$c: unpack failed" | tee -a $log; return; }
  ( cd $P/code && CUDA_VISIBLE_DEVICES=$gpu python pipeline/nchc/uq_v23_sp_protocol.py --head $c --targets $tf --also-thal $xf --protocols $PROTO ${ONLY:+--only $ONLY} ) >> $log 2>&1
  echo "$c: $(ls $U/runs_sp_protocol/$c/*.npz 2>/dev/null | wc -l) runs, $( [ -e $U/runs_sp_protocol/$c/DONE ] && echo complete || echo INCOMPLETE ) | $(grep -c -i "error\|traceback" $log) errors logged | $(date +%H:%M)"
  rm -rf $SCR/data/m2m_$c $SCR/cache/${c}__*
}
source ~/spicehead_setup/gpu_queue.sh
run_on_gpus one "$NGPU" $mine
echo "task $K finished $(date): $(for c in $mine; do [ -e $U/runs_sp_protocol/$c/DONE ] && echo $c; done | wc -l) of $(echo "$mine" | wc -w) heads complete"
