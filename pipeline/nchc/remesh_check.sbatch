#!/usr/bin/env bash
# Re-mesh check (18 Oct 2026; the PI approved it after the 300-head collection).
# Eight head x target rows kept < 90 % of the FEM optimum with the SP1 winner, and 6 of those 8 heads
# were meshed in the first run (the one that filled /work). This job re-meshes those 8 heads and
# 16 matched controls (2 per outlier, same site and mesh batch, every target >= 95 %) from their verified
# input tarballs, with a fresh charm mesh (4 threads), and reruns all stages. Outputs stay separate
# from the collected 300-head set:
#   $P/remesh_check/results/<code>__{sp2,sp1,fem}.json + <code>.done,  logs in $P/remesh_check/logs,
#   outboxes in $P/remesh_check/outbox (not watched by pull_outputs.sh; fetched by fetch_remesh.sh).
#   sbatch ~/spicehead_setup/remesh_check.sbatch
#SBATCH --job-name=remesh_check
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour. Reduced from measured mean use; this job type has CPU stages and was not run at this size. Check speed on a small array first.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/work/<account>/spicehead/remesh_check/job_%j.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
export PREFIX=/work/<account>/spicehead
P=$PREFIX; R=$P/remesh_check
export SPICEHEAD_PIPE_OUT=$R/results SPICEHEAD_PIPE_LOGS=$R/logs SPICEHEAD_OUTBOXES=$R/outbox SPICEHEAD_FORCE_REMESH=1
export SPICEHEAD_SCRATCH=${TMPDIR:-/tmp}/remesh_${SLURM_JOB_ID:-$$}
mkdir -p $R/results $R/logs $R/outbox $SPICEHEAD_SCRATCH; trap 'rm -rf "$SPICEHEAD_SCRATCH"' EXIT
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -gt 0 ] || NGPU=1
( source $P/activate_gpu.sh
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g python -c "import torch; assert torch.cuda.is_available(); (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()" 2>/dev/null \
      && echo "GPU $g ok" || { echo "GPU $g FAILED CUDA init on $(hostname)"; exit 1; }
  done ) || { echo "preflight failed on $(hostname)"; exit 3; }
HEADS=$(tr -d '\r' < $R/heads.txt)
echo "re-mesh check on $(hostname): $(echo $HEADS | wc -w) heads | $(date)"
i=0
for h in $HEADS; do
  [ -f $R/results/$h.done ] && continue
  bash ~/spicehead_setup/head_pipeline_v3.sh "$h" 4 $(( i % NGPU )) > /dev/null 2>&1 &
  i=$((i+1))
done
wait
echo "finished $(date)"
for h in $HEADS; do [ -f $R/results/$h.done ] && echo "  $h DONE" || echo "  $h INCOMPLETE (see $R/logs/$h.log)"; done
