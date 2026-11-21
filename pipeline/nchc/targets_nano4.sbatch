#!/usr/bin/env bash
# Region-target scoring on Nano 4 (18 Oct 2026; tier 1 = AAL90, tier 2 = psychiatric targets, ref6 check).
# One worker per GPU. Each worker:
#   1. claims the next uploaded bundle (targets/bundles/<code>.npz plus its .sha256, written last);
#   2. checks the checksum, waiting for the head's tier-2 files when REQUIRE_T2=1;
#   3. runs score_targets.py, with the result going to /home/<account>/spicehead/targets_out/<code>__targets.npz;
#   4. deletes the bundle (a copy of the verified data on L:) so upload_bundles.sh can send more within
#      the /work quota.
# Claims are atomic (mkdir); a claim whose owner job has left the queue is taken over atomically.
# A task ends when UPLOAD_DONE exists and nothing is left to score, or after 2 h without work.
#   sbatch --array=1-2 ~/spicehead_setup/targets_nano4.sbatch
#SBATCH --job-name=targets
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour; GPUs and memory are not billed. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/work/<account>/spicehead/targets/job_%A_%a.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; T=$P/targets; B=$T/bundles; OUTD=/home/<account>/spicehead/targets_out
REQUIRE_T2=${REQUIRE_T2:-1}
ME=${SLURM_JOB_ID:-$$}
mkdir -p $B $T/t2 $T/claims $T/logs $OUTD
source $P/activate_gpu.sh
export SPICEHEAD_CODE_ROOT=$P/code OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -gt 0 ] || NGPU=1
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python -c "import torch; assert torch.cuda.is_available(); (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()" 2>/dev/null \
    || { echo "GPU $g FAILED CUDA init on $(hostname); stopping"; exit 3; }
done
echo "targets task ${SLURM_ARRAY_TASK_ID:-0} (job $ME) on $(hostname), $NGPU GPUs, REQUIRE_T2=$REQUIRE_T2 | $(date)"

claim() {
  local d=$T/claims/$1
  if mkdir "$d" 2>/dev/null; then echo "$ME" > "$d/owner"; return 0; fi
  local o; o=$(cat "$d/owner" 2>/dev/null)
  # alive = the owner's job ID is among this user's running or pending job IDs (%A). `squeue -j <id>` is not
  # enough: the first array task's job ID equals the array ID, which lists every task (18 Oct 2026).
  local q; q=$(squeue -h -u "$USER" -o %A 2>/dev/null) && [ -n "$q" ] || return 1   # squeue failure: never take over
  if [ -n "$o" ] && [ "$o" != "$ME" ] && ! grep -qx "$o" <<< "$q"; then
    mkdir "$d/takeover_from_$o" 2>/dev/null || return 1
    echo "$ME" > "$d/owner"; return 0
  fi
  return 1
}

worker() {
  local gpu=$1 idle=0 c f
  while :; do
    c=""
    for f in $(ls $B/h[0-9]*.sha256 2>/dev/null | sort); do
      local k; k=$(basename $f .sha256)
      [ -s $OUTD/${k}__targets.npz ] && continue
      [ -s $B/$k.npz ] || continue
      if [ "$REQUIRE_T2" = 1 ]; then [ -s $T/t2/${k}__t2labels.npz ] && [ -s $T/t2/${k}__centres.json ] || continue; fi
      claim $k && { c=$k; break; }
    done
    if [ -z "$c" ]; then
      left=$(for f in $(ls $B/h[0-9]*.npz 2>/dev/null); do k=$(basename $f .npz); [ -s $OUTD/${k}__targets.npz ] || echo $k; done | wc -l)
      [ -f $T/UPLOAD_DONE ] && [ "$left" -eq 0 ] && break
      [ $idle -ge 7200 ] && { echo "worker $gpu: 2 h without work, stopping"; break; }
      sleep 60; idle=$((idle+60)); continue
    fi
    idle=0
    if [ "$(sha256sum $B/$c.npz | cut -d' ' -f1)" != "$(cut -d' ' -f1 $B/$c.sha256)" ]; then
      echo "$(date +%T) $c checksum mismatch: bundle removed for re-upload"; rm -f $B/$c.npz $B/$c.sha256; rm -rf $T/claims/$c; continue
    fi
    local t0=$(date +%s) args=""
    [ -s $T/t2/${c}__centres.json ] && args="$args --centres $T/t2/${c}__centres.json"
    [ -s $T/t2/${c}__t2labels.npz ] && args="$args --extra-labels $T/t2/${c}__t2labels.npz"
    CUDA_VISIBLE_DEVICES=$gpu python ~/spicehead_setup/score_targets.py $B/$c.npz $OUTD $args \
        --ref-json-dir $P/results/v22_fem_fullsearch > $T/logs/$c.log 2>&1
    if [ -s $OUTD/${c}__targets.npz ]; then
      rm -f $B/$c.npz; echo "$(date +%T) $c scored on GPU $gpu in $(( $(date +%s) - t0 )) s"
    else
      echo "$(date +%T) $c FAILED on GPU $gpu (see $T/logs/$c.log); bundle kept"
    fi
  done
}

for g in $(seq 0 $((NGPU-1))); do worker $g & done
wait
echo "task finished $(date) | scored in total: $(ls $OUTD/*__targets.npz 2>/dev/null | wc -l)"
