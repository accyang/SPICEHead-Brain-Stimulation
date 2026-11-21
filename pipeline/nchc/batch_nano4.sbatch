#!/usr/bin/env bash
# Nano 4 production batch, full-node tasks: 8 H200 + 48 CPUs per task (mean use was 22 CPUs; the allowance is 12 per GPU),
# PAR heads running at once (round-robin over the 8 GPUs), THREADS CPU threads each.
# Scoring and the SPICEhead bases run on the GPU; meshing and the FEM leadfield on CPU.
#
# Head selection (changed 17 Oct, 20:45): each task CLAIMS the next HEADS_PER_TASK heads of
# $P/heads_all.txt that are unpacked, not finished and not claimed by a running job (a claim is the
# directory $P/claims/<code> holding the owner's job id; claims of jobs no longer in the queue are taken
# over). Array indices therefore only number the tasks, and tasks can be submitted as uploads arrive
# without overlapping. HEADS_PER_TASK = PAR = 24: one wave per task, no idle tail.
#
#   sbatch --array=1 ~/spicehead_setup/batch_nano4.sbatch          # one task (24 heads)
#   sbatch --array=2-5%10 ~/spicehead_setup/batch_nano4.sbatch     # several; at most 10 at once
#
# Measured per head on Nano 4 (single head, 4 threads): mesh 20 min, FEM leadfield 11 min, GPU bases
# ~1 min, GPU scoring ~1 min. Resumable: a re-run only redoes unfinished stages.
#SBATCH --job-name=spicehead_fem
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour. Reduced from measured mean use; this job type has CPU stages and was not run at this size. Check speed on a small array first.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/work/<account>/spicehead/jobs/fem_%A_%a.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
export PREFIX=/work/<account>/spicehead
P=$PREFIX
HEADS_PER_TASK=${HEADS_PER_TASK:-24}
PAR=${PAR:-24}
THREADS=${THREADS:-4}
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -gt 0 ] || NGPU=1
LIST=${HEADS_LIST:-$P/heads_all.txt}
TID=${SLURM_ARRAY_TASK_ID:-0}
ME=${SLURM_JOB_ID:-$$}
mkdir -p $P/claims

claim() {   # claim() <code>: succeed if this job now owns the head
  local h=$1 d=$P/claims/$1
  if mkdir "$d" 2>/dev/null; then echo "$ME" > "$d/owner"; return 0; fi
  local owner; owner=$(cat "$d/owner" 2>/dev/null)
  [ "$owner" = "$ME" ] && return 0
  # alive = the owner's job ID is among this user's running or pending job IDs (%A). `squeue -j <id>` is not
  # enough: the first array task's job ID equals the array ID, which lists every task (18 Oct 2026).
  local q; q=$(squeue -h -u "$USER" -o %A 2>/dev/null) && [ -n "$q" ] || return 1   # squeue failure: never take over
  if [ -n "$owner" ] && ! grep -qx "$owner" <<< "$q"; then
    # owner job gone: take over. The mkdir makes this atomic. On 17 Oct 23:42 tasks 10 and 14 started
    # in the same second, both took over the same 24 stale claims, and ran those heads twice.
    mkdir "$d/takeover_from_$owner" 2>/dev/null || return 1
    echo "$ME" > "$d/owner"; return 0
  fi
  return 1
}

# GPU preflight (17 Oct 22:50): on 25a-hgpn062 every CUDA init failed ("CUDA unknown error"). Test each
# allocated GPU before claiming any head; stop the task early instead of failing 24 heads.
( source $P/activate_gpu.sh
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g python -c "import torch; assert torch.cuda.is_available(); (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()" 2>/dev/null       && echo "GPU $g ok" || { echo "GPU $g FAILED CUDA init on $(hostname)"; exit 1; }
  done ) || { echo "preflight failed on $(hostname); no heads claimed"; exit 3; }

# Storage (17 Oct 23:45): heads run in node-local scratch ($TMPDIR, 28 TB NVMe), so /work only receives
# each finished head's outbox (~0.6 GB) until pull_outputs.sh on the desktop has copied and verified it.
# head_pipeline_v2.sh waits when neither outbox has room. Start only if /work or /home has 10 GB free.
STORE=${SPICEHEAD_TAR_STORE:-/home/<account>/spicehead/tar}
export SPICEHEAD_SCRATCH=${TMPDIR:-/tmp}/spicehead_$ME
mkdir -p $SPICEHEAD_SCRATCH; trap 'rm -rf "$SPICEHEAD_SCRATCH"' EXIT
echo "scratch $SPICEHEAD_SCRATCH: $(df -h --output=avail ${TMPDIR:-/tmp} | tail -1) free"
fw=$(df -B1G --output=avail $P | tail -1); fh=$(df -B1G --output=avail $HOME | tail -1)
[ $fw -ge 10 ] || [ $fh -ge 10 ] || { echo "/work ${fw} GB and /home ${fh} GB free: too little for outboxes; no heads claimed"; exit 4; }
HEADS=()
while read -r h; do
  [ ${#HEADS[@]} -ge $HEADS_PER_TASK ] && break
  [ -n "$h" ] || continue
  [ -f $P/results/v22_fem_fullsearch/$h.done ] && continue
  [ -d $P/data/m2m_$h ] || [ -s $STORE/m2m_$h.tar.gz ] || continue
  claim "$h" && HEADS+=("$h")
done < "$LIST"
echo "task $TID (job $ME) on $(hostname): ${#HEADS[@]} heads (${HEADS[*]}) | PAR=$PAR THREADS=$THREADS GPUs=$NGPU | $(date)"
[ ${#HEADS[@]} -gt 0 ] || { echo "no unclaimed heads"; exit 0; }
i=0
for h in "${HEADS[@]}"; do
  gpu=$(( i % NGPU ))
  bash ~/spicehead_setup/head_pipeline_v2.sh "$h" $THREADS $gpu > /dev/null 2>&1 &
  i=$(( i + 1 ))
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
done
wait
echo "task $TID finished $(date)"
for h in "${HEADS[@]}"; do [ -f $P/results/v22_fem_fullsearch/$h.done ] && echo "  $h DONE" || echo "  $h INCOMPLETE (see $P/logs/$h.log)"; done
