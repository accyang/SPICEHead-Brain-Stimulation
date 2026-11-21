#!/usr/bin/env bash
# AAL3 thalamic nuclei and supracallosal ACC for the 300-head atlas on Nano 4 (25 Oct 2026; PI: "add the thalamic nuclei
# and ACC_sup on nano4"). One head per GPU, 8 per node; workers claim heads as their inputs arrive:
#   $X/fem/<code>__fem_basis.npz + .sha256   the published FEM grey-matter basis (uploaded by the PI; sha written last)
#   $P/atlas_ext/targets/<code>__exttargets.npz (+ SHA256SUMS.txt)  already on the cluster (AAL3 labels on grey matter)
#   the /home tarball store supplies the m2m folder (electrodes, label volume)
# Per head: unpack m2m (SHA-256 checked) -> SPICEhead v22 2/1/5 mm and v23 2/1 mm bases and scoring of 22 targets for
# 6 models (atlas_thal_score.py, GPU env) -> $X/out/<code>__thal.npz. After a head succeeds its FEM basis copy is
# deleted (the L: copy is the source), so the upload can continue within the /work quota. A task ends when
# UPLOAD_DONE exists and nothing is left, or after 2 h without work.
#   sbatch --array=0-3 ~/spicehead_setup/atlas_thal_nano4.sbatch
#SBATCH --job-name=atlas_thal
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour; GPUs and memory are not billed. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/work/<account>/spicehead/atlas_thal/job_%A_%a.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; X=$P/atlas_thal; TD=$P/atlas_ext/targets
SHA=$P/jobs/code_sha.txt; [ -s $SHA ] || SHA=/home/<account>/code_sha.txt
ME=${SLURM_JOB_ID:-$$}
SCR=${TMPDIR:-/tmp}/atlas_thal_$ME
mkdir -p $X/out $X/logs $X/claims $X/done $X/failed $SCR; trap 'rm -rf "$SCR"' EXIT
NGPU=$(nvidia-smi -L | wc -l)
echo "thalamic nuclei task ${SLURM_ARRAY_TASK_ID:-0} (job $ME) on $(hostname), $NGPU GPUs | $(date)"
echo "code: $(cd $P/code && sha256sum pipeline/nchc/atlas_thal_score.py pipeline/nchc/atlas_ext_score.py pipeline/nchc/score_targets_v2.py spicehead/homogenize.py spicehead/mri.py | cut -c1-16 | tr '\n' ' ')"
claim() {
  local d=$X/claims/$1
  if mkdir "$d" 2>/dev/null; then echo "$ME" > "$d/owner"; return 0; fi
  local o; o=$(cat "$d/owner" 2>/dev/null)
  local q; q=$(squeue -h -u "$USER" -o %A 2>/dev/null) && [ -n "$q" ] || return 1
  if [ -n "$o" ] && [ "$o" != "$ME" ] && ! grep -qx "$o" <<< "$q"; then
    mkdir "$d/takeover_from_$o" 2>/dev/null || return 1; echo "$ME" > "$d/owner"; return 0
  fi
  return 1
}
one() {
  local c=$1 gpu=$2 log=$X/logs/$1.log d=$SCR/$1
  mkdir -p $d/data
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $SHA); got=$(sha256sum $STORE/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; return 1; }
  tar -xzf $STORE/m2m_$c.tar.gz -C $d/data || { echo "$c: unpack failed" | tee -a $log; return 1; }
  local ff=$X/fem/${c}__fem_basis.npz
  [ "$(sha256sum $ff | cut -d' ' -f1)" = "$(cut -d' ' -f1 $ff.sha256)" ] || { echo "$c: FEM basis checksum mismatch" | tee -a $log; return 1; }
  local tf=$TD/${c}__exttargets.npz
  [ "$(sha256sum $tf | cut -d' ' -f1)" = "$(awk -v f=$(basename $tf) '$2==f{print $1}' $TD/SHA256SUMS.txt)" ] || { echo "$c: target file checksum mismatch" | tee -a $log; return 1; }
  ( source $P/activate_gpu.sh; export CUDA_VISIBLE_DEVICES=$gpu SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu SPICEHEAD_SCORE=gpu PYTHONHASHSEED=0 \
      OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=$P/code:$P/code/pipeline SPICEHEAD_FULLSEARCH_CACHE=$d/cache
    cd $P/code && python pipeline/nchc/atlas_thal_score.py --head $c --m2m $d/data/m2m_$c --targets $tf --fem $ff --out $X/out ) >> $log 2>&1 \
    || { echo "$c: scoring step failed (see $log)"; return 1; }
  rm -f $ff $ff.sha256 && touch $X/done/$c
  echo "$(date +%T) $c done on GPU $gpu ($(grep -o 'ACC_sup scored ([0-9]* s)' $log | tail -1))"
  rm -rf $d
}
worker() {
  local gpu=$1 idle=0 c f
  while :; do
    c=""
    for f in $(ls $X/fem/h[0-9]*__fem_basis.npz.sha256 2>/dev/null | sort); do
      local k; k=$(basename $f __fem_basis.npz.sha256)
      [ -e $X/done/$k ] && continue
      [ -e $X/failed/$k.2 ] && continue                     # two failures: left for inspection
      [ -s $X/fem/${k}__fem_basis.npz ] || continue
      claim $k && { c=$k; break; }
    done
    if [ -z "$c" ]; then
      left=$(for f in $(ls $X/fem/h[0-9]*__fem_basis.npz.sha256 2>/dev/null); do k=$(basename $f __fem_basis.npz.sha256); [ -e $X/done/$k ] || [ -e $X/failed/$k.2 ] || echo $k; done | wc -l)
      [ -f $X/UPLOAD_DONE ] && [ "$left" -eq 0 ] && break
      [ $idle -ge 7200 ] && { echo "worker $gpu: 2 h without work, stopping"; break; }
      sleep 60; idle=$((idle+60)); continue
    fi
    idle=0
    one $c $gpu || { n=1; [ -e $X/failed/$c.1 ] && n=2; touch $X/failed/$c.$n; echo "$(date +%T) $c FAILED on GPU $gpu (failure $n); basis kept"; rm -rf $X/claims/$c $SCR/$c; sleep 30; }
  done
}
for g in $(seq 0 $((NGPU-1))); do worker $g & done
wait
echo "task finished $(date) | heads done in total: $(ls $X/done | wc -l)"
