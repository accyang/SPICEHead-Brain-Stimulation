#!/usr/bin/env bash
# Cerebellum and brainstem extension of the 300-head atlas on Nano 4 (24 Oct 2026; PI: "build the FEM pipeline for all
# 300 heads (also on nano4)"). One head per GPU, 8 per node; workers claim heads as their inputs arrive:
#   $X/meshes/<code>.msh + <code>.msh.sha256 (the FEM mesh of the 300-head run, uploaded by the PI; sha written last)
#   $X/targets/<code>__exttargets.npz (+ SHA256SUMS.txt)            the /home tarball store supplies the m2m folder
# Per head: unpack m2m (SHA-256 checked) -> mesh into m2m -> FEM leadfield with WM+GM stored (atlas_ext_fem.py,
# SimNIBS env, 12 threads) -> SPICEhead v22 2/1/5 mm and v23 2/1 mm bases and scoring of 44 targets for 6 models
# (atlas_ext_score.py, GPU env) -> $X/out/<code>__ext.npz. The FEM brainstem basis of every head goes to $X/fem_bs/,
# the full FEM extension of heads with code number % 30 == 1 to $X/fem_check/ (desktop check against the published
# FEM basis). After a head succeeds its mesh copy is deleted (the L: copy is the source), so the upload can continue
# within the /work quota. A task ends when UPLOAD_DONE exists and nothing is left, or after 2 h without work.
#   sbatch --array=0-4 ~/spicehead_setup/atlas_ext_nano4.sbatch
#SBATCH --job-name=atlas_ext
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour. Reduced from measured mean use; this job type has CPU stages and was not run at this size. Check speed on a small array first.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=10:00:00
#SBATCH --output=/work/<account>/spicehead/atlas_ext/job_%A_%a.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; X=$P/atlas_ext
SHA=$P/jobs/code_sha.txt; [ -s $SHA ] || SHA=/home/<account>/code_sha.txt
ME=${SLURM_JOB_ID:-$$}
SCR=${TMPDIR:-/tmp}/atlas_ext_$ME
mkdir -p $X/out $X/fem_bs $X/fem_check $X/logs $X/claims $X/done $X/failed $SCR; trap 'rm -rf "$SCR"' EXIT
NGPU=$(nvidia-smi -L | wc -l)
echo "atlas extension task ${SLURM_ARRAY_TASK_ID:-0} (job $ME) on $(hostname), $NGPU GPUs | $(date)"
echo "code: $(cd $P/code && sha256sum pipeline/nchc/atlas_ext_fem.py pipeline/nchc/atlas_ext_score.py pipeline/nchc/score_targets_v2.py spicehead/homogenize.py spicehead/mri.py | cut -c1-16 | tr '\n' ' ')"
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
  mkdir -p $d/data $d/out
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $SHA); got=$(sha256sum $STORE/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; return 1; }
  tar -xzf $STORE/m2m_$c.tar.gz -C $d/data || { echo "$c: unpack failed" | tee -a $log; return 1; }
  [ "$(sha256sum $X/meshes/$c.msh | cut -d' ' -f1)" = "$(cut -d' ' -f1 $X/meshes/$c.msh.sha256)" ] || { echo "$c: mesh checksum mismatch" | tee -a $log; return 1; }
  cp $X/meshes/$c.msh $d/data/m2m_$c/$c.msh
  local tf=$X/targets/${c}__exttargets.npz
  [ "$(sha256sum $tf | cut -d' ' -f1)" = "$(awk -v f=$(basename $tf) '$2==f{print $1}' $X/targets/SHA256SUMS.txt)" ] || { echo "$c: target file checksum mismatch" | tee -a $log; return 1; }
  ( source $P/activate_simnibs.sh; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$P/code:$P/code/pipeline
    cd $P/code && python pipeline/nchc/atlas_ext_fem.py --head $c --m2m $d/data/m2m_$c --targets $tf --out $d/out --work $d/lf ) >> $log 2>&1 \
    || { echo "$c: FEM step failed (see $log)"; return 1; }
  ( source $P/activate_gpu.sh; export CUDA_VISIBLE_DEVICES=$gpu SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu SPICEHEAD_SCORE=gpu PYTHONHASHSEED=0 \
      OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=$P/code:$P/code/pipeline SPICEHEAD_FULLSEARCH_CACHE=$d/cache
    cd $P/code && python pipeline/nchc/atlas_ext_score.py --head $c --m2m $d/data/m2m_$c --targets $tf --fem $d/out/${c}__fem_ext.npz --out $X/out ) >> $log 2>&1 \
    || { echo "$c: scoring step failed (see $log)"; return 1; }
  ( source $P/activate_gpu.sh; cd $d/out && python -c "
import numpy as np, sys
z = np.load('${c}__fem_ext.npz'); np.savez('$X/fem_bs/${c}__fem_bs.npz.part.npz', basis_bs=z['basis_bs'], labs=z['labs'], meta=z['meta'])" ) && mv $X/fem_bs/${c}__fem_bs.npz.part.npz $X/fem_bs/${c}__fem_bs.npz
  [ $(( 10#${c#h} % 30 )) -eq 1 ] && cp $d/out/${c}__fem_ext.npz $X/fem_check/
  rm -f $X/meshes/$c.msh $X/meshes/$c.msh.sha256 && touch $X/done/$c
  echo "$(date +%T) $c done on GPU $gpu ($(grep -o 'extension scored ([0-9]* s)' $log | tail -1))"
  rm -rf $d
}
worker() {
  local gpu=$1 idle=0 c f
  while :; do
    c=""
    for f in $(ls $X/meshes/h[0-9]*.msh.sha256 2>/dev/null | sort); do
      local k; k=$(basename $f .msh.sha256)
      [ -e $X/done/$k ] && continue
      [ -e $X/failed/$k.2 ] && continue                     # two failures: left for inspection
      [ -s $X/meshes/$k.msh ] || continue
      claim $k && { c=$k; break; }
    done
    if [ -z "$c" ]; then
      left=$(for f in $(ls $X/meshes/h[0-9]*.msh.sha256 2>/dev/null); do k=$(basename $f .msh.sha256); [ -e $X/done/$k ] || [ -e $X/failed/$k.2 ] || echo $k; done | wc -l)
      [ -f $X/UPLOAD_DONE ] && [ "$left" -eq 0 ] && break
      [ $idle -ge 7200 ] && { echo "worker $gpu: 2 h without work, stopping"; break; }
      sleep 60; idle=$((idle+60)); continue
    fi
    idle=0
    one $c $gpu || { n=1; [ -e $X/failed/$c.1 ] && n=2; touch $X/failed/$c.$n; echo "$(date +%T) $c FAILED on GPU $gpu (failure $n); mesh kept"; rm -rf $X/claims/$c $SCR/$c; sleep 30; }
  done
}
for g in $(seq 0 $((NGPU-1))); do worker $g & done
wait
echo "task finished $(date) | heads done in total: $(ls $X/done | wc -l)"
