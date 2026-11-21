#!/usr/bin/env bash
# Nano 4 version of the four-pad FEM check (31 Oct / 1 Nov 2026, review decision D4; Forerunner unreachable).
# Per head: unpack the m2m tarball from the /home store (SHA-256 checked against ~/code_sha.txt), rebuild the mesh with
# `charm --mesh` as head_pipeline.sh did for the 19-pad reference (threads = 8 from the staged settings.ini), then
# pipeline/nchc/pad4_fem_check.py --leadfield under simnibs_env with 8 threads: twelve 4-pad direct runs AND one 19-pad
# leadfield on the same rebuilt mesh (charm meshing is not bit-reproducible, so the 19-pad reference must share the
# mesh), six montages scored both ways. SimNIBS work in node-local scratch; only the small JSON goes to /work.
# FEM only: no GPU is used, but Nano 4 refuses a job without one, so each task holds the minimum of 1 GPU (2 Nov 2026;
# it held 8 before, 225 GPU-hours unused). Prefer Forerunner (pad4_f1.sbatch) when it is up. Three heads per task on 12
# CPUs (mean use was 1 CPU per head). Mesh threads stay 8 (settings.ini), matching the reference meshes; FEM threads 4.
# Not run at this size: check speed on a small array first. Heads are claimed atomically ($W/claims/<code>) so a second array may be added with ORDER=rev; resumable
# (a finished head has pad4/results/<code>__pad4.json). User cap: 20 queued jobs incl. the 18 UQ tasks -> 2 nodes.
#   sbatch --array=0-1 ~/spicehead_setup/pad4_nano4.sbatch          # ORDER=rev sbatch ... for a later second array
#SBATCH --job-name=pad4
#SBATCH --account=<project>
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/work/<account>/spicehead/pad4/job_%A_%a.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; SHA=/home/<account>/code_sha.txt
W=$P/pad4; OUT=$W/results; LOGS=$W/logs; CL=$W/claims; mkdir -p $OUT $LOGS $CL
NT=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}; CONC=${CONC:-3}; ORDER=${ORDER:-fwd}
SCR=${TMPDIR:-/tmp}/pad4_${SLURM_JOB_ID:-$$}; mkdir -p $SCR/data $SCR/cache $SCR/out; trap 'rm -rf "$SCR"' EXIT
source $P/activate_simnibs.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 SPICEHEAD_MODEL=v23 PYTHONPATH=$P/code:$P/code/pipeline PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export SPICEHEAD_CHARM_ROOT=$SCR/data SPICEHEAD_FULLSEARCH_CACHE=$SCR/cache
for f in $W/heads300.txt $W/uq_fixed_montages.json $P/code/pipeline/nchc/pad4_fem_check.py $SHA; do [ -s $f ] || { echo "missing $f"; exit 2; }; done
mine=$(tr -d '\r' < $W/heads300.txt | grep . | awk -v n=$NT -v k=$K '(NR-1)%n==k'); [ "$ORDER" = rev ] && mine=$(echo "$mine" | tac)
echo "pad4 task $K of $NT on $(hostname): $(echo "$mine" | wc -w) heads ($ORDER), $CONC in parallel | $(date)"
echo "code: $(cd $P/code && sha256sum pipeline/nchc/pad4_fem_check.py pipeline/v22_fem_fullsearch.py pipeline/v21_simnibs_compare.py spicehead/envelope_fast.py | cut -c1-16 | tr '\n' ' ') | simnibs $(python -c 'import simnibs; print(simnibs.__version__)')"
one() {
  local c=$1 log=$LOGS/$1.log m2m=$SCR/data/m2m_$1 t0=$(date +%s)
  [ -e $OUT/${c}__pad4.json ] && { echo "$c done before"; return; }
  mkdir $CL/$c 2>/dev/null || { echo "$c claimed by another task"; return; }
  echo "${SLURM_JOB_ID:-$$} $(hostname) $(date +%FT%T)" > $CL/$c/owner
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $SHA); got=$(sha256sum $STORE/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; rm -rf $CL/$c; return; }
  tar -xzf $STORE/m2m_$c.tar.gz -C $SCR/data || { echo "$c: unpack failed" | tee -a $log; rm -rf $CL/$c; return; }
  echo "== $c mesh start $(date +%T)" >> $log
  ( cd $SCR/data && charm $c --mesh ) >> $log 2>&1                       # exit code is unreliable (toMNI not staged); judge by the file
  [ -s $m2m/$c.msh ] || { echo "$c: mesh FAILED ($(( ($(date +%s)-t0)/60 )) min)" | tee -a $log; rm -rf $m2m $CL/$c; return; }
  sha256sum $m2m/$c.msh | awk -v c=$c '{print $1"  "c".msh"}' > $OUT/${c}.meshsha
  echo "== $c mesh OK $(( ($(date +%s)-t0)/60 )) min; pad4 + leadfield start $(date +%T)" >> $log
  ( cd $P/code && python pipeline/nchc/pad4_fem_check.py --head $c --m2m $m2m --montages $W/uq_fixed_montages.json --out $SCR/out --threads 4 --leadfield ) >> $log 2>&1
  if [ -e $SCR/out/${c}__pad4.json ]; then cp $SCR/out/${c}__pad4.json $OUT/${c}__pad4.json.part && mv $OUT/${c}__pad4.json.part $OUT/${c}__pad4.json
  elif [ -e $SCR/out/${c}__pad4.json.pad4part.json ]; then cp $SCR/out/${c}__pad4.json.pad4part.json $OUT/${c}__pad4.PARTIAL.json; rm -rf $CL/$c
  else rm -rf $CL/$c; fi
  echo "$c: $( [ -e $OUT/${c}__pad4.json ] && echo done || echo FAILED ) in $(( ($(date +%s)-t0)/60 )) min | $(date +%T)"
  rm -rf $m2m $SCR/cache/${c}__* $SCR/out/${c}__runs
}
for c in $mine; do
  one $c &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 30; done
done
wait
echo "task $K finished $(date): $(for c in $mine; do [ -e $OUT/${c}__pad4.json ] && echo $c; done | wc -l) of $(echo "$mine" | wc -w) heads done"
