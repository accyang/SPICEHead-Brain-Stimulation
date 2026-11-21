#!/usr/bin/env bash
# Forerunner 1: four-pad FEM check of the ref6 cohort montages for the 300 heads (31 Oct 2026; review decision D4).
# Per head: the unpacked m2m folder (data/m2m_<code>, from the 26 Oct tarballs) plus its mesh (mesh/<code>.msh,
# uploaded by the PI, SHA-256 checked) -> pipeline/nchc/pad4_fem_check.py under simnibs_env, 8 threads, 14 heads per
# node in parallel. Heads sharded over the array tasks; resumable (a finished head has results/pad4/<code>__pad4.json).
#   sbatch --array=0-2 ~/pad4_f1.sbatch
#SBATCH --job-name=pad4
#SBATCH --account=<project>
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=112
#SBATCH --time=24:00:00
#SBATCH --output=/work1/<account>/spicehead/logs/pad4_%A_%a.log
set -uo pipefail
P=/work1/<account>/spicehead; RD=${RUNDIR:-$P}; OUT=$P/results/pad4; MESH=$P/mesh
NT=${SLURM_ARRAY_TASK_COUNT:-1}; K=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $OUT $P/logs/pad4
source $P/miniforge3/etc/profile.d/conda.sh && conda activate simnibs_env || { echo "simnibs_env missing"; exit 2; }
export SPICEHEAD_MODEL=v23 PYTHONPATH=$P/code:$P/code/pipeline PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export SPICEHEAD_FULLSEARCH_CACHE=${TMPDIR:-/tmp}/pad4_cache_$$; mkdir -p $SPICEHEAD_FULLSEARCH_CACHE
mine=$(tr -d '\r' < $P/jobs/heads300.csv | tail -n +2 | cut -d, -f1 | grep . | awk -v n=$NT -v k=$K '(NR-1)%n==k')
echo "pad4 task $K of $NT on $(hostname): $(echo "$mine" | wc -w) heads | $(date)"
one() {
  local c=$1 log=$P/logs/pad4/$1.log
  [ -e $OUT/${c}__pad4.json ] && { echo "$c done before"; return; }
  local m2m=$RD/data/m2m_$c
  if [ ! -e $m2m/.ok ]; then
    local want got; want=$(awk -v c=$c '$1==c{print $2}' $P/jobs/code_sha.txt); got=$(sha256sum $P/tar/m2m_$c.tar.gz | cut -d' ' -f1)
    [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: tarball checksum mismatch" | tee -a $log; return; }
    rm -rf $m2m && tar -xzf $P/tar/m2m_$c.tar.gz -C $RD/data && touch $m2m/.ok || { echo "$c: unpack failed" | tee -a $log; return; }
  fi
  want=$(awk -v f=$c.msh '$2==f{print $1}' $MESH/SHA256SUMS.txt); got=$(sha256sum $MESH/$c.msh 2>/dev/null | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c: mesh missing or checksum mismatch" | tee -a $log; return; }
  [ -e $m2m/$c.msh ] || ln -s $MESH/$c.msh $m2m/$c.msh
  ( cd $P/code && OMP_NUM_THREADS=8 python pipeline/nchc/pad4_fem_check.py --head $c --m2m $m2m --montages $P/jobs/uq_fixed_montages.json --out $OUT --threads 8 ) >> $log 2>&1
  echo "$c: $( [ -e $OUT/${c}__pad4.json ] && echo done || echo FAILED ) | $(date +%T)"
}
i=0
for c in $mine; do
  one $c &
  i=$((i+1))
  while [ "$(jobs -rp | wc -l)" -ge 14 ]; do sleep 30; done
done
wait
echo "task $K finished $(date): $(for c in $mine; do [ -e $OUT/${c}__pad4.json ] && echo $c; done | wc -l) of $(echo "$mine" | wc -w) heads done"
