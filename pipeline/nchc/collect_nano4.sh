#!/usr/bin/env bash
# Collect the full-search comparison over every finished head and pack the small outputs for download.
#   bash ~/spicehead_setup/collect_nano4.sh        (login node is fine: a few minutes of numpy)
set -uo pipefail
P=${PREFIX:-/work/<account>/spicehead}
export SPICEHEAD_CHARM_ROOT=$P/data SPICEHEAD_FEM_BASIS_ROOT=$P/fem_basis SPICEHEAD_FULLSEARCH_CACHE=$P/cache
export SPICEHEAD_FULLSEARCH_OUT=$P/results/v22_fem_fullsearch SPICEHEAD_MODEL=v22 PYTHONHASHSEED=0
# pinned list = every head with all three score files
( echo CaseID; for f in $P/results/v22_fem_fullsearch/*__fem.json; do c=$(basename $f __fem.json); [ -s $P/results/v22_fem_fullsearch/${c}__sp2.json ] && [ -s $P/results/v22_fem_fullsearch/${c}__sp1.json ] && echo $c; done ) > $P/heads/complete.csv
export SPICEHEAD_FS_HEADS=$P/heads/complete.csv
echo "complete heads: $(( $(wc -l < $P/heads/complete.csv) - 1 ))"
source $P/activate_gpu.sh
cd $P/code && python pipeline/v22_fem_fullsearch.py --collect 2>&1 | tail -20
OUT=$P/results/fullsearch_export_$(date +%Y%m%d).tar.gz
tar -czf $OUT -C $P/results v22_fem_fullsearch --exclude='*.part' && ls -la $OUT
echo "download with:  rsync -av nano4-dtn:$OUT /mnt/c/spicehead_v22_logs/nchc_stage/"
