#!/usr/bin/env bash
# v23 package check on Nano 4 (23 Oct 2026, after the PI approved v23 and its addition to the spicehead package).
# 1. tests/test_homogenize.py and tests/test_validation.py with the deployed package;
# 2. the production pipeline (v22_fem_fullsearch.py --sp-basis 2/1, --score sp2/sp1) under SPICEHEAD_MODEL=v23 on
#    one head, unpacked from the /home tarball store (SHA-256 checked);
# 3. its score files against the 300-head homogenized run ($E/homog300/<code>__sp{2,1}hom.json, prototype code).
# Report: $E/v23_check/<code>.json and the job log.
#   sbatch ~/spicehead_setup/v23_check_nano4.sbatch h001
#SBATCH --job-name=v23_check
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/<account>/spicehead/experiments/v23_check_%j.log
set -uo pipefail
CODE=${1:?usage: sbatch v23_check_nano4.sbatch <code>}
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; E=$P/experiments
SCR=${TMPDIR:-/tmp}/v23check_${SLURM_JOB_ID:-$$}
mkdir -p $SCR/data $SCR/cache $SCR/out $E/v23_check; trap 'rm -rf "$SCR"' EXIT
source $P/activate_gpu.sh
export PYTHONPATH=$P/code:$P/code/pipeline PYTHONHASHSEED=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "v23 check on $(hostname), $CODE | $(date)"
cd $P/code
echo "== unit tests"; python tests/test_homogenize.py 2>&1 | grep -v Warning | tail -8; python tests/test_validation.py 2>&1 | tail -3
want=$(awk -v c=$CODE '$1==c{print $2}' $P/jobs/code_sha.txt); got=$(sha256sum $STORE/m2m_$CODE.tar.gz | cut -d' ' -f1)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "$CODE: tarball checksum mismatch"; exit 2; }
tar -xzf $STORE/m2m_$CODE.tar.gz -C $SCR/data || exit 2
printf "CaseID\n%s\n" "$CODE" > $SCR/head.csv
export SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu SPICEHEAD_SCORE=gpu SPICEHEAD_FS_HEADS=$SCR/head.csv \
       SPICEHEAD_CHARM_ROOT=$SCR/data SPICEHEAD_COHORT_CSV=$P/data/cohort.csv \
       SPICEHEAD_FULLSEARCH_CACHE=$SCR/cache SPICEHEAD_FULLSEARCH_OUT=$SCR/out
echo "== pipeline under v23"
for dx in 2.0 1.0; do python pipeline/v22_fem_fullsearch.py --sp-basis $dx --heads $CODE 2>&1 | grep -v Warning | grep -E "nodes|superposition|OK" ; done
for m in sp2 sp1; do python pipeline/v22_fem_fullsearch.py --score $m --heads $CODE 2>&1 | grep -v Warning | tail -1; done
echo "== compare with the 300-head run"
python - $CODE $SCR $E <<'PY'
import json, os, sys, time
import numpy as np
code, scr, E = sys.argv[1:4]
rep = dict(code=code, host=os.uname().nodename, checked=time.strftime("%Y-%m-%d %H:%M"))
for tag in ("sp2", "sp1"):
    z = np.load(os.path.join(scr, "cache", f"{code}__{tag}_v23_basis.npz"), allow_pickle=True); meta = json.loads(str(z["meta"]))
    s = json.load(open(os.path.join(scr, "out", f"{code}__{tag}.json")))["rois"]
    s0 = json.load(open(os.path.join(E, "homog300", f"{code}__{tag}hom.json")))["rois"]
    fd, same = 0.0, 0
    for roi in s0:
        k = sorted(s0[roi]); f1 = np.array([s[roi][x][0] for x in k]); f0 = np.array([s0[roi][x][0] for x in k])
        fd = max(fd, float(np.abs(f1 - f0).max() / np.abs(f0).max())); same += int(f1.argmax() == f0.argmax())
    rep[tag] = dict(F_max_rel_diff=fd, same_winner=f"{same}/{len(s0)}", spicehead_model=meta.get("spicehead_model"),
                    network=meta.get("network"), n_nodes=meta.get("n_nodes"), basis_check=meta.get("basis_check"),
                    t_build_s=meta.get("t_build_s"), t_basis_s=meta.get("t_basis_s"))
    print(tag, rep[tag])
json.dump(rep, open(os.path.join(E, "v23_check", f"{code}.json"), "w"), indent=1)
PY
echo "finished $(date)"
