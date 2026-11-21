#!/bin/bash
# CPU-per-GPU trial (2 Nov 2026): the same two uncertainty runs on one head, one GPU, with the CPU count given to sbatch.
# Measures elapsed time, billed CPU, busy CPU and GPU use, and checks that the numbers do not depend on the CPU count.
# Outputs stay in node scratch; only the summary line goes to $P/cpu_trial. Billing is 2.5 SU per allocated core-hour.
#   for c in 2 4 12; do sbatch -c $c ~/spicehead_setup/cpu_trial_nano4.sbatch; done
#SBATCH --job-name=cpu_trial
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/work/<account>/spicehead/cpu_trial/job_%j.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
C=${TRIAL_HEAD:-h001}; RUNS=${TRIAL_RUNS:-nominal,cond_00}
P=/work/<account>/spicehead; STORE=/home/<account>/spicehead/tar; U=$P/uq_v23; T=$P/cpu_trial
N=${SLURM_CPUS_PER_TASK:-1}; SCR=${TMPDIR:-/tmp}/cpu_trial_${SLURM_JOB_ID}
mkdir -p $T $SCR/data $SCR/cache $SCR/a $SCR/t; trap 'kill $SAMPLER 2>/dev/null; rm -rf "$SCR"' EXIT
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu SPICEHEAD_SCORE=gpu PYTHONPATH=$P/code:$P/code/pipeline PYTHONHASHSEED=0
export SPICEHEAD_CHARM_ROOT=$SCR/data SPICEHEAD_FULLSEARCH_CACHE=$SCR/cache \
       SPICEHEAD_UQ_ATLAS_OUT=$SCR/a SPICEHEAD_UQ_ATLAS_FIXED=$U/uq_atlas_fixed_montages.json \
       SPICEHEAD_UQ_THAL_OUT=$SCR/t SPICEHEAD_UQ_THAL_FIXED=$U/uq_atlas_fixed_montages_thal.json
export OMP_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N MKL_NUM_THREADS=$N
tar -xzf $STORE/m2m_$C.tar.gz -C $SCR/data || { echo "unpack failed"; exit 2; }
( while :; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' '; sleep 5; done ) > $SCR/gpu.txt &
SAMPLER=$!
t0=$(date +%s)
( cd $P/code && python pipeline/nchc/uq_v23_atlas.py --head $C --targets $U/targets/${C}__uqtargets2.npz \
    --also-thal $P/atlas_ext/targets/${C}__exttargets.npz --only $RUNS )
rc=$?; t1=$(date +%s); kill $SAMPLER 2>/dev/null
digest=$(python - "$SCR/a/$C" "$SCR/t/$C" <<'PY'
import glob, hashlib, os, sys
import numpy as np
h = hashlib.sha256()
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(f) as z:
            h.update(np.round(z["fixed_vals"].astype(np.float64), 5).tobytes())
print(h.hexdigest()[:16])
PY
)
gpu=$(awk -F, '{s+=$1; n++; if($1>10)b++; if($2>m)m=$2} END{printf "gpu_util_mean=%.0f%% gpu_busy_samples=%.0f%% gpu_mem_peak=%dMiB samples=%d", s/n, 100*b/n, m, n}' $SCR/gpu.txt)
line="job=$SLURM_JOB_ID node=$(hostname) cpus=$N rc=$rc elapsed_s=$((t1-t0)) $gpu digest=$digest"
echo "$line"; echo "$line" >> $T/results.txt
