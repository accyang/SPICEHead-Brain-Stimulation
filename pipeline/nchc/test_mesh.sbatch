#!/usr/bin/env bash
# One-head mesh test: `charm <code> --mesh` from an existing label image, inside a Slurm job.
# Fill in ACCOUNT and PARTITION, then:   sbatch --export=ALL,PREFIX=/work1/$USER/spicehead test_mesh.sbatch
# Expected: 35-46 min on one head with 4 threads (measured on our desktop); needs ~8-16 GB RAM.
#SBATCH --job-name=charm_mesh_test
#SBATCH --account=__PROJECT_ID__        # e.g. <project>
#SBATCH --partition=__PARTITION__       # a CPU or test partition on the host you are on
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=charm_mesh_test_%j.log
set -euo pipefail
: "${PREFIX:=/work1/$USER/spicehead}"
: "${CODE:=h001}"                       # neutral head code; folder m2m_$CODE must sit in $DATA
: "${DATA:=$PREFIX/data}"
source "$PREFIX/activate_simnibs.sh"
cd "$DATA"
[ -d "m2m_$CODE" ] || { echo "m2m_$CODE not found in $DATA"; exit 1; }
echo "host $(hostname) | cpus ${SLURM_CPUS_PER_TASK:-1} | start $(date)"
t0=$(date +%s)
charm "$CODE" --mesh          # thread count comes from OMP_NUM_THREADS (set in activate_simnibs.sh)
t1=$(date +%s)
ls -la "m2m_$CODE/$CODE.msh"
python - <<PY
from simnibs import mesh_io
m = mesh_io.read_msh("m2m_$CODE/$CODE.msh")
print("mesh OK:", m.nodes.nr, "nodes,", m.elm.nr, "elements; tags", sorted(set(m.elm.tag1.tolist()))[:12])
PY
echo "mesh time $(( (t1-t0)/60 )) min | end $(date)"
