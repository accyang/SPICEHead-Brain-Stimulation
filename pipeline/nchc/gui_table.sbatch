#!/usr/bin/env bash
# Optimal-montage table for the SPICEhead GUI on Nano 4, for any model version (24 Oct 2026; generalized from
# gui_table_v23.sbatch when the PI asked for the v22 table too). montage_aal.py under SPICEHEAD_MODEL=<model> on the
# MNI152 CHARM template head, with the settings of the existing
# (pre-v22) table: dx 4 mm, grey matter only, 20 mm halo; all 11,628 two-pair 10-20 montages for every AAL region.
# Inputs (public, no subject data) are staged in $G by wsl_gui_table_v23.sh: the template segmentation, the SPM12
# AAL atlas, SimNIBS's EEG10-10_UI_Jurak_2007.csv cap file and pipeline/aal/aal_regions.json.
#   sbatch ~/spicehead_setup/gui_table.sbatch <model>      (v22 or v23)
#SBATCH --job-name=gui_table
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour; GPUs and memory are not billed. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/<account>/spicehead/gui_v23/job_%j.log
set -uo pipefail
M=${1:?usage: sbatch gui_table.sbatch <model>}
P=/work/<account>/spicehead; G=$P/gui_v23     # inputs staged on 24 Oct; the folder holds the tables of every model
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=$M PYTHONPATH=$P/code:$P/code/pipeline SPICEHEAD_CAP_CSV=$G/EEG10-10_UI_Jurak_2007.csv \
       OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=0
cd $P/code
echo "$(hostname) | $(date)"; python -c "from spicehead import config; print(config.describe())"
echo "code: $(sha256sum pipeline/montage_aal.py pipeline/montage_search.py spicehead/homogenize.py spicehead/mri.py | cut -c1-16 | tr '\n' ' ')"
python pipeline/montage_aal.py $G/m2m_MNItempl $G/AAL.nii $G/aal_regions.json $G/aal_optimal_montages_$M.json 4.0 gm on 2>&1 | grep -v Warning
sha256sum $G/aal_optimal_montages_$M.json > $G/aal_optimal_montages_$M.json.sha256 && cat $G/aal_optimal_montages_$M.json.sha256
echo "finished $(date)"
