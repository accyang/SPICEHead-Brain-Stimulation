#!/usr/bin/env bash
# v23 optimal-montage table for the SPICEhead GUI, on Nano 4 (24 Oct 2026; PI: "generate the v23 GUI table on nano4").
# montage_aal.py under SPICEHEAD_MODEL=v23 on the MNI152 CHARM template head, with the settings of the existing
# (pre-v22) table: dx 4 mm, grey matter only, 20 mm halo; all 11,628 two-pair 10-20 montages for every AAL region.
# Inputs (public, no subject data) are staged in $G by wsl_gui_table_v23.sh: the template segmentation, the SPM12
# AAL atlas, SimNIBS's EEG10-10_UI_Jurak_2007.csv cap file and pipeline/aal/aal_regions.json.
#   sbatch ~/spicehead_setup/gui_table_v23.sbatch
#SBATCH --job-name=gui_v23
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour; GPUs and memory are not billed. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/<account>/spicehead/gui_v23/job_%j.log
set -uo pipefail
P=/work/<account>/spicehead; G=$P/gui_v23
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=v23 PYTHONPATH=$P/code:$P/code/pipeline SPICEHEAD_CAP_CSV=$G/EEG10-10_UI_Jurak_2007.csv \
       OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=0
cd $P/code
echo "$(hostname) | $(date)"; python -c "from spicehead import config; print(config.describe())"
echo "code: $(sha256sum pipeline/montage_aal.py pipeline/montage_search.py spicehead/homogenize.py spicehead/mri.py | cut -c1-16 | tr '\n' ' ')"
python pipeline/montage_aal.py $G/m2m_MNItempl $G/AAL.nii $G/aal_regions.json $G/aal_optimal_montages_v23.json 4.0 gm on 2>&1 | grep -v Warning
sha256sum $G/aal_optimal_montages_v23.json > $G/aal_optimal_montages_v23.json.sha256 && cat $G/aal_optimal_montages_v23.json.sha256
echo "finished $(date)"
