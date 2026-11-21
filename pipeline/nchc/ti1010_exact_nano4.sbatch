#!/usr/bin/env bash
# Exhaustive 10-10 two-pair search on the MNI template (31 Oct 2026; review decision D9: certify the screened optimum).
# One GPU: basis (75 solves at 2 mm, cached) then pipeline/ti_template_1010_exact.py for the six insula/hippocampus
# targets and the two insula subregions. Public inputs only (MNI152 template segmentation, AAL atlas, SimNIBS cap).
#   sbatch ~/spicehead_setup/ti1010_exact_nano4.sbatch
#SBATCH --job-name=ti1010x
#SBATCH --account=<project>
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
# CPUs (2 Nov 2026): billing is 2.5 SU per allocated CPU core-hour; GPUs and memory are not billed. Trial jobs 672284-6: 2, 4, 12 CPUs on one GPU gave the same time and identical numbers.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/work/<account>/spicehead/template/job_%j.log
#SBATCH --exclude=25a-hgpn062,25a-hgpn071
set -uo pipefail
P=/work/<account>/spicehead; T=$P/template
source $P/activate_gpu.sh
export SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu SPICEHEAD_SCORE=gpu PYTHONPATH=$P/code:$P/code/pipeline PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export SPICEHEAD_MNI_M2M=$T/m2m_MNItempl SPICEHEAD_AAL_NII=$T/AAL.nii SPICEHEAD_TI1010_CACHE=$T/ti1010_v23
export SPICEHEAD_CAP_CSV=$P/miniforge3/envs/simnibs_env/lib/python3.11/site-packages/simnibs/resources/ElectrodeCaps_MNI/EEG10-10_UI_Jurak_2007.csv
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
[ -s $SPICEHEAD_CAP_CSV ] || { echo "cap file missing"; exit 2; }
cd $P/code
echo "code: $(sha256sum pipeline/ti_template_1010.py pipeline/ti_template_1010_exact.py pipeline/fig_ti_targets_mni.py pipeline/nchc/score_targets_v2.py spicehead/mri.py spicehead/score_gpu.py | cut -c1-16 | tr '\n' ' ') | $(date)"
for sub in "" "--subregions"; do
  python pipeline/ti_template_1010.py --basis $sub || { echo "basis failed $sub"; exit 3; }
  python pipeline/ti_template_1010_exact.py $sub --out $T/exhaustive_1010${sub:+_subregions} || { echo "exhaustive failed $sub"; exit 4; }
done
echo "finished $(date)"
