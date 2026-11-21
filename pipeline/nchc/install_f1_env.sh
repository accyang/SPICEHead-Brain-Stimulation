#!/usr/bin/env bash
# CPU environment for SPICEhead on NCHC Forerunner 1 (創進一號), 26 Oct 2026 (PI: "Use forerunner"): Miniforge plus
# python/numpy/scipy/nibabel at the desktop's versions (3.12.10 / 2.4.6 / 1.18.0 / 5.4.2), so v23 results computed here
# match a desktop run as closely as possible. No subject data is involved.
#   PREFIX=/work1/<account>/spicehead bash install_f1_env.sh
set -euo pipefail
: "${PREFIX:=/work1/$USER/spicehead}"
mkdir -p "$PREFIX"
if [ ! -x "$PREFIX/miniforge3/bin/conda" ]; then
  curl -fsSL -o "$PREFIX/miniforge.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$PREFIX/miniforge.sh" -b -p "$PREFIX/miniforge3"
  rm -f "$PREFIX/miniforge.sh"
fi
source "$PREFIX/miniforge3/etc/profile.d/conda.sh"
if ! conda env list | grep -q '^spicehead_cpu '; then
  conda create -y -n spicehead_cpu -c conda-forge --override-channels python=3.12.10 numpy=2.4.6 scipy=1.18.0 nibabel=5.4.2
fi
conda activate spicehead_cpu
# matplotlib (imported by montage_full.py) at the desktop's 3.11.1: conda cannot solve it with this python pin, pip can
python -c "import matplotlib" 2>/dev/null || pip install matplotlib==3.11.1 numpy==2.4.6
python -c "import sys, numpy, scipy, nibabel, matplotlib; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| nibabel', nibabel.__version__, '| matplotlib', matplotlib.__version__)"
cat > "$PREFIX/activate_cpu.sh" <<EOF
source "$PREFIX/miniforge3/etc/profile.d/conda.sh"
conda activate spicehead_cpu
export SPICEHEAD_MODEL=v23 SPICEHEAD_ENVELOPE=maxdir PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PREFIX/code:$PREFIX/code/pipeline:\${PYTHONPATH:-}"
# the env's libstdc++ first: the CPU torch wheel otherwise loads the older system one, and conda's numpy then fails
# (GLIBCXX_3.4.29 not found) when torch is imported
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
EOF
# CPU PyTorch at the desktop's version, for HD-TI steering (ti_template_hdti.py falls back to the CPU without CUDA)
source "$PREFIX/activate_cpu.sh"
python -c "import torch" 2>/dev/null || pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"
echo "== done: source $PREFIX/activate_cpu.sh"
