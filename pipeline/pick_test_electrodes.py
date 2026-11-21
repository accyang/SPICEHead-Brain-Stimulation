"""
Pick four scalp contact voxels from a CHARM final_tissues volume for a quick
two-pair TI validation montage.

These are NOT anatomically precise EEG-10-10 positions — just crossing
left/right and front/back points on the scalp, on the same 3 mm working grid
that examples/head_model.py builds. Use it to sanity-check the whole MRI path
(CHARM label volume -> SPICEhead resistor network -> TI envelope) end to end;
swap in real EEG coordinates via spicehead.mri.world_to_voxel for science runs.

Usage:
    python pipeline/pick_test_electrodes.py <m2m_dir>/final_tissues.nii.gz [dx_mm]

Prints ready-to-paste --pair1/--pair2 arguments for examples/head_model.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spicehead import mri
from spicehead.conductivity import CHARM_LABELS

SCALP_LABEL = 5  # CHARM 'scalp'


def s(v):
    return f"{int(v[0])} {int(v[1])} {int(v[2])}"


def main(seg_path, dx=3.0):
    labels, affine, zooms, _ = mri.head_grid(seg_path, dx)     # grid of the active model version
    print(f"working grid: shape={labels.shape}  dx={dx} mm")
    present = sorted(set(np.unique(labels)) & set(CHARM_LABELS))
    print("tissue labels present:", [f"{v}:{CHARM_LABELS[v]}" for v in present])

    scalp = np.argwhere(labels == SCALP_LABEL)
    if scalp.size == 0:
        raise SystemExit("No scalp voxels (label 5) found — check the segmentation.")

    # Restrict to a mid-height axial band so contacts sit around ear level,
    # then take the extreme scalp voxels in x (L/R) and y (A/P).
    kmid = int(np.median(scalp[:, 2]))
    band = scalp[np.abs(scalp[:, 2] - kmid) <= 2]
    if band.shape[0] < 4:
        band = scalp
    left  = band[np.argmin(band[:, 0])]
    right = band[np.argmax(band[:, 0])]
    front = band[np.argmax(band[:, 1])]
    back  = band[np.argmin(band[:, 1])]

    print()
    print(f"pair1 (L-R): {s(left)}   {s(right)}")
    print(f"pair2 (A-P): {s(front)}  {s(back)}")
    print()
    print("Run SPICEhead end-to-end with:")
    print(f'  python examples/head_model.py "{seg_path}" --dx {dx} \\')
    print(f'      --pair1 {s(left)} {s(right)} \\')
    print(f'      --pair2 {s(front)} {s(back)} \\')
    print('      --current1 2e-3 --current2 2e-3 --radius 4 --out validation_env')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    dx = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    main(sys.argv[1], dx)
