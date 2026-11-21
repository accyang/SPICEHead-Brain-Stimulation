"""
Is O (the focality denominator) measuring real off-target field, or discretisation artefact?

T/O uses O = p99(envelope in brain >20 mm from ROI). On a voxel grid, E = -grad(V) by central
differences straddles tissue conductivity jumps at every boundary, and the normal component of E
genuinely jumps across a sigma discontinuity too. Either way the brain-edge / CSF-adjacent voxels
carry the largest envelope values - exactly the voxels that set a 99th percentile.

If p99 is dominated by a 1-voxel rim, then T/O is (a) pessimistic and (b) sensitive to segmentation
and dx rather than to the montage. This audits that, and reports T/O under several O definitions.
It does NOT propose picking whichever definition looks best - the point is to find out whether the
number is robust.

  python focality_metric_audit.py [region] [dx]
"""
import os
import sys

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead_gui.engine import TemplateEngine

M2M = r"L:\BrainProbe\SPICEhead\charm\m2m_MNItempl"
AAL = r"C:\Users\ALBERT YANG\nilearn_data\aal_SPM12\aal\atlas\AAL.nii"
OPTIMA = os.path.join(ROOT, "pipeline", "aal", "aal_optimal_montages.json")
GM, WM = 2, 1
EXCLUDE_MM = 20.0


def main(region="Thalamus_L", dx=4.0):
    E = TemplateEngine(M2M, AAL, OPTIMA, dx=dx)
    o = E.optimal(region)
    env = np.nan_to_num(E.simulate(o["pairs"]))
    roi = E.region_mask(o["value"])
    T = float(np.median(env[roi]))
    print(f"{region}  montage {o['montage']}   T = {T:.4f} V/m   (dx={dx})\n")

    dist = distance_transform_edt(~roi, sampling=(dx, dx, dx))
    far = dist > EXCLUDE_MM
    brain = E.brain                      # GM + WM
    gm = (E.labels == GM)

    # a voxel is "rim" if any 6-neighbour leaves the brain mask -> its central difference
    # straddles a conductivity jump
    interior = binary_erosion(brain, structure=np.array(
        [[[0,0,0],[0,1,0],[0,0,0]], [[0,1,0],[1,1,1],[0,1,0]], [[0,0,0],[0,1,0],[0,0,0]]], bool))
    rim = brain & ~interior

    defs = {
        "brain, p99            (current)": brain & far,
        "brain interior, p99   (rim removed)": interior & far,
        "grey matter, p99": gm & far,
        "grey interior, p99": (gm & interior) & far,
    }
    print(f"{'O definition':<38} {'n vox':>8} {'O':>8} {'T/O':>7}")
    for name, m in defs.items():
        v = env[m]
        O = float(np.percentile(v, 99))
        print(f"{name:<38} {v.size:>8} {O:>8.4f} {T/O:>7.3f}")

    # how much of the top 1% off-target sits on the rim?
    off = brain & far
    v = env[off]
    thr = np.percentile(v, 99)
    hot = off & (env >= thr)
    frac_rim = float((hot & rim).sum()) / max(hot.sum(), 1)
    print(f"\ntop-1% off-target voxels: {int(hot.sum())}, of which {100*frac_rim:.1f}% are brain-rim")
    print(f"rim is {100*rim.sum()/brain.sum():.1f}% of the brain mask by volume")

    # CONSISTENCY CHECK. Restricting only O to grey matter is not a like-for-like comparison:
    # the thalamus ROI at 4 mm also contains white matter, and WM carries a higher envelope
    # (sigma_WM < sigma_GM so E = J/sigma is larger). Numerator and denominator must use the
    # SAME tissue class or the ratio just measures the GM/WM conductivity contrast.
    print("\nlike-for-like (same tissue class in T and O):")
    roi_gm = roi & gm
    roi_gm_int = roi & gm & interior
    rows = [
        ("GM+WM  / GM+WM   (current)", roi, brain & far),
        ("GM     / GM", roi_gm, gm & far),
        ("GM int / GM int", roi_gm_int, (gm & interior) & far),
    ]
    for name, rmask, omask in rows:
        if rmask.sum() == 0:
            print(f"   {name:<28} ROI empty at this dx")
            continue
        Tn = float(np.median(env[rmask]))
        On = float(np.percentile(env[omask], 99))
        print(f"   {name:<28} nROI={int(rmask.sum()):>4}  T={Tn:.4f}  O={On:.4f}  T/O={Tn/On:.3f}")
    print(f"   [ROI tissue mix: {100*roi_gm.sum()/max(roi.sum(),1):.0f}% GM, "
          f"{100*(roi & (E.labels == WM)).sum()/max(roi.sum(),1):.0f}% WM]")
    print(f"   [median env: GM {np.median(env[gm & far]):.4f} vs "
          f"WM {np.median(env[(E.labels == WM) & far]):.4f} V/m  <- conductivity contrast]")

    # percentile sweep - how fast does O fall away from the extreme tail?
    print("\nO and T/O vs percentile (brain, far):")
    for p in (50, 90, 95, 97, 99, 99.5, 99.9):
        Op = float(np.percentile(v, p))
        print(f"   p{p:<5} O={Op:.4f}  T/O={T/Op:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Thalamus_L",
         float(sys.argv[2]) if len(sys.argv) > 2 else 4.0)
