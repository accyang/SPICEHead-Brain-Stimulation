# -*- coding: utf-8 -*-
"""Compare SPICEhead against the SimNIBS FEM reference (reviewer Essential item 2, stage 3).

Runs under the PROJECT interpreter (needs spicehead); stage 2 runs under the SimNIBS interpreter.

For each subject and each SPICEhead resolution (2 mm and 1 mm, per the review's suggestion), the
same montage and the same rotation-invariant envelope are applied to both vector fields, and the
comparison is made on the SimNIBS output grid inside grey matter.

Reported: regression slope/intercept, R^2 and Pearson r, normalized MAE, Bland-Altman bias and
95% limits, Dice of the top-decile field, peak-location distance, and per-ROI target error.

  python pipeline/v21_simnibs_compare.py [dx list] [n_subjects]
"""
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)
from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components
from spicehead.envelope_fast import envelope_max_direction_flat
from montage_search import (snap, roi_mask, ELEC_RADIUS_MM, ROI_RADIUS_MM,
                            SCALP, SCORE, ROI_ORDER)
from v21_head_convergence import make_solver
from v21_convergence_validation import agreement_metrics
from v21_simnibs_validation import MONTAGE, CURRENT, OUT, subjects

DXS = [2.0, 1.0]


def load_simnibs_E(fem_dir):
    """Return (Ex,Ey,Ez) on the subject voxel grid, plus the affine, from a SimNIBS run."""
    vol = os.path.join(fem_dir, "subject_volumes")
    # NOT glob: the project path contains "[New]", which glob reads as a character class matching
    # a single N/e/w, so an absolute pattern silently matches nothing. os.listdir has no such
    # syntax. (This has now caused three silent failures in this project.)
    if not os.path.isdir(vol):
        raise FileNotFoundError(f"no subject_volumes in {fem_dir}")
    names = sorted(os.listdir(vol))
    cand = [os.path.join(vol, n) for n in names if n.endswith("_E.nii.gz")]
    if not cand:                       # some versions name the vector field differently
        cand = [os.path.join(vol, n) for n in names
                if n.endswith(".nii.gz") and "magn" not in n and "vector" in n]
    if not cand:
        raise FileNotFoundError(f"no vector E volume in {vol}; found {names}")
    img = nib.load(cand[0])
    d = np.asanyarray(img.dataobj)
    if d.ndim != 4 or d.shape[-1] != 3:
        raise ValueError(f"expected a 4-D vector field, got {d.shape} from {cand[0]}")
    d = np.nan_to_num(d.astype(np.float32))
    return (d[..., 0], d[..., 1], d[..., 2]), img.affine, img.shape[:3]


def spicehead_E(m2m, dx, labels_cache={}):
    """SPICEhead vector fields for both channels at this resolution, on its own grid."""
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, affine = mri.head_model(seg, dx)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    scalp_vox = np.argwhere(labels == SCALP)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    need = {e for pair in MONTAGE for e in pair}
    elecs = {}
    for lab in need:
        v = snap(scalp_vox, mri.world_to_voxel(affine, ej[lab]))
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                               radius_nodes=ELEC_RADIUS_MM / dx), lab)
    solver, _ = make_solver(model.G)
    fields = []
    for (a, b) in MONTAGE:
        V = solver(pair_injection(model, elecs[a], elecs[b], CURRENT))
        fields.append(tuple(np.nan_to_num(f).astype(np.float32)
                            for f in efield_components(model, V)))
    return fields, affine, labels


def resample_to(src_field, src_affine, dst_affine, dst_shape):
    """Trilinear-resample a scalar volume onto another grid."""
    inv = np.linalg.inv(src_affine)
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in dst_shape], indexing="ij")
    world = (dst_affine[:3, :3] @ np.stack([ii, jj, kk]).reshape(3, -1)).T + dst_affine[:3, 3]
    vox = (inv[:3, :3] @ world.T).T + inv[:3, 3]
    return map_coordinates(src_field, vox.T, order=1, mode="nearest").reshape(dst_shape)


def envelope(F1, F2):
    sh = F1[0].shape
    return envelope_max_direction_flat(*[c.ravel() for c in F1],
                                       *[c.ravel() for c in F2]).reshape(sh)


def main(dxs=DXS, n=8):
    rows = []
    subs = subjects(n)
    for si, (cid, m2m, age, site) in enumerate(subs, 1):
        fem1 = os.path.join(OUT, cid, "ch1")
        fem2 = os.path.join(OUT, cid, "ch2")
        try:
            E1, aff, shape = load_simnibs_E(fem1)
            E2, _, _ = load_simnibs_E(fem2)
        except Exception as ex:
            print(f"[{si}/{len(subs)}] {cid}: no FEM output ({type(ex).__name__}) - skip",
                  flush=True)
            continue
        fem_env = np.nan_to_num(envelope(E1, E2))

        # grey matter and ROIs on the FEM grid, from the native-resolution labels
        lab_img = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
        lab = np.asanyarray(lab_img.dataobj)
        gm_native = np.isin(lab, SCORE).astype(np.float32)
        gm = resample_to(gm_native, lab_img.affine, aff, shape) > 0.5
        rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
        gidx = np.argwhere(gm)
        gworld = (aff[:3, :3] @ gidx.T).T + aff[:3, 3]
        rmask = {r: roi_mask(shape, gidx, gworld, rois[r], ROI_RADIUS_MM)
                 for r in ROI_ORDER if r in rois}

        for dx in dxs:
            t = time.time()
            try:
                (S1, S2), saff, _ = spicehead_E(m2m, dx)
            except Exception as ex:
                print(f"[{si}] {cid} dx={dx}: SPICEhead failed {type(ex).__name__}: {ex}",
                      flush=True)
                continue
            sp_env_own = np.nan_to_num(envelope(S1, S2))
            sp_env = resample_to(sp_env_own, saff, aff, shape)

            x = sp_env[gm].astype(float)
            y = fem_env[gm].astype(float)
            keep = np.isfinite(x) & np.isfinite(y) & (y > 0)
            m = agreement_metrics(x[keep], y[keep])
            mx = x[keep] >= np.quantile(x[keep], 0.90)
            my = y[keep] >= np.quantile(y[keep], 0.90)
            m["dice"] = float(2 * np.sum(mx & my) / (mx.sum() + my.sum()))
            w = gworld[keep[np.arange(len(x))]] if len(gworld) == len(x) else gworld
            pm, pr = int(np.argmax(x[keep])), int(np.argmax(y[keep]))
            m["peak_err_mm"] = float(np.linalg.norm(w[pm] - w[pr])) if len(w) > max(pm, pr) else np.nan
            m.update(subject=cid, age=age, site=site, dx_mm=dx, n_vox=int(keep.sum()))
            for r, mk in rmask.items():
                sv, fv = sp_env[mk], fem_env[mk]
                if sv.size:
                    m[f"roi_{r}_sp"] = float(np.median(sv))
                    m[f"roi_{r}_fem"] = float(np.median(fv))
                    m[f"roi_{r}_relerr_pct"] = 100 * (np.median(sv) / np.median(fv) - 1) \
                        if np.median(fv) > 0 else np.nan
            rows.append(m)
            print(f"[{si}/{len(subs)}] {cid} dx={dx:g}: r={m['r']:.3f} R2={m['r2']:.3f} "
                  f"nMAE={m['nmae']:.3f} slope={m['slope']:.3f} Dice={m['dice']:.3f} "
                  f"peak={m['peak_err_mm']:.0f}mm ({time.time()-t:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("no comparisons produced - run --mesh and --sim first")
    os.makedirs(OUT, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    order = ["subject", "age", "site", "dx_mm", "n_vox", "r", "r2", "slope", "intercept",
             "nmae", "ba_bias", "ba_lo", "ba_hi", "dice", "peak_err_mm"]
    keys = order + [k for k in keys if k not in order]
    agree_dir = OUT if config.MODEL == "v21" else os.path.join(config.RESULTS, "v21_simnibs_validation")
    os.makedirs(agree_dir, exist_ok=True)
    with open(os.path.join(agree_dir, "agreement.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k) for k in keys})
    print("\nwrote", os.path.join(agree_dir, "agreement.csv"))
    for dx in dxs:
        sub = [r for r in rows if r["dx_mm"] == dx]
        if not sub:
            continue
        f = lambda k: np.mean([r[k] for r in sub if np.isfinite(r.get(k, np.nan))])
        print(f"  dx={dx:g} mm (n={len(sub)}): r={f('r'):.3f} R2={f('r2'):.3f} "
              f"slope={f('slope'):.3f} nMAE={f('nmae'):.3f} Dice={f('dice'):.3f} "
              f"peak={f('peak_err_mm'):.0f} mm")


if __name__ == "__main__":
    dxs = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else DXS
    main(dxs, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
