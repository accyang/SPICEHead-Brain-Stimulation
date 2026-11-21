# -*- coding: utf-8 -*-
"""Worked example: the paper's exhaustive two-pair 10-20 montage search on the example head.

Builds the resistor network from examples/mni152_head/labels_1mm.nii.gz at the requested spacing,
solves the 18 single-electrode basis fields against Cz, reconstructs all 11,628 two-pair montages
by superposition, and scores each for the six targets with the paper's objective:

  T  median TI envelope (maximum over directions) in the target's grey matter (10 mm sphere)
  O  99th percentile of the grey-matter envelope farther than 20 mm from the target centre
  F  T / O     (1 mA per channel; F does not depend on total current, so T and O are printed)

  python examples/mni152_full_search.py --dx 3            a few minutes on an idle workstation
  python examples/mni152_full_search.py --dx 2 --out mni152_search_2mm.csv     (tens of minutes)

The network, electrode model and interface rule follow spicehead.config (default v23: network homogenized
from the fine segmentation, skin-surface 12 mm contacts, harmonic interface conductances). "Optimal" below means the highest F among these 11,628
candidates at this spacing; see the paper's Methods 2.6 for the scope of that statement.
"""
import argparse
import csv
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from spicehead import mri                                              # noqa: E402
from spicehead.electrodes import Electrode, contact_nodes, pair_injection   # noqa: E402
from spicehead.field import efield_components                          # noqa: E402
from spicehead.envelope_fast import envelope_max_direction_flat       # noqa: E402
from montage_search import snap, roi_mask, make_multiground_solver    # noqa: E402

HEAD = os.path.join(HERE, "mni152_head")
REF = "Cz"
CURRENT = 1e-3
ELEC_RADIUS_MM, MIN_DIST_MM, ROI_RADIUS_MM, EXCLUDE_MM = 12.0, 40.0, 10.0, 20.0
GM, SCALP = 2, 5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dx", type=float, default=3.0, help="grid spacing in mm (default 3)")
    ap.add_argument("--out", default=None, help="CSV of every montage's F, T, O per target")
    ap.add_argument("--top", type=int, default=3, help="montages to print per target")
    a = ap.parse_args()

    t0 = time.time()
    model, aff = mri.head_model(os.path.join(HEAD, "labels_1mm.nii.gz"), a.dx)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    gm = labels == GM
    gm_idx = np.argwhere(gm)
    gm_world = (aff[:3, :3] @ gm_idx.T).T + aff[:3, 3]
    rois = json.load(open(os.path.join(HEAD, "roi_coords_mni.json")))
    roi_sel = {r: roi_mask(labels.shape, gm_idx, gm_world, c, ROI_RADIUS_MM)[gm] for r, c in rois.items()}
    out_sel = {r: ~roi_mask(labels.shape, gm_idx, gm_world, c, EXCLUDE_MM)[gm] for r, c in rois.items()}

    ej = json.load(open(os.path.join(HEAD, "electrodes_1020_mni.json")))
    scalp_vox = np.argwhere(labels == SCALP)
    elecs = {}
    for lab, xyz in ej.items():
        v = snap(scalp_vox, mri.world_to_voxel(aff, xyz))
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                             radius_nodes=ELEC_RADIUS_MM / a.dx), lab)
    labs = sorted(ej)
    pairs = [(p, q) for p, q in combinations(labs, 2)
             if np.linalg.norm(np.subtract(ej[p], ej[q])) >= MIN_DIST_MM]
    montages = [(pa, pb) for pa, pb in combinations(pairs, 2) if len(set(pa) | set(pb)) == 4]
    print(f"network: {model.n_nodes:,} nodes at {a.dx:g} mm; {int(gm.sum()):,} GM voxels; "
          f"{len(pairs)} pairs, {len(montages)} montages  ({time.time()-t0:.0f} s)", flush=True)

    t1 = time.time()
    solver, ncomp = make_multiground_solver(model.G)

    def field(inj):
        return np.stack([np.nan_to_num(c[gm]) for c in efield_components(model, solver(inj))])

    basis64 = {lab: field(pair_injection(model, elecs[lab], elecs[REF], CURRENT))
               for lab in labs if lab != REF}
    print(f"18 basis solves in {time.time()-t1:.0f} s ({ncomp} conductor components)", flush=True)

    # numerical check of the superposition on one directly solved pair, in double precision
    p, q = next(pr for pr in pairs if REF not in pr)
    direct = field(pair_injection(model, elecs[p], elecs[q], CURRENT))
    rel = float(np.abs(basis64[p] - basis64[q] - direct).max() / np.abs(direct).max())
    print(f"superposition check on {p}-{q}: largest relative difference {rel:.1e}", flush=True)

    basis = {k: v.astype(np.float32) for k, v in basis64.items()}      # scoring in single precision
    del basis64

    def pair_field(p, q):
        return (basis[p] if p != REF else 0.0) - (basis[q] if q != REF else 0.0)

    t2 = time.time()
    flat = {pr: pair_field(*pr) for pr in pairs}
    rows, best = [], {r: [] for r in rois}
    for i, (pa, pb) in enumerate(montages, 1):
        A, B = flat[pa], flat[pb]
        env = envelope_max_direction_flat(A[0], A[1], A[2], B[0], B[1], B[2])
        name = " / ".join(sorted("+".join(sorted(x)) for x in (pa, pb)))
        row = {"montage": name}
        for r in rois:
            T = float(np.median(env[roi_sel[r]]))
            O = float(np.percentile(env[out_sel[r]], 99))
            F = T / O if O > 0 else 0.0
            row[f"{r}_F"], row[f"{r}_T"], row[f"{r}_O"] = round(F, 4), round(T, 5), round(O, 5)
            best[r].append((F, T, O, name))
        rows.append(row)
        if i % 3000 == 0:
            print(f"  scored {i}/{len(montages)} ({time.time()-t2:.0f} s)", flush=True)
    print(f"scored {len(montages)} montages x {len(rois)} targets in {time.time()-t2:.0f} s\n")

    print(f"{'target':16s} {'rank':>4s}  {'montage':24s} {'F':>7s} {'T (V/m)':>9s} {'O (V/m)':>9s}")
    for r in rois:
        for k, (F, T, O, name) in enumerate(sorted(best[r], reverse=True)[:a.top], 1):
            print(f"{r if k == 1 else '':16s} {k:4d}  {name:24s} {F:7.3f} {T:9.4f} {O:9.4f}")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out}")
    print(f"total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
