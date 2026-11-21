# -*- coding: utf-8 -*-
"""Mesh convergence on a REAL head (reviewer comment B).

The cylinder study (v21_convergence_validation.py) checks the discrete solution against an exact
analytic reference. No analytic solution exists for a head, so this is a SELF-CONVERGENCE study:
each resolution is compared against the finest grid (dx = 1 mm), which is the standard substitute
when a reference solution is unavailable. Convergence is demonstrated by the agreement improving
monotonically as dx decreases.

Reports the same agreement suite as the cylinder study (r, R^2, slope, intercept, nMAE,
Bland-Altman, Dice of the top-10% field, peak-location error), plus the quantity the paper
actually reports -- the median envelope in each target ROI -- as a function of dx.

Usage:  python pipeline/v21_head_convergence.py [subject] [dx list]
Resumable: each resolution's envelope volume is cached outside the Dropbox tree.
"""
import csv
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)
from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components, ti_envelope_magnitude
from montage_search import (make_multiground_solver, snap, roi_mask, ELEC_RADIUS_MM,
                            ROI_RADIUS_MM, SCALP, SCORE, ROI_ORDER)
from v21_convergence_validation import agreement_metrics

SUBJECT = "SUBJ01"
DXS = [5.0, 4.0, 3.0, 2.0, 1.0]
MONTAGE = (("T8", "P8"), ("F7", "T7"))     # Table 3 thalamic optimum, 10-20 sites only
CH_MA = 1.0
CACHE = os.path.join(os.environ.get("TEMP", "."), config.tagged("spicehead_headconv_cache"))
OUT = os.path.join(config.RESULTS, "v21_head_convergence")
CH = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")


DIRECT_MAX_NODES = 100_000     # SuperLU fill-in is brutal on a 3-D Laplacian. Measured on this
                               # head: dx=3 (148k nodes) took 292 s to factorize but 3 s by CG;
                               # dx=2 (0.5M) took 81 min; dx=1 (4M) raised MemoryError. CG matches
                               # the direct solution to 4e-11 relative, so prefer it above ~100k.


def make_solver(G):
    """Multi-ground solver, direct for small systems and preconditioned CG for large ones.

    Same contract as montage_search.make_multiground_solver -- one ground per connected component
    so isolated conductor islands don't make the system singular -- but above DIRECT_MAX_NODES it
    switches to Jacobi-preconditioned CG, because direct factorization of the 1 mm head does not
    fit in memory.
    """
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import LinearOperator, cg

    N = G.shape[0]
    # SPICEHEAD_SOLVER=gpu: same multi-ground Jacobi-CG, run on a CUDA device (spicehead.solve_gpu);
    # added 17 Oct 2026 for the NCHC GPU nodes. Small systems keep the direct solver either way.
    if os.environ.get("SPICEHEAD_SOLVER", "").lower() == "gpu" and N > DIRECT_MAX_NODES:
        from spicehead.solve_gpu import make_gpu_solver, available
        if not available():
            raise RuntimeError("SPICEHEAD_SOLVER=gpu but no CUDA device/torch is available")
        return make_gpu_solver(G, rtol=1e-10, maxiter=20000)
    Gc = csr_matrix(G)
    ncomp, comp = connected_components(csgraph=Gc, directed=False)
    if N <= DIRECT_MAX_NODES:
        return make_multiground_solver(G)[0], ncomp

    grounds = [int(np.flatnonzero(comp == c)[0]) for c in range(ncomp)]
    keep = np.ones(N, bool)
    keep[grounds] = False
    idx = np.flatnonzero(keep)
    A = Gc[idx][:, idx].tocsr()
    d = A.diagonal().copy()
    d[d == 0] = 1.0
    M = LinearOperator(A.shape, matvec=lambda x: x / d)   # Jacobi

    def solve(current):
        b = np.asarray(current, float)[idx]
        t0 = time.time()
        x, info = cg(A, b, rtol=1e-10, atol=0.0, maxiter=20000, M=M)
        res = np.linalg.norm(A @ x - b) / (np.linalg.norm(b) or 1.0)
        print(f"      CG info={info} relres={res:.2e} in {time.time()-t0:.0f}s", flush=True)
        if res > 1e-6:
            raise RuntimeError(f"CG failed to converge (relres={res:.2e})")
        V = np.zeros(N)
        V[idx] = x
        return V
    return solve, ncomp


CMEAN = os.environ.get("SPICEHEAD_CMEAN", config.CONDUCTANCE_MEAN)   # 'arithmetic' | 'harmonic'


def envelope_at(m2m, dx):
    """TI envelope volume at this resolution, plus affine, ROI masks and GM mask."""
    model, affine = mri.head_model(   # v21/v22: nearest-label network; v23: homogenized
        os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"), dx, conductance_mean=CMEAN)
    labels = model.labels
    gm = np.isin(labels, SCORE)
    bidx = np.argwhere(gm)
    bw = (affine[:3, :3] @ bidx.T).T + affine[:3, 3]
    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    rmask = {r: roi_mask(labels.shape, bidx, bw, rois[r], ROI_RADIUS_MM) for r in ROI_ORDER}

    scalp_vox = np.argwhere(labels == SCALP)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    need = {e for pair in MONTAGE for e in pair}
    elecs = {}
    for lab in need:
        v = snap(scalp_vox, mri.world_to_voxel(affine, ej[lab]))
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                               radius_nodes=ELEC_RADIUS_MM / dx), lab)
    t = time.time()
    solver, ncomp = make_solver(model.G)
    print(f"    prepared in {time.time()-t:.0f}s ({ncomp} components)", flush=True)

    fields = []
    for (a, b) in MONTAGE:
        t = time.time()
        V = solver(pair_injection(model, elecs[a], elecs[b], CH_MA * 1e-3))
        Ex, Ey, Ez = efield_components(model, V)
        fields.append((Ex.astype(np.float32), Ey.astype(np.float32), Ez.astype(np.float32)))
        print(f"    pair {a}-{b} solved in {time.time()-t:.0f}s", flush=True)
    env = np.nan_to_num(ti_envelope_magnitude(fields[0], fields[1])).astype(np.float32)
    roiT = {r: float(np.median(env[rmask[r]])) for r in ROI_ORDER if rmask[r].any()}
    return {"env": env, "affine": affine, "gm": gm, "shape": labels.shape,
            "n_nodes": int(model.n_nodes), "roiT": roiT}


def sample_reference(ref, coords_world):
    """Trilinear-sample the fine-grid envelope at arbitrary world coordinates."""
    inv = np.linalg.inv(ref["affine"])
    hom = np.concatenate([coords_world, np.ones((len(coords_world), 1))], axis=1)
    vox = (inv @ hom.T).T[:, :3]
    return map_coordinates(ref["env"], vox.T, order=1, mode="nearest")


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    m2m = os.path.join(CH, "m2m_" + SUBJECT)
    got = {}
    for dx in sorted(DXS, reverse=True):          # coarse first: fail fast, cheap first
        cp = os.path.join(CACHE, f"{SUBJECT}__dx{dx:g}" + ("" if CMEAN == "arithmetic" else "_" + CMEAN) + ".pkl")
        if os.path.exists(cp):
            with open(cp, "rb") as fh:
                got[dx] = pickle.load(fh)
            print(f"  dx={dx:g} mm (cached, {got[dx]['n_nodes']:,} nodes)", flush=True)
            continue
        print(f"  dx={dx:g} mm ...", flush=True)
        t = time.time()
        got[dx] = envelope_at(m2m, dx)
        tmp = cp + ".part"
        with open(tmp, "wb") as fh:
            pickle.dump(got[dx], fh, protocol=4)
        os.replace(tmp, cp)
        print(f"  dx={dx:g} mm done in {(time.time()-t)/60:.1f} min "
              f"({got[dx]['n_nodes']:,} nodes)", flush=True)

    fine = min(got)
    ref = got[fine]
    print(f"\nreference grid: dx={fine:g} mm, {ref['n_nodes']:,} nodes")

    rows = []
    for dx in sorted(got, reverse=True):
        g = got[dx]
        # compare on this grid's GM voxels, sampling the fine solution at the same world points
        idx = np.argwhere(g["gm"])
        world = (g["affine"][:3, :3] @ idx.T).T + g["affine"][:3, 3]
        x = g["env"][g["gm"]].astype(float)
        y = sample_reference(ref, world).astype(float)
        keep = np.isfinite(x) & np.isfinite(y) & (y > 0)
        met = agreement_metrics(x[keep], y[keep])
        mx = x[keep] >= np.quantile(x[keep], 0.90)
        my = y[keep] >= np.quantile(y[keep], 0.90)
        met["dice"] = float(2 * np.sum(mx & my) / (mx.sum() + my.sum()))
        pm = np.argmax(x[keep]); pr = np.argmax(y[keep])
        met["peak_err_mm"] = float(np.linalg.norm(world[keep][pm] - world[keep][pr]))
        met.update(dx_mm=dx, n_nodes=g["n_nodes"], n_compared=int(keep.sum()))
        for r in ROI_ORDER:
            met[f"T_{r}"] = g["roiT"].get(r)
        rows.append(met)

    keys = ["dx_mm", "n_nodes", "n_compared", "r", "r2", "slope", "intercept", "nmae",
            "ba_bias", "ba_lo", "ba_hi", "dice", "peak_err_mm"] + [f"T_{r}" for r in ROI_ORDER]
    with open(os.path.join(OUT, "head_convergence.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    json.dump({"subject": SUBJECT, "montage": [list(p) for p in MONTAGE],
               "reference_dx_mm": fine, "channel_mA": CH_MA, "rows": rows},
              open(os.path.join(OUT, "head_convergence.json"), "w"), indent=2, default=float)

    print(f"\n{'dx':>4} {'nodes':>10} {'r':>8} {'R2':>8} {'nMAE':>8} {'Dice':>6} {'peak(mm)':>9}")
    for r in rows:
        print(f"{r['dx_mm']:>4.0f} {r['n_nodes']:>10,} {r['r']:>8.4f} {r['r2']:>8.4f} "
              f"{r['nmae']:>8.4f} {r['dice']:>6.3f} {r['peak_err_mm']:>9.1f}")
    print(f"\n{'dx':>4} " + "".join(f"{r.split('_')[0][:9]:>10}" for r in ROI_ORDER))
    for r in rows:
        print(f"{r['dx_mm']:>4.0f} " + "".join(
            f"{(r.get('T_'+q) or float('nan')):>10.4f}" for q in ROI_ORDER))
    print(f"\nwrote {OUT}/head_convergence.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SUBJECT = sys.argv[1]
        OUT = os.path.join(config.RESULTS, "v21_head_convergence", SUBJECT)
    if len(sys.argv) > 2:
        DXS = [float(x) for x in sys.argv[2].split(",")]
    main()
