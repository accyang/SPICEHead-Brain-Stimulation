# -*- coding: utf-8 -*-
"""Is the montage RANKING stable against grid resolution?

The head-convergence study (v21_head_convergence.py) shows the absolute envelope field at
dx = 5 mm -- the resolution used for Tables 3-4 -- correlates only r ~ 0.64 with the 1 mm solution
and overestimates ROI intensity by up to ~36%. That damages absolute-intensity and peak-location
claims. It does not automatically damage the montage ranking, because F = T/O is a ratio compared
between montages AT THE SAME resolution, so systematic bias largely cancels.

This tests that directly: take the top-K montages per ROI from the published dx = 5 cohort ranking,
re-score them on one subject at dx = 5 and at dx = 1, and ask whether the ordering survives
(Spearman rho, whether rank 1 is preserved, and how far the dx=5 winner falls at dx=1).

Solves each unique electrode pair once and reuses the field across every montage that needs it --
the same factorize-once trick as the cohort search, which is what makes this affordable at 1 mm.

Usage:  python pipeline/v21_rank_stability.py [subject] [K]
"""
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy import stats as sstats

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)
from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components, ti_envelope_magnitude
from montage_search import (snap, roi_mask, ELEC_RADIUS_MM, ROI_RADIUS_MM, EXCLUDE_MM,
                            SCALP, SCORE, ROI_ORDER)
from v21_head_convergence import make_solver

SUBJECT = "SUBJ01"
K = 8
DXS = [5.0, 1.0]
CH_MA = 1.0
T3 = os.path.join(ROOT, "results", "montage_optimization", "cohort_gm300_dx5",
                  "cohort_montage_optima.json")
OUT = os.path.join(config.RESULTS, "v21_rank_stability")
CACHE = os.path.join(os.environ.get("TEMP", "."), config.tagged("spicehead_rank_cache"))
CH = r"L:\BrainProbe\SPICEhead\charm"


def parse(s):
    a, b = [p.strip() for p in s.split("/")]
    return (tuple(sorted(x.strip() for x in a.split("+"))),
            tuple(sorted(x.strip() for x in b.split("+"))))


def score_all(m2m, dx, montages):
    """montages: {roi: [montage_str,...]} -> {roi: {montage_str: (F, T, O)}}"""
    model, affine = mri.head_model(   # v21/v22: nearest-label network; v23: homogenized
        os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"), dx)
    labels = model.labels
    gm = np.isin(labels, SCORE)
    bidx = np.argwhere(gm)
    bw = (affine[:3, :3] @ bidx.T).T + affine[:3, 3]
    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    rmask = {r: roi_mask(labels.shape, bidx, bw, rois[r], ROI_RADIUS_MM) for r in ROI_ORDER}
    outside = {r: (gm & ~roi_mask(labels.shape, bidx, bw, rois[r], EXCLUDE_MM)) for r in ROI_ORDER}

    scalp_vox = np.argwhere(labels == SCALP)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    need_pairs = sorted({p for lst in montages.values() for s in lst for p in parse(s)})
    need_elec = sorted({e for p in need_pairs for e in p})
    elecs = {}
    for lab in need_elec:
        v = snap(scalp_vox, mri.world_to_voxel(affine, ej[lab]))
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                               radius_nodes=ELEC_RADIUS_MM / dx), lab)
    solver, ncomp = make_solver(model.G)
    print(f"  dx={dx:g}: {model.n_nodes:,} nodes, {ncomp} comps, "
          f"{len(need_pairs)} unique pairs to solve", flush=True)

    pairE = {}
    for i, (a, b) in enumerate(need_pairs, 1):
        t = time.time()
        V = solver(pair_injection(model, elecs[a], elecs[b], CH_MA * 1e-3))
        Ex, Ey, Ez = efield_components(model, V)
        pairE[(a, b)] = (Ex.astype(np.float32), Ey.astype(np.float32), Ez.astype(np.float32))
        print(f"    [{i}/{len(need_pairs)}] {a}-{b} {time.time()-t:.0f}s", flush=True)

    out = {}
    for r, lst in montages.items():
        out[r] = {}
        for s in lst:
            pa, pb = parse(s)
            env = np.nan_to_num(ti_envelope_magnitude(pairE[pa], pairE[pb]))
            rv = env[rmask[r]]
            if rv.size == 0:
                continue
            T = float(np.median(rv))
            O = float(np.percentile(env[outside[r]], 99))
            out[r][s] = (T / O if O > 0 else 0.0, T, O)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    top = json.load(open(T3, encoding="utf-8"))["top"]
    montages = {r: [e["montage"] for e in top[r][:K]] for r in ROI_ORDER}
    m2m = os.path.join(CH, "m2m_" + SUBJECT)

    res = {}
    for dx in DXS:
        cp = os.path.join(CACHE, f"{SUBJECT}__rank_dx{dx:g}_K{K}.pkl")
        if os.path.exists(cp):
            res[dx] = pickle.load(open(cp, "rb"))
            print(f"  dx={dx:g} (cached)", flush=True)
            continue
        t = time.time()
        res[dx] = score_all(m2m, dx, montages)
        tmp = cp + ".part"
        pickle.dump(res[dx], open(tmp, "wb"), protocol=4)
        os.replace(tmp, cp)
        print(f"  dx={dx:g} done in {(time.time()-t)/60:.1f} min", flush=True)

    coarse, fine = max(DXS), min(DXS)
    rows = []
    print(f"\n{'ROI':<15}{'rho':>7}{'rank1 kept':>12}{'dx5 winner rank at dx1':>25}")
    for r in ROI_ORDER:
        ms = [m for m in montages[r] if m in res[coarse].get(r, {}) and m in res[fine].get(r, {})]
        if len(ms) < 3:
            continue
        fc = np.array([res[coarse][r][m][0] for m in ms])
        ff = np.array([res[fine][r][m][0] for m in ms])
        rho, p = sstats.spearmanr(fc, ff)
        wc = ms[int(np.argmax(fc))]
        wf = ms[int(np.argmax(ff))]
        order_f = [ms[i] for i in np.argsort(-ff)]
        pos = order_f.index(wc) + 1
        rows.append({"roi": r, "n_montages": len(ms), "spearman_rho": round(float(rho), 4),
                     "rank1_preserved": bool(wc == wf), "coarse_winner": wc, "fine_winner": wf,
                     "coarse_winner_rank_at_fine": pos,
                     "F_coarse_winner": round(float(fc.max()), 4),
                     "F_fine_winner": round(float(ff.max()), 4),
                     "F_fine_of_coarse_winner": round(float(ff[ms.index(wc)]), 4)})
        print(f"{r:<15}{rho:>7.3f}{('yes' if wc == wf else 'NO'):>12}{pos:>25}")

    json.dump({"subject": SUBJECT, "K": K, "dx_coarse": coarse, "dx_fine": fine,
               "rows": rows,
               "raw": {str(dx): {r: {m: list(v) for m, v in res[dx][r].items()} for r in res[dx]}
                       for dx in res}},
              open(os.path.join(OUT, "rank_stability.json"), "w"), indent=2)
    keep = sum(1 for x in rows if x["rank1_preserved"])
    med = float(np.median([x["spearman_rho"] for x in rows])) if rows else float("nan")
    print(f"\nrank-1 preserved in {keep}/{len(rows)} ROIs; median Spearman rho = {med:.3f}")
    print("wrote", os.path.join(OUT, "rank_stability.json"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SUBJECT = sys.argv[1]
    if len(sys.argv) > 2:
        K = int(sys.argv[2])
    main()
