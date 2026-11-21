"""
Comprehensive cohort montage analysis (10/20). For every subject, score every 2-pair montage per
ROI keeping BOTH intensity T (median envelope in ROI) and off-target O (99th pctile >20mm away).
From the per-subject (age, T, O) records produce:
  (1) overall optimal montage per ROI (median T/O),
  (2) AGE-STRATIFIED optima (young <50 vs old >=50) + age-slope of the overall-best montage,
  (3) intensity-vs-focality PARETO scatter per ROI (PNG).

  python montage_full.py <cohort.csv> <charm_outroot> <out_dir> [dx]
"""
import os
import sys
import csv
import json
import time
from itertools import combinations

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here)); sys.path.insert(0, _here)
from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components, ti_envelope_magnitude
# TI envelope definition. 'maxdir' = rotation-invariant maximum-over-direction (Grossman et al.);
# 'peraxis' = superseded coordinate-dependent form, retained only to reproduce pre-2026-08 numbers.
ENVELOPE = os.environ.get("SPICEHEAD_ENVELOPE", "maxdir")
from spicehead.envelope_fast import envelope_max_direction_flat

from montage_search import (make_multiground_solver, snap, roi_mask, ELEC_RADIUS_MM,
                            MIN_DIST_MM, ROI_RADIUS_MM, EXCLUDE_MM, SCALP, SCORE, ROI_ORDER, EASY)


def score_TO(m2m, dx):
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, affine = mri.head_model(seg, dx)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    # scoring mask = grey matter only, in BOTH T and O -- must match montage_search.SCORE or this
    # analysis lands in a different metric than Tables 3/4 and the numbers are not comparable.
    brain = np.isin(labels, SCORE); brain_idx = np.argwhere(brain)
    bw = (affine[:3, :3] @ brain_idx.T).T + affine[:3, 3]
    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    rmask = {r: roi_mask(labels.shape, brain_idx, bw, rois[r], ROI_RADIUS_MM) for r in ROI_ORDER}
    outside = {r: (brain & ~roi_mask(labels.shape, brain_idx, bw, rois[r], EXCLUDE_MM)) for r in ROI_ORDER}
    scalp_vox = np.argwhere(labels == SCALP); epos = {}; elecs = {}
    for lab, world in json.load(open(os.path.join(m2m, "electrodes_1020.json"))).items():
        v = snap(scalp_vox, mri.world_to_voxel(affine, world)); epos[lab] = v.astype(float)
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v), radius_nodes=ELEC_RADIUS_MM / dx), lab)
    labs = list(elecs)
    pairs = [(a, b) for a, b in combinations(labs, 2) if np.linalg.norm(epos[a] - epos[b]) * dx >= MIN_DIST_MM]
    solver, _ = make_multiground_solver(model.G)
    pE = {}
    for (a, b) in pairs:
        Ex, Ey, Ez = efield_components(model, solver(pair_injection(model, elecs[a], elecs[b], 1e-3)))
        pE[(a, b)] = (Ex.astype(np.float32), Ey.astype(np.float32), Ez.astype(np.float32))
    # flat scored-tissue arrays: the envelope is evaluated once per montage, so restricting it to
    # the ~18% of voxels actually scored is what makes the rotation-invariant form affordable
    _flat = {k: tuple(c[brain] for c in v) for k, v in pE.items()}
    _rsel = {r: rmask[r][brain] for r in rmask}
    _osel = {r: outside[r][brain] for r in outside}
    out = {r: {} for r in ROI_ORDER}
    for pa, pb in combinations(pairs, 2):
        if len(set(pa) | set(pb)) < 4:
            continue
        if ENVELOPE == "maxdir":
            a, b = _flat[pa], _flat[pb]
            env = envelope_max_direction_flat(a[0], a[1], a[2], b[0], b[1], b[2])
        else:
            env = np.nan_to_num(ti_envelope_magnitude(pE[pa], pE[pb]))[brain]
        key = "%s+%s / %s+%s" % (pa[0], pa[1], pb[0], pb[1])
        for r in ROI_ORDER:
            rv = env[_rsel[r]]
            if rv.size:
                out[r][key] = (float(np.median(rv)), float(np.percentile(env[_osel[r]], 99)))
    return out


def convenience(key):
    es = set(key.replace(" / ", "+").split("+"))
    return sum(1 for e in es if e in EASY)


def rank(accum_TO, keys_ok):
    ranked = []
    for k in keys_ok:
        vals = accum_TO[k]
        foc = [T / O for (T, O) in vals if O > 0]
        ranked.append((float(np.median(foc)), float(np.median([T for T, O in vals])), len(foc), convenience(k), k))
    ranked.sort(key=lambda x: -x[0])
    return ranked


def main(cohort_csv, outroot, out_dir, dx=5.0):
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(cohort_csv, encoding="utf-8-sig")))
    subs = []
    for r in rows:
        cid = r["CaseID"]
        m2m = os.path.join(outroot, "m2m_SUBJ01asc") if cid.startswith("SUBJ01") else os.path.join(outroot, "m2m_" + cid)
        if os.path.exists(os.path.join(m2m, "electrodes_1020.json")) and os.path.exists(os.path.join(m2m, "roi_coords.json")):
            subs.append((cid, float(r["Age_Scan"]), m2m))
    print(f"scoring {len(subs)} subjects at dx={dx}")
    # accum[roi][key] = list of (age, T, O)
    accum = {r: {} for r in ROI_ORDER}
    for i, (cid, age, m2m) in enumerate(subs):
        t = time.time(); sc = score_TO(m2m, dx)
        for r in ROI_ORDER:
            for k, (T, O) in sc[r].items():
                accum[r].setdefault(k, []).append((age, T, O))
        print(f"  [{i+1}/{len(subs)}] {cid} ({time.time()-t:.0f}s)")
    nsub = len(subs)
    ages = np.array([a for _, a, _ in subs])

    result = {"n": nsub, "dx": dx, "overall": {}, "young": {}, "old": {}, "age_slope_of_best": {}}
    for r in ROI_ORDER:
        keys_ok = [k for k, v in accum[r].items() if len(v) >= 0.8 * nsub]
        # overall
        TO_all = {k: [(T, O) for (_, T, O) in accum[r][k]] for k in keys_ok}
        ov = rank(TO_all, keys_ok)
        result["overall"][r] = [{"score": round(s, 3), "T": round(T, 4), "n": n, "easy": c, "montage": k} for s, T, n, c, k in ov[:10]]
        # young/old
        yTO = {k: [(T, O) for (a, T, O) in accum[r][k] if a < 50] for k in keys_ok}
        oTO = {k: [(T, O) for (a, T, O) in accum[r][k] if a >= 50] for k in keys_ok}
        result["young"][r] = [{"score": round(s, 3), "easy": c, "montage": k} for s, T, n, c, k in rank(yTO, [k for k in keys_ok if len(yTO[k]) >= 5])[:5]]
        result["old"][r] = [{"score": round(s, 3), "easy": c, "montage": k} for s, T, n, c, k in rank(oTO, [k for k in keys_ok if len(oTO[k]) >= 5])[:5]]
        # age slope of the overall-best montage's focality
        best_k = ov[0][4]
        rec = accum[r][best_k]
        a = np.array([x[0] for x in rec]); foc = np.array([T / O if O > 0 else 0 for (_, T, O) in rec])
        slope = float(np.polyfit(a, foc, 1)[0])
        result["age_slope_of_best"][r] = {"montage": best_k, "slope_per_yr": round(slope, 5),
                                          "young_median": round(float(np.median(foc[a < 50])), 3),
                                          "old_median": round(float(np.median(foc[a >= 50])), 3)}
    json.dump(result, open(os.path.join(out_dir, "montage_full.json"), "w"), indent=2)
    print("wrote montage_full.json")

    # Pareto: intensity (median T) vs focality (median T/O), per ROI
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # A hardcoded 2x2 grid silently truncates ROI_ORDER to its first four entries, so the
        # hippocampus panels were computed but never drawn. Size the grid from the ROI list,
        # 2 columns wide on a portrait canvas that scales down into a manuscript column.
        ncol = 2
        nrow = int(np.ceil(len(ROI_ORDER) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.9 * nrow), squeeze=False)
        flat = axes.ravel()
        for ax in flat[len(ROI_ORDER):]:
            ax.set_visible(False)
        for ax, r in zip(flat, ROI_ORDER):
            keys_ok = [k for k, v in accum[r].items() if len(v) >= 0.8 * nsub]
            T = np.array([np.median([t for (_, t, o) in accum[r][k]]) for k in keys_ok])
            F = np.array([np.median([t / o if o > 0 else 0 for (_, t, o) in accum[r][k]]) for k in keys_ok])
            ax.scatter(T, F, s=8, alpha=0.35, edgecolor="none")
            # Pareto frontier (max F for >= each T)
            order = np.argsort(-T)
            best = -np.inf; fx, fy = [], []
            for idx in order:
                if F[idx] > best:
                    best = F[idx]; fx.append(T[idx]); fy.append(F[idx])
            ax.plot(fx, fy, "r.-", ms=6, lw=1, label="Pareto frontier")
            ax.set_title(r, fontsize=10); ax.set_xlabel("intensity T (median V/m)", fontsize=9)
            ax.set_ylabel("focality T/O", fontsize=9); ax.legend(fontsize=6.5)
        fig.suptitle(f"Intensity vs focality per ROI (n={nsub}, each dot = a 10/20 montage)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pareto_intensity_focality.png"), dpi=200)
        print("wrote pareto_intensity_focality.png")
    except Exception as e:
        print("pareto plot skipped:", e)

    for r in ROI_ORDER:
        b = result["age_slope_of_best"][r]
        print(f"  {r}: best={b['montage']}  young={b['young_median']} old={b['old_median']} slope={b['slope_per_yr']}/yr")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    dx = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    main(sys.argv[1], sys.argv[2], sys.argv[3], dx)
