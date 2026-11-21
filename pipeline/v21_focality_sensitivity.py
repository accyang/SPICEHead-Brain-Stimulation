# -*- coding: utf-8 -*-
"""Sensitivity of the focality ratio F = T/O to the two arbitrary choices in its definition.

Review item I: "a focality value below one represents an intrinsic limit of temporal-interference
stimulation" is too strong -- it is a result under a particular 20 mm exclusion zone and
99th-percentile off-target denominator. This sweeps both and also reports the ABSOLUTE target
intensity T, because T/O is invariant to total injected current and so says nothing about whether
the target receives a biologically meaningful field.

For each subject we score only that ROI's Table 3 montage (not the full 11,600-montage sweep), so
this costs ~2 pair solves per ROI on top of one factorization.

Usage:  python pipeline/v21_focality_sensitivity.py [dx] [n_subjects]
Resumable: per-subject results are cached and re-runs skip what is done.
"""
import csv
import json
import os
import pickle
import sys
import time
from itertools import combinations

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)
from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components, ti_envelope_magnitude
# TI envelope definition. 'maxdir' = rotation-invariant maximum-over-direction (Grossman et al.);
# 'peraxis' = superseded coordinate-dependent form, retained only to reproduce pre-2026-08 numbers.
ENVELOPE = os.environ.get("SPICEHEAD_ENVELOPE", "maxdir")
from spicehead.envelope_fast import envelope_max_direction_flat

from montage_search import (make_multiground_solver, snap, roi_mask, ELEC_RADIUS_MM,
                            ROI_RADIUS_MM, SCALP, SCORE, ROI_ORDER)

EXCLUSIONS = [10.0, 15.0, 20.0, 25.0, 30.0]      # mm; published value is 20
PERCENTILES = [95.0, 97.0, 99.0, 99.5, 100.0]    # published value is 99 (100 = max)
CH_MA = 1.0                                       # per-channel current, mA
CACHE = os.path.join(os.environ.get("TEMP", "."),
                     config.tagged("spicehead_sens_cache_"
                                   + os.environ.get("SPICEHEAD_ENVELOPE", "maxdir")))
OUT = os.path.join(config.RESULTS, "v21_sensitivity_"
                   + os.environ.get("SPICEHEAD_ENVELOPE", "maxdir"))
T3 = os.environ.get("SPICEHEAD_T3") or os.path.join(
    config.RESULTS, "montage_optimization",
    "cohort_gm300_dx5" if config.MODEL == "v21" else "cohort_gm300_dx5_maxdir",
    "cohort_montage_optima.json")
CSVP = os.environ.get("SPICEHEAD_COHORT_CSV", r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv")
CH = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")


def parse_montage(s):
    """'Pz+O2 / F3+F4' -> (('Pz','O2'), ('F3','F4'))"""
    a, b = [p.strip() for p in s.split("/")]
    return tuple(x.strip() for x in a.split("+")), tuple(x.strip() for x in b.split("+"))


def score_subject(m2m, dx, montages):
    model, affine = mri.head_model(   # v21/v22: nearest-label network; v23: homogenized
        os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"), dx)
    labels = model.labels
    brain = np.isin(labels, SCORE)                    # grey matter, matching Tables 3-4
    bidx = np.argwhere(brain)
    bw = (affine[:3, :3] @ bidx.T).T + affine[:3, 3]
    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    rmask = {r: roi_mask(labels.shape, bidx, bw, rois[r], ROI_RADIUS_MM) for r in ROI_ORDER}
    # one off-target mask per exclusion distance
    outside = {(r, e): (brain & ~roi_mask(labels.shape, bidx, bw, rois[r], e))
               for r in ROI_ORDER for e in EXCLUSIONS}

    scalp_vox = np.argwhere(labels == SCALP)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    elecs = {}
    for lab, world in ej.items():
        v = snap(scalp_vox, mri.world_to_voxel(affine, world))
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                               radius_nodes=ELEC_RADIUS_MM / dx), lab)
    solver, _ = make_multiground_solver(model.G)

    need = {p for m in montages.values() for p in m}
    pairE = {}
    for (a, b) in need:
        if a not in elecs or b not in elecs:
            continue
        V = solver(pair_injection(model, elecs[a], elecs[b], CH_MA * 1e-3))
        Ex, Ey, Ez = efield_components(model, V)
        pairE[(a, b)] = (Ex.astype(np.float32), Ey.astype(np.float32), Ez.astype(np.float32))

    out = {}
    for r, (pa, pb) in montages.items():
        if pa not in pairE or pb not in pairE:
            continue
        env = (envelope_max_direction_flat(*[c.ravel() for c in pairE[pa]],
                                          *[c.ravel() for c in pairE[pb]]
                                          ).reshape(pairE[pa][0].shape)
               if ENVELOPE == "maxdir"
               else np.nan_to_num(ti_envelope_magnitude(pairE[pa], pairE[pb])))
        rv = env[rmask[r]]
        if rv.size == 0:
            continue
        T = float(np.median(rv))
        rec = {"T": T}
        for e in EXCLUSIONS:
            ov = env[outside[(r, e)]]
            for q in PERCENTILES:
                O = float(ov.max() if q >= 100 else np.percentile(ov, q))
                rec[f"F_e{e:g}_p{q:g}"] = (T / O) if O > 0 else 0.0
                rec[f"O_e{e:g}_p{q:g}"] = O     # reported beside T and F (review, 1 October)
        out[r] = rec
    return out


def main(dx=5.0, nmax=None):
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    top = json.load(open(T3, encoding="utf-8"))["top"]
    montages = {r: parse_montage(top[r][0]["montage"]) for r in ROI_ORDER}
    print("Table 3 montages under test:")
    for r in ROI_ORDER:
        print(f"  {r:<15} {top[r][0]['montage']}")

    ids = [r["CaseID"] for r in csv.DictReader(open(CSVP, encoding="utf-8-sig"))]
    if nmax:
        ids = ids[:nmax]
    recs = {}
    t0 = time.time()
    for i, cid in enumerate(ids, 1):
        cp = os.path.join(CACHE, f"{cid}__dx{dx:g}.pkl")
        if os.path.exists(cp):
            with open(cp, "rb") as fh:
                recs[cid] = pickle.load(fh)
            print(f"  [{i}/{len(ids)}] {cid} (cached)", flush=True)
            continue
        m2m = os.path.join(CH, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)
        if not os.path.exists(os.path.join(m2m, "electrodes_1020.json")):
            print(f"  [{i}/{len(ids)}] {cid} - no electrodes, skip", flush=True)
            continue
        t = time.time()
        try:
            r = score_subject(m2m, dx, montages)
        except Exception as ex:
            print(f"  [{i}/{len(ids)}] {cid} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        tmp = cp + ".part"
        with open(tmp, "wb") as fh:
            pickle.dump(r, fh, protocol=4)
        os.replace(tmp, cp)
        recs[cid] = r
        print(f"  [{i}/{len(ids)}] {cid} ({time.time()-t:.0f}s)", flush=True)
    print(f"\n{len(recs)} subjects in {(time.time()-t0)/60:.1f} min")

    # ---- aggregate
    rows = []
    for r in ROI_ORDER:
        vals = [v[r] for v in recs.values() if r in v]
        if not vals:
            continue
        Tmed = float(np.median([v["T"] for v in vals]))
        for e in EXCLUSIONS:
            for q in PERCENTILES:
                k = f"F_e{e:g}_p{q:g}"
                f = np.array([v[k] for v in vals])
                ko = f"O_e{e:g}_p{q:g}"
                # caches written before O was recorded have no O; report it only when every
                # subject has it, never from a subset
                Omed = (round(float(np.median([v[ko] for v in vals])), 5)
                        if all(ko in v for v in vals) else "")
                rows.append({"roi": r, "exclusion_mm": e, "percentile": q,
                             "n": len(f), "F_median": round(float(np.median(f)), 4),
                             "F_iqr_lo": round(float(np.percentile(f, 25)), 4),
                             "F_iqr_hi": round(float(np.percentile(f, 75)), 4),
                             "frac_above_1": round(float((f > 1).mean()), 4),
                             "T_median_Vm": round(Tmed, 5), "O_median_Vm": Omed})
    with open(os.path.join(OUT, "sensitivity.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    json.dump({"dx": dx, "n_subjects": len(recs), "channel_mA": CH_MA,
               "exclusions_mm": EXCLUSIONS, "percentiles": PERCENTILES,
               "published_setting": {"exclusion_mm": 20.0, "percentile": 99.0},
               "montages": {r: top[r][0]["montage"] for r in ROI_ORDER},
               "rows": rows}, open(os.path.join(OUT, "sensitivity.json"), "w"), indent=2)
    print("wrote", os.path.join(OUT, "sensitivity.csv"))

    print(f"\n{'ROI':<15}{'T (V/m)':>9}   F at each (exclusion mm, p99) and (20 mm, percentile)")
    for r in ROI_ORDER:
        sub = [x for x in rows if x["roi"] == r]
        if not sub:
            continue
        byE = {x["exclusion_mm"]: x["F_median"] for x in sub if x["percentile"] == 99}
        byP = {x["percentile"]: x["F_median"] for x in sub if x["exclusion_mm"] == 20}
        print(f"{r:<15}{sub[0]['T_median_Vm']:>9.4f}   "
              f"excl {'/'.join(f'{byE[e]:.2f}' for e in EXCLUSIONS)}   "
              f"pct {'/'.join(f'{byP[q]:.2f}' for q in PERCENTILES)}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 5.0,
         int(sys.argv[2]) if len(sys.argv) > 2 else None)
