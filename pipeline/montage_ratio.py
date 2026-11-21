"""
Current-ratio steering: for each ROI's overall-best 10/20 montage, sweep the current ratio
between the two pairs per subject; report the cohort-median optimal ratio and the focality gain
over equal currents. (Envelope shape steers toward the lower-current pair; focality T/O is
scale-invariant to a global scaling, so only the ratio matters.)

  python montage_ratio.py <cohort.csv> <outroot> <cohort_montage_optima.json> <out.json> [dx]
"""
import os
import sys
import csv
import json
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
                            ROI_RADIUS_MM, EXCLUDE_MM, SCALP, SCORE, ROI_ORDER)

RATIOS = [0.5, 0.63, 0.79, 1.0, 1.26, 1.59, 2.0]  # current(pair1) / current(pair2)


def parse_montage(s):
    p1, p2 = s.split(" / ")
    a, b = p1.split("+"); c, d = p2.split("+")
    return (a, b), (c, d)


def main(cohort_csv, outroot, optima_json, outjson, dx=5.0):
    opt = json.load(open(optima_json))
    best = {r: parse_montage(opt["top"][r][0]["montage"]) for r in ROI_ORDER}
    needed = sorted({e for r in ROI_ORDER for pair in best[r] for e in pair})

    rows = list(csv.DictReader(open(cohort_csv, encoding="utf-8-sig")))
    subs = []
    for r in rows:
        cid = r["CaseID"]
        m2m = os.path.join(outroot, "m2m_SUBJ01asc") if cid.startswith("SUBJ01") else os.path.join(outroot, "m2m_" + cid)
        if os.path.exists(os.path.join(m2m, "electrodes_1020.json")) and os.path.exists(os.path.join(m2m, "roi_coords.json")):
            subs.append((cid, m2m))
    print(f"steering {len(subs)} subjects; montages: " + "; ".join(f"{r}:{opt['top'][r][0]['montage']}" for r in ROI_ORDER))

    agg = {r: {"best_ratio": [], "foc_eq": [], "foc_best": []} for r in ROI_ORDER}
    for i, (cid, m2m) in enumerate(subs):
        seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
        model, affine = mri.head_model(seg, dx)   # v21/v22: nearest-label network; v23: homogenized
        labels = model.labels
        # grey matter only in BOTH T and O -- keep in sync with montage_search.SCORE
        brain = np.isin(labels, SCORE); bidx = np.argwhere(brain)
        bw = (affine[:3, :3] @ bidx.T).T + affine[:3, 3]
        rj = json.load(open(os.path.join(m2m, "roi_coords.json")))
        rmask = {r: roi_mask(labels.shape, bidx, bw, rj[r], ROI_RADIUS_MM) for r in ROI_ORDER}
        outside = {r: (brain & ~roi_mask(labels.shape, bidx, bw, rj[r], EXCLUDE_MM)) for r in ROI_ORDER}
        scalp_vox = np.argwhere(labels == SCALP)
        ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
        elecs = {}
        for lab in needed:
            v = snap(scalp_vox, mri.world_to_voxel(affine, ej[lab]))
            elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v), radius_nodes=ELEC_RADIUS_MM / dx), lab)
        solver, _ = make_multiground_solver(model.G)
        fieldcache = {}
        for r in ROI_ORDER:
            (a, b), (c, d) = best[r]
            for (x, y) in ((a, b), (c, d)):
                if (x, y) not in fieldcache:
                    Ex, Ey, Ez = efield_components(model, solver(pair_injection(model, elecs[x], elecs[y], 1e-3)))
                    fieldcache[(x, y)] = (Ex, Ey, Ez)
            EA = fieldcache[(a, b)]; EB = fieldcache[(c, d)]
            def foc(rr):
                _A = (rr * EA[0], rr * EA[1], rr * EA[2])
                env = (envelope_max_direction_flat(_A[0].ravel(), _A[1].ravel(), _A[2].ravel(),
                                                   EB[0].ravel(), EB[1].ravel(), EB[2].ravel()
                                                   ).reshape(_A[0].shape)
                       if ENVELOPE == "maxdir"
                       else np.nan_to_num(ti_envelope_magnitude(_A, EB)))
                O = float(np.percentile(env[outside[r]], 99))
                return (float(np.median(env[rmask[r]])) / O) if O > 0 else 0.0
            focs = {rr: foc(rr) for rr in RATIOS}
            br = max(focs, key=focs.get)
            agg[r]["best_ratio"].append(br); agg[r]["foc_eq"].append(focs[1.0]); agg[r]["foc_best"].append(focs[br])
        print(f"  [{i+1}/{len(subs)}] {cid}")

    out = {"n": len(subs), "ratios": RATIOS, "per_roi": {}}
    for r in ROI_ORDER:
        eq = np.array(agg[r]["foc_eq"]); bf = np.array(agg[r]["foc_best"])
        out["per_roi"][r] = {
            "montage": opt["top"][r][0]["montage"],
            "median_best_ratio": float(np.median(agg[r]["best_ratio"])),
            "focality_equal": round(float(np.median(eq)), 3),
            "focality_steered": round(float(np.median(bf)), 3),
            "median_gain_pct": round(float(np.median((bf - eq) / eq * 100)), 1),
        }
    json.dump(out, open(outjson, "w"), indent=2)
    for r in ROI_ORDER:
        o = out["per_roi"][r]
        print(f"  {r}: {o['montage']}  eq={o['focality_equal']} -> steered={o['focality_steered']} "
              f"(+{o['median_gain_pct']}%, best ratio~{o['median_best_ratio']})")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    dx = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], dx)
