# -*- coding: utf-8 -*-
"""External validation of the Table 3 montages on subjects genuinely disjoint from the 300.

Review item G asks whether the "independent 97-subject subset" is independent of, held out from,
or drawn from the 300-subject cohort. It is none of those cleanly: of the 97 (all from the v1
100-case cohort, scored at dx=5 mm on the grey-matter metric), 64 are also in the 300 and 33 are
not. The 33 come from two sites -- NARI and BSNIP -- that appear nowhere in the 300.

Those 33 are therefore a true out-of-sample, out-of-site validation set. This script re-aggregates
the ALREADY CACHED per-subject scores over that disjoint subset, so it needs no recomputation and
uses exactly the ranking logic of run_montage_cohort.main().

Usage:  python pipeline/v21_external_validation.py
"""
import csv
import json
import os
import pickle
import re
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)
from montage_search import ROI_ORDER
from run_montage_cohort import convenience, montage_str

CACHE = os.path.join(ROOT, "results", "montage_optimization", "cohort_dx5_gm", "_subject_scores")
T3 = os.path.join(ROOT, "results", "montage_optimization", "cohort_gm300_dx5",
                  "cohort_montage_optima.json")
OUT = os.path.join(ROOT, "results", "montage_optimization", "external_val_disjoint")
V1 = r"L:\BrainProbe\SPICEhead\SPICEhead_cohort.csv"
V2 = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"
MIN_FRAC = 0.8          # same presence threshold as the cohort run
DX, TISSUE = 5.0, "gm"


def rank(accum, nsub):
    """Identical ranking to run_montage_cohort.main()."""
    top = {}
    for r in ROI_ORDER:
        ranked = []
        for key, vals in accum[r].items():
            if len(vals) >= MIN_FRAC * nsub:
                ranked.append((float(np.median(vals)), float(np.percentile(vals, 25)),
                               float(np.percentile(vals, 75)), len(vals),
                               convenience(key), montage_str(key)))
        ranked.sort(key=lambda x: -x[0])
        top[r] = [{"rank": i + 1, "median_score": round(m, 3), "iqr": [round(q1, 3), round(q3, 3)],
                   "n": n, "easy_electrodes": c, "montage": s}
                  for i, (m, q1, q3, n, c, s) in enumerate(ranked[:15])]
    return top


def aggregate(ids):
    accum = {r: {} for r in ROI_ORDER}
    used = 0
    for cid in sorted(ids):
        p = os.path.join(CACHE, f"{cid}__dx{DX:g}_{TISSUE}.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            sc = pickle.load(fh)
        for r in sc:
            for k, v in sc[r].items():
                accum[r].setdefault(k, []).append(v)
        used += 1
    return accum, used


def norm(m):
    return " / ".join(sorted(p.strip() for p in m.split("/")))


def main():
    os.makedirs(OUT, exist_ok=True)
    v1 = {r["CaseID"] for r in csv.DictReader(open(V1, encoding="utf-8-sig"))}
    v2 = {r["CaseID"] for r in csv.DictReader(open(V2, encoding="utf-8-sig"))}
    site1 = {r["CaseID"]: r["Site"] for r in csv.DictReader(open(V1, encoding="utf-8-sig"))}
    cached = {re.sub(r"__dx.*", "", f) for f in os.listdir(CACHE) if f.endswith(".pkl")}

    disjoint = cached - v2                 # in the 97, not in the 300
    overlap = cached & v2
    print(f"cached (the 97): {len(cached)}   disjoint from the 300: {len(disjoint)}   "
          f"overlapping: {len(overlap)}")
    sites = {}
    for c in disjoint:
        sites[site1.get(c, "?")] = sites.get(site1.get(c, "?"), 0) + 1
    print(f"disjoint sites: {sites}")

    t3 = json.load(open(T3, encoding="utf-8"))["top"]
    res = {"disjoint": None, "overlap": None}
    for tag, ids in (("disjoint", disjoint), ("overlap", overlap)):
        accum, n = aggregate(ids)
        res[tag] = {"n": n, "top": rank(accum, n)}
        print(f"\n== {tag}: n={n} ==")
        agree = 0
        for r in ROI_ORDER:
            t = t3[r][0]
            e = res[tag]["top"][r][0]
            hit = norm(e["montage"]) == norm(t["montage"])
            agree += hit
            # where does the Table 3 montage itself rank here?
            pos = next((x["rank"] for x in res[tag]["top"][r]
                        if norm(x["montage"]) == norm(t["montage"])), None)
            print(f"  {r:<15} best={e['montage']:<22} F={e['median_score']:.3f}   "
                  f"{'MATCHES Table 3' if hit else 'differs; T3 montage ranks ' + (str(pos) if pos else '>15')}")
        res[tag]["n_agree"] = agree
        print(f"  -> {agree}/{len(ROI_ORDER)} reproduce the Table 3 optimum")

    out = {
        "purpose": "external validation of Table 3 montages on subjects disjoint from the 300",
        "dx": DX, "tissue": TISSUE, "min_present_frac": MIN_FRAC,
        "disjoint_n": res["disjoint"]["n"], "disjoint_sites": sites,
        "overlap_n": res["overlap"]["n"],
        "note": ("The 97-subject sample is NOT independent of and NOT a subset of the 300: "
                 f"{len(overlap)} of the 97 are also in the 300 and {len(disjoint)} are not. "
                 "The disjoint subjects come from two sites absent from the 300."),
        "results": res,
    }
    json.dump(out, open(os.path.join(OUT, "external_validation.json"), "w"), indent=2)
    print("\nwrote", os.path.join(OUT, "external_validation.json"))


if __name__ == "__main__":
    main()
