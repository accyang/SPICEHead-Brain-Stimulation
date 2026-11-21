# -*- coding: utf-8 -*-
"""Aggregate the per-head field-error maps (field_error_map.py) into results_v22/v22_field_error_map/.

Medians across heads of the per-head medians: by electrode, depth bin, distance to the injecting electrode,
CSF and bone quartiles, AAL region; whole-GM r / CCC / bias per grid. Aggregates only (no head codes).

    python pipeline/nchc/field_error_summary.py
"""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PERHEAD = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\field_error_perhead"
OUT = os.path.join(ROOT, "results_v22", "v22_field_error_map")
AAL = os.path.join(r"L:\BrainProbe\SPICEhead\nchc_fullsearch", "bundles", "aal90_targets.json")
DEPTH = ["0-10", "10-20", "20-30", "30-40", ">40"]
DIST = ["<20", "20-40", "40-60", "60-80", ">80"]
Q = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def main():
    files = sorted(f for f in os.listdir(PERHEAD) if f.endswith(".json"))
    R = [json.load(open(os.path.join(PERHEAD, f))) for f in files]
    names = {d["index"]: d["name"] for d in json.load(open(AAL))} if os.path.exists(AAL) else {}
    os.makedirs(OUT, exist_ok=True)
    L = [f"# Field-level error of SPICEhead against FEM: {len(R)} heads, 18 single-electrode bases each", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `pipeline/nchc/field_error_summary.py` from per-head maps "
         "(`field_error_map.py`; grey-matter voxels of the shared 1 mm grid; FEM leadfield vs SPICEhead 1 mm and 2 mm bases, "
         "E(a, Cz) at 1 mA). rel = |E_sp − E_fem| / |E_fem| per voxel; ratio = |E_sp| / |E_fem|; angle in degrees. "
         "Each cell is the median across heads of the per-head median.", ""]
    for m, lab in (("sp1", "SPICEhead 1 mm"), ("sp2", "SPICEhead 2 mm")):
        L += [f"## {lab}", ""]
        e = [v[m] for r in R for v in r["by_electrode"].values()]
        L += [f"Whole grey matter, all bases: rel {med([x['rel_median'] for x in e]):.3f} (90th pct within head {med([x['rel_p90'] for x in e]):.3f}); "
              f"magnitude ratio {med([x['ratio_median'] for x in e]):.3f}; bias {med([x['bias_pct'] for x in e]):+.1f} %; angle {med([x['ang_median'] for x in e]):.1f}°; "
              f"r {med([x['r'] for x in e]):.3f}; CCC {med([x['ccc'] for x in e]):.3f}.", ""]
        for key, labels, title in (("by_depth", DEPTH, "Depth below the scalp surface (mm)"), ("by_dist_this", DIST, "Distance to the injecting electrode (mm)"),
                                   ("by_dist_any", DIST, "Distance to the nearest electrode (mm)"), ("by_csf5", Q, "CSF fraction within 5 mm (quartiles within head)"),
                                   ("by_bone10", Q, "Bone fraction within 10 mm (quartiles within head)")):
            L += [f"### {title}", "", "| Bin | rel | ratio | angle | voxel share |", "|---|---|---|---|---|"]
            tot = sum(r[key].get(f"{m}:{b}", {}).get("n", 0) for r in R for b in range(len(labels)))
            for b, name in enumerate(labels):
                k = f"{m}:{b}"
                rows = [r[key][k] for r in R if k in r[key]]
                if rows:
                    L.append(f"| {name} | {med([x['rel'] for x in rows]):.3f} | {med([x['ratio'] for x in rows]):.3f} | {med([x['ang'] for x in rows]):.1f} | "
                             f"{100 * sum(x['n'] for x in rows) / max(tot, 1):.1f} % |")
            L.append("")
        L += ["### Electrodes (median over heads)", "", "| Electrode | rel | ratio | bias % | r |", "|---|---|---|---|---|"]
        for el in sorted(R[0]["by_electrode"]):
            x = [r["by_electrode"][el][m] for r in R if el in r["by_electrode"]]
            L.append(f"| {el} | {med([y['rel_median'] for y in x]):.3f} | {med([y['ratio_median'] for y in x]):.3f} | {med([y['bias_pct'] for y in x]):+.1f} | {med([y['r'] for y in x]):.3f} |")
        L.append("")
        reg = []
        for i in range(1, 91):
            k = f"{m}:{i}"
            rows = [r["by_aal"][k] for r in R if k in r["by_aal"]]
            if rows:
                reg.append((med([x["rel"] for x in rows]), med([x["ratio"] for x in rows]), names.get(i, f"AAL{i:02d}")))
        reg.sort()
        L += ["### AAL regions with the largest and smallest error", "", "| Largest rel | ratio | Smallest rel | ratio |", "|---|---|---|---|"]
        for (a, ra, na), (b, rb, nb) in zip(reg[::-1][:10], reg[:10]):
            L.append(f"| {na} {a:.3f} | {ra:.3f} | {nb} {b:.3f} | {rb:.3f} |")
        L.append("")
    json.dump(dict(n_heads=len(R), generated=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(OUT, "PROVENANCE.json"), "w"), indent=1)
    open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"... wrote {OUT}/SUMMARY.md ({len(R)} heads)")


if __name__ == "__main__":
    main()
