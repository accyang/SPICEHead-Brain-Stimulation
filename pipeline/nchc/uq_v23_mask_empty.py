# -*- coding: utf-8 -*-
"""Target-mask scenario of the v23 uncertainty analysis (amendment 2, decision D7): count, per region, the heads in which
one-voxel erosion or dilation leaves the target mask without a valid T at the fixed montage (2 Nov 2026).

The collector (uq_v23_atlas_collect.py) skips non-finite T in its percentiles, so an emptied mask either drops out (T
not finite) or enters as a ratio of 0 (T = 0). This script reports both cases explicitly so the mask results can be
read correctly. A head-target is counted when the nominal run has a valid T there and the mask run does not.
Input: L:/BrainProbe/SPICEhead/nchc_fullsearch/uq_v23/{runs_atlas,runs_thal}/<code>/{nominal,mask_erode,mask_dilate}.npz
Output: results_v23/v23_uq/mask_empty_300.csv (aggregate counts only; no h-codes).
    python pipeline/nchc/uq_v23_mask_empty.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\uq_v23"
OUT = os.path.join(ROOT, "results_v23", "v23_uq", "mask_empty_300.csv")


def main():
    counts = {}
    for sub in ("runs_atlas", "runs_thal"):
        d0 = os.path.join(BASE, sub)
        heads = sorted(c for c in os.listdir(d0) if os.path.exists(os.path.join(d0, c, "nominal.npz")))
        for c in heads:
            d = os.path.join(d0, c)
            names = json.load(open(os.path.join(d, "targets.json")))["targets"]
            with np.load(os.path.join(d, "nominal.npz")) as z:
                fv0 = z["fixed_vals"].astype(np.float64)
            fc = np.where(np.isfinite(fv0[:, 0, 1]), 0, 3)
            ix = np.arange(len(fc))
            T0 = fv0[ix, fc, 1]
            ok = np.isfinite(T0) & (T0 > 0)
            for run in ("mask_erode", "mask_dilate"):
                with np.load(os.path.join(d, f"{run}.npz")) as z:
                    T = z["fixed_vals"].astype(np.float64)[ix, fc, 1]
                for t in np.where(ok)[0]:
                    if sub == "runs_thal" and names[t].startswith("ref6:"):
                        continue                                   # the six spheres are counted from runs_atlas
                    k = counts.setdefault(names[t], dict(heads=0, mask_erode_nonfinite=0, mask_erode_zero=0,
                                                         mask_dilate_nonfinite=0, mask_dilate_zero=0))
                    if run == "mask_erode":
                        k["heads"] += 1
                    if not np.isfinite(T[t]):
                        k[f"{run}_nonfinite"] += 1
                    elif T[t] <= 0:
                        k[f"{run}_zero"] += 1
    rows = [dict(target=t, **v) for t, v in sorted(counts.items())]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        n = r["mask_erode_nonfinite"] + r["mask_erode_zero"] + r["mask_dilate_nonfinite"] + r["mask_dilate_zero"]
        if n:
            print(r["target"], r)
    print(f"wrote {OUT}: {len(rows)} targets")


if __name__ == "__main__":
    main()
