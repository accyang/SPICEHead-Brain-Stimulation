# -*- coding: utf-8 -*-
"""B1 of Rev0.5/10_ANALYSIS_REQUEST_v23_2026-11-02.md (optional, record only): SPICEhead T paired with four-pad and 19-pad FEM T at
the exact six montages of results_v23/v23_pad4_check, in all 300 heads.

SPICEhead scores come from the 300-head homogenized full search (identical to the packaged v23, results_v23/v23_package_check): same
target, montage, 1 mm scoring grid and units. Ratios of T only; a target-envelope ratio is never subtracted from a basis-field bias.
    python pipeline/rev06_b1_pad4_pair.py
Output: results_v23/v23_uq_recollect_2026-11-04/b1_pad4_paired.csv (aggregates) and b1_pad4_paired_per_head.csv (h-codes only).
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAD = os.path.join(ROOT, "results_v23", "v23_pad4_check", "per_head.csv")
HOM = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\homog300"
OUT = os.path.join(ROOT, "results_v23", "v23_uq_recollect_2026-11-04")
NBOOT, SEED = 2000, 20261103


def main():
    p = pd.read_csv(PAD)
    rows = []
    for code, g in p.groupby("code"):
        rois = json.load(open(os.path.join(HOM, f"{code}__sp1hom.json"), encoding="utf-8"))["rois"]
        for r in g.itertuples():
            f, t, o = rois[r.target][r.montage]
            rows.append(dict(code=code, target=r.target, montage=r.montage, n_roi=r.n_roi, T_spicehead=t, F_spicehead=f, T_4pad=r.T_4pad, T_19pad_samemesh=r.T_19pad_samemesh, F_4pad=r.F_4pad, F_19pad_samemesh=r.F_19pad_samemesh))
    d = pd.DataFrame(rows)
    for k, b in (("sp_over_4pad_T", "T_4pad"), ("sp_over_19pad_T", "T_19pad_samemesh")):
        d[k] = d.T_spicehead / d[b]
    d["sp_over_4pad_F"] = d.F_spicehead / d.F_4pad; d["sp_over_19pad_F"] = d.F_spicehead / d.F_19pad_samemesh
    d.to_csv(os.path.join(OUT, "b1_pad4_paired_per_head.csv"), index=False)
    heads = sorted(d.code.unique()); assert len(heads) == 300 and len(d) == 1800
    rng = np.random.default_rng(SEED)
    res = []
    for tg, s in list(d.groupby("target")) + [("all six (head-clustered)", d)]:
        for k in ("sp_over_4pad_T", "sp_over_19pad_T", "sp_over_4pad_F", "sp_over_19pad_F"):
            piv = s.pivot(index="code", columns="target", values=k).loc[heads].values
            med = np.median(piv); p5, p95 = np.percentile(piv, [5, 95])
            b = [np.median(piv[rng.integers(0, 300, 300)]) for _ in range(NBOOT)]
            res.append(dict(target=tg, ratio=k, heads=300, median=round(float(med), 4), p5=round(float(p5), 4), p95=round(float(p95), 4), median_ci_lo=round(float(np.percentile(b, 2.5)), 4), median_ci_hi=round(float(np.percentile(b, 97.5)), 4)))
    res = pd.DataFrame(res); res.to_csv(os.path.join(OUT, "b1_pad4_paired.csv"), index=False)
    print(res.to_string())


if __name__ == "__main__":
    main()
