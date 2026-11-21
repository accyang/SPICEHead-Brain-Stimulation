# -*- coding: utf-8 -*-
"""Raw (non-rescaled) agreement against the FEM reference — second-round review, comment 4.

The first pass reported Bland-Altman on scale-matched fields, which drives the bias to ~0 by
construction and so removes exactly the systematic disagreement Bland-Altman exists to reveal. This
reports the raw quantities:

  * raw Bland-Altman bias and 95% limits of agreement, in V/m, SPICEhead minus FEM
  * proportional bias (slope of difference on mean) — is the disagreement constant or amplitude-
    dependent?
  * Lin's concordance correlation coefficient, which penalises both scale and location shift
  * regression slope and intercept, with the direction stated explicitly

Regression direction: FEM (reference) is regressed on SPICEhead (model), y_FEM = a + b*x_SPICEhead,
so b < 1 means SPICEhead reads high.

Caches the SPICEhead envelope per subject/resolution so re-runs are cheap.

  python pipeline/v21_agreement_raw.py [dx list] [n]
"""
import csv
import os
import pickle
import sys
import time

import numpy as np
import nibabel as nib

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)
from montage_search import SCORE
from v21_simnibs_compare import load_simnibs_E, envelope, spicehead_E, resample_to
from v21_simnibs_validation import OUT, subjects

CACHE = os.path.join(os.environ.get("TEMP", "."), config.tagged("spicehead_femcmp_cache"))


def stats(x, y):
    """x = SPICEhead (model), y = FEM (reference). All raw, no rescaling."""
    d = x - y
    m = (x + y) / 2.0
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    b, a = np.polyfit(x, y, 1)              # y_FEM = a + b*x_SPICEhead
    pslope = float(np.polyfit(m, d, 1)[0])  # proportional bias
    vx, vy = x.var(), y.var()
    ccc = float(2 * np.cov(x, y)[0, 1] / (vx + vy + (x.mean() - y.mean()) ** 2))
    ss = float(1 - ((y - (a + b * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return dict(raw_bias=bias, loa_lo=bias - 1.96 * sd, loa_hi=bias + 1.96 * sd,
                prop_bias_slope=pslope, ccc=ccc, slope=float(b), intercept=float(a),
                r=float(np.corrcoef(x, y)[0, 1]), r2=ss,
                mean_spicehead=float(x.mean()), mean_fem=float(y.mean()))


def main(dxs=(2.0, 1.0), n=8):
    os.makedirs(CACHE, exist_ok=True)
    rows = []
    for si, (cid, m2m, age, site) in enumerate(subjects(n), 1):
        try:
            E1, aff, sh = load_simnibs_E(os.path.join(OUT, cid, "ch1"))
            E2, _, _ = load_simnibs_E(os.path.join(OUT, cid, "ch2"))
        except Exception as ex:
            print(f"[{si}] {cid}: no FEM ({type(ex).__name__}) - skip", flush=True)
            continue
        fem = np.nan_to_num(envelope(E1, E2))
        li = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
        gm = resample_to(np.isin(np.asanyarray(li.dataobj), SCORE).astype(np.float32),
                         li.affine, aff, sh) > 0.5
        for dx in dxs:
            cp = os.path.join(CACHE, f"{cid}__dx{dx:g}_env.pkl")
            if os.path.exists(cp):
                with open(cp, "rb") as fh:
                    sp = pickle.load(fh)
                note = "cached"
            else:
                t = time.time()
                (S1, S2), saff, _ = spicehead_E(m2m, dx)
                sp = resample_to(np.nan_to_num(envelope(S1, S2)), saff, aff, sh)
                tmp = cp + ".part"
                with open(tmp, "wb") as fh:
                    pickle.dump(sp.astype(np.float32), fh, protocol=4)
                os.replace(tmp, cp)
                note = f"{time.time()-t:.0f}s"
            x = np.asarray(sp)[gm].astype(float)
            y = fem[gm].astype(float)
            k = np.isfinite(x) & np.isfinite(y) & (y > 0)
            st = stats(x[k], y[k])
            st.update(subject=cid, age=age, site=site, dx_mm=dx, n_vox=int(k.sum()))
            rows.append(st)
            print(f"[{si}] {cid} dx={dx:g}: bias={st['raw_bias']:+.5f} "
                  f"LoA[{st['loa_lo']:+.5f},{st['loa_hi']:+.5f}] "
                  f"prop={st['prop_bias_slope']:+.3f} CCC={st['ccc']:.3f} "
                  f"slope={st['slope']:.3f} ({note})", flush=True)

    keys = ["subject", "age", "site", "dx_mm", "n_vox", "raw_bias", "loa_lo", "loa_hi",
            "prop_bias_slope", "ccc", "slope", "intercept", "r", "r2",
            "mean_spicehead", "mean_fem"]
    out_dir = OUT if config.MODEL == "v21" else os.path.join(config.RESULTS, "v21_simnibs_validation")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "agreement_raw.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print("\nwrote", out)
    for dx in dxs:
        s = [r for r in rows if r["dx_mm"] == dx]
        if not s:
            continue
        f = lambda k: float(np.mean([r[k] for r in s]))
        print(f"  dx={dx:g} mm (n={len(s)}): raw bias {f('raw_bias'):+.5f} V/m, "
              f"LoA [{f('loa_lo'):+.5f}, {f('loa_hi'):+.5f}], "
              f"proportional bias {f('prop_bias_slope'):+.3f}, CCC {f('ccc'):.3f}, "
              f"slope {f('slope'):.3f}, r {f('r'):.3f}")


if __name__ == "__main__":
    dxs = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else (2.0, 1.0)
    main(dxs, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
