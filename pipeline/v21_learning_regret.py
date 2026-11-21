# -*- coding: utf-8 -*-
"""Performance-based learning curve — third-round review, comment 7.

The first learning curve scored a training set as successful only if it recovered the *identical*
all-300 winning montage. That is a brittle endpoint when several montages are near-tied: a draw
that picks the second-best montage counts as a total failure even when the second-best performs
indistinguishably. It made the motor hand-knob look like an unstable optimization when the real
situation may be many interchangeable solutions.

This replaces identity with performance. For each training size, over random draws:

  regret        focality shortfall of the selected montage against the all-300 optimum, both
                evaluated on the held-out complement so the estimate is not self-referential
  P(within x%)  probability the selected montage lands within 1 / 2 / 5 % of that optimum
  rank          position of the selected montage in the all-300 ranking (1 = the global optimum)
  regret CI     bootstrap 95% interval over draws
  Jaccard       overlap of the training set's top-5 with the all-300 top-5, which distinguishes
                "found a different member of the same near-optimal set" from "searched elsewhere"

No new field solves: this re-aggregates the cached per-subject scores.

  python pipeline/v21_learning_regret.py [reps]
"""
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)
from montage_search import ROI_ORDER
from v21_loso_generalization import load_all, best_montage, MIN_FRAC

OUT = os.path.join(config.RESULTS, "v21_generalization")
SIZES = (5, 10, 20, 50, 100, 200)
SEED = 0


def dense(data, ids, roi):
    """(subjects x montages) score matrix, NaN where a subject lacks that montage.

    Pure-Python re-medianing of ~11,600 montages per draw is far too slow (billions of dict
    lookups). Building the matrix once turns each draw into one nanmedian over a numpy slice.
    """
    keys = sorted({k for c in ids for k in data[c].get(roi, {})})
    idx = {k: i for i, k in enumerate(keys)}
    M = np.full((len(ids), len(keys)), np.nan, np.float32)
    for i, c in enumerate(ids):
        for k, v in data[c].get(roi, {}).items():
            M[i, idx[k]] = v
    return keys, M


def score(M, rows_, min_frac=MIN_FRAC):
    """Median over the given subject rows, NaN for montages present in too few of them."""
    sub = M[rows_]
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(sub, axis=0)
    med[np.count_nonzero(~np.isnan(sub), axis=0) < min_frac * len(rows_)] = np.nan
    return med


def main(reps=40):
    data, _ = load_all()
    ids = sorted(data)
    rng = np.random.default_rng(SEED)
    print(f"{len(ids)} subjects, {reps} draws per size", flush=True)

    K, M, gmed, gkey, grank, gtop5 = {}, {}, {}, {}, {}, {}
    allrows = np.arange(len(ids))
    for roi in ROI_ORDER:
        K[roi], M[roi] = dense(data, ids, roi)
        gmed[roi] = score(M[roi], allrows)
        order = np.argsort(-np.nan_to_num(gmed[roi], nan=-np.inf))
        gkey[roi] = int(order[0])
        grank[roi] = np.empty(len(order), int)
        grank[roi][order] = np.arange(1, len(order) + 1)
        gtop5[roi] = set(order[:5].tolist())
        print(f"  {roi:<16}{M[roi].shape[1]} montages", flush=True)

    rows = []
    for n_train in SIZES:
        acc = {r: dict(regret=[], rank=[], jac=[], exact=0) for r in ROI_ORDER}
        for _ in range(reps):
            tr = rng.choice(len(ids), size=n_train, replace=False)
            held = np.setdiff1d(allrows, tr)
            for roi in ROI_ORDER:
                mt = score(M[roi], tr)
                if np.all(np.isnan(mt)):
                    continue
                sel = int(np.nanargmax(mt))
                # evaluate BOTH on the held-out complement, so the comparison is out-of-sample
                ev = score(M[roi], held)
                ref = ev[gkey[roi]]
                if not np.isfinite(ev[sel]) or not np.isfinite(ref) or ref == 0:
                    continue
                acc[roi]["regret"].append(100 * (ref - ev[sel]) / ref)
                acc[roi]["rank"].append(int(grank[roi][sel]))
                top5 = set(np.argsort(-np.nan_to_num(mt, nan=-np.inf))[:5].tolist())
                acc[roi]["jac"].append(len(gtop5[roi] & top5) / len(gtop5[roi] | top5))
                acc[roi]["exact"] += (sel == gkey[roi])
        for roi in ROI_ORDER:
            g = np.array(acc[roi]["regret"], float)
            g = g[np.isfinite(g)]
            if not g.size:
                continue
            boot = [float(np.median(rng.choice(g, size=g.size, replace=True))) for _ in range(1000)]
            rows.append(dict(
                n_train=n_train, roi=roi, n_draws=int(g.size),
                median_regret_pct=round(float(np.median(g)), 4),
                mean_regret_pct=round(float(np.mean(g)), 4),
                regret_ci_lo=round(float(np.percentile(boot, 2.5)), 4),
                regret_ci_hi=round(float(np.percentile(boot, 97.5)), 4),
                p_within_1pct=round(float(np.mean(g <= 1.0)), 4),
                p_within_2pct=round(float(np.mean(g <= 2.0)), 4),
                p_within_5pct=round(float(np.mean(g <= 5.0)), 4),
                median_rank=round(float(np.nanmedian(acc[roi]["rank"])), 1),
                jaccard_top5=round(float(np.mean(acc[roi]["jac"])), 4),
                exact_match_pct=round(100 * acc[roi]["exact"] / reps, 1)))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "learning_curve_regret.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'n':>5}{'ROI':<16}{'exact':>7}{'regret':>9}{'95% CI':>16}"
          f"{'P<1%':>7}{'P<5%':>7}{'rank':>7}{'Jacc':>7}")
    for r in rows:
        print(f"{r['n_train']:>5}{r['roi']:<16}{r['exact_match_pct']:>6.0f}%"
              f"{r['median_regret_pct']:>8.2f}%"
              f"  [{r['regret_ci_lo']:>5.2f},{r['regret_ci_hi']:>5.2f}]"
              f"{100*r['p_within_1pct']:>6.0f}%{100*r['p_within_5pct']:>6.0f}%"
              f"{r['median_rank']:>7.0f}{r['jaccard_top5']:>7.2f}")
    print("\nwrote", p)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
