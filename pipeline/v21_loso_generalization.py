# -*- coding: utf-8 -*-
"""Leave-one-site-out generalization of cohort-derived montages (second-round review, comment 8).

"Does a population-derived TI montage generalize to unseen subjects and unseen sites?" The cohort
spans four heterogeneous acquisition sites, which makes leave-one-site-out the strict test: derive
the optimum on three sites, apply it to the fourth, and measure what is lost against that site's own
optimum.

Three references per held-out set:
  ORACLE     the best montage for the held-out subjects, chosen with knowledge of them (a ceiling
             that cannot be achieved prospectively)
  TRANSFER   the montage derived on the training sites, evaluated on the held-out site
  GLOBAL     the all-300 optimum (what Table 3 reports), evaluated on the held-out site

Also reports a learning curve over training-set size (5, 10, 20, 50, 100, 200 subjects, drawn
without regard to site) to estimate how many heads are needed before a group montage stabilizes.

No new field solves: this re-aggregates the cached per-subject scores from the rotation-invariant
Table 3 run, so which subjects are in the training set is the only thing that changes.

  python pipeline/v21_loso_generalization.py
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
from spicehead import config
sys.path.insert(0, _here)
from montage_search import ROI_ORDER
from run_montage_cohort import montage_str

CACHE = os.path.join(config.RESULTS, "montage_optimization",
                     "cohort_gm300_dx5_maxdir", "_subject_scores")
OUT = os.path.join(config.RESULTS, "v21_generalization")
CSVP = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"
MIN_FRAC = 0.8
SEED = 0


def load_all():
    site = {r["CaseID"]: r["Site"] for r in csv.DictReader(open(CSVP, encoding="utf-8-sig"))}
    data, sites = {}, {}
    for f in sorted(os.listdir(CACHE)):
        if not f.endswith("_maxdir.pkl"):
            continue
        cid = re.sub(r"__dx.*", "", f)
        with open(os.path.join(CACHE, f), "rb") as fh:
            data[cid] = pickle.load(fh)
        sites[cid] = site.get(cid, "?")
    return data, sites


def best_montage(data, ids, roi):
    """Median focality across `ids`, returning (montage_key, median) for the winner."""
    acc = {}
    for c in ids:
        sc = data[c].get(roi, {})
        for k, v in sc.items():
            acc.setdefault(k, []).append(v)
    n = len(ids)
    ranked = [(float(np.median(v)), k) for k, v in acc.items() if len(v) >= MIN_FRAC * n]
    if not ranked:
        return None, None
    ranked.sort(key=lambda x: -x[0])
    return ranked[0][1], ranked[0][0]


def evaluate(data, ids, roi, key):
    vals = [data[c][roi][key] for c in ids if key in data[c].get(roi, {})]
    return float(np.median(vals)) if vals else float("nan")


def main():
    os.makedirs(OUT, exist_ok=True)
    data, sites = load_all()
    ids = sorted(data)
    by_site = {}
    for c in ids:
        by_site.setdefault(sites[c], []).append(c)
    print(f"{len(ids)} subjects across {len(by_site)} sites: "
          + ", ".join(f"{s} n={len(v)}" for s, v in sorted(by_site.items())))

    glob_key = {r: best_montage(data, ids, r)[0] for r in ROI_ORDER}

    rows = []
    print(f"\n{'held-out site':<12}{'ROI':<15}{'oracle':>8}{'transfer':>10}{'global':>8}"
          f"{'transfer loss':>15}{'same as oracle':>16}")
    for site in sorted(by_site):
        held = by_site[site]
        train = [c for c in ids if sites[c] != site]
        for roi in ROI_ORDER:
            ok, ov = best_montage(data, held, roi)          # oracle on the held-out site
            tk, _ = best_montage(data, train, roi)          # derived on training sites
            tv = evaluate(data, held, roi, tk)
            gv = evaluate(data, held, roi, glob_key[roi])
            loss = 100 * (ov - tv) / ov if ov else float("nan")
            same = (tk == ok)
            rows.append(dict(held_out_site=site, n_held=len(held), roi=roi,
                             oracle=round(ov, 4), transfer=round(tv, 4), global_=round(gv, 4),
                             transfer_loss_pct=round(loss, 2), transfer_is_oracle=bool(same),
                             transfer_montage=montage_str(tk), oracle_montage=montage_str(ok)))
            print(f"{site:<12}{roi:<15}{ov:>8.3f}{tv:>10.3f}{gv:>8.3f}{loss:>14.1f}%"
                  f"{('yes' if same else 'no'):>16}")

    with open(os.path.join(OUT, "loso.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    loss = np.array([r["transfer_loss_pct"] for r in rows])
    same = sum(1 for r in rows if r["transfer_is_oracle"])
    print(f"\ntransfer reproduces the held-out optimum in {same}/{len(rows)} site-target pairs")
    print(f"median transfer loss {np.median(loss):.2f}%, worst {loss.max():.2f}%")

    # ---- learning curve: how many training heads before the montage stabilizes?
    rng = np.random.default_rng(SEED)
    print(f"\n{'n_train':>8}" + "".join(f"{r.split('_')[0][:9]:>11}" for r in ROI_ORDER))
    lc = []
    for n_train in (5, 10, 20, 50, 100, 200):
        agree = {r: 0 for r in ROI_ORDER}
        REPS = 20
        for _ in range(REPS):
            tr = list(rng.choice(ids, size=n_train, replace=False))
            for roi in ROI_ORDER:
                k, _ = best_montage(data, tr, roi)
                agree[roi] += (k == glob_key[roi])
        row = {"n_train": n_train}
        for roi in ROI_ORDER:
            row[roi] = round(100 * agree[roi] / REPS, 1)
        lc.append(row)
        print(f"{n_train:>8}" + "".join(f"{row[r]:>10.0f}%" for r in ROI_ORDER))
    with open(os.path.join(OUT, "learning_curve.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lc[0]))
        w.writeheader()
        w.writerows(lc)
    print("\n(percent of random training sets of that size that recover the all-300 optimum)")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
