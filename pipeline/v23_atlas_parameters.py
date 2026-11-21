# -*- coding: utf-8 -*-
"""Per-region targeting parameters of the 300-head, 163-target atlas under v23 at 1 mm and FEM (30 Oct 2026).

The PI reframed Paper A on 30 Oct 2026 around region-wise targeting parameters with optimal montage selection, using
the 300 heads only. The atlas collectors (pipeline/nchc/atlas_ext_collect.py) report the medians over heads. This
script adds what a lookup table of targeting parameters needs, from the same per-head arrays and the same rule
(collect_targets_v2.choose: admissible = T >= 80 % of the head's best T; constrained optimum = highest F among them;
reported choice = most easy-to-place electrodes within 3 % of it):

  spread over heads   T and F at each head's reported choice: 5th, 25th, 50th, 75th and 95th percentiles
  constraint          per head, T and F of the constrained optimum relative to the unconstrained F optimum
  cohort montage      the rule applied to the cohort medians; the share of heads for which that montage is admissible
                      (T >= 80 % of the head's own best T), and its T and F per head (median)

Models: sp1v23 (SPICEhead v23, 1 mm) and fem (SimNIBS leadfield). Targets: AAL90, psychiatric tier 2 and the six
spheres (atlas_v23/out + targets_out), cerebellum and brainstem (atlas_ext/out), thalamic nuclei and supracallosal
ACC (atlas_thal/out); the six spheres are taken from the first set only. Targets with < 5 voxels in a head are
skipped in that head; targets valid in < 10 heads are dropped (VTA in the first set has no grey matter; it is scored
on its own tissue points in the brainstem set).

Input: L:/BrainProbe/SPICEhead/nchc_fullsearch (h-codes only). Output: results_v23/v23_atlas_parameters/.
Reads about 30 GB; below-normal priority on 4 CPUs.
    python pipeline/v23_atlas_parameters.py
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(HERE, "nchc")]
from collect_targets_v2 import choose, EASY, TMIN_FRAC          # noqa: E402  (the approved v2 rule, unchanged)
from v23_fullsearch_collect import low_footprint               # noqa: E402

L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
OUT = os.path.join(ROOT, "results_v23", "v23_atlas_parameters")
MODELS = ("sp1v23", "fem")
# (set, files, suffix, where each model lives: (dir, suffix, model name in that file))
SETS = [
    ("main", {"sp1v23": (os.path.join(L, "atlas_v23", "out"), "__v23atlas.npz", "sp1v23"),
              "fem": (os.path.join(L, "targets_out"), "__targets.npz", "fem")}),
    ("ext", {"sp1v23": (os.path.join(L, "atlas_ext", "out"), "__ext.npz", "sp1v23"),
             "fem": (os.path.join(L, "atlas_ext", "out"), "__ext.npz", "fem")}),
    ("thal", {"sp1v23": (os.path.join(L, "atlas_thal", "out"), "__thal.npz", "sp1v23"),
              "fem": (os.path.join(L, "atlas_thal", "out"), "__thal.npz", "fem")}),
]


PERHEAD = os.path.join(L, "atlas_v23_parameters_perhead.npz")
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read locally only
COHORT = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"          # private; read locally only


def demographics(PH, codes, tnames):
    """Per target and model: log T at the constrained optimum ~ age (per decade) + sex + site (OLS), and the age
    effect alone (Spearman), with BH-FDR across targets per model. Aggregates only leave this function."""
    from scipy.stats import t as tdist, spearmanr
    from collect_targets_v2 import bh
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    crow = {r["CaseID"]: r for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))}
    age = np.array([float(crow[cmap[c]]["Age_Scan"]) for c in codes])
    male = np.array([crow[cmap[c]]["Sex_1M0F"] == "1" for c in codes], float)
    site = [crow[cmap[c]]["Site"] for c in codes]
    sites = sorted(set(site))
    X0 = np.column_stack([np.ones(len(codes)), age / 10, male] + [[s == k for s in site] for k in sites[1:]]).astype(float)
    out = {}
    for m in MODELS:
        res = []
        for t in tnames:
            y = np.log(PH[(m, "T_wc")][t].astype(float)); ok = np.isfinite(y)
            X = X0[ok]; b, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
            e = y[ok] - X @ b; dof = ok.sum() - X.shape[1]
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / dof)
            p = 2 * tdist.sf(np.abs(b / se), dof)
            res.append((t, 100 * (np.exp(b[1]) - 1), p[1], 100 * (np.exp(b[2]) - 1), p[2],
                        spearmanr(age[ok], y[ok]).correlation))
        qa, qs = bh(np.array([r[2] for r in res])), bh(np.array([r[4] for r in res]))
        for (t, a, pa, s, ps, rho), q1, q2 in zip(res, qa, qs):
            out[(m, t)] = dict(age_T_pct_per_decade=round(float(a), 2), age_q=float(q1), sex_T_pct_men=round(float(s), 2),
                               sex_q=float(q2), rho_age_T_wc=round(float(rho), 3))
    return out


def load(path, model):
    with np.load(path) as z:
        mi = list(z["models"]).index(model)
        return z["F"][mi], z["T"][mi], list(z["targets"]), list(z["tiers"]), z["n_roi"], list(z["montages"])


def main():
    if os.name == "nt":
        low_footprint()
    t0 = time.time()
    rows, seen = [], set()
    PH, CODES = {}, [None]                                 # per-head values for the demographic regressions
    for sname, src in SETS:
        d0, suf0, _ = src["sp1v23"]
        codes = sorted(f.split("__")[0] for f in os.listdir(d0) if f.endswith(suf0))
        assert len(codes) == 300, (sname, len(codes))
        per, monts = {}, []
        for m in MODELS:
            d, suf, name = src[m]
            F0, T0, names, tiers, _, mont = load(os.path.join(d, codes[0] + suf), name)
            monts.append(mont)
            assert monts[0] == mont, (sname, m, "montage order differs between models")
            H, NT, NM = len(codes), len(names), len(mont)
            F = np.zeros((H, NT, NM), np.float32); T = np.zeros_like(F); nroi = np.zeros((H, NT), np.int64)
            for hi, c in enumerate(codes):
                Fh, Th, nm, _, nr, mo = load(os.path.join(d, c + suf), name)
                assert nm == names and mo == mont, (c, m)
                F[hi], T[hi], nroi[hi] = Fh, Th, nr
            per[m] = (F, T, nroi, names, tiers)
            print(f"{sname} {m}: {H} heads x {NT} targets loaded ({time.time() - t0:.0f} s)", flush=True)
        easy = np.array([sum(e in EASY for p in mo.split(" / ") for e in p.split("+")) for mo in mont], np.float32)
        names = per["sp1v23"][3]
        for name in names:
            key = name.replace("ref6:", "")
            if key in seen and name.startswith("ref6:"):
                continue                                   # the six spheres are rescored in every set; keep the first
            for m in MODELS:
                F, T, nroi, nm, tiers = per[m]
                if name not in nm:
                    continue
                t = nm.index(name)                         # each file keeps its own target order
                v = nroi[:, t] >= 5
                if v.sum() < 10:
                    continue
                Ft, Tt = F[v, t], T[v, t]                  # (heads, montages)
                ch = choose(Ft, Tt, easy)
                hi = np.arange(Ft.shape[0])
                w0, wc, wa = ch["w0"], ch["wc"], ch["wa"]
                co = choose(np.median(Ft, 0)[None], np.median(Tt, 0)[None], easy)
                cc, c0 = int(co["wa"][0]), int(co["w0"][0])
                tmax = Tt.max(1)
                pc = lambda a: [round(float(np.percentile(a, p)), 5) for p in (5, 25, 50, 75, 95)]
                r = dict(target=name, tier=tiers[t], set=sname, model=m, heads=int(v.sum()),
                         cohort_choice=mont[cc], cohort_unconstrained=mont[c0],
                         **{f"T_choice_p{p}": x for p, x in zip((5, 25, 50, 75, 95), pc(Tt[hi, wa]))},
                         **{f"F_choice_p{p}": x for p, x in zip((5, 25, 50, 75, 95), pc(Ft[hi, wa]))},
                         T_constrained_over_unconstrained_median=round(float(np.median(Tt[hi, wc] / Tt[hi, w0])), 4),
                         F_constrained_over_unconstrained_median=round(float(np.median(Ft[hi, wc] / Ft[hi, w0])), 4),
                         cohort_admissible_pct=round(100 * float(np.mean(Tt[:, cc] >= TMIN_FRAC * tmax)), 1),
                         T_cohort_median=round(float(np.median(Tt[:, cc])), 5), F_cohort_median=round(float(np.median(Ft[:, cc])), 4),
                         band_size_median=int(np.median(ch["nband"])))
                rows.append(r)
                # per head (all 300, NaN where the target has < 5 voxels): T and F at the constrained optimum
                hv = np.flatnonzero(v)
                for k, a in (("T_wc", Tt[hi, wc]), ("F_wc", Ft[hi, wc]), ("T_wa", Tt[hi, wa]), ("F_wa", Ft[hi, wa])):
                    x = np.full(300, np.nan, np.float32); x[hv] = a; PH.setdefault((m, k), {})[name] = x
            seen.add(key)
        if CODES[0] is None:
            CODES[0] = codes
        assert codes == CODES[0], "head order differs between sets"
        del per
    os.makedirs(OUT, exist_ok=True)
    # per-head arrays stay on L: (h-codes only), like every other per-head atlas output
    tnames = sorted(PH[("sp1v23", "T_wc")])
    np.savez_compressed(PERHEAD, codes=np.array(CODES[0]), targets=np.array(tnames),
                        **{f"{m}__{k}": np.stack([PH[(m, k)][t] for t in tnames], 1)
                           for (m, k) in PH if all(t in PH[(m, k)] for t in tnames)})
    demo = demographics(PH, CODES[0], tnames)
    for r in rows:
        r.update(demo.get((r["model"], r["target"]), {}))
    with open(os.path.join(OUT, "target_parameters.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = {m: sum(r["model"] == m for r in rows) for m in MODELS}
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M"), script="pipeline/v23_atlas_parameters.py", inputs=L,
                   rule=dict(admissible=f"T >= {TMIN_FRAC} x head max T", band="3 % of the constrained optimum",
                             report="most easy-to-place electrodes in the band, then F", easy=sorted(EASY)),
                   targets_per_model=n), open(os.path.join(OUT, "PROVENANCE.json"), "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT}: {n} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
