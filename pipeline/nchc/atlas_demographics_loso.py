# -*- coding: utf-8 -*-
"""Sex, site and leave-one-site-out transfer for all 163 targets of the 300-head atlas (26 Oct 2026; PI: "run sex/site and
LOSO for all 163 targets").

Targets: the 111 of the 22 Oct atlas (targets_out; 109 resolvable), the 38 cerebellar and brainstem targets (atlas_ext)
and the 16 thalamic-nucleus and ACC_sup targets (atlas_thal); their ref6 copies are dropped. Per head and target the
approved v2 rule (collect_targets_v2.choose) gives the constrained optimum (FC, TC); the cohort montage is the same rule
on the cohort medians.

1. Sex and site (age-adjusted). Per target, OLS on the resolvable heads: y ~ age + sex + site (three site dummies), for
   y = log TC (target field at the constrained optimum), FC and log gain (gain = FC / F at the all-300 cohort montage).
   Sex: coefficient (log scale: per cent difference men vs women), t-test p. Site: partial F-test of the three site
   terms (p) and the spread of the adjusted site means. BH-FDR across targets, per model and outcome. Also the medians
   by sex and site, as in the 22 Oct demographics table.
2. Leave-one-site-out (LOSO). For each held-out site: the cohort montage is chosen by the v2 rule on the other three
   sites' medians (transfer), on all 300 heads (global) and on the held-out site's own medians (site optimum, which
   needs knowledge of that site). On the held-out heads: transfer loss = 1 - median F(transfer) / median F(site optimum)
   (the earlier definition, v21_loso_generalization.py), and the per-head retention F(montage) / FC, transfer against
   global.
Outputs (aggregates only; demographics are read from the private cohort files and never written per head):
  <out>/demographics_163.csv, loso_163.csv, SUMMARY.md, PROVENANCE.json
    python pipeline/nchc/atlas_demographics_loso.py [--models fem,sp1] [--out DIR]
    (v22 models -> results_v22/v22_target_atlas_300/demographics_loso_163; with v23 models pass --out results_v23/...)
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline"), HERE]
from collect_targets_v2 import choose, bh, EASY            # noqa: E402  (the approved v2 rule, unchanged)

L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only
COHORT = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"          # private; read only
OUT = os.path.join(ROOT, "results_v22", "v22_target_atlas_300", "demographics_loso_163")
# where each model's scores of each target set live: (folder, file suffix)
MAIN = {"fem": ("targets_out", "__targets.npz"), "sp2": ("targets_out", "__targets.npz"), "sp1": ("targets_out", "__targets.npz"),
        "sp5": ("targets_out_sp5", "__targets.npz"), "sp2v23": (os.path.join("atlas_v23", "out"), "__v23atlas.npz"),
        "sp1v23": (os.path.join("atlas_v23", "out"), "__v23atlas.npz")}
EXT = [(os.path.join("atlas_ext", "out"), "__ext.npz"), (os.path.join("atlas_thal", "out"), "__thal.npz")]


def load_model(m, codes):
    """F, T (heads x targets x montages, float32), names, tiers, n_roi for one model over all 163 targets."""
    F = T = None; names = tiers = None; nroi = []
    for hi, c in enumerate(codes):
        parts = []
        for folder, suf, drop_ref6 in [(*MAIN[m], False)] + [(f, s, True) for f, s in EXT]:
            with np.load(os.path.join(L, folder, f"{c}{suf}")) as z:
                mi = list(z["models"]).index(m)
                keep = [i for i, n in enumerate(z["targets"]) if not (drop_ref6 and str(n).startswith("ref6:"))]
                parts.append((z["F"][mi][keep], z["T"][mi][keep], [str(z["targets"][i]) for i in keep], [str(z["tiers"][i]) for i in keep],
                              z["n_roi"][keep], list(z["montages"])))
        mont = parts[0][5]
        assert all(p[5] == mont for p in parts), c
        if F is None:
            names = sum((p[2] for p in parts), []); tiers = sum((p[3] for p in parts), [])
            F = np.zeros((len(codes), len(names), len(mont)), np.float32); T = np.zeros_like(F); M = mont
        assert sum((p[2] for p in parts), []) == names and mont == M, c
        F[hi] = np.concatenate([p[0] for p in parts]); T[hi] = np.concatenate([p[1] for p in parts])
        nroi.append(np.concatenate([p[4] for p in parts]))
        if (hi + 1) % 50 == 0:
            print(f"    {m}: loaded {hi+1}/{len(codes)}", flush=True)
    return F, T, names, tiers, np.array(nroi), M


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return beta, float(res @ res)


def main():
    a = sys.argv[1:]
    models = (a[a.index("--models") + 1] if "--models" in a else "fem,sp1").split(",")
    out = a[a.index("--out") + 1] if "--out" in a else OUT
    t0 = time.time()
    codes = sorted(f.split("__")[0] for f in os.listdir(os.path.join(L, "atlas_thal", "out")) if f.endswith("__thal.npz"))
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    crow = {r["CaseID"]: r for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))}
    age = np.array([float(crow[cmap[c]]["Age_Scan"]) for c in codes])
    sex = np.array([crow[cmap[c]]["Sex_1M0F"] for c in codes]).astype(int)            # 1 = man
    site = np.array([crow[cmap[c]]["Site"] for c in codes]); sites = sorted(set(site))
    H = len(codes)
    dem, loso, fixed_names = [], [], None
    for m in models:
        F, T, names, tiers, nroi, mont = load_model(m, codes)
        fixed_names = fixed_names or names
        assert names == fixed_names
        easy = np.array([sum(e in EASY for p in mm.split(" / ") for e in p.split("+")) for mm in mont], np.float32)
        NT = len(names)
        wc = np.array([choose(F[hi], T[hi], easy)["wc"] for hi in range(H)])           # (H, NT) constrained optimum
        ar = np.arange(NT)
        FC = np.stack([F[hi][ar, wc[hi]] for hi in range(H)]); TC = np.stack([T[hi][ar, wc[hi]] for hi in range(H)])
        valid = nroi >= 5
        print(f"  {m}: rule applied ({time.time()-t0:.0f} s)", flush=True)

        def cohort(sel, t):
            co = choose(np.median(F[sel, t], 0)[None], np.median(T[sel, t], 0)[None], easy)
            return int(co["wa"][0])
        rows_m = []
        for t in range(NT):
            v = valid[:, t]
            if v.sum() < 10:
                continue
            cg = cohort(v, t)
            gain = FC[v, t] / F[v, t, cg]
            Xd = [np.ones(v.sum()), age[v], sex[v]] + [(site[v] == s).astype(float) for s in sites[1:]]
            X = np.column_stack(Xd); Xr = X[:, :3]
            d = dict(target=names[t], tier=tiers[t], model=m, n=int(v.sum()), cohort_choice=mont[cg],
                     F_median_men=round(float(np.median(FC[v & (sex == 1), t])), 4), F_median_women=round(float(np.median(FC[v & (sex == 0), t])), 4),
                     T_median_men=round(float(np.median(TC[v & (sex == 1), t])), 5), T_median_women=round(float(np.median(TC[v & (sex == 0), t])), 5))
            for s in sites:
                d[f"T_median_{s}"] = round(float(np.median(TC[v & (site == s), t])), 5)
            for key, y in (("logT", np.log(TC[v, t].astype(float))), ("F", FC[v, t].astype(float)), ("loggain", np.log(gain.astype(float)))):
                beta, rss = ols(y, X); _, rss_r = ols(y, Xr)
                dof = len(y) - X.shape[1]; s2 = rss / dof
                cov = s2 * np.linalg.inv(X.T @ X); tsex = beta[2] / np.sqrt(cov[2, 2])
                d[f"sex_coef_{key}"] = round(float(beta[2]), 5); d[f"p_sex_{key}"] = float(2 * stats.t.sf(abs(tsex), dof))
                k = X.shape[1] - 3; Fst = ((rss_r - rss) / k) / s2
                d[f"p_site_{key}"] = float(stats.f.sf(Fst, k, dof))
                adj = np.r_[0.0, beta[3:]]                                              # site means relative to the first site
                d[f"site_spread_{key}"] = round(float(adj.max() - adj.min()), 5)
            rows_m.append(d)
            # leave-one-site-out
            for s in sites:
                ho = v & (site == s); tr = v & (site != s)
                if ho.sum() < 5 or tr.sum() < 10:
                    continue
                ct, co = cohort(tr, t), cohort(ho, t)
                ft, fo, fg = (float(np.median(F[ho, t, c])) for c in (ct, co, cg))
                rt = F[ho, t, ct] / FC[ho, t]; rg = F[ho, t, cg] / FC[ho, t]
                loso.append(dict(target=names[t], tier=tiers[t], model=m, held_out_site=s, n_held_out=int(ho.sum()),
                                 transfer_montage=mont[ct], site_optimum_montage=mont[co], global_montage=mont[cg],
                                 transfer_is_site_optimum=bool(ct == co), transfer_is_global=bool(ct == cg),
                                 transfer_loss_pct=round(100 * (fo - ft) / fo, 3) if fo > 0 else float("nan"),
                                 retention_transfer_median=round(float(np.median(rt)), 4), retention_global_median=round(float(np.median(rg)), 4),
                                 retention_transfer_p5=round(float(np.percentile(rt, 5)), 4)))
        for key in ("logT", "F", "loggain"):
            for kind in ("sex", "site"):
                for d, qq in zip(rows_m, bh([d[f"p_{kind}_{key}"] for d in rows_m])):
                    d[f"q_{kind}_{key}"] = float(qq)
        dem += rows_m
        del F, T
        print(f"  {m}: done ({time.time()-t0:.0f} s)", flush=True)
    os.makedirs(out, exist_ok=True)

    def write(name, rr):
        ks = list(rr[0]); [ks.append(k) for r in rr for k in r if k not in ks]
        with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ks); w.writeheader(); w.writerows(rr)
    write("demographics_163.csv", dem); write("loso_163.csv", loso)
    # ---- summary
    setname = {"tier1": "AAL90 cerebrum", "tier2": "Psychiatric (tier 2)", "ref6": "Paper A spheres", "cerebellum": "Cerebellum",
               "brainstem": "Brainstem nuclei", "thalamus": "Thalamic nuclei", "cingulate": "ACC_sup"}
    order = list(setname.values())
    nsite = {s: int((site == s).sum()) for s in sites}
    Lm = [f"# Sex, site and leave-one-site-out transfer: 300 heads, {len({d['target'] for d in dem})} targets", "",
          f"Collected {time.strftime('%Y-%m-%d %H:%M')} by `pipeline/nchc/atlas_demographics_loso.py`. Models: {', '.join(models)}. "
          f"Sites (heads): {', '.join(f'{s} ({n})' for s, n in nsite.items())}; men {int(sex.sum())}, women {int((1 - sex).sum())}.",
          "Sex and site effects are age-adjusted (OLS: y ~ age + sex + site), BH-FDR across targets per model and outcome. "
          "T = target field at the head's constrained optimum (log scale: a coefficient of 0.05 is about 5 %).", "",
          "This is a new result set. It is not in any Results record or manuscript.", ""]
    for m in models:
        D = [d for d in dem if d["model"] == m]; Lo = [r for r in loso if r["model"] == m]
        Lm += [f"## {m}", "", "| Set | Targets | Sex effect on T, q < 0.05 (men higher / lower) | Sex effect on T, median | Site effect on T, q < 0.05 | Site spread of T, median | Sex on F, q < 0.05 | Site on F, q < 0.05 | LOSO transfer loss, median [max] | Transfer = all-300 montage | Retention, transfer vs global (median) |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for lab in order + ["All"]:
            sub = [d for d in D if setname[d["tier"]] == lab or lab == "All"]
            if not sub:
                continue
            tg = {d["target"] for d in sub}; lo = [r for r in Lo if r["target"] in tg]
            sexT = [d for d in sub if d["q_sex_logT"] < 0.05]
            Lm.append(f"| {lab} | {len(sub)} | {len(sexT)} ({sum(d['sex_coef_logT'] > 0 for d in sexT)} / {sum(d['sex_coef_logT'] < 0 for d in sexT)}) | "
                      f"{100*np.median([d['sex_coef_logT'] for d in sub]):+.1f} % | {sum(d['q_site_logT'] < 0.05 for d in sub)} | {100*np.median([d['site_spread_logT'] for d in sub]):.1f} % | "
                      f"{sum(d['q_sex_F'] < 0.05 for d in sub)} | {sum(d['q_site_F'] < 0.05 for d in sub)} | "
                      f"{np.nanmedian([r['transfer_loss_pct'] for r in lo]):.1f} % [{np.nanmax([r['transfer_loss_pct'] for r in lo]):.1f} %] | "
                      f"{100*np.mean([r['transfer_is_global'] for r in lo]):.0f} % | "
                      f"{np.median([r['retention_transfer_median'] for r in lo]):.3f} vs {np.median([r['retention_global_median'] for r in lo]):.3f} |")
        worst = sorted([r for r in Lo if np.isfinite(r["transfer_loss_pct"])], key=lambda r: -r["transfer_loss_pct"])[:10]
        Lm += ["", f"Largest LOSO transfer losses ({m}):", "", "| Target | Held-out site | Loss | Transfer montage | Site optimum |", "|---|---|---|---|---|"]
        Lm += [f"| {r['target']} | {r['held_out_site']} | {r['transfer_loss_pct']:.1f} % | {r['transfer_montage']} | {r['site_optimum_montage']} |" for r in worst]
        Lm.append("")
    Lm += ["Per target: `demographics_163.csv` (sex and site coefficients, p and q for log T, F and log gain; medians by sex and site); "
           "per target and held-out site: `loso_163.csv`."]
    open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8", newline="\n").write("\n".join(Lm) + "\n")
    json.dump(dict(collected=time.strftime("%Y-%m-%d %H:%M"), models=models, heads=H, sites=nsite, script="pipeline/nchc/atlas_demographics_loso.py"),
              open(os.path.join(out, "PROVENANCE.json"), "w"), indent=1)
    print("\n".join(Lm)); print(f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
