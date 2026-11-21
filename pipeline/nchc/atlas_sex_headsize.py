# -*- coding: utf-8 -*-
"""Sex effect on the delivered field, adjusted for head size, for all 163 targets of the 300-head atlas (26 Oct 2026;
PI: "adjust the sex effect for head size"). Follows atlas_demographics_loso.py (same scores, v2 rule and covariates).

Head size per head, from the files the simulations use (field-of-view independent):
  ICV       intracranial volume = white matter + grey matter + CSF in CHARM's tissue_labeling_upsampled (0.5 mm)
  R_cap     head radius at the cap = mean distance of the 10-20 electrode positions (electrodes_1020.json) from their centroid
  t_extra   extracranial thickness = R_cap - (3 ICV / 4 pi)^(1/3), i.e. scalp + skull + the gap to the cap
Per target, OLS of log TC (target field at the constrained optimum) and FC on age + sex + site (M0, as before), + log ICV
(M1), + log R_cap (M2), + log ICV + t_extra (M3). Reported: the sex coefficient and its BH-FDR q under each model, the
head-size coefficients, and the share of the M0 sex effect that the head-size terms account for.
Per-head sizes are cached outside Dropbox (C:/spicehead_v22_logs/head_size_cache.csv, h-codes); outputs are aggregates.
    python pipeline/nchc/atlas_sex_headsize.py [--models fem,sp1] [--out DIR]
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline"), HERE]
from atlas_demographics_loso import load_model, CODEMAP, COHORT, OUT, L     # noqa: E402
from collect_targets_v2 import choose, bh, EASY                              # noqa: E402

CACHE = r"C:\spicehead_v22_logs\head_size_cache.csv"


def size_one(args):
    code, cid = args
    import nibabel as nib
    from stage_heads import m2m_dir                                          # the CaseID -> folder map used for staging
    m2m = m2m_dir(cid)
    img = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    cnt = np.bincount(np.asarray(img.dataobj).ravel(), minlength=11)
    vox = float(np.prod(img.header.get_zooms()[:3])) / 1000.0                    # cm3 per voxel
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    P = np.array([ej[k] for k in sorted(ej)], float)
    return dict(code=code, icv_cm3=round(float(cnt[1:4].sum()) * vox, 2), r_cap_mm=round(float(np.linalg.norm(P - P.mean(0), axis=1).mean()), 3),
                n_electrodes=len(P))


def head_sizes(codes, cmap):
    have = {r["code"]: r for r in csv.DictReader(open(CACHE, encoding="utf-8"))} if os.path.exists(CACHE) else {}
    todo = [(c, cmap[c]) for c in codes if c not in have]
    if todo:
        with ProcessPoolExecutor(4) as ex:
            for r in ex.map(size_one, todo):
                have[r["code"]] = r
        with open(CACHE, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["code", "icv_cm3", "r_cap_mm", "n_electrodes"]); w.writeheader()
            w.writerows(have[c] for c in sorted(have))
    icv = np.array([float(have[c]["icv_cm3"]) for c in codes]); rcap = np.array([float(have[c]["r_cap_mm"]) for c in codes])
    return icv, rcap


def fit(y, X, j):
    """OLS; coefficient j, its two-sided p, and all coefficients."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; dof = len(y) - X.shape[1]
    cov = (res @ res / dof) * np.linalg.inv(X.T @ X)
    return float(beta[j]), float(2 * stats.t.sf(abs(beta[j] / np.sqrt(cov[j, j])), dof)), beta


def main():
    a = sys.argv[1:]
    models = (a[a.index("--models") + 1] if "--models" in a else "fem,sp1").split(",")
    out = a[a.index("--out") + 1] if "--out" in a else OUT
    t0 = time.time()
    codes = sorted(f.split("__")[0] for f in os.listdir(os.path.join(L, "atlas_thal", "out")) if f.endswith("__thal.npz"))
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    crow = {r["CaseID"]: r for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))}
    age = np.array([float(crow[cmap[c]]["Age_Scan"]) for c in codes])
    sex = np.array([crow[cmap[c]]["Sex_1M0F"] for c in codes]).astype(int)
    site = np.array([crow[cmap[c]]["Site"] for c in codes]); sites = sorted(set(site))
    icv, rcap = head_sizes(codes, cmap)
    textra = rcap - (3 * icv * 1000 / (4 * np.pi)) ** (1 / 3)                    # mm
    print(f"head sizes ready ({time.time()-t0:.0f} s)", flush=True)
    # cohort description (aggregates)
    desc = {}
    for lab, x in (("ICV (cm3)", icv), ("head radius at the cap (mm)", rcap), ("extracranial thickness (mm)", textra)):
        tt = stats.ttest_ind(x[sex == 1], x[sex == 0])
        desc[lab] = (float(np.median(x[sex == 1])), float(np.median(x[sex == 0])), float(tt.pvalue))
    base = [np.ones(len(codes)), age, sex] + [(site == s).astype(float) for s in sites[1:]]
    zs = lambda x: (x - x.mean()) / x.std()
    covs = {"M0 age+sex+site": [], "M1 +log ICV": [zs(np.log(icv))], "M2 +log head radius": [zs(np.log(rcap))],
            "M3 +log ICV +extracranial thickness": [zs(np.log(icv)), zs(textra)]}
    rows = []
    for m in models:
        F, T, names, tiers, nroi, mont = load_model(m, codes)
        easy = np.array([sum(e in EASY for p in mm.split(" / ") for e in p.split("+")) for mm in mont], np.float32)
        NT = len(names); ar = np.arange(NT)
        wc = np.array([choose(F[i], T[i], easy)["wc"] for i in range(len(codes))])
        FC = np.stack([F[i][ar, wc[i]] for i in range(len(codes))]); TC = np.stack([T[i][ar, wc[i]] for i in range(len(codes))])
        del F, T
        valid = nroi >= 5
        for t in range(NT):
            v = valid[:, t]
            if v.sum() < 10:
                continue
            r = dict(target=names[t], tier=tiers[t], model=m, n=int(v.sum()))
            for key, y in (("logT", np.log(TC[v, t].astype(float))), ("F", FC[v, t].astype(float))):
                for mk, extra in covs.items():
                    X = np.column_stack([c[v] for c in base] + [c[v] for c in extra])
                    b, p, beta = fit(y, X, 2)
                    tag = mk.split()[0]
                    r[f"{key}_sex_{tag}"] = round(b, 5); r[f"{key}_p_sex_{tag}"] = p
                    for k, c in enumerate(extra):                                     # standardized head-size coefficients
                        r[f"{key}_{tag}_size{k+1}"] = round(float(beta[len(base) + k]), 5)
            rows.append(r)
        print(f"  {m}: done ({time.time()-t0:.0f} s)", flush=True)
    for m in models:
        sub = [r for r in rows if r["model"] == m]
        for key in ("logT", "F"):
            for tag in ("M0", "M1", "M2", "M3"):
                for r, qq in zip(sub, bh([r[f"{key}_p_sex_{tag}"] for r in sub])):
                    r[f"{key}_q_sex_{tag}"] = float(qq)
    os.makedirs(out, exist_ok=True)
    ks = list(rows[0]); [ks.append(k) for r in rows for k in r if k not in ks]
    with open(os.path.join(out, "sex_headsize_163.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ks); w.writeheader(); w.writerows(rows)
    setname = {"tier1": "AAL90 cerebrum", "tier2": "Psychiatric (tier 2)", "ref6": "Paper A spheres", "cerebellum": "Cerebellum",
               "brainstem": "Brainstem nuclei", "thalamus": "Thalamic nuclei", "cingulate": "ACC_sup"}
    Lm = [f"# Sex effect on the delivered field, adjusted for head size: 300 heads, {len({r['target'] for r in rows})} targets", "",
          f"Collected {time.strftime('%Y-%m-%d %H:%M')} by `pipeline/nchc/atlas_sex_headsize.py`. Models: {', '.join(models)}. "
          "Outcome: log T at the head's constrained optimum (a sex coefficient of -0.10 is about 10 % less field in men). "
          "All models include age and site; BH-FDR across targets per model.", "",
          "This is a new result set. It is not in any Results record or manuscript.", "",
          "## Head size by sex (medians; t-test)", "", "| Measure | Men | Women | p |", "|---|---|---|---|"]
    Lm += [f"| {k} | {mm:.1f} | {ww:.1f} | {pp:.1e} |" for k, (mm, ww, pp) in desc.items()]
    Lm.append(f"\nCorrelation of ICV with head radius at the cap: r = {np.corrcoef(icv, rcap)[0, 1]:.2f}; electrodes per head: 19.\n")
    for m in models:
        sub = [r for r in rows if r["model"] == m]
        Lm += [f"## {m}: sex coefficient on log T (median over targets) and targets with q < 0.05", "",
               "| Set | Targets | M0 age + sex + site | M1 + log ICV | M2 + log head radius | M3 + log ICV + extracranial thickness |", "|---|---|---|---|---|---|"]
        for lab in list(setname.values()) + ["All"]:
            S = [r for r in sub if setname[r["tier"]] == lab or lab == "All"]
            if not S:
                continue
            cell = lambda tag: (f"{100*np.median([r[f'logT_sex_{tag}'] for r in S]):+.1f} % ({sum(r[f'logT_q_sex_{tag}'] < 0.05 for r in S)}"
                                f"; men lower {sum(r[f'logT_q_sex_{tag}'] < 0.05 and r[f'logT_sex_{tag}'] < 0 for r in S)})")
            Lm.append(f"| {lab} | {len(S)} | " + " | ".join(cell(t) for t in ("M0", "M1", "M2", "M3")) + " |")
        att = [1 - r["logT_sex_M3"] / r["logT_sex_M0"] for r in sub if r["logT_sex_M0"] < 0]
        Lm += ["", f"Share of the M0 sex effect on log T accounted for by ICV + extracranial thickness (M3): median {100*np.median(att):.0f} % "
               f"(IQR {100*np.percentile(att, 25):.0f}–{100*np.percentile(att, 75):.0f} %).",
               f"Standardized coefficients in M3 (per 1 SD, median over targets): log ICV {np.median([r['logT_M3_size1'] for r in sub]):+.3f}, "
               f"extracranial thickness {np.median([r['logT_M3_size2'] for r in sub]):+.3f}; M2 log head radius {np.median([r['logT_M2_size1'] for r in sub]):+.3f}.",
               f"Focality F, sex effect significant: M0 {sum(r['F_q_sex_M0'] < 0.05 for r in sub)}, M1 {sum(r['F_q_sex_M1'] < 0.05 for r in sub)}, "
               f"M2 {sum(r['F_q_sex_M2'] < 0.05 for r in sub)}, M3 {sum(r['F_q_sex_M3'] < 0.05 for r in sub)} of {len(sub)}.", ""]
    Lm.append("Per target: `sex_headsize_163.csv`.")
    open(os.path.join(out, "SUMMARY_sex_headsize.md"), "w", encoding="utf-8", newline="\n").write("\n".join(Lm) + "\n")
    print("\n".join(Lm)); print(f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
