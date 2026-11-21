# -*- coding: utf-8 -*-
"""One head-level mixed model for age, sex, site and head size across the 163-region, 300-head v23 atlas (31 Oct 2026).

A reviewer asked that the 163 separate per-region OLS regressions (pipeline/v23_atlas_review.py demographics block;
pipeline/nchc/atlas_sex_headsize.py for the head-size covariates) be replaced by a single linear mixed model with
head as random intercept and region as fixed effect:

    log y[h, r] = region[r] + b_age * age_h/10 + b_sex * men_h + site_h + size_h + u_h + e[h, r],   u_h ~ N(0, s_u^2)

Outcomes y (SPICEhead v23, 1 mm): T at the head's reported montage (T_wa), T at the region's cohort montage (T_coh; the
fixed-protocol case) and F at the reported montage (F_wa); FEM T_wa is fitted as a supplementary check. Head size
enters as in atlas_sex_headsize.py: M0 none, M1 log ICV, M2 log head radius at the cap, M3 log ICV + extracranial
thickness (each standardised to 1 SD over the 300 heads). Fits: statsmodels MixedLM, REML for the coefficients and
variance components; ML for the likelihood-ratio test of a region x sex interaction. Because every head is scored in
every region, the design is balanced and the REML/ML solution has a closed form (ANOVA estimators); it is used as the
starting value of every statsmodels fit and reported next to it as a check. Design VIFs of the head-level covariates
are reported; in the balanced design they equal the VIFs of the full row-level design, which is checked numerically.

Inputs: L:/BrainProbe/SPICEhead/nchc_fullsearch/atlas_v23_parameters_perhead.npz (per-head T and F at the reported and
constrained montages, h-codes); per-head T and F at the cohort montage are re-scored from the per-head atlas files on L:
(as v23_atlas_review.py; the sp1v23 model only, about 15 min) and cached, h-codes only, next to the parameters file
(atlas_v23_cohort_perhead.npz). Demographics come from the private files via v23_atlas_review.demographics() and head
sizes from the cache written by atlas_sex_headsize.py; nothing identifying is printed or written.

Output: results_v23/v23_atlas_review/mixed_model/ (CSV tables, SUMMARY.md, PROVENANCE.json). Below-normal priority;
the fits run in 4 single-threaded worker processes (4 threads in total).
    python pipeline/v23_mixed_model.py [--no-interaction] [--workers 4]
"""
import csv
import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, "nchc")]
from collect_targets_v2 import EASY                                  # noqa: E402
from v23_atlas_parameters import SETS, load, PERHEAD, L, CODEMAP, COHORT   # noqa: E402
from v23_atlas_review import choose, demographics, OUT as REVIEW     # noqa: E402
from v23_fullsearch_collect import low_footprint                     # noqa: E402

OUT = os.path.join(REVIEW, "mixed_model")
COHORT_CACHE = os.path.join(L, "atlas_v23_cohort_perhead.npz")
SIZE_CACHE = r"C:\spicehead_v22_logs\head_size_cache.csv"           # per head, h-codes (atlas_sex_headsize.py)
REGION_REVIEW = os.path.join(REVIEW, "region_review.csv")
SIZE_MODELS = {"M0": [], "M1": ["zlogICV"], "M2": ["zlogRcap"], "M3": ["zlogICV", "ztextra"]}
SIZE_LABEL = {"M0": "none", "M1": "+ log ICV", "M2": "+ log head radius", "M3": "+ log ICV + extracranial thickness"}
OUTCOME_LABEL = {"T_wa": "T at the head's reported montage", "T_coh": "T at the region's cohort montage (fixed protocol)",
                 "F_wa": "F at the head's reported montage"}
H, R = 300, 163


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def cohort_perhead(codes, targets, t0):
    """Per head and region: T and F of the region's cohort montage (sp1v23), cached on L: (h-codes only)."""
    if os.path.exists(COHORT_CACHE):
        with np.load(COHORT_CACHE) as z:
            assert list(z["codes"]) == list(codes) and list(z["targets"]) == list(targets)
            return z["T_coh"], z["F_coh"], list(z["montage"]), "cache"
    Tc = np.full((H, len(targets)), np.nan, np.float32)
    Fc = np.full_like(Tc, np.nan)
    mont_of = [""] * len(targets)
    seen = set()
    for sname, src in SETS:
        d, suf, name = src["sp1v23"]
        cs = sorted(f.split("__")[0] for f in os.listdir(d) if f.endswith(suf))
        assert cs == list(codes), sname
        F0, T0, names, tiers, _, mont = load(os.path.join(d, cs[0] + suf), name)
        F = np.zeros((H, len(names), len(mont)), np.float32)
        T = np.zeros_like(F)
        nroi = np.zeros((H, len(names)), np.int64)
        for hi, c in enumerate(cs):
            Fh, Th, nm, _, nr, mo = load(os.path.join(d, c + suf), name)
            assert nm == names and mo == mont, c
            F[hi], T[hi], nroi[hi] = Fh, Th, nr
        print(f"{sname} sp1v23: loaded ({time.time() - t0:.0f} s)", flush=True)
        easy = np.array([sum(e in EASY for p in mo.split(" / ") for e in p.split("+")) for mo in mont], np.float32)
        for t, nm in enumerate(names):
            key = nm.replace("ref6:", "")
            if key in seen and nm.startswith("ref6:"):
                continue
            v = nroi[:, t] >= 5
            if v.sum() < 10:
                continue                                       # VTA of the first set (no grey matter)
            assert v.all(), (nm, int(v.sum()))
            _, cc = choose(np.median(F[:, t], 0)[None], np.median(T[:, t], 0)[None], easy)
            cc = int(cc[0])
            j = list(targets).index(nm)
            Tc[:, j], Fc[:, j], mont_of[j] = T[:, t, cc], F[:, t, cc], mont[cc]
            seen.add(key)
        del F, T
    assert np.isfinite(Tc).all() and np.isfinite(Fc).all() and all(mont_of)
    np.savez_compressed(COHORT_CACHE, codes=np.array(codes), targets=np.array(targets), T_coh=Tc, F_coh=Fc,
                        montage=np.array(mont_of), model="sp1v23", rule="floor 0.8, band 0.03, tie-break on; cohort medians")
    return Tc, Fc, mont_of, "rescored"


def head_covariates(codes):
    """Age (per decade, centred), men, site dummies and standardised head-size terms per head; aggregates only leave."""
    cmap, crow = demographics()
    age = np.array([float(crow[cmap[c]]["Age_Scan"]) for c in codes])
    men = np.array([crow[cmap[c]]["Sex_1M0F"] == "1" for c in codes], float)
    site = np.array([str(crow[cmap[c]]["Site"]) for c in codes])
    sites = sorted(set(site))
    hs = {r["code"]: r for r in csv.DictReader(open(SIZE_CACHE, encoding="utf-8"))}
    icv = np.array([float(hs[c]["icv_cm3"]) for c in codes])
    rcap = np.array([float(hs[c]["r_cap_mm"]) for c in codes])
    textra = rcap - (3 * icv * 1000 / (4 * np.pi)) ** (1 / 3)        # mm, as atlas_sex_headsize.py
    assert np.isfinite(age).all() and np.isfinite(icv).all() and np.isfinite(rcap).all()
    zs = lambda x: (x - x.mean()) / x.std()
    cov = {"age_dec": age / 10 - (age / 10).mean(), "men": men}
    for s in sites[1:]:
        cov[f"site[{s}]"] = (site == s).astype(float)
    cov["zlogICV"], cov["zlogRcap"], cov["ztextra"] = zs(np.log(icv)), zs(np.log(rcap)), zs(textra)
    scale = {"age_dec": "per decade of age", "men": "men vs women",
             "zlogICV": f"per SD of log ICV (SD {np.log(icv).std():.4f}, i.e. x{np.exp(np.log(icv).std()):.3f} in ICV)",
             "zlogRcap": f"per SD of log head radius (SD {np.log(rcap).std():.4f}, i.e. x{np.exp(np.log(rcap).std()):.3f})",
             "ztextra": f"per SD of extracranial thickness (SD {textra.std():.3f} mm)"}
    desc = dict(sites={s: int((site == s).sum()) for s in sites}, men=int(men.sum()), women=int((1 - men).sum()),
                age_mean=round(float(age.mean()), 2), age_sd=round(float(age.std()), 2),
                age_min=int(age.min()), age_max=int(age.max()),
                icv_cm3_median=round(float(np.median(icv)), 1), rcap_mm_median=round(float(np.median(rcap)), 2),
                textra_mm_median=round(float(np.median(textra)), 2))
    return cov, sites, scale, desc


def vifs(X, names, groups):
    """VIF per column and the generalised VIF (Fox & Monette) per covariate group; X without the constant."""
    Xc = X - X.mean(0)
    Rm = np.corrcoef(Xc, rowvar=False)
    out = dict(zip(names, np.diag(np.linalg.inv(Rm)).tolist()))     # VIF_j = [R^-1]_jj
    gv = {}
    for g, cols in groups.items():
        idx = [names.index(c) for c in cols]
        rest = [k for k in range(X.shape[1]) if k not in idx]
        det = lambda ii: float(np.linalg.det(Rm[np.ix_(ii, ii)])) if ii else 1.0
        gvif = det(idx) * det(rest) / det(list(range(X.shape[1])))
        gv[g] = (gvif, gvif ** (1 / (2 * len(idx))), len(idx))
    return out, gv


def design(Ymat, cov, names, interaction=False):
    """Row-level design: constant, 162 region dummies, head-level covariates (and region x men), head-major rows."""
    head_idx = np.repeat(np.arange(H), R)
    reg_idx = np.tile(np.arange(R), H)
    Dreg = np.zeros((H * R, R - 1))
    Dreg[np.arange(H * R), np.clip(reg_idx - 1, 0, None)] = (reg_idx > 0)
    Xh = np.column_stack([np.ones(H)] + [cov[n] for n in names])
    cols = [np.ones(H * R), Dreg, Xh[head_idx][:, 1:]]
    xn = ["const"] + [f"region[{i}]" for i in range(1, R)] + list(names)
    if interaction:
        cols.append(Dreg * cov["men"][head_idx][:, None])
        xn += [f"region[{i}]:men" for i in range(1, R)]
    return np.log(Ymat.ravel()), np.column_stack(cols), Xh, head_idx, xn


def closed_form(y, X, Xh, reml):
    """Exact REML/ML of the balanced random-intercept model: OLS fixed effects; s_e^2 from the within-head residual SS,
    s_u^2 + s_e^2/R from the residual SS of the head means on the head-level design (ANOVA = REML/ML when positive)."""
    N = len(y)
    ybar = y.reshape(H, R).mean(1)
    bb = np.linalg.lstsq(Xh, ybar, rcond=None)[0]
    SSb = float(np.sum((ybar - Xh @ bb) ** 2))
    bf = np.linalg.lstsq(X, y, rcond=None)[0]
    SSw = float(np.sum((y - X @ bf) ** 2)) - R * SSb
    kw = X.shape[1] - Xh.shape[1]
    if reml:
        se2, tau2 = SSw / (N - H - kw), SSb / (H - Xh.shape[1])
    else:
        se2, tau2 = SSw / (N - H), SSb / H
    return bf, tau2 - se2 / R, se2


def fit_job(args):
    """One statsmodels MixedLM fit (random intercept per head) in a worker process; returns plain arrays."""
    tag, Ymat, cov, names, reml, interaction, site_names = args[:7]
    default_start = len(args) > 7 and args[7]                       # check: statsmodels' own start instead of the closed form
    if os.name == "nt":
        low_footprint()
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLMParams
    t = time.time()
    y, X, Xh, groups, xn = design(Ymat, cov, names, interaction)
    bf, su2, se2 = closed_form(y, X, Xh, reml)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = sm.MixedLM(y, X, groups=groups)
        start = None if default_start else MixedLMParams.from_components(fe_params=bf, cov_re=np.array([[su2 / se2]]))
        res = md.fit(start_params=start, reml=reml, method=["lbfgs", "powell"], maxiter=2000)
    k = len(xn)
    idx = [xn.index(s) for s in site_names]
    Cm = np.zeros((len(idx), len(res.params)))
    Cm[np.arange(len(idx)), idx] = 1
    wt = res.wald_test(Cm, scalar=True)
    return dict(tag=tag, xn=xn, fe=np.asarray(res.fe_params)[:k], se=np.asarray(res.bse_fe)[:k],
                ci=np.asarray(res.conf_int())[:k], p=np.asarray(res.pvalues)[:k],
                var_u=float(np.asarray(res.cov_re).ravel()[0]), var_e=float(res.scale), llf=float(res.llf),
                converged=bool(res.converged), reml=reml, n=len(y), k_fe=k, seconds=time.time() - t,
                site_chi2=float(np.asarray(wt.statistic)), site_p=float(np.asarray(wt.pvalue)), site_df=len(idx),
                closed=dict(fe=bf[:k], var_u=su2, var_e=se2))


def main():
    if os.name == "nt":
        low_footprint()
    a = sys.argv[1:]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 4
    do_inter = "--no-interaction" not in a
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    import statsmodels

    with np.load(PERHEAD) as z:
        codes, targets = [str(c) for c in z["codes"]], [str(t) for t in z["targets"]]
        PH = {k: z[k].astype(float) for k in z.files if "__" in k}
    assert len(codes) == H and len(targets) == R
    Tcoh, Fcoh, mont_coh, how = cohort_perhead(codes, targets, t0)
    Tcoh, Fcoh = Tcoh.astype(float), Fcoh.astype(float)
    print(f"cohort T/F per head: {how} ({time.time() - t0:.0f} s)", flush=True)

    # check the re-scored cohort values and the per-head file against the review table (medians per region)
    rr = {r["target"]: r for r in csv.DictReader(open(REGION_REVIEW, encoding="utf-8"))}
    assert set(rr) == set(targets), "region set differs from region_review.csv"
    chk = {}
    for lab, arr, col in (("T_coh", Tcoh, "sp1v23__T_cohort_median"), ("F_coh", Fcoh, "sp1v23__F_cohort_median"),
                          ("T_wa", PH["sp1v23__T_wa"], "sp1v23__T_reported_median"), ("F_wa", PH["sp1v23__F_wa"], "sp1v23__F_reported_median"),
                          ("fem_T_wa", PH["fem__T_wa"], "fem__T_reported_median")):
        d = np.array([np.median(arr[:, j]) - float(rr[t][col]) for j, t in enumerate(targets)])
        chk[lab] = float(np.abs(d).max())
    assert all(v < 2e-4 for v in chk.values()), chk
    print("medians agree with region_review.csv:", {k: f"{v:.1e}" for k, v in chk.items()}, flush=True)

    cov, sites, scale, desc = head_covariates(codes)
    base = ["age_dec", "men"] + [f"site[{s}]" for s in sites[1:]]
    site_names = [f"site[{s}]" for s in sites[1:]]
    outcomes = {("T_wa", "sp1v23"): PH["sp1v23__T_wa"], ("T_coh", "sp1v23"): Tcoh, ("F_wa", "sp1v23"): PH["sp1v23__F_wa"],
                ("T_wa", "fem"): PH["fem__T_wa"]}
    for Y in outcomes.values():
        assert Y.shape == (H, R) and np.isfinite(Y).all() and (Y > 0).all()

    # design VIFs: head level, and the full row-level design (with the region dummies) for M0 and M3
    vrows = []
    for mk, extra in SIZE_MODELS.items():
        names = base + extra
        Xh = np.column_stack([cov[n] for n in names])
        grp = {"age": ["age_dec"], "sex": ["men"], "site": site_names}
        if extra:
            grp["size"] = extra
        v, gv = vifs(Xh, names, grp)
        full = {}
        if mk in ("M0", "M3"):
            _, Xr, _, _, xn = design(outcomes[("T_wa", "sp1v23")], cov, names)
            fv, _ = vifs(Xr[:, 1:], xn[1:], {})
            full = {n: fv[n] for n in names}
        for n in names:
            vrows.append(dict(size_model=mk, term=n, vif=round(v[n], 3), gvif_scaled="",
                              vif_full_rowlevel_design=round(full[n], 3) if n in full else ""))
        for g, (g1, g2, df) in gv.items():
            vrows.append(dict(size_model=mk, term=f"GVIF[{g}] (df {df})", vif=round(g1, 3), gvif_scaled=round(g2, 3), vif_full_rowlevel_design=""))
    with open(os.path.join(OUT, "vif.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["size_model", "term", "vif", "gvif_scaled", "vif_full_rowlevel_design"]); w.writeheader(); w.writerows(vrows)
    print(f"VIFs done ({time.time() - t0:.0f} s)", flush=True)

    # the fits: REML per outcome x size model; ML main and ML region x sex for the three SPICEhead outcomes
    jobs = [((oc, m, mk, "REML"), Y, cov, base + SIZE_MODELS[mk], True, False, site_names)
            for (oc, m), Y in outcomes.items() for mk in SIZE_MODELS]
    if do_inter:
        for oc in ("T_wa", "T_coh", "F_wa"):
            jobs.append(((oc, "sp1v23", "M0", "ML"), outcomes[(oc, "sp1v23")], cov, base, False, False, site_names))
            jobs.append(((oc, "sp1v23", "M0", "ML x sex"), outcomes[(oc, "sp1v23")], cov, base, False, True, site_names))
    if "--no-default-start-check" not in a:                        # independent check: statsmodels from its own default start
        jobs.insert(0, (("T_wa", "sp1v23", "M0", "REML default-start check"), outcomes[("T_wa", "sp1v23")], cov, base, True, False, site_names, True))
        jobs.insert(1, (("T_wa", "sp1v23", "M0", "ML default-start check"), outcomes[("T_wa", "sp1v23")], cov, base, False, False, site_names, True))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"                                            # workers: one thread each
    results = {}
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(fit_job, jobs):
            results[r["tag"]] = r
            print(f"{' '.join(r['tag'][:3])} {r['tag'][3]}: {r['seconds']:.0f} s, converged {r['converged']}, "
                  f"sd_head {np.sqrt(r['var_u']):.4f} sd_resid {np.sqrt(r['var_e']):.4f} ({time.time() - t0:.0f} s)", flush=True)

    dchk = {}
    for kind in ("REML", "ML"):
        k0, k1 = ("T_wa", "sp1v23", "M0", kind), ("T_wa", "sp1v23", "M0", f"{kind} default-start check")
        if k1 in results:
            r0, r1 = results[k0], results[k1]
            dchk[kind] = dict(seconds=round(r1["seconds"], 1), converged=r1["converged"], llf_default_start=round(r1["llf"], 5),
                              llf_closed_form_start=round(r0["llf"], 5),
                              max_abs_diff_fixed_effects=float(np.abs(r1["fe"] - r0["fe"]).max()),
                              rel_diff_var_head=float(r1["var_u"] / r0["var_u"] - 1), rel_diff_var_resid=float(r1["var_e"] / r0["var_e"] - 1))
    fe_rows, vc_rows, reg_rows = [], [], []
    for (oc, model, mk, kind), r in results.items():
        if kind != "REML":
            continue
        xn, b, se, ci, p = r["xn"], r["fe"], r["se"], r["ci"], r["p"]
        cf = r["closed"]
        vc_rows.append(dict(outcome=oc, model=model, size_model=mk, n_rows=r["n"], n_heads=H, n_regions=R, fixed_effects=r["k_fe"],
                            sd_head=round(float(np.sqrt(r["var_u"])), 5), sd_resid=round(float(np.sqrt(r["var_e"])), 5),
                            icc=round(r["var_u"] / (r["var_u"] + r["var_e"]), 4), loglik_reml=round(r["llf"], 2),
                            converged=r["converged"], method="REML (statsmodels MixedLM, lbfgs from the closed-form start)",
                            sd_head_closed_form=round(float(np.sqrt(cf["var_u"])), 5), sd_resid_closed_form=round(float(np.sqrt(cf["var_e"])), 5),
                            max_abs_diff_fixed_effects_vs_closed_form=float(np.abs(b - cf["fe"]).max()),
                            max_rel_diff_variances_vs_closed_form=float(max(abs(r["var_u"] / cf["var_u"] - 1), abs(r["var_e"] / cf["var_e"] - 1))),
                            seconds=round(r["seconds"], 1)))
        for k, n in enumerate(xn):
            if n.startswith("region["):
                if mk == "M0":
                    reg_rows.append(dict(outcome=oc, model=model, region=targets[int(n[7:-1])], coef_log=round(float(b[k]), 5),
                                         pct_vs_reference=round(100 * (np.exp(b[k]) - 1), 2),
                                         ci_lo_pct=round(100 * (np.exp(ci[k, 0]) - 1), 2), ci_hi_pct=round(100 * (np.exp(ci[k, 1]) - 1), 2)))
                continue
            fe_rows.append(dict(outcome=oc, model=model, size_model=mk, term=n, coef_log=round(float(b[k]), 6), se=round(float(se[k]), 6),
                                pct_change=round(100 * (np.exp(b[k]) - 1), 3), ci_lo_pct=round(100 * (np.exp(ci[k, 0]) - 1), 3),
                                ci_hi_pct=round(100 * (np.exp(ci[k, 1]) - 1), 3), p=float(p[k]),
                                scale=scale.get(n, "vs reference site" if n.startswith("site") else "")))
        fe_rows.append(dict(outcome=oc, model=model, size_model=mk, term="site (joint Wald)", coef_log="", se="", pct_change="",
                            ci_lo_pct="", ci_hi_pct="", p=r["site_p"], scale=f"chi2 {r['site_chi2']:.2f}, df {r['site_df']}"))
    for name, rows in (("fixed_effects.csv", fe_rows), ("variance_components.csv", vc_rows), ("region_effects_M0.csv", reg_rows)):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)

    # likelihood-ratio test of region x sex (ML fits)
    from scipy.stats import chi2
    inter_rows = []
    if do_inter:
        for oc in ("T_wa", "T_coh", "F_wa"):
            r0, r1 = results[(oc, "sp1v23", "M0", "ML")], results[(oc, "sp1v23", "M0", "ML x sex")]
            lr, df = 2 * (r1["llf"] - r0["llf"]), R - 1
            bi = r1["fe"][-df:]
            sex_main = float(r1["fe"][r1["xn"].index("men")])
            inter_rows.append(dict(outcome=oc, model="sp1v23", size_model="M0", status="fitted", loglik_main_ML=round(r0["llf"], 2),
                                   loglik_interaction_ML=round(r1["llf"], 2), lr_chi2=round(float(lr), 2), df=df, p=float(chi2.sf(lr, df)),
                                   converged_main=r0["converged"], converged_interaction=r1["converged"],
                                   sex_effect_reference_region_pct=round(100 * (np.exp(sex_main) - 1), 2),
                                   region_sex_deviation_sd_log=round(float(np.std(np.concatenate([[0.0], bi]))), 5),
                                   region_sex_effect_min_pct=round(100 * (np.exp(sex_main + min(0.0, bi.min())) - 1), 2),
                                   region_sex_effect_max_pct=round(100 * (np.exp(sex_main + max(0.0, bi.max())) - 1), 2),
                                   seconds=round(r0["seconds"] + r1["seconds"], 1)))
    else:
        inter_rows.append(dict(outcome="all", model="sp1v23", size_model="M0", status="skipped: --no-interaction"))
    ks = []
    [ks.append(k) for r in inter_rows for k in r if k not in ks]
    with open(os.path.join(OUT, "interaction_region_x_sex.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ks); w.writeheader(); w.writerows(inter_rows)

    # comparison with the per-region OLS approach (M0 and M3): per-region slopes vs the mixed-model estimate and CI
    from scipy.stats import t as tdist
    cmp_rows, cmp_sum = [], []
    for (oc, model), Y in outcomes.items():
        for mk in ("M0", "M3"):
            names = base + SIZE_MODELS[mk]
            Xh = np.column_stack([np.ones(H)] + [cov[n] for n in names])
            mixed = {r["term"]: r for r in fe_rows if r["outcome"] == oc and r["model"] == model and r["size_model"] == mk}
            B = np.linalg.lstsq(Xh, np.log(Y), rcond=None)[0]                 # (p, R): all 163 regressions at once
            E = np.log(Y) - Xh @ B
            dof = H - Xh.shape[1]
            SE = np.sqrt(np.outer(np.diag(np.linalg.inv(Xh.T @ Xh)), (E ** 2).sum(0) / dof))
            P = 2 * tdist.sf(np.abs(B / SE), dof)
            for term in ("age_dec", "men") + tuple(SIZE_MODELS[mk]):
                j = 1 + names.index(term)
                pct = 100 * (np.exp(B[j]) - 1)
                inside = (pct >= mixed[term]["ci_lo_pct"]) & (pct <= mixed[term]["ci_hi_pct"])
                for ri, t in enumerate(targets):
                    cmp_rows.append(dict(outcome=oc, model=model, size_model=mk, term=term, region=t, ols_pct=round(float(pct[ri]), 3),
                                         ols_p=float(P[j, ri]), inside_mixed_ci=bool(inside[ri])))
                cmp_sum.append(dict(outcome=oc, model=model, size_model=mk, term=term, mixed_pct=mixed[term]["pct_change"],
                                    mixed_ci_lo_pct=mixed[term]["ci_lo_pct"], mixed_ci_hi_pct=mixed[term]["ci_hi_pct"],
                                    ols_mean_of_log_slopes_pct=round(100 * (np.exp(B[j].mean()) - 1), 3),
                                    ols_median_pct=round(float(np.median(pct)), 3), ols_min_pct=round(float(pct.min()), 2),
                                    ols_max_pct=round(float(pct.max()), 2), ols_sd_of_log_slopes=round(float(B[j].std()), 5),
                                    regions_inside_mixed_ci=int(inside.sum()), regions_ols_p_lt_0_05_unadjusted=int((P[j] < 0.05).sum()),
                                    regions=R))
    mx = {r["term"]: r for r in fe_rows if (r["outcome"], r["model"], r["size_model"]) == ("T_coh", "sp1v23", "M0")}
    rev_age = np.array([float(rr[t]["cohortT_age_pct_per_decade"]) for t in targets])
    rev_sex = np.array([float(rr[t]["cohortT_sex_pct_men"]) for t in targets])
    rev_in = dict(age=int(np.sum((rev_age >= mx["age_dec"]["ci_lo_pct"]) & (rev_age <= mx["age_dec"]["ci_hi_pct"]))),
                  sex=int(np.sum((rev_sex >= mx["men"]["ci_lo_pct"]) & (rev_sex <= mx["men"]["ci_hi_pct"]))),
                  age_q05=int(np.sum(np.array([float(rr[t]["cohortT_age_q"]) for t in targets]) < 0.05)),
                  sex_q05=int(np.sum(np.array([float(rr[t]["cohortT_sex_q"]) for t in targets]) < 0.05)),
                  age_median=round(float(np.median(rev_age)), 2), sex_median=round(float(np.median(rev_sex)), 2),
                  age_min=round(float(rev_age.min()), 2), age_max=round(float(rev_age.max()), 2),
                  sex_min=round(float(rev_sex.min()), 2), sex_max=round(float(rev_sex.max()), 2))
    for name, rows in (("per_region_vs_mixed.csv", cmp_rows), ("per_region_vs_mixed_summary.csv", cmp_sum)):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "cohort_montages.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["region", "cohort_montage_sp1v23", "T_coh_median", "F_coh_median"])
        w.writerows([t, m, round(float(np.median(Tcoh[:, j])), 5), round(float(np.median(Fcoh[:, j])), 4)]
                    for j, (t, m) in enumerate(zip(targets, mont_coh)))

    # SUMMARY.md
    g = lambda oc, mk, term, model="sp1v23": next(r for r in fe_rows if (r["outcome"], r["model"], r["size_model"], r["term"]) == (oc, model, mk, term))
    vc = lambda oc, mk, model="sp1v23": next(r for r in vc_rows if (r["outcome"], r["model"], r["size_model"]) == (oc, model, mk))
    pf = lambda p, d=1: f"= {p:.{d}e}" if p > 0 else "< 1e-300 (underflow)"
    fmt = lambda r: (f"{r['pct_change']:+.2f} % ({r['ci_lo_pct']:+.2f} to {r['ci_hi_pct']:+.2f}; p {pf(r['p'])})" if r["pct_change"] != ""
                     else f"p {pf(r['p'])} ({r['scale']})")
    dtxt = " ".join(f"Refitting T_wa M0 ({k}) from statsmodels' default start instead gives the same fixed effects to {d['max_abs_diff_fixed_effects']:.0e} and "
                    f"variances within {max(abs(d['rel_diff_var_head']), abs(d['rel_diff_var_resid'])):.0e} (relative), log-likelihood {d['llf_default_start']} vs "
                    f"{d['llf_closed_form_start']}, in {d['seconds']:.0f} s." for k, d in dchk.items()) or "The default-start check was not run."
    Lm = ["# Head-level mixed model of age, sex, site and head size: 300 heads x 163 regions (SPICEhead v23)", "",
          f"Built {time.strftime('%Y-%m-%d %H:%M')} by `pipeline/v23_mixed_model.py` (statsmodels {statsmodels.__version__} MixedLM). "
          "Answers the reviewer's request for one head-level mixed model in place of the 163 separate per-region OLS regressions of "
          "`pipeline/v23_atlas_review.py` and `pipeline/nchc/atlas_sex_headsize.py`. This is a new result set; it is not in any Results "
          "record or manuscript.", "",
          "## Model", "",
          f"log y[h, r] = region[r] + b_age (age_h / 10, centred) + b_sex men_h + site_h (3 dummies vs {sites[0]}) + size_h + u_h + e[h, r]; "
          "u_h ~ N(0, sd_head^2) is the random intercept of head h and e ~ N(0, sd_resid^2) the residual. Region is a fixed effect (162 dummies vs "
          f"{targets[0]}). Head size: M0 none; M1 log ICV; M2 log head radius at the cap; M3 log ICV + extracranial thickness, each standardised "
          "to 1 SD over the 300 heads (ICV = WM + GM + CSF of CHARM's upsampled labels; head radius = mean distance of the 19 10-20 electrodes from "
          "their centroid; extracranial thickness = head radius - (3 ICV / 4 pi)^(1/3); all as pipeline/nchc/atlas_sex_headsize.py). Coefficients "
          "are REML estimates reported as % change = 100 (exp(b) - 1) with Wald 95 % CI and p; site is also tested jointly (Wald, 3 df). The "
          f"region x sex test compares ML fits. Rows: {H * R} = 300 heads x 163 regions, no missing values; ICC = sd_head^2 / (sd_head^2 + sd_resid^2). "
          "Outcomes (SPICEhead v23, 1 mm): " + "; ".join(f"{k} = {v}" for k, v in OUTCOME_LABEL.items()) + "; FEM T_wa is a supplementary check. "
          "The cohort montage of a region is the approved rule (floor 0.8, band 3 %, tie-break) applied to the cohort medians, re-scored from the "
          f"per-head files on L: ({how}; regional medians agree with region_review.csv to {max(chk.values()):.1e}). Because the design is "
          "balanced, REML/ML has a closed form (ANOVA estimators); statsmodels is started from it and the two agree to "
          f"{max(v['max_rel_diff_variances_vs_closed_form'] for v in vc_rows):.0e} (relative) in the variance components and "
          f"{max(v['max_abs_diff_fixed_effects_vs_closed_form'] for v in vc_rows):.0e} in the fixed effects (log scale). " + dtxt, "",
          f"Cohort: {desc['men']} men, {desc['women']} women; age {desc['age_mean']} +/- {desc['age_sd']} y ({desc['age_min']}-{desc['age_max']}); sites " +
          ", ".join(f"{k} {v}" for k, v in desc["sites"].items()) + f"; median ICV {desc['icv_cm3_median']} cm3, head radius {desc['rcap_mm_median']} mm, "
          f"extracranial thickness {desc['textra_mm_median']} mm.", ""]
    for oc in ("T_wa", "T_coh", "F_wa"):
        Lm += [f"## {OUTCOME_LABEL[oc]} (log {oc.split('_')[0]}, SPICEhead v23)", "",
               "| Size model | Age, per decade | Men vs women | Site (joint) | Size term(s), per SD | sd_head | sd_resid | ICC |", "|---|---|---|---|---|---|---|---|"]
        for mk, extra in SIZE_MODELS.items():
            sz = "; ".join(f"{n}: {fmt(g(oc, mk, n))}" for n in extra) or "-"
            v = vc(oc, mk)
            Lm.append(f"| {mk} ({SIZE_LABEL[mk]}) | {fmt(g(oc, mk, 'age_dec'))} | {fmt(g(oc, mk, 'men'))} | {fmt(g(oc, mk, 'site (joint Wald)'))} | {sz} | "
                      f"{v['sd_head']:.5f} | {v['sd_resid']:.5f} | {v['icc']:.4f} |")
        Lm += ["", f"Site dummies vs {sites[0]} (M0): " + "; ".join(f"{s} {fmt(g(oc, 'M0', f'site[{s}]'))}" for s in sites[1:]) + ".", ""]
    Lm += ["## FEM, T at the reported montage (supplementary)", "", "| Size model | Age, per decade | Men vs women | Site (joint) | Size term(s), per SD | ICC |", "|---|---|---|---|---|---|"]
    for mk, extra in SIZE_MODELS.items():
        sz = "; ".join(f"{n}: {fmt(g('T_wa', mk, n, 'fem'))}" for n in extra) or "-"
        Lm.append(f"| {mk} | {fmt(g('T_wa', mk, 'age_dec', 'fem'))} | {fmt(g('T_wa', mk, 'men', 'fem'))} | {fmt(g('T_wa', mk, 'site (joint Wald)', 'fem'))} | {sz} | {vc('T_wa', mk, 'fem')['icc']} |")
    Lm += ["", "## Region x sex interaction (likelihood ratio of adding region x sex to M0; ML fits)", ""]
    for r in inter_rows:
        if r["status"] == "fitted":
            Lm.append(f"- {r['outcome']}: chi2 = {r['lr_chi2']} on {r['df']} df, p {pf(r['p'], 2)} (log-likelihood {r['loglik_main_ML']} -> {r['loglik_interaction_ML']}); "
                      f"region-specific sex effects range from {r['region_sex_effect_min_pct']:+.2f} % to {r['region_sex_effect_max_pct']:+.2f} % "
                      f"(reference region {r['sex_effect_reference_region_pct']:+.2f} %; SD of the regional log deviations {r['region_sex_deviation_sd_log']}); "
                      f"converged {r['converged_main']}/{r['converged_interaction']}; {r['seconds']} s.")
        else:
            Lm.append(f"- {r['outcome']}: {r['status']}.")
    Lm += ["", "## Design VIFs (head-level covariates; identical in the full row-level design with the region dummies, checked for M0 and M3)", "",
           "| Size model | " + " | ".join(base) + " | size terms | GVIF^(1/2df) site |", "|---|" + "---|" * (len(base) + 2)]
    for mk, extra in SIZE_MODELS.items():
        vv = {r["term"]: r for r in vrows if r["size_model"] == mk}
        Lm.append(f"| {mk} | " + " | ".join(f"{vv[n]['vif']:.2f}" for n in base) + " | " + ("; ".join(f"{n} {vv[n]['vif']:.2f}" for n in extra) or "-") +
                  f" | {vv['GVIF[site] (df 3)']['gvif_scaled']:.2f} |")
    Lm += ["", "## Comparison with the per-region OLS approach", ""]
    cs = lambda oc, mk, term, model="sp1v23": next(r for r in cmp_sum if (r["outcome"], r["model"], r["size_model"], r["term"]) == (oc, model, mk, term))
    for oc in ("T_wa", "T_coh", "F_wa"):
        for mk in ("M0", "M3"):
            parts = []
            for term in ("age_dec", "men") + tuple(SIZE_MODELS[mk]):
                c = cs(oc, mk, term)
                parts.append(f"{term}: mixed {c['mixed_pct']:+.2f} % = mean of the 163 regional log-slopes ({c['ols_mean_of_log_slopes_pct']:+.2f} %); "
                             f"regional range {c['ols_min_pct']:+.2f} to {c['ols_max_pct']:+.2f} %, {c['regions_inside_mixed_ci']}/163 inside the mixed-model CI, "
                             f"{c['regions_ols_p_lt_0_05_unadjusted']}/163 with regional p < 0.05 (unadjusted)")
            Lm.append(f"- {oc} {mk}: " + "; ".join(parts) + ".")
    Lm += ["", f"Existing review table (region_review.csv, cohort T, M0, per-region OLS with BH-FDR): regional age slopes median {rev_in['age_median']:+.2f} % per decade "
           f"(range {rev_in['age_min']:+.2f} to {rev_in['age_max']:+.2f}; {rev_in['age_q05']}/163 with q < 0.05), regional sex effects median {rev_in['sex_median']:+.2f} % "
           f"(range {rev_in['sex_min']:+.2f} to {rev_in['sex_max']:+.2f}; {rev_in['sex_q05']}/163 with q < 0.05). Of these, {rev_in['age']}/163 age slopes and "
           f"{rev_in['sex']}/163 sex effects lie inside the mixed-model 95 % CI (age {mx['age_dec']['ci_lo_pct']:+.2f} to {mx['age_dec']['ci_hi_pct']:+.2f} %; "
           f"sex {mx['men']['ci_lo_pct']:+.2f} to {mx['men']['ci_hi_pct']:+.2f} %).", "",
           "Reading. With every head scored in every region, the mixed model's head-level coefficient is exactly the average of the 163 regional "
           "OLS slopes (the GLS estimate of a head-level covariate in a balanced random-intercept design is OLS on the head means), so the two approaches "
           "cannot disagree on the pooled effect; what the mixed model adds is one CI for that pooled effect that accounts for the head-level correlation "
           "of the 163 outcomes, one between-head variance, and a formal test of regional heterogeneity. Its CI is the uncertainty of the average, not a "
           "range for regions, so regional slopes outside it reflect regional heterogeneity (quantified by the region x sex LR test and the regional range), "
           "not a conflict between the approaches. The per-region table answers 'does region r show the effect' (with FDR control); the mixed model answers "
           "'what is the common head-level effect after pooling the 163 regions'.", "",
           "Files: fixed_effects.csv, variance_components.csv (with the closed-form check), region_effects_M0.csv, interaction_region_x_sex.csv, vif.csv, "
           "per_region_vs_mixed.csv, per_region_vs_mixed_summary.csv, cohort_montages.csv, PROVENANCE.json. "
           f"Total {time.time() - t0:.0f} s."]
    open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8", newline="\n").write("\n".join(Lm) + "\n")

    inputs = {os.path.relpath(p, ROOT) if p.startswith(ROOT) else p: sha256(p) for p in (PERHEAD, COHORT_CACHE, SIZE_CACHE, REGION_REVIEW)}
    inputs.update({"private head-code map (read locally only)": sha256(CODEMAP), "private cohort table (read locally only)": sha256(COHORT)})
    json.dump(dict(script="pipeline/v23_mixed_model.py", date=time.strftime("%Y-%m-%d %H:%M"), statsmodels=statsmodels.__version__,
                   numpy=np.__version__, python=sys.version.split()[0], rows=H * R, heads=H, regions=R, region_reference=targets[0],
                   site_reference=sites[0], size_models=SIZE_LABEL, outcomes={f"{oc} ({m})": OUTCOME_LABEL[oc] for oc, m in outcomes},
                   cohort_perhead=how, cohort_rescore_source=f"{L} (atlas_v23/out, atlas_ext/out, atlas_thal/out; sp1v23; h-codes only)",
                   median_check_vs_region_review_max_abs=chk, inputs_sha256=inputs, workers=workers, default_start_check=dchk,
                   fits={" ".join(k): dict(seconds=round(r["seconds"], 1), converged=r["converged"], llf=round(r["llf"], 3)) for k, r in results.items()},
                   seconds=round(time.time() - t0)),
              open(os.path.join(OUT, "PROVENANCE.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(Lm))
    print(f"wrote {OUT} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
