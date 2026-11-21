# -*- coding: utf-8 -*-
"""Every number in the Rev0.4 manuscript package (30 Oct 2026), computed from result files.

The PI chose Rev0.3 as the base and asked on 30 Oct 2026 for an extensive v23 update that uses the 300 heads only and
is framed around region-wise targeting parameters with optimal montage selection (not computational speed). Sources,
all v23 or model-independent:
  values_v23            analytic disk and layered slab; the 300-head six-target FEM decision comparison (unconstrained F)
  fields_perhead.csv    per-head field agreement with FEM, 300 heads (results_v23/v23_fem_fullsearch_300: the
                        homogenized network of 23 Oct, identical to the packaged v23; results_v23/v23_package_check;
                        rows copied by pipeline/v23_fields_collect.py)
  atlas summaries       results_v23/v23_target_atlas_300{,_ext,_thal}/target_summary_ext.csv (163 targets; sp1v23, fem)
  parameters            results_v23/v23_atlas_parameters/target_parameters.csv (pipeline/v23_atlas_parameters.py)
  demographics          results_v23/v23_target_atlas_300/demographics_loso_163/ (sex, site, LOSO, head size)

  SPICEHEAD_MODEL=v23 python pipeline/rev04_numbers.py     prints the numbers and writes evidence JSON
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R23 = os.path.join(ROOT, "results_v23")
SUMMARIES = [os.path.join(R23, d, "target_summary_ext.csv")
             for d in ("v23_target_atlas_300", "v23_target_atlas_300_ext", "v23_target_atlas_300_thal")]
PARAMS = os.path.join(R23, "v23_atlas_parameters", "target_parameters.csv")
DEMO = os.path.join(R23, "v23_target_atlas_300", "demographics_loso_163")
FIELDS = os.path.join(R23, "v23_fem_fullsearch_300", "fields_perhead.csv")
REVIEW = os.path.join(R23, "v23_atlas_review", "region_review.csv")   # review round of 31 Oct 2026 (v23_atlas_review.py)
# 2 Nov 2026, decision D3: the uncertainty analysis in all 300 heads (pipeline/nchc/uq_v23_atlas_collect.py); the
# thalamic file repeats the six spheres, so the brain-region file is read first and wins
UQ300 = [os.path.join(R23, "v23_uq", d, "per_target.csv") for d in ("atlas_300", "atlas_thal_300")]
SETS = [("tier1", "Cerebral (AAL)"), ("tier2", "Psychiatric"), ("ref6", "Reference spheres"),
        ("cerebellum", "Cerebellum"), ("brainstem", "Brainstem nuclei"), ("thalamus", "Thalamic nuclei"),
        ("cingulate", "Supracallosal ACC")]
ROI6 = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def atlas():
    """Per target and model, the atlas summary row merged with the parameter row (one row per target; the six
    spheres once)."""
    A = {}
    for p in SUMMARIES:
        for r in rows(p):
            A.setdefault((r["target"], r["model"]), r)
    P = {(r["target"], r["model"]): r for r in rows(PARAMS)} if os.path.exists(PARAMS) else {}
    D = {(r["target"], r["model"]): r for r in rows(os.path.join(DEMO, "demographics_163.csv"))}
    S = {(r["target"], r["model"]): r for r in rows(os.path.join(DEMO, "sex_headsize_163.csv"))}
    RV = {r["target"]: r for r in rows(REVIEW)} if os.path.exists(REVIEW) else {}
    # 31 Oct 2026, decisions D5/D6 (pipeline/v23_site_size_review.py): per-site medians and the size check
    SS = os.path.join(os.path.dirname(REVIEW), "site_size")
    PS = {r["target"]: r for r in rows(os.path.join(SS, "per_site_T.csv"))} if os.path.exists(os.path.join(SS, "per_site_T.csv")) else {}
    SZ = {r["target"]: r for r in rows(os.path.join(SS, "size_threshold.csv"))} if os.path.exists(os.path.join(SS, "size_threshold.csv")) else {}
    UQ = {}
    for p in UQ300:
        if os.path.exists(p):
            for r in rows(p):
                UQ.setdefault(r["target"], r)
    lo = {}
    for r in rows(os.path.join(DEMO, "loso_163.csv")):
        lo.setdefault((r["target"], r["model"]), []).append(r)
    out = {}
    for k, r in A.items():
        x = dict(r)
        x.update({f"p_{a}": b for a, b in P.get(k, {}).items()})
        x.update({f"d_{a}": b for a, b in D.get(k, {}).items()})
        x.update({f"s_{a}": b for a, b in S.get(k, {}).items()})
        L = lo.get(k, [])
        x["loso_max"] = max((f(y["transfer_loss_pct"]) for y in L), default=float("nan"))
        x["loso_changed"] = sum(y["transfer_is_global"] != "True" for y in L)
        x["loso_n"] = len(L)
        rv = RV.get(k[0])
        if rv is not None:
            x.update({f"rv_{a}": b for a, b in rv.items()})
            if k[1] in ("sp1v23", "fem"):
                # the review found the summaries' share was taken at the constrained optimum; report it at the
                # reported montage, as the tables say, and keep the other under its own name
                x["heads_F_above_1_constrained_pct"] = x["heads_F_above_1_pct"]
                x["heads_F_above_1_pct"] = rv[f"{k[1]}__F_above_1_reported_pct"]
        if k[0] in PS:
            x.update({f"ps_{a}": b for a, b in PS[k[0]].items()})
        if k[0] in SZ:
            x.update({f"sz_{a}": b for a, b in SZ[k[0]].items()})
            x["sz_small_flag"] = "yes" if f(SZ[k[0]]["voxels_median"]) < 60 else ""
        if k[0] in UQ:
            x.update({f"uq_{a}": b for a, b in UQ[k[0]].items()})
        out[k] = x
    return out


def nice(t):
    """Readable region name."""
    t = t.replace("ref6:", "").replace("brainstem:", "")
    special = {"M1_handknob": "M1 hand knob (sphere)", "thalamus": "Thalamus (sphere)", "insula": "Insula (sphere)",
               "ACC": "Anterior cingulate (sphere)", "hippocampus_L": "Hippocampus L (sphere)",
               "hippocampus_R": "Hippocampus R (sphere)", "DLPFC_L_Fox2012_optimized": "DLPFC L, Fox optimized",
               "DLPFC_L_Fox2012_sgACC_anticorr": "DLPFC L, Fox sgACC-anticorrelated",
               "DLPFC_L_5cm_site_Fox2012": "DLPFC L, 5 cm rule (Fox)"}
    if t in special:
        return special[t]
    return t.replace("_", " ")


def compute():
    sys.path.insert(0, HERE)
    import values_v23 as VV
    V, _ = VV.values()
    assert not V["missing"], V["missing"]
    N = {}
    # analytic verification
    for k in ("disk.nmae_5mm", "disk.nmae_1mm", "disk.slope_1mm", "slab.v23_abserr_max", "slab.thin.v22_err_lo",
              "slab.thin.v22_err_hi", "slab.aligned.v22_abserr_max"):
        N[k] = V[k]
    # six reference targets, unconstrained F, 300 heads
    for k in ("fs.agree_fem_sp1", "fs.agree_fem_sp2", "fs.agree_sp2_sp1", "fs.n_fem_sp1", "fs.rho_fem_sp1.median",
              "fs.rho_fem_sp2.median", "fs.keep_sp1_under_fem.median", "fs.keep_sp1_under_fem.p5", "fs.keep_sp1_under_fem.min",
              "fs.keep_sp2_under_fem.median", "fs.keep_sp2_under_fem.p5", "fs.low90_sp1_under_fem", "fs.low90_sp2_under_fem",
              "fs.coh.agree_fem_sp1", "fs.coh.keep_sp1_under_fem.min", "fs.low90_rows", "fs.dis_keep_sp1_under_fem.median"):
        N[k] = V[k]
    dec = rows(os.path.join(R23, "v23_fem_fullsearch_300", "fullsearch_decisions.csv"))
    low = [100 - f(r["regret_sp1winner_under_fem_pct"]) for r in dec if f(r["regret_sp1winner_under_fem_pct"]) > 10]
    lowM1 = [100 - f(r["regret_sp1winner_under_fem_pct"]) for r in dec
             if f(r["regret_sp1winner_under_fem_pct"]) > 10 and r["roi"] == "M1_handknob"]
    N["six.low_M1_n"], N["six.low_M1_lo"], N["six.low_M1_hi"] = len(lowM1), min(lowM1), max(lowM1)
    N["six.low_other"] = sorted(set(low) - set(lowM1))
    six = []
    for roi in ROI6:
        for m, g in (("sp1", "1 mm"), ("sp2", "2 mm")):
            s = [r for r in dec if r["roi"] == roi]
            k = np.array([100 - f(r[f"regret_{m}winner_under_fem_pct"]) for r in s])
            six.append(dict(target=roi, grid=g, same=sum(r[f"agree_fem_{m}"] == "True" for r in s),
                            rho=float(np.median([f(r[f"rho_fem_{m}"]) for r in s])),
                            keep=(float(np.median(k)), float(np.percentile(k, 5)), float(k.min())), below90=int((k < 90).sum())))
    N["six_table"] = six
    # field agreement per head
    fr = rows(FIELDS)
    for m, g in (("sp1hom", "1mm"), ("sp2hom", "2mm")):
        s = [x for x in fr if x["model"] == m]
        for k in ("rel", "ratio", "bias_pct", "r", "angle_deg"):
            a = np.array([f(x[k]) for x in s])
            N[f"field.{g}.{k}"] = (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)),
                                   float(a.min()), float(a.max()))
        N[f"field.{g}.n"] = len(s)
        N[f"field.{g}.r_ge95"] = int(sum(f(x["r"]) >= 0.95 for x in s))
    # atlas
    A = atlas()
    T = sorted({t for t, m in A})
    N["atlas.n"] = len(T)
    sp, fe = "sp1v23", "fem"
    g = lambda m, key: np.array([f(A[(t, m)][key]) for t in T])
    Fm, Tm = g(sp, "F_choice_median"), g(sp, "T_choice_median_Vm_per_mA")
    N["atlas.F"] = (float(np.median(Fm)), float(Fm.min()), float(Fm.max()))
    N["atlas.T"] = (float(np.median(Tm)), float(Tm.min()), float(Tm.max()))
    N["atlas.FEM.T"] = (float(np.median(g(fe, "T_choice_median_Vm_per_mA"))),)
    a1 = g(sp, "heads_F_above_1_pct")
    N["atlas.F1_majority"], N["atlas.F1_rare"] = int((a1 > 50).sum()), int((a1 < 5).sum())
    sc, k50, k5 = g(sp, "same_choice_as_fem_pct"), g(sp, "fem_optimum_kept_median"), g(sp, "fem_optimum_kept_p5")
    N["atlas.same"] = (float(np.median(sc)), float(sc.min()), float(sc.max()))
    N["atlas.kept50_min"] = float(k50.min())
    N["atlas.kept5"] = (float(np.median(k5)), int((k5 >= 90).sum()))
    N["atlas.kept5_low"] = [(nice(t), float(A[(t, sp)]["fem_optimum_kept_p5"])) for t in T if f(A[(t, sp)]["fem_optimum_kept_p5"]) < 90]
    N["atlas.kept5_low_raw"] = [(t, float(A[(t, sp)]["fem_optimum_kept_p5"])) for t in T if f(A[(t, sp)]["fem_optimum_kept_p5"]) < 90]
    N["atlas.cohort_same"] = int(sum(A[(t, sp)]["cohort_choice"] == A[(t, fe)]["cohort_choice"] for t in T))
    sc2, k52 = g("sp2v23", "same_choice_as_fem_pct"), g("sp2v23", "fem_optimum_kept_p5")
    N["atlas.2mm.same"], N["atlas.2mm.kept5_below90"] = float(np.median(sc2)), int((k52 < 90).sum())
    gn, g10 = g(sp, "gain_median"), g(sp, "heads_gain_ge_10pct")
    N["atlas.gain"] = (float(np.median(gn)), float(gn.max()))
    N["atlas.g10_lt5"], N["atlas.g10_ge20"] = int((g10 < 5).sum()), int((g10 >= 20).sum())
    N["atlas.g10_top"] = [(nice(t), float(A[(t, sp)]["heads_gain_ge_10pct"]))
                          for t in sorted(T, key=lambda t: -f(A[(t, sp)]["heads_gain_ge_10pct"]))[:8]]
    N["atlas.g10_top_raw"] = [(t, float(A[(t, sp)]["heads_gain_ge_10pct"]))
                              for t in sorted(T, key=lambda t: -f(A[(t, sp)]["heads_gain_ge_10pct"]))[:8]]
    rho, q = g(sp, "rho_age_T"), g(sp, "q_age_T")
    N["atlas.age_decline"], N["atlas.age_rise"] = int(((q < 0.05) & (rho < 0)).sum()), int(((q < 0.05) & (rho > 0)).sum())
    N["atlas.age_rho"] = float(np.median(rho))
    top = sorted(T, key=lambda t: -f(A[(t, sp)]["F_choice_median"]))
    N["atlas.most_focal"] = [(nice(t), f(A[(t, sp)]["F_choice_median"]), f(A[(t, sp)]["T_choice_median_Vm_per_mA"])) for t in top[:8]]
    N["atlas.most_focal_raw"] = top[:8]
    N["atlas.least_focal"] = [(nice(t), f(A[(t, sp)]["F_choice_median"])) for t in top[-8:]]
    # per set
    bys = []
    for key, name in SETS:
        s = [t for t in T if A[(t, fe)]["tier"] == key]
        h = lambda m, c: np.array([f(A[(t, m)][c]) for t in s])
        bys.append(dict(set=name, n=len(s), F=h(sp, "F_choice_median"), T=h(sp, "T_choice_median_Vm_per_mA"),
                        F1=int((h(sp, "heads_F_above_1_pct") > 50).sum()), g20=int((h(sp, "heads_gain_ge_10pct") >= 20).sum()),
                        # D2 (31 Oct 2026): the same counts by FEM, and the regions where both methods agree
                        F1_fem=int((h(fe, "heads_F_above_1_pct") > 50).sum()), g20_fem=int((h(fe, "heads_gain_ge_10pct") >= 20).sum()),
                        F1_both=int(((h(sp, "heads_F_above_1_pct") > 50) & (h(fe, "heads_F_above_1_pct") > 50)).sum()),
                        g20_both=int(((h(sp, "heads_gain_ge_10pct") >= 20) & (h(fe, "heads_gain_ge_10pct") >= 20)).sum()),
                        same=float(np.median(h(sp, "same_choice_as_fem_pct"))), k5=h(sp, "fem_optimum_kept_p5"),
                        age=int(sum(f(A[(t, sp)]["q_age_T"]) < 0.05 and f(A[(t, sp)]["rho_age_T"]) < 0 for t in s)),
                        FEM_F=h(fe, "F_choice_median"), FEM_T=h(fe, "T_choice_median_Vm_per_mA")))
        if all("rv_sp_choice_fails_fem_floor_pct" in A[(t, sp)] for t in s):
            fl = h(sp, "rv_sp_choice_fails_fem_floor_pct")
            bys[-1].update(fail=float(np.median(fl)), fail_max=float(fl.max()))
    N["by_set"] = bys
    for key, name in SETS:
        s = [t for t in T if A[(t, fe)]["tier"] == key]
        N[f"set.{key}.F"] = (min(f(A[(t, sp)]["F_choice_median"]) for t in s), max(f(A[(t, sp)]["F_choice_median"]) for t in s))
    # parameters (constraint, cohort admissibility, spread, age per decade)
    if all(f"p_T_choice_p5" in A[(t, sp)] for t in T):
        tg = g(sp, "p_T_constrained_over_unconstrained_median"); fc = 100 * (1 - g(sp, "p_F_constrained_over_unconstrained_median"))
        N["param.Tgain"] = (float(np.median(tg)), float(tg.min()), float(tg.max()))
        N["param.Fcost"] = (float(np.median(fc)), float(fc.min()), float(fc.max()))
        adm = g(sp, "p_cohort_admissible_pct")
        N["param.cohort_adm"] = (float(np.median(adm)), float(adm.min()), int((adm >= 90).sum()))
        spread = g(sp, "p_T_choice_p95") / g(sp, "p_T_choice_p5")
        N["param.T_spread"] = (float(np.median(spread)), float(spread.min()), float(spread.max()))
        ad, aq = g(sp, "p_age_T_pct_per_decade"), g(sp, "p_age_q")
        N["param.age_decade"] = (float(np.median(ad)), int(((aq < 0.05) & (ad < 0)).sum()), float(ad.min()), float(ad.max()))
        for key, name in SETS:
            s = [t for t in T if A[(t, fe)]["tier"] == key]
            N[f"set.{key}.Tgain"] = float(np.median([f(A[(t, sp)]["p_T_constrained_over_unconstrained_median"]) for t in s]))
    # demographics summaries (sp1v23)
    dsex = np.array([f(A[(t, sp)]["s_logT_sex_M0"]) for t in T]); qsex = np.array([f(A[(t, sp)]["s_logT_q_sex_M0"]) for t in T])
    N["sex.M0"] = (float(100 * (np.exp(np.median(dsex)) - 1)), int(((qsex < 0.05) & (dsex < 0)).sum()))
    d3 = np.array([f(A[(t, sp)]["s_logT_sex_M3"]) for t in T]); q3 = np.array([f(A[(t, sp)]["s_logT_q_sex_M3"]) for t in T])
    N["sex.M3"] = (float(100 * (np.exp(np.median(d3)) - 1)), int((q3 < 0.05).sum()))
    share = 100 * (dsex - d3) / dsex
    N["sex.share_explained"] = (float(np.median(share)), float(np.percentile(share, 25)), float(np.percentile(share, 75)))
    qsite = np.array([f(A[(t, sp)]["d_q_site_logT"]) for t in T]); ssp = np.array([f(A[(t, sp)]["d_site_spread_logT"]) for t in T])
    N["site"] = (int((qsite < 0.05).sum()), float(100 * np.median(ssp)))
    lm = np.array([A[(t, sp)]["loso_max"] for t in T]); lc = sum(A[(t, sp)]["loso_changed"] for t in T); ln = sum(A[(t, sp)]["loso_n"] for t in T)
    lo_all = [f(r["transfer_loss_pct"]) for r in rows(os.path.join(DEMO, "loso_163.csv")) if r["model"] == sp]
    N["loso"] = (float(np.median(lo_all)), float(lm.max()), 100 * lc / ln, nice(T[int(np.argmax(lm))]))
    N["loso_raw"] = T[int(np.argmax(lm))]
    # regression checks of the atlas runs (each SUMMARY.md: ref6 maxima against the 300-head comparison)
    import re
    reg = {}
    for p in SUMMARIES:
        line = next(l for l in open(os.path.join(os.path.dirname(p), "SUMMARY.md"), encoding="utf-8") if l.startswith("Regression check"))
        for m, v in re.findall(r"(fem|sp2v23|sp1v23) ([0-9.]+e-[0-9]+)", line):
            reg.setdefault(m, []).append(float(v))
    N["reg.sp1v23"] = max(reg["sp1v23"])
    N["reg.fem_max"] = max(reg["fem"])
    N["reg.fem_main"] = sorted(reg["fem"])[-2]                 # all runs but the one that re-solved FEM
    # ---- review round of 31 Oct 2026 (review/RESULTS_v23_REVIEW_2026-10-31_01.md)
    fr1 = [x for x in fr if x["model"] == "sp1hom"]
    N["field.1mm.low_r"] = sorted(f(x["r"]) for x in fr1 if f(x["r"]) < 0.95)
    N["field.1mm.higher"] = int(sum(f(x["bias_pct"]) > 0 for x in fr1))
    rT, rF = g(sp, "T_choice_median_Vm_per_mA") / g(fe, "T_choice_median_Vm_per_mA"), g(sp, "F_choice_median") / g(fe, "F_choice_median")
    N["offset.T"] = (float(np.median(rT)), float(rT.min()), float(rT.max()))
    N["offset.F"] = (float(rF.min()), float(rF.max()))
    N["fem.F1_majority"] = int((g(fe, "heads_F_above_1_pct") > 50).sum())
    N["fem.g10_ge20"] = int((g(fe, "heads_gain_ge_10pct") >= 20).sum())
    # D2: regions where SPICEhead's count is confirmed by FEM (both methods), and FEM-only
    N["fem.F1_both"] = int(((g(sp, "heads_F_above_1_pct") > 50) & (g(fe, "heads_F_above_1_pct") > 50)).sum())
    N["fem.g10_both"] = int(((g(sp, "heads_gain_ge_10pct") >= 20) & (g(fe, "heads_gain_ge_10pct") >= 20)).sum())
    N["fem.F1_rare"] = int((g(fe, "heads_F_above_1_pct") < 5).sum())
    g2 = 100 * (g("sp2v23", "T_choice_median_Vm_per_mA") / g(sp, "T_choice_median_Vm_per_mA") - 1)
    N["grid.T_2_over_1"] = (float(np.median(g2)), float(g2.min()), float(g2.max()))
    N["fs.coh.keep_sp1_under_fem.min_floor"] = np.floor(10 * N["fs.coh.keep_sp1_under_fem.min"]) / 10
    qa, ra = g(sp, "q_age_T"), g(sp, "rho_age_T")
    aq, ad_ = g(sp, "p_age_q"), g(sp, "p_age_T_pct_per_decade")
    N["age.both"] = int(((qa < 0.05) & (ra < 0) & (aq < 0.05) & (ad_ < 0)).sum())
    N["age.adj_rise"] = int(((aq < 0.05) & (ad_ > 0)).sum())
    for M in ("M1", "M2", "M3"):
        sM, qM = g(sp, f"s_logT_sex_{M}"), g(sp, f"s_logT_q_sex_{M}")
        N[f"sex.{M}.sig"], N[f"sex.{M}.men_higher"] = int((qM < 0.05).sum()), int(((qM < 0.05) & (sM > 0)).sum())
    N["sex.F_sig"] = int((g(sp, "d_q_sex_F") < 0.05).sum())
    N["site"] = (N["site"][0], float(100 * np.median(np.exp(ssp) - 1)))       # one percent convention (exp)
    if all("rv_target" in A[(t, sp)] for t in T):
        r = lambda k: np.array([f(A[(t, sp)][f"rv_{k}"]) for t in T])
        sv = r("sp1v23__cohort_served_pct")
        N["rv.served"] = (float(np.median(sv)), int((sv >= 80).sum()), int((sv < 50).sum()), float(sv.min()))
        fail = r("sp_choice_fails_fem_floor_pct")
        N["rv.fem_fail"] = (float(np.median(fail)), float(fail.max()), int((fail > 10).sum()))
        N["rv.ok_feasible"] = (float(np.median(r("retention_ok_and_feasible_pct"))), float(r("retention_ok_and_feasible_pct").min()))
        rs = r("sp1v23__cohort_reselected_pct")
        N["rv.reselected"] = (float(np.median(rs)), int((rs < 50).sum()), int((rs >= 90).sum()))
        wT = 100 * (r("sp1v23__T_reported_ci_hi") - r("sp1v23__T_reported_ci_lo")) / r("sp1v23__T_reported_median")
        N["rv.T_ci_width"] = (float(np.median(wT)), float(wT.max()))
        vox = r("voxels_median")
        from scipy.stats import spearmanr
        N["rv.voxels"] = (int(vox.min()), int(vox.max()), float(spearmanr(vox, g(sp, "F_choice_median")).correlation))
        ca, cq = r("cohortT_age_pct_per_decade"), r("cohortT_age_q")
        cs, csq = r("cohortT_sex_pct_men"), r("cohortT_sex_q")
        N["rv.cohort_age"] = (int(((cq < 0.05) & (ca < 0)).sum()), float(np.median(ca)))
        N["rv.cohort_sex"] = (int(((csq < 0.05) & (cs < 0)).sum()), float(np.median(cs)))
        S = rows(os.path.join(os.path.dirname(REVIEW), "rule_sensitivity.csv"))
        N["rv.rule"] = S
        alt = [x for x in S if x["default"] != "True"]
        N["rv.rule_range"] = {k: (min(f(x[k]) for x in S), max(f(x[k]) for x in S)) for k in
                              ("same_choice_median", "retention_p5_ge90", "cohort_same_as_fem", "F_above_1_majority",
                               "gain10_ge20", "T_median", "F_median")}
        N["rv.rule_changed"] = (min(int(x["cohort_changed_vs_default"]) for x in alt),
                                max(int(x["cohort_changed_vs_default"]) for x in alt))
    # ---- 31 Oct 2026, decisions D5-D7: head-level mixed model, site vs anatomy, region size
    MM = os.path.join(os.path.dirname(REVIEW), "mixed_model")
    SS = os.path.join(os.path.dirname(REVIEW), "site_size")
    if os.path.exists(os.path.join(MM, "fixed_effects.csv")):
        N["mm"] = {f"{x['outcome']}|{x['size_model']}|{x['term']}": (f(x["pct_change"]), f(x["ci_lo_pct"]), f(x["ci_hi_pct"]), f(x["p"]))
                   for x in rows(os.path.join(MM, "fixed_effects.csv")) if x["model"] == sp}
        N["mm.icc"] = {f"{x['outcome']}|{x['size_model']}": f(x["icc"]) for x in rows(os.path.join(MM, "variance_components.csv")) if x["model"] == sp}
        ix = {x["outcome"]: x for x in rows(os.path.join(MM, "interaction_region_x_sex.csv"))} if os.path.exists(os.path.join(MM, "interaction_region_x_sex.csv")) else {}
        N["mm.interaction"] = ix
    if os.path.exists(os.path.join(SS, "site_T_summary.csv")):
        st = [x for x in rows(os.path.join(SS, "site_T_summary.csv")) if "sp1v23__T_wa" in x["outcome"]]
        N["site.models"] = st
        bym = {x["model"][:2]: x for x in st}
        N["site.ge"] = {m: (f(bym[m]["Bai_TVGH_vs_others_pct_median"]), int(f(bym[m]["Bai_TVGH_vs_others_q05"])), int(f(bym[m]["site_F_q05"])),
                            f(bym[m]["site_spread_pct_median"])) for m in bym}
        an = {x["measure"]: x for x in rows(os.path.join(SS, "site_anatomy_ancova.csv"))}
        N["site.anat"] = {m: (f(an[m]["Bai_TVGH_vs_others_pct"]), f(an[m]["Bai_TVGH_vs_others_p"])) for m in
                          ("icv_ml", "vol_csf_ml", "vol_spongy_bone_ml", "vol_compact_bone_ml", "skull_thickness_mm", "soft_thickness_mm") if m in an}
        ps = rows(os.path.join(SS, "per_site_T.csv"))
        rt = np.array([f(x["sp1v23__T_wa__median_Bai_TVGH_over_others"]) for x in ps])
        N["site.ge_ratio_unadj"] = (float(np.median(rt)), float(np.percentile(rt, 25)), float(np.percentile(rt, 75)))
    if os.path.exists(os.path.join(SS, "size_threshold.csv")):
        from scipy.stats import spearmanr
        sz = rows(os.path.join(SS, "size_threshold.csv"))
        vox = np.array([f(x["voxels_median"]) for x in sz]); Fm_ = np.array([f(x["F_median"]) for x in sz])
        ciw = np.array([f(x["relT_ci_width_pct"]) for x in sz])
        a90 = [i for i, x in enumerate(sz) if x["tier"] == "tier1"]
        N["size.small"] = [(x["target"], int(f(x["voxels_median"])), int(f(x["voxels_min"]))) for x in sz if f(x["voxels_median"]) < 60]
        N["size.ret_low"] = [(x["target"], f(x["retention_p5"])) for x in sz if f(x["retention_p5"]) < 90]
        r_all, r_90 = spearmanr(vox, Fm_), spearmanr(vox[a90], Fm_[a90])
        r_ci = spearmanr(vox, ciw)
        N["size.rho_F"] = (float(r_all.correlation), float(r_all.pvalue), float(r_90.correlation), float(r_90.pvalue), len(a90))
        N["size.ciw"] = (float(np.median(ciw)), float(ciw.min()), float(ciw.max()), float(r_ci.correlation), float(r_ci.pvalue))
    # ---- 1 Nov 2026, decision D4: four-pad FEM check of the six cohort montages in all 300 heads (pipeline/nchc/pad4_collect.py)
    P4 = os.path.join(R23, "v23_pad4_check", "per_head.csv")
    if os.path.exists(P4):
        p4 = rows(P4)
        p4 = [x for x in p4 if x.get("pad_T_ratio")]
        q = lambda k: np.array([f(x[k]) for x in p4])
        stat = lambda a: (float(np.median(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95)), float(a.min()), float(a.max()))
        N["pad4.n"] = len(p4)
        N["pad4.heads"] = len({x["code"] for x in p4})
        N["pad4.pad.T"], N["pad4.pad.O"], N["pad4.pad.F"] = stat(q("pad_T_ratio")), stat(q("pad_O_ratio")), stat(q("pad_F_ratio"))
        N["pad4.remesh.T"], N["pad4.remesh.F"] = stat(q("remesh_T_ratio")), stat(q("remesh_F_ratio"))
        N["pad4.total.T"], N["pad4.total.F"] = stat(q("total_T_ratio")), stat(q("total_F_ratio"))
        N["pad4.pad.F_within3"] = float(100 * np.mean(np.abs(q("pad_F_ratio") - 1) <= 0.03))
        N["pad4.pad.T_within5"] = float(100 * np.mean(np.abs(q("pad_T_ratio") - 1) <= 0.05))
        N["pad4.remesh.within1"] = float(100 * np.mean((np.abs(q("remesh_T_ratio") - 1) <= 0.01) & (np.abs(q("remesh_F_ratio") - 1) <= 0.01)))
        N["pad4.remesh.T_maxdev"] = float(np.abs(q("remesh_T_ratio") - 1).max())
        N["pad4.remesh.F_maxdev"] = float(np.abs(q("remesh_F_ratio") - 1).max())
        N["pad4.mesh_identical"] = int(sum(x["mesh_identical_to_archive"] == "1" for x in p4) / 6)
        by_t = {}
        for x in p4:
            by_t.setdefault(x["target"], []).append(x)
        def tstat(rows_, k):
            return stat(np.array([f(x[k]) for x in rows_]))
        N["pad4.by_target"] = [dict(target=t, montage=by_t[t][0]["montage"], heads=len(by_t[t]),
                                    pad_T=tstat(by_t[t], "pad_T_ratio"), pad_O=tstat(by_t[t], "pad_O_ratio"), pad_F=tstat(by_t[t], "pad_F_ratio"),
                                    remesh_T=tstat(by_t[t], "remesh_T_ratio"), remesh_F=tstat(by_t[t], "remesh_F_ratio"),
                                    pad_F_within3=float(100 * np.mean(np.abs(np.array([f(x["pad_F_ratio"]) for x in by_t[t]]) - 1) <= 0.03)))
                               for t in ROI6 if t in by_t]
        tm = [b["pad_T"][0] for b in N["pad4.by_target"]]
        N["pad4.pad.T_target_range"] = (min(tm), max(tm))
    # ---- 2 Nov 2026, decision D3: uncertainty analysis in all 300 heads (75 physical runs + 2 target-mask runs per head)
    U = [A[(t, "sp1v23")] for t in T if "uq_heads" in A[(t, "sp1v23")]]
    if U and min(int(f(u["uq_heads"])) for u in U) >= 290:
        uq = lambda k: np.array([f(u[f"uq_{k}"]) for u in U])
        N["uq.targets"] = len(U)
        N["uq.heads"] = (int(min(uq("heads"))), int(max(uq("heads"))))
        for g in ("cond", "jit", "rot", "seg", "mask"):
            if f"uq_{g}_T_ratio_p5" in U[0]:
                lo_, hi_ = uq(f"{g}_T_ratio_p5"), uq(f"{g}_T_ratio_p95")
                N[f"uq.{g}.T"] = (float(np.median(lo_)), float(np.median(hi_)), float(lo_.min()), float(hi_.max()))
                # the mask scenario has no variance where a one-voxel erosion empties a small region's mask (T = 0)
                N[f"uq.{g}.share"] = float(np.nanmedian(uq(f"{g}_uncertainty_share_pct")))
                N[f"uq.{g}.T_zero_regions"] = int(np.sum(lo_ <= 0))
                N[f"uq.{g}.fixed_ret_p5"] = float(np.median(uq(f"{g}_fixed_retention_p5")))
                N[f"uq.{g}.kept"] = float(np.median(uq(f"{g}_reported_kept_pct")))
                N[f"uq.{g}.nomret_p5"] = float(np.nanmedian(uq(f"{g}_nominal_choice_retention_p5")))
        ret = uq("fixed_retention_all_runs_p5"); sh = uq("share_runs_retention_ge_90pct")
        rob = np.array([u["uq_fixed_protocol_robust"] in ("True", "yes", "1") for u in U])
        N["uq.ret_p5"] = (float(np.median(ret)), float(ret.min()), float(ret.max()))
        N["uq.share90"] = (float(np.median(sh)), float(sh.min()), float(sh.max()))
        N["uq.robust"] = int(rob.sum())
        N["uq.ret_nominal"] = float(np.median(uq("fixed_retention_nominal_median")))
        bys = []
        for key, name in SETS + [("all", "All regions")]:
            us = [A[(t, "sp1v23")] for t in T if "uq_heads" in A[(t, "sp1v23")] and (key == "all" or A[(t, "fem")]["tier"] == key)]
            if not us:
                continue
            m = lambda k: float(np.nanmedian([f(u[f"uq_{k}"]) for u in us]))
            bys.append(dict(set=name, n=len(us), **{g: (m(f"{g}_T_ratio_p5"), m(f"{g}_T_ratio_p95")) for g in ("cond", "jit", "rot", "seg", "mask")},
                            cond_share=m("cond_uncertainty_share_pct"), ret_p5=m("fixed_retention_all_runs_p5"),
                            share90=m("share_runs_retention_ge_90pct"), robust=sum(u["uq_fixed_protocol_robust"] == "True" for u in us)))
        N["uq.by_set"] = bys
        # heads in which one-voxel erosion empties the target mask (pipeline/nchc/uq_v23_mask_empty.py)
        ME = os.path.join(R23, "v23_uq", "mask_empty_300.csv")
        me = {r["target"]: int(r["mask_erode_zero"]) + int(r["mask_erode_nonfinite"]) for r in rows(ME)} if os.path.exists(ME) else {}
        for t in T:
            A[(t, "sp1v23")]["uq_mask_emptied_heads"] = me.get(t, "")
        N["uq.mask_empty"] = sorted(((t, n) for t, n in me.items() if n > 0), key=lambda z: -z[1])
        N["uq.least"] = sorted(((t, f(A[(t, "sp1v23")]["uq_share_runs_retention_ge_90pct"])) for t in T if "uq_heads" in A[(t, "sp1v23")]),
                               key=lambda z: z[1])[:5]
    N["A"] = A
    N["targets"] = T
    return N


if __name__ == "__main__":
    N = compute()
    for k, v in N.items():
        if k not in ("A", "targets", "by_set", "six_table"):
            print(f"{k:28s} {v}")
    for b in N["by_set"]:
        print(b["set"], b["n"], f"F {np.median(b['F']):.2f} [{b['F'].min():.2f}-{b['F'].max():.2f}]",
              f"T {np.median(b['T']):.3f} [{b['T'].min():.3f}-{b['T'].max():.3f}]", b["F1"], b["g20"], b["same"],
              f"k5 {np.median(b['k5']):.1f} [{b['k5'].min():.1f}]", b["age"])
