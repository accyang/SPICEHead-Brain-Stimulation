# -*- coding: utf-8 -*-
"""Every number in the v22 methods paper, computed from result files and never typed.

The v22 paper (October 2026 review framing; skin-surface contacts; harmonic interface rule) is
generated the way the v21 manuscript was: `render_v22.py` substitutes these values into
`Paper_A_SPICEhead/Brain Stimulation/v22/04_manuscript_v22.tmpl.md`. The v21 generator (values.py) is left
untouched, because the v21 supplement and audit still depend on it.

Where the inputs live
  model-dependent      RES = results_v22/ (spicehead.config.RESULTS under SPICEHEAD_MODEL=v22)
  model-independent    RES0 = results/: analytic disk and layered-slab benchmarks, the SimNIBS
                       reference fields and their solver logs, the homogeneous-cylinder benchmark
  MNI-template maps    Paper_A_SPICEhead/Brain Stimulation/figures/TI_targets_MNI/

A result that is not there yet is recorded in MISSING instead of raising, so a draft can be
rendered while the recompute runs; render_v22.py refuses a final build while anything is missing.

  python pipeline/values_v22.py [substr]
  SPICEHEAD_VALUES_ROOT=results SPICEHEAD_MNI_DIR=TI_targets_MNI_v21 python pipeline/values_v22.py
      (dry run against the v21 outputs, to test this module before the v22 results exist)
"""
import csv
import json
import os
import re
import statistics as st
import sys

import numpy as np
from scipy import stats as sstats

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
if __name__ == "__main__":
    os.environ.setdefault("SPICEHEAD_MODEL", "v22")   # v22 record script; the package default is v23 since 2 Nov 2026
from spicehead import config                                     # noqa: E402

RES = os.path.join(ROOT, os.environ.get("SPICEHEAD_VALUES_ROOT", "")) \
    if os.environ.get("SPICEHEAD_VALUES_ROOT") else config.RESULTS
RES0 = os.path.join(ROOT, "results")
FEM = os.path.join(RES0, "v21_simnibs_validation")
MNI = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "figures",
                   os.environ.get("SPICEHEAD_MNI_DIR", "TI_targets_MNI"))
COHORT = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"
HEADS4 = ["SUBJ01", "SUBJ02", "SUBJ03", "SUBJ04"]

ROIS = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]
NICE = {"M1_handknob": "M1 hand knob", "thalamus": "Thalamus", "insula": "Insula",
        "ACC": "Anterior cingulate", "hippocampus_L": "Hippocampus, left",
        "hippocampus_R": "Hippocampus, right"}
# targets as they read inside a sentence
PROSE = {"M1_handknob": "the motor hand knob", "thalamus": "the thalamus", "insula": "the insula",
         "ACC": "the ACC", "hippocampus_L": "the left hippocampus",
         "hippocampus_R": "the right hippocampus"}
MM_ORDER = ["M1", "THAL", "HIPP", "INFERIOR"]
MM_NICE = {"M1": "Fronto-parietal (F3+F4 / O1+Pz)", "THAL": "Bilateral temporal (F7+T7 / P8+T8)",
           "HIPP": "Fronto-occipital (F7+Fp2 / O2+P8)",
           "INFERIOR": "Inferior chain (F9+T9 / F10+T10)"}
MISSING = []
MINUS = "−"
# set by values_v23.py, which reuses this module with v23 sources; the defaults are the v22 record's
FS8_DIR = None          # eight-head desktop run (None: RES/v22_fem_fullsearch)
FS300_DIR = None        # 300-head run (None: RES/v22_fem_fullsearch_300)
FS_V22_PARTS = True     # the parts that only exist for v22: four-pad check of the SPICEhead optima, desktop
                        # timings and benchmark, desktop-vs-cluster consistency
TD_DIR = "v22_template_design"


# ---------------------------------------------------------------- formatting helpers
_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x, sig=1):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    s = f"{abs(x / 10 ** e):.{sig}f}".rstrip("0").rstrip(".")
    return f"{MINUS if x < 0 else ''}{s}×10{str(e).translate(_SUP)}"


def pct(x, dp=1):
    return f"{MINUS if x < 0 else '+'}{abs(x):.{dp}f}%"


def num(x, dp=3):
    return f"{MINUS if x < 0 else ''}{abs(x):.{dp}f}"


def pfmt(p):
    if p < 0.001:
        return "< .001"
    return f"= {p:.3g}".replace("0.", ".", 1) if p < 1 else f"= {p:.3g}"


def table(header, rows, align=None):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(align or ["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- readers
def _p(*parts):
    return os.path.join(*parts)


def ls(d, suffix):
    """Files in d ending with suffix. Not glob: the project path contains "[New]", which glob
    reads as a character class and so silently matches nothing."""
    return sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith(suffix))         if os.path.isdir(d) else []


def rows(path, what):
    if not os.path.exists(path):
        MISSING.append(f"{what}: {os.path.relpath(path, ROOT)}")
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def jload(path, what):
    if not os.path.exists(path):
        MISSING.append(f"{what}: {os.path.relpath(path, ROOT)}")
        return None
    return json.load(open(path, encoding="utf-8"))


def f(x):
    return float(x) if x not in (None, "") else float("nan")


def mean(rs, k):
    v = [f(r[k]) for r in rs if r.get(k) not in (None, "")]
    return float(np.mean(v)) if v else float("nan")


def label(s):
    """Canonical montage label: pairs sorted internally and against each other."""
    pairs = [tuple(sorted(p.strip().split("+"))) for p in s.split("/")]
    return " / ".join("+".join(p) for p in sorted(pairs))


# ---------------------------------------------------------------- sections
def cohort(V, T):
    rs = list(csv.DictReader(open(COHORT, encoding="utf-8-sig")))
    age = np.array([float(r["Age_Scan"]) for r in rs])
    male = sum(r["Sex_1M0F"] == "1" for r in rs)
    V.update({"coh.n": len(rs), "coh.age_mean": age.mean(), "coh.age_sd": age.std(ddof=1),
              "coh.age_min": age.min(), "coh.age_max": age.max(), "coh.male": male,
              "coh.female": len(rs) - male, "coh.n_sites": len({r["Site"] for r in rs})})
    out = []
    for s in sorted({r["Site"] for r in rs}):
        sub = [r for r in rs if r["Site"] == s]
        a = np.array([float(r["Age_Scan"]) for r in sub])
        m = sum(r["Sex_1M0F"] == "1" for r in sub)
        out.append([s, len(sub), f"{a.mean():.1f} ± {a.std(ddof=1):.1f}",
                    f"{a.min():.0f}–{a.max():.0f}", f"{m} / {len(sub) - m}"])
    out.append(["All", len(rs), f"{age.mean():.1f} ± {age.std(ddof=1):.1f}",
                f"{age.min():.0f}–{age.max():.0f}", f"{male} / {len(rs) - male}"])
    T["sites"] = table(["Site", "n", "Age, mean ± SD", "Range", "Men / women"], out)


def verification(V, T):
    # analytic disk: model against the discrete gradient of the exact potential, and that
    # discrete reference against the exact closed-form gradient
    # The principal analytic result is the network against the exact closed-form gradient
    # (review of 2 October, R5.2); every disk figure is a normalized mean absolute error (nMAE),
    # not a pointwise maximum, and the reference's p95 absolute relative error is kept beside it.
    for r in rows(_p(RES0, "v21_validation", "disk_analytic_gradient.csv"), "disk benchmark"):
        d = f"{float(r['dx_mm']):g}mm"
        V[f"disk.nmae_{d}"] = f(r["model_vs_exact_nmae_pct"])
        V[f"disk.slope_{d}"] = f(r["model_vs_exact_slope"])
        V[f"disk.nmae_disc_{d}"] = f(r["model_vs_discrete_nmae_pct"])
        V[f"disk.ref_nmae_{d}"] = f(r["discrete_vs_exact_nmae_pct"])
        V[f"disk.ref_p95_{d}"] = f(r["discrete_vs_exact_p95_abs_rel_pct"])
    ks = [k for k in V if k.startswith("disk.ref_nmae_")]
    if ks:
        V["disk.ref_nmae_max"] = max(V[k] for k in ks)
        V["disk.ref_p95_max"] = max(V[k] for k in V if k.startswith("disk.ref_p95_"))
    for r in rows(_p(RES0, "v21_validation", "convergence.csv"), "disk convergence"):
        d = f"{float(r['dx_mm']):g}mm"
        V[f"disk.peak_err_{d}"] = f(r["peak_err_mm"])
    # layered slab: total resistance against the exact series value, by interface rule
    for r in rows(_p(RES0, "v21_interface", "interface_rule.csv"), "layered slab"):
        V[f"slab.{r['rule']}_{float(r['dx_mm']):g}mm"] = f(r["R_ratio"])
        V[f"slab.{r['rule']}_err_{float(r['dx_mm']):g}mm"] = 100 * abs(1 - f(r["R_ratio"]))

    # realistic-head refinement, four heads, each against its own 1 mm solution
    per = {}
    for h in HEADS4:
        p = _p(RES, "v21_head_convergence", "head_convergence.csv") if h == "SUBJ01" else \
            _p(RES, "v21_head_convergence", h, "head_convergence.csv")
        rs = rows(p, f"grid study {h}")
        if rs:
            per[h] = {float(r["dx_mm"]): r for r in rs}
    V["conv.n_heads"] = len(per)
    if not per:
        return
    dxs = sorted({d for h in per for d in per[h]}, reverse=True)
    nodes1 = [f(per[h][1.0]["n_nodes"]) for h in per if 1.0 in per[h]]
    if nodes1:
        V["conv.nodes1_min_M"] = min(nodes1) / 1e6
        V["conv.nodes1_max_M"] = max(nodes1) / 1e6
    out = []
    for d in dxs:
        if d == 1.0:
            continue
        hs = [h for h in per if d in per[h] and 1.0 in per[h]]
        rr = [f(per[h][d]["r"]) for h in hs]
        sl = [f(per[h][d]["slope"]) for h in hs]
        nm = [f(per[h][d]["nmae"]) for h in hs]
        tag = f"{d:g}mm"
        V[f"conv.r_{tag}"] = float(np.mean(rr))
        V[f"conv.r_{tag}_lo"], V[f"conv.r_{tag}_hi"] = min(rr), max(rr)
        V[f"conv.slope_{tag}"] = float(np.mean(sl))
        V[f"conv.nmae_{tag}"] = 100 * float(np.mean(nm))
        exc = {}
        for roi in ROIS:
            e = [100 * (f(per[h][d][f"T_{roi}"]) / f(per[h][1.0][f"T_{roi}"]) - 1) for h in hs]
            exc[roi] = e
            V[f"conv.exc_{tag}.{roi}"] = float(np.mean(e))
        out.append([f"{d:g} mm", f"{np.mean(rr):.3f} ({min(rr):.3f}–{max(rr):.3f})",
                    f"{np.mean(sl):.3f}", f"{100 * np.mean(nm):.1f}%"]
                   + [pct(float(np.mean(exc[r]))) for r in ROIS])
    T["conv"] = table(["Grid", "r vs 1 mm, mean (range)", "Slope", "nMAE"]
                      + [f"T, {NICE[r]}" for r in ROIS], out)
    # how much the ROI intensities still move between 2 mm and 1 mm (paired, relative to 2 mm)
    for roi in ROIS:
        ch = [100 * (f(per[h][1.0][f"T_{roi}"]) / f(per[h][2.0][f"T_{roi}"]) - 1)
              for h in per if 2.0 in per[h] and 1.0 in per[h]]
        if ch:
            V[f"conv.step21.{roi}"] = float(np.mean(ch))
    # step21_lo/hi span the target-wise MEANS across the four heads (R5.3); the individual
    # head-target changes are kept separately, and the worst head per target below
    vals = [V[f"conv.step21.{r}"] for r in ROIS if f"conv.step21.{r}" in V]
    if vals:
        V["conv.step21_lo"], V["conv.step21_hi"] = min(vals), max(vals)
    ind = [100 * (f(per[h][1.0][f"T_{roi}"]) / f(per[h][2.0][f"T_{roi}"]) - 1)
           for h in per if 2.0 in per[h] and 1.0 in per[h] for roi in ROIS]
    if ind:
        V["conv.step21_ind_lo"], V["conv.step21_ind_hi"] = min(ind), max(ind)
    # worst head per target, for the proposed grid-stability criterion
    for roi in ROIS:
        ch = [abs(100 * (f(per[h][1.0][f"T_{roi}"]) / f(per[h][2.0][f"T_{roi}"]) - 1))
              for h in per if 2.0 in per[h] and 1.0 in per[h]]
        if ch:
            V[f"conv.step21_absmax.{roi}"] = max(ch)
    am = [V[f"conv.step21_absmax.{r}"] for r in ROIS if f"conv.step21_absmax.{r}" in V]
    if am:
        V["conv.step21_absmax_lo"], V["conv.step21_absmax_hi"] = min(am), max(am)
    # monotone convergence of ROI intensity, counted per head and target over 5-4-3-2-1 mm
    mono = tot = 0
    for h in per:
        for roi in ROIS:
            seq = [f(per[h][d][f"T_{roi}"]) for d in sorted(per[h], reverse=True)]
            dif = np.diff(seq)
            tot += 1
            mono += bool(np.all(dif <= 0) or np.all(dif >= 0))
    V["conv.monotone"], V["conv.monotone_n"] = mono, tot


def fem_qc(V, T):
    """SimNIBS's electrode-flux disagreement diagnostic for every reference run (64)."""
    pat = re.compile(r"(?:calibration error:|Estimated error value:)\s*([0-9.]+)%")
    runs = []
    for cid in sorted(os.listdir(FEM)):
        d = _p(FEM, cid)
        if not os.path.isdir(d):
            continue
        for sub in sorted(os.listdir(d)):
            m = re.fullmatch(r"(?:(M1|HIPP|INFERIOR)_)?(ch[12])", sub)
            if not m:
                continue
            vals = []
            for lg in ls(_p(d, sub), ".log"):
                vals += [float(x) for x in pat.findall(open(lg, encoding="utf-8",
                                                            errors="replace").read())]
            if vals:
                runs.append((cid, m.group(1) or "THAL", m.group(2), vals[-1]))
    V["qc.n_runs"] = len(runs)
    if not runs:
        MISSING.append("FEM calibration diagnostics: no SimNIBS logs found")
        return
    V["qc.n_over10"] = sum(r[3] > 10 for r in runs)
    V["qc.min"], V["qc.max"] = min(r[3] for r in runs), max(r[3] for r in runs)
    for mm in MM_ORDER:
        s = [r for r in runs if r[1] == mm]
        V[f"qc.{mm}.n"], V[f"qc.{mm}.over10"] = len(s), sum(r[3] > 10 for r in s)
        V[f"qc.{mm}.median"] = float(np.median([r[3] for r in s])) if s else float("nan")
    V["qc.flagged"] = sorted({(r[0], r[1]) for r in runs if r[3] > 10})


def fem(V, T):
    ag = rows(_p(RES, "v21_simnibs_validation", "agreement.csv"), "FEM agreement")
    raw = rows(_p(RES, "v21_simnibs_validation", "agreement_raw.csv"), "FEM raw agreement")
    for dx, tag in (("1.0", "1mm"), ("2.0", "2mm")):
        s = [x for x in ag if x["dx_mm"] == dx]
        for k in ("r", "r2", "slope", "nmae", "dice", "peak_err_mm"):
            V[f"fem.{tag}.{k}"] = mean(s, k)
        q = [x for x in raw if x["dx_mm"] == dx]
        for k in ("raw_bias", "loa_lo", "loa_hi", "prop_bias_slope", "ccc", "slope", "r",
                  "mean_fem", "mean_spicehead"):
            V[f"raw.{tag}.{k}"] = mean(q, k)
        if q:
            rel = [100 * (f(x["mean_spicehead"]) / f(x["mean_fem"]) - 1) for x in q]
            V[f"raw.{tag}.relbias"] = float(np.mean(rel))
            V[f"raw.{tag}.relbias_lo"], V[f"raw.{tag}.relbias_hi"] = min(rel), max(rel)
        for roi in ROIS:
            V[f"fem.{tag}.{roi}.relerr"] = mean(s, f"roi_{roi}_relerr_pct")

    mm = rows(_p(RES, "v21_multimontage", "agreement_by_montage.csv"), "multi-montage agreement")
    rk_i = rows(_p(RES, "v21_multimontage", "rank_preservation.csv"), "intensity ranks")
    rk_f = rows(_p(RES, "v21_multimontage", "rank_preservation_focality.csv"), "focality ranks")
    out = []
    for dx in (2.0, 1.0):
        tag = f"{dx:g}mm"
        sub = [x for x in mm if float(x["dx_mm"]) == dx]
        for t in MM_ORDER:
            s = [x for x in sub if x["montage"] == t]
            if not s:
                continue
            rel = [100 * (f(x["mean_spicehead"]) / f(x["mean_fem"]) - 1) for x in s]
            for k in ("r", "slope", "ccc"):
                V[f"mm.{tag}.{t}.{k}"] = mean(s, k)
            V[f"mm.{tag}.{t}.relbias"] = float(np.mean(rel))
            V[f"mm.{tag}.{t}.relbias_lo"], V[f"mm.{tag}.{t}.relbias_hi"] = min(rel), max(rel)
            out.append([MM_NICE[t], f"{dx:g} mm", f"{mean(s, 'r'):.3f}", f"{mean(s, 'slope'):.3f}",
                        f"{mean(s, 'ccc'):.3f}",
                        f"{pct(float(np.mean(rel)))} ({pct(min(rel))} to {pct(max(rel))})",
                        f"{V.get(f'qc.{t}.over10', '')}/{V.get(f'qc.{t}.n', '')}"])
        if sub:
            sl = [V[f"mm.{tag}.{t}.slope"] for t in MM_ORDER if f"mm.{tag}.{t}.slope" in V]
            rb = [V[f"mm.{tag}.{t}.relbias"] for t in MM_ORDER if f"mm.{tag}.{t}.relbias" in V]
            V[f"mm.{tag}.slope_lo"], V[f"mm.{tag}.slope_hi"] = min(sl), max(sl)
            V[f"mm.{tag}.relbias_lo"], V[f"mm.{tag}.relbias_hi"] = min(rb), max(rb)
            V[f"mm.{tag}.r_lo"] = min(V[f"mm.{tag}.{t}.r"] for t in MM_ORDER if f"mm.{tag}.{t}.r" in V)
            V[f"mm.{tag}.r_hi"] = max(V[f"mm.{tag}.{t}.r"] for t in MM_ORDER if f"mm.{tag}.{t}.r" in V)
        for lab, rk in (("int", rk_i), ("foc", rk_f)):
            g_ = [x for x in rk if float(x.get("dx_mm", 0)) == dx]
            if g_:
                V[f"mm.{tag}.rank_{lab}.agree"] = sum(x["agree"] == "True" for x in g_)
                V[f"mm.{tag}.rank_{lab}.n"] = len(g_)
                V[f"mm.{tag}.rank_{lab}.rho"] = float(np.nanmean([f(x["spearman"]) for x in g_]))
                V[f"mm.{tag}.rank_{lab}.n_heads"] = len({x["subject"] for x in g_})
                for roi in ROIS:
                    q = [x for x in g_ if x["roi"] == roi]
                    V[f"mm.{tag}.rank_{lab}.{roi}"] = sum(x["agree"] == "True" for x in q)
                    V[f"mm.{tag}.rank_{lab}.{roi}.n"] = len(q)
                # targets where the methods disagreed, as "the insula (2 of 8)"
                dis = [(roi, V[f"mm.{tag}.rank_{lab}.{roi}.n"] - V[f"mm.{tag}.rank_{lab}.{roi}"])
                       for roi in ROIS]
                V[f"mm.{tag}.rank_{lab}.disagree"] = ", ".join(
                    f"{PROSE[r]} ({k} of {V[f'mm.{tag}.rank_{lab}.{r}.n']})" for r, k in dis if k) or "none"
                # sensitivity (review of 2 October, R4): best of three, the inferior chain removed
                # from both methods' candidates (13 of its 16 reference runs carry the warning);
                # computed from unrounded scores in v21_multimontage_fem.py --focality
                if lab == "foc" and all(x.get("agree3") in ("True", "False") for x in g_):
                    V[f"mm.{tag}.rank3.agree"] = sum(x["agree3"] == "True" for x in g_)
                    V[f"mm.{tag}.rank3.n"] = len(g_)
                    dis3 = []
                    for roi in ROIS:
                        q = [x for x in g_ if x["roi"] == roi]
                        a3 = sum(x["agree3"] == "True" for x in q)
                        V[f"mm.{tag}.rank3.{roi}"] = a3
                        if a3 < len(q):
                            dis3.append(f"{PROSE[roi]} ({len(q) - a3} of {len(q)})")
                    V[f"mm.{tag}.rank3.disagree"] = ", ".join(dis3) or "none"
                    # how often the flagged montage was itself a method's best (what removal tests)
                    V[f"mm.{tag}.inferior_best_sp"] = sum(x["best_spicehead"] == "INFERIOR" for x in g_)
                    V[f"mm.{tag}.inferior_best_fem"] = sum(x["best_fem"] == "INFERIOR" for x in g_)
                    # the manuscript says removal left "the same disagreements"; hold it to that
                    if V[f"mm.{tag}.rank3.disagree"] != V[f"mm.{tag}.rank_foc.disagree"]:
                        MISSING.append(f"best-of-three disagreements at {dx:g} mm differ from "
                                       "best-of-four: reword manuscript §3.2")
                    V[f"mm.{tag}.inferior_best_rois"] = " and ".join(
                        PROSE[r] for r in ROIS if any(x["roi"] == r and "INFERIOR" in
                                                      (x["best_spicehead"], x["best_fem"]) for x in g_))
                elif lab == "foc":
                    MISSING.append(f"best-of-three sensitivity at {dx:g} mm: rerun "
                                   "v21_multimontage_fem.py --focality 8 " + f"{dx:g}")
    T["fem"] = table(["Montage", "Grid", "r", "Slope (FEM on SPICEhead)", "CCC",
                      "Mean bias, % of FEM (range across heads)", "FEM runs flagged"], out)


def sweep(V, T):
    d = _p(RES, "v21_dx2_check_full")
    s = jload(_p(d, "dx2_full_summary.json"), "full sweep summary")
    vs = rows(_p(d, "dx2_full_vs_dx5.csv"), "full sweep table")
    if not s or not vs:
        return
    V["sweep.status"] = s.get("status")
    V["sweep.n"] = len(s.get("expected", []))
    V["sweep.missing"] = len(s.get("missing", []))
    V["sweep.superpos"] = s.get("superposition_worst")
    V["sweep.crosscheck"] = s.get("cross_check_max_rel")
    js = [json.load(open(p, encoding="utf-8")) for p in ls(d, "__dx2_full.json")]
    if js:
        # the superposition identity spans all 171 pairs; numerically it was checked on two
        # directly solved pairs per head (R5.1)
        V["sweep.n_checks"] = sum(len(j.get("basis_check", {})) for j in js)
        V["sweep.checks_per_head"] = max(len(j.get("basis_check", {})) for j in js)
        nn = [j["n_nodes"] for j in js]
        V["sweep.nodes_med_k"] = float(np.median(nn)) / 1e3
        V["sweep.nodes_lo_k"], V["sweep.nodes_hi_k"] = min(nn) / 1e3, max(nn) / 1e3
        V["sweep.gm_med_k"] = float(np.median([j["n_gm_voxels"] for j in js])) / 1e3
        V["sweep.solve_min_med"] = float(np.median([j["solve_s"] for j in js])) / 60
        V["sweep.score_min_med"] = float(np.median([j["score_s"] for j in js])) / 60
    # per-head 2 mm scores ([F, T, O] for every montage), for the exposure of any montage
    sc = {}
    for j in js:
        for roi, m in j["rois"].items():
            sc.setdefault(roi, []).append({label(k): v for k, v in m.items()})

    def fto(roi, mont):
        """Cohort medians of F, T and O at 2 mm for one montage over the 20 heads."""
        v = [q[mont] for q in sc.get(roi, []) if mont in q and q[mont][0] == q[mont][0]]
        return [float(np.median([x[i] for x in v])) for i in range(3)] if v else [float("nan")] * 3

    def fto_cell(x):
        return f"{x[0]:.3f} / {x[1]:.3f} / {x[2]:.3f}"

    out1, out2 = [], []
    kept = 0
    for r in vs:
        roi = r["roi"]
        w2, w5, w5s = label(r["winner_2mm"]), label(r["winner_5mm_300"]), label(r["winner_5mm_same_subjects"])
        V[f"sweep.{roi}.rho300"] = f(r["spearman_vs_300rank"])
        V[f"sweep.{roi}.rho20"] = f(r["spearman_same_subjects"])
        V[f"sweep.{roi}.kept"] = r["winner_kept"] == "True"
        V[f"sweep.{roi}.rank"] = int(r["rank_of_5mm_winner_at_2mm"])
        # where the 2 mm winner sat in the 300-head 5 mm ranking (would a 5 mm shortlist hold it?)
        V[f"sweep.{roi}.rank5_of_2"] = int(r["rank_at_5mm_of_2mm_winner"])
        V[f"sweep.{roi}.in_shortlist"] = r["shortlist_contains_2mm_winner"] == "True"
        V[f"sweep.{roi}.sl_loss"] = f(r["shortlist_cohort_loss_pct"])
        V[f"sweep.{roi}.sl_subj"] = s["per_target"][roi]["shortlist_contains_subject_optimum"]
        V[f"sweep.{roi}.sl_subj_n"] = int(V[f"sweep.{roi}.sl_subj"].split("/")[0])
        # transfer: the 300-head 5 mm winner scored in the 20 heads at 2 mm (resolution AND cohort)
        V[f"sweep.{roi}.loss"] = f(r["cohort_regret_5mm_winner_pct"])
        # paired: the same 20 heads' 5 mm winner at 2 mm (resolution alone; R1, the primary result)
        V[f"sweep.{roi}.same_loss"] = f(r["same_subject_regret_pct"])
        V[f"sweep.{roi}.same_kept"] = r["same_subject_winner_kept"] == "True"
        V[f"sweep.{roi}.same_montage"] = w5s
        V[f"sweep.{roi}.win2"], V[f"sweep.{roi}.win5"] = w2, w5
        for k in ("F", "T", "O"):
            V[f"sweep.{roi}.{k}2"] = f(r[f"{k}_2mm_winner"])
            V[f"sweep.{roi}.{k}5"] = f(r[f"{k}_5mm_winner_at_2mm"])
        for k, x in zip("FTO", fto(roi, w5s)):
            V[f"sweep.{roi}.{k}5s"] = x
        # exposure behind a focality ratio: the 5 mm winner's T and O relative to the 2 mm winner's
        V[f"sweep.{roi}.T_ratio"] = V[f"sweep.{roi}.T5"] / V[f"sweep.{roi}.T2"]
        V[f"sweep.{roi}.O_ratio"] = V[f"sweep.{roi}.O5"] / V[f"sweep.{roi}.O2"]
        # consistency of this module's medians with the sweep's own table
        V[f"sweep.{roi}.fto_dev"] = max(abs(a / b - 1) for a, b in zip(
            fto(roi, w2), (V[f"sweep.{roi}.F2"], V[f"sweep.{roi}.T2"], V[f"sweep.{roi}.O2"])))
        kept += r["winner_kept"] == "True"
        pt = s["per_target"][roi]
        for k in ("cohort5", "own5", "cohort2", "shortlist"):
            g = pt[f"regret_{k}"]
            V[f"sweep.{roi}.{k}.median"] = g["median"]
            V[f"sweep.{roi}.{k}.q1"], V[f"sweep.{roi}.{k}.q3"] = g["q1"], g["q3"]
            V[f"sweep.{roi}.{k}.max"] = g["max"]
            V[f"sweep.{roi}.{k}.over5"] = g["n_over_5pct"]
        same5 = w5s == w2
        out1.append([NICE[roi], w2, fto_cell(fto(roi, w2)),
                     "= 2 mm optimum" if same5 else w5s,
                     "—" if same5 else fto_cell(fto(roi, w5s)),
                     f"{V[f'sweep.{roi}.same_loss']:.2f}%",
                     ("same montage" if w5 == w5s else w5) + f"; {V[f'sweep.{roi}.loss']:.2f}%"])

        def cell(k):
            g = pt[f"regret_{k}"]
            return f"{g['median']:.1f} [{g['q1']:.1f}–{g['q3']:.1f}]; {g['max']:.1f}; {g['n_over_5pct']}"
        out2.append([NICE[roi], r["rank_at_5mm_of_2mm_winner"], f"{V[f'sweep.{roi}.sl_loss']:.2f}%",
                     V[f"sweep.{roi}.sl_subj"], cell("cohort5"), cell("own5"), cell("shortlist")])
    V["sweep.kept"] = kept
    V["sweep.changed"] = len(vs) - kept
    V["sweep.n_roi"] = len(vs)
    V["sweep.same_changed"] = sum(not V[f"sweep.{r}.same_kept"] for r in ROIS)
    V["sweep.fto_dev_max"] = max(V[f"sweep.{r}.fto_dev"] for r in ROIS)
    missed = [r for r in ROIS if f"sweep.{r}.in_shortlist" in V and not V[f"sweep.{r}.in_shortlist"]]
    V["sweep.n_missed_shortlist"] = len(missed)
    V["sweep.missed_shortlist"] = " and ".join(PROSE[r] for r in missed)
    V["sweep.missed_ranks"] = " and ".join(str(V[f"sweep.{r}.rank5_of_2"]) for r in missed)
    V["sweep.sl_loss_max"] = max(V[f"sweep.{r}.sl_loss"] for r in ROIS)
    V["sweep.sl_loss_max_roi"] = PROSE[max(ROIS, key=lambda r: V[f"sweep.{r}.sl_loss"])]
    V["sweep.loss_max_roi"] = PROSE[max(ROIS, key=lambda r: V.get(f"sweep.{r}.loss", -1))]
    lost = [V[f"sweep.{r}.loss"] for r in ROIS if f"sweep.{r}.loss" in V]
    V["sweep.loss_max"] = max(lost) if lost else float("nan")
    sl = [V[f"sweep.{r}.same_loss"] for r in ROIS]
    V["sweep.same_loss_max"] = max(sl)
    V["sweep.same_loss_max_roi"] = PROSE[max(ROIS, key=lambda r: V[f"sweep.{r}.same_loss"])]
    V["sweep.same_lossy"] = sum(x > 0 for x in sl)
    rho = [V[f"sweep.{r}.rho300"] for r in ROIS if f"sweep.{r}.rho300" in V]
    V["sweep.rho300_lo"], V["sweep.rho300_hi"] = min(rho), max(rho)
    rho = [V[f"sweep.{r}.rho20"] for r in ROIS]
    V["sweep.rho20_lo"], V["sweep.rho20_hi"] = min(rho), max(rho)
    for k in ("cohort5", "own5", "shortlist"):
        med = [V[f"sweep.{r}.{k}.median"] for r in ROIS]
        mx = [V[f"sweep.{r}.{k}.max"] for r in ROIS]
        V[f"sweep.{k}.median_lo"], V[f"sweep.{k}.median_hi"] = min(med), max(med)
        V[f"sweep.{k}.max_lo"], V[f"sweep.{k}.max_hi"] = min(mx), max(mx)
        V[f"sweep.{k}.max_roi"] = PROSE[max(ROIS, key=lambda r: V[f"sweep.{r}.{k}.max"])]
    V["sweep.own5_worse"] = sum(V[f"sweep.{r}.own5.median"] > V[f"sweep.{r}.cohort5.median"]
                                for r in ROIS)
    V["sweep.own5_equal"] = sum(V[f"sweep.{r}.own5.median"] == V[f"sweep.{r}.cohort5.median"]
                                for r in ROIS)
    T["sweep_cohort"] = table(["Target", "2 mm optimum", "F / T / O at 2 mm",
                               "5 mm choice, same 20 heads", "Its F / T / O at 2 mm",
                               "Loss, same heads", "300-head 5 mm choice; loss at 2 mm"], out1)
    T["sweep_subject"] = table(["Target", "Rank of 2 mm optimum, 300-head 5 mm ranking",
                                "Best top-25 candidate: cohort loss",
                                "Heads whose own optimum is in the top 25",
                                "Per-head regret: 300-head 5 mm choice",
                                "Per-head regret: head's own 5 mm optimum",
                                "Per-head regret: best top-25 candidate"], out2)


def bench(V, T):
    rs = rows(_p(RES, "v22_benchmark", "heads.csv"), "head benchmark")
    out = []
    for r in rs:
        tag = f"{float(r['dx_mm']):g}mm"
        for k in ("n_nodes", "t_build_s", "t_setup_s", "t_per_solve_s", "t_basis_s", "t_check_s",
                  "t_score_s", "t_total_s", "peak_gb", "n_gm_voxels"):
            V[f"bench.{tag}.{k}"] = f(r[k])
        V[f"bench.{tag}.t_total_min"] = f(r["t_total_s"]) / 60
        V[f"bench.{tag}.t_total_h"] = f(r["t_total_s"]) / 3600
        out.append([f"{float(r['dx_mm']):g} mm", f"{int(f(r['n_nodes'])):,}", r["solver"],
                    f"{f(r['t_build_s']):.1f}", f"{f(r['t_setup_s']):.1f}",
                    f"{f(r['t_per_solve_s']):.1f}", f"{f(r['t_score_s']) / 60:.1f}",
                    f"{f(r['t_total_s']) / 60:.1f}", f"{f(r['peak_gb']):.1f}"])
    if out:
        # labels per review R5.6: per-solve time includes field extraction; "total" is the sum
        # of the instrumented stages, not a separately measured wall clock; memory is the
        # process peak working set
        T["bench"] = table(["Grid", "Nodes", "Solver", "Build (s)", "Setup (s)",
                            "Per basis solve incl. field extraction (s)",
                            "Scoring 11,628 montages (min)", "Sum of stages (min)",
                            "Peak working set (GB)"], out)
    info = jload(_p(RES, "v22_benchmark", "heads.json"), "benchmark machine record")
    if info:
        m = info["machine"]
        V["bench.cpu"], V["bench.cores"] = m["cpu"], m["physical_cores"]
        V["bench.ram_gb"] = m["ram_gb"]
        # Windows lists its idle time as a "process"; it is not competing work
        busy = [b for b in m.get("busy_processes_at_start", []) if not b.startswith("System Idle")]
        V["bench.busy"] = len(busy)
        # their combined load at start, in logical processors (100% = one)
        V["bench.busy_cpus"] = sum(float(b.rsplit(":", 1)[1].rstrip("%")) for b in busy) / 100
        V["bench.logical"] = m.get("logical_cores")
        # BLAS threads of the benchmark process (OPENBLAS_NUM_THREADS); the solver itself is serial
        V["bench.threads"] = m.get("threads")
        V["bench.threads_n"] = (int(m["threads"]) if str(m.get("threads", "")).isdigit()
                                else m.get("threads"))
    cyl = rows(_p(RES0, "v21_benchmark", "stages.csv"), "cylinder benchmark")
    b = next((x for x in cyl if x["nodes"] == "78600"), None)
    if b:
        V["cyl.t_fact_s"], V["cyl.t_per_config_s"] = f(b["t_fact"]), f(b["t_per_config"])


def scale(V, T):
    rs = rows(_p(RES, "v22_scale", "census_dx5.csv"), "5 mm census")
    if rs:
        nn = np.array([f(r["n_nodes"]) for r in rs])
        V["scale.n5"] = len(rs)
        V["scale.nodes5_med_k"] = float(np.median(nn)) / 1e3
        V["scale.nodes5_lo_k"], V["scale.nodes5_hi_k"] = nn.min() / 1e3, nn.max() / 1e3
        V["scale.contact5_med"] = float(np.median([f(r["contact_median"]) for r in rs]))
        V["scale.contact5_min"] = min(int(f(r["contact_min"])) for r in rs)
        V["scale.empty_contacts"] = sum(int(f(r["n_empty_contacts"])) for r in rs)
    for dx in (2, 1):
        q = rows(_p(RES, "v22_scale", f"census_fem8_dx{dx}.csv"), f"FEM-head census {dx} mm")
        if q:
            nn = [f(r["n_nodes"]) for r in q]
            V[f"scale.fem8_{dx}mm_lo"], V[f"scale.fem8_{dx}mm_hi"] = min(nn), max(nn)


def optima(V, T):
    t3 = jload(_p(RES, "montage_optimization", "cohort_gm300_dx5_maxdir",
                  "cohort_montage_optima.json"), "300-head optima")
    sens = rows(_p(RES, "v21_sensitivity_maxdir", "sensitivity.csv"), "focality sensitivity")
    if not t3:
        return
    V["opt.n"] = t3["n_subjects"]
    pub = {r["roi"]: r for r in sens
           if float(r["exclusion_mm"]) == 20.0 and float(r["percentile"]) == 99.0}
    out = []
    for roi in ROIS:
        top = t3["top"][roi][0]
        V[f"opt.{roi}.montage"] = label(top["montage"])
        V[f"opt.{roi}.F"] = top["median_score"]
        V[f"opt.{roi}.F_q1"], V[f"opt.{roi}.F_q3"] = top["iqr"]
        p = pub.get(roi)
        if p:
            V[f"opt.{roi}.T"] = f(p["T_median_Vm"])
            V[f"opt.{roi}.O"] = f(p.get("O_median_Vm"))
            V[f"opt.{roi}.frac_above_1"] = f(p["frac_above_1"])
        out.append([NICE[roi], V[f"opt.{roi}.montage"],
                    f"{top['median_score']:.2f} [{top['iqr'][0]:.2f}–{top['iqr'][1]:.2f}]",
                    f"{V.get(f'opt.{roi}.T', float('nan')):.3f}",
                    f"{V.get(f'opt.{roi}.O', float('nan')):.3f}"])
    V["opt.n_F_above_1"] = sum(V[f"opt.{r}.F"] > 1 for r in ROIS)
    lo, hi = min(ROIS, key=lambda r: V[f"opt.{r}.F"]), max(ROIS, key=lambda r: V[f"opt.{r}.F"])
    V["opt.F_lo"], V["opt.F_lo_roi"] = V[f"opt.{lo}.F"], PROSE[lo]
    V["opt.F_hi"], V["opt.F_hi_roi"] = V[f"opt.{hi}.F"], PROSE[hi]
    T["optima"] = table(["Target", "Cohort-optimal 10-20 montage", "F, median [IQR]",
                         "T, median (V/m)", "O, median (V/m)"], out)
    lo = rows(_p(RES, "v21_generalization", "loso.csv"), "leave-one-site-out")
    if lo:
        loss = [f(r["transfer_loss_pct"]) for r in lo]
        V["loso.n"], V["loso.exact"] = len(lo), sum(r["transfer_is_oracle"] == "True" for r in lo)
        V["loso.median"], V["loso.worst"] = float(np.median(loss)), max(loss)
        V["loso.worst_roi"] = NICE[max(lo, key=lambda r: f(r["transfer_loss_pct"]))["roi"]]
    lc = rows(_p(RES, "v21_generalization", "learning_curve_regret.csv"), "learning curve")
    for n in (5, 20):
        s = [r for r in lc if int(r["n_train"]) == n]
        if s:
            V[f"lc.n{n}.p1"] = 100 * float(np.mean([f(r["p_within_1pct"]) for r in s]))
            V[f"lc.n{n}.median_regret_max"] = max(f(r["median_regret_pct"]) for r in s)


def mni(V, T):
    s = jload(_p(MNI, "TI_targets_MNI152_summary.json"), "MNI target maps")
    if not s:
        return
    V["mni.model"], V["mni.nodes_k"] = s["model"], s["n_nodes"] / 1e3
    V["mni.superpos"] = s["superposition_check"]
    for t in s["targets"]:
        k = t["target"]
        V[f"mni.{k}.montage"] = label(t["montage"])
        V[f"mni.{k}.F"], V[f"mni.{k}.T"], V[f"mni.{k}.O"] = t["F"], t["T_V_per_m"], t["O_V_per_m"]
        V[f"mni.{k}.peak_in"] = t["peak_in_target"]
        V[f"mni.{k}.runner_up"], V[f"mni.{k}.F_runner_up"] = label(t["runner_up"]), t["F_runner_up"]
        # how much exposure can differ among near-equal focality: the five best montages
        top = t.get("top5") or []
        if top:
            Fs, Ts = [e["F"] for e in top], [e["T"] for e in top]
            V[f"mni.{k}.top5_F_lo"], V[f"mni.{k}.top5_F_hi"] = min(Fs), max(Fs)
            V[f"mni.{k}.top5_T_lo"], V[f"mni.{k}.top5_T_hi"] = min(Ts), max(Ts)
            V[f"mni.{k}.top5_T_ratio"] = max(Ts) / min(Ts)


def circuit(V, T):
    d = _p(RES, "v21_twochannel")
    tc = rows(_p(d, "twochannel_exact_dx2.csv"), "co-simulation 2 mm")
    if not tc:
        return
    V["tc.n"] = len({r["subject"] for r in tc})
    sweep_ = sorted({f(r["r2_contact_ohm"]) for r in tc})
    V["tc.sweep_k"] = [x / 1e3 for x in sweep_]
    for k, col in (("fullnet", "chk_fullnet_rel"), ("field", "chk_field_rel"),
                   ("contact", "chk_contact_rel"), ("driver", "chk_driver_rel"),
                   ("recip", "reciprocity_err_pct")):
        v = [f(r[col]) for r in tc if r.get(col) not in (None, "") and np.isfinite(f(r[col]))]
        V[f"tc.chk_{k}"] = max(v) if v else float("nan")
    # head contribution to each channel path: port resistance minus the two contacts in series
    base = [r for r in tc if f(r["r2_contact_ohm"]) == sweep_[0]]
    rhead = [f(r["R11"]) - 2 * 1e3 for r in base]           # channel 1 contacts are 1 kOhm
    V["tc.rhead_mean"] = float(np.mean(rhead))
    V["tc.rhead_lo"], V["tc.rhead_hi"] = min(rhead), max(rhead)
    out = []
    unclamped_max = None
    for r2 in sweep_:
        s = [r for r in tc if f(r["r2_contact_ohm"]) == r2]
        k = f"tc.{r2 / 1e3:g}k"
        A2 = np.array([f(r["A2_td_mA"]) for r in s])
        com = np.array([f(r["com_shift_td_mm"]) for r in s])
        ncl = sum(r["ch2_clamped"] == "True" for r in s)
        V[f"{k}.I2"], V[f"{k}.clamped"] = float(A2.mean()), ncl
        V[f"{k}.dT"] = mean(s, "dT_td_pct")
        V[f"{k}.dF"] = mean(s, "dF_td_pct")
        V[f"{k}.com"], V[f"{k}.com_sd"] = float(com.mean()), float(com.std(ddof=1))
        V[f"{k}.com_lo"], V[f"{k}.com_hi"] = float(com.min()), float(com.max())
        V[f"{k}.comtop"] = mean(s, "com_top_shift_td_mm")
        V[f"{k}.com_static"] = mean(s, "com_shift_mm")
        V[f"{k}.gain"] = mean(s, "td_over_static_I2")
        if ncl == 0:
            unclamped_max = r2
        out.append([f"{r2 / 1e3:g} kΩ", f"{A2.mean():.3f} mA", f"{ncl}/{len(s)}",
                    pct(V[f"{k}.dT"]), pct(V[f"{k}.dF"]),
                    f"{com.mean():.1f} ± {com.std(ddof=1):.1f} mm",
                    f"{V[f'{k}.comtop']:.1f} mm"])
    V["tc.unclamped_max_k"] = unclamped_max / 1e3 if unclamped_max else float("nan")
    # largest shortfall of delivered current, either channel and any head, while none was clamped
    unc = [f(r[c]) for r in tc for c in ("A1_td_mA", "A2_td_mA")
           if unclamped_max and f(r["r2_contact_ohm"]) <= unclamped_max]
    V["tc.unclamped_dev_pct"] = 100 * (1 - min(unc)) if unc else float("nan")
    com_unc = [f(r["com_shift_td_mm"]) for r in tc
               if unclamped_max and f(r["r2_contact_ohm"]) <= unclamped_max]
    V["tc.unclamped_com_max"] = max(com_unc) if com_unc else float("nan")
    first_clamped = next((x for x in sweep_ if x > (unclamped_max or 0)), None)
    V["tc.first_clamped_k"] = first_clamped / 1e3 if first_clamped else float("nan")
    # predicted per-contact boundary for the channel-2 path: I (R_head + 2 R_el) = V_cc
    sys.path.insert(0, _here)
    from v21_twochannel_cosim import I_PROG, V_CC              # noqa: E402
    V["tc.i_prog_mA"], V["tc.vcc"] = 1e3 * I_PROG, V_CC
    V["tc.boundary_k"] = (V_CC / I_PROG - V["tc.rhead_mean"]) / 2 / 1e3
    T["circuit"] = table(["Channel-2 contact", "Delivered I₂", "Clamped", "Δ thalamic T",
                          "Δ focality", "Center-of-mass shift", "Top-decile shift"], out)
    t1 = rows(_p(d, "twochannel_exact_dx1.csv"), "co-simulation 1 mm")
    m2 = {(r["subject"], f(r["r2_contact_ohm"])): r for r in tc}
    for r2 in sorted({f(r["r2_contact_ohm"]) for r in t1}):
        pr = [(f(x["com_shift_mm"]), f(m2[(x["subject"], r2)]["com_shift_mm"])) for x in t1
              if f(x["r2_contact_ohm"]) == r2 and (x["subject"], r2) in m2]
        if len(pr) < 2:
            continue
        a, b = np.array([q[0] for q in pr]), np.array([q[1] for q in pr])
        k = f"tc1.{r2 / 1e3:g}k"
        V[f"{k}.com1"], V[f"{k}.com2"] = float(a.mean()), float(b.mean())
        V[f"{k}.diff"], V[f"{k}.diff_sd"] = float((a - b).mean()), float((a - b).std(ddof=1))
        if b.std() > 0 and a.std() > 0:
            V[f"{k}.r"] = float(np.corrcoef(a, b)[0, 1])
            V[f"{k}.p"] = float(sstats.ttest_rel(a, b).pvalue)
    smp = rows(_p(d, "twocarrier_sampling.csv"), "two-carrier sampling")
    if smp:
        V["tc.sampling_100"] = max(abs(f(r["reldiff_spc100"])) for r in smp)
    gain = rows(_p(d, "twocarrier_dx1_gain.csv"), "two-carrier 1 mm gain")
    if gain:
        V["tc.gain_dx_diff"] = max(abs(f(r["gain_diff"])) for r in gain)


def fullsearch(V, T):
    """Matched full-space comparison with FEM (plan of 2 October 2026, item 1; Supplement S11):
    every montage scored by FEM, SPICEhead 2 mm and SPICEhead 1 mm on the same grid with the same
    masks (v22_fem_fullsearch.py).

    Since 22 October 2026 (PI decision) the decision tables come from the 300-head run on NCHC
    Nano 4 (results_v22/v22_fem_fullsearch_300, collected 18 October). The eight-head desktop run
    (results_v22/v22_fem_fullsearch) still supplies what only exists there: the leadfield
    validation, the passive-pad and four-pad checks, and the same-machine timings."""
    d8 = FS8_DIR or _p(RES, "v22_fem_fullsearch")         # eight desktop heads (checks, timings)
    d = FS300_DIR or _p(RES, "v22_fem_fullsearch_300")    # 300 heads (decisions, cohort level)
    if not os.path.isdir(d) and FS300_DIR is None:    # v23 must never fall back to the v22 desktop run
        d = d8
    # one-off validation of the leadfield basis (SUBJ05, T8-P8): superposition within one 19-pad
    # geometry, the passive-pad effect, and the leadfield mapping against a direct 19-pad run
    val = jload(_p(d8, "fem_validation_SUBJ05.json"), "FEM leadfield validation")
    if val:
        V["fsv.superpos"] = val["superposition_allpads"]["max_rel_of_max"]
        V["fsv.pad_max"] = 100 * val["passive_pad_effect"]["max_rel_of_max"]
        V["fsv.pad_median"] = 100 * val["passive_pad_effect"]["median_voxel_rel"]
        V["fsv.pad_ratio"] = val["passive_pad_effect"]["magnitude_ratio_median"]
        V["fsv.lf_max"] = 100 * val["leadfield_vs_allpads_direct"]["max_rel_of_max"]
        V["fsv.lf_median"] = 100 * val["leadfield_vs_allpads_direct"]["median_voxel_rel"]
        V["fsv.lf_ratio"] = val["leadfield_vs_allpads_direct"]["magnitude_ratio_median"]
        tr = list(val["T_ratio_leadfield_over_allpads"].values())
        orr = list(val["O_ratio_leadfield_over_allpads"].values())
        fr = list(val["F_ratio_leadfield_over_allpads"].values())
        V["fsv.T_dev_max"] = 100 * max(abs(x - 1) for x in tr)
        V["fsv.O_dev_max"] = 100 * max(abs(x - 1) for x in orr)
        V["fsv.F_dev_max"] = 100 * max(abs(x - 1) for x in fr)
        if val.get("solve_s_per_electrode"):
            V["fsv.solve_s"] = float(np.median(val["solve_s_per_electrode"]))
    # direct four-pad FEM of the three models' optima (v22_fem_4pad_check.py): the leadfield
    # carries 15 unused pads per montage, a real montage and SPICEhead carry none
    pads = rows(_p(d8, "fem_passive_pads.csv"), "passive-pad effect")
    if pads:
        r_ = [f(x["magnitude_ratio_median"]) for x in pads]
        V["fsp.n"] = len(pads)
        V["fsp.ratio_lo"], V["fsp.ratio_hi"] = min(r_), max(r_)
        V["fsp.weaker_max"] = 100 * (1 - min(r_))          # leadfield weaker than the 2-pad run
        V["fsp.stronger_max"] = 100 * (max(r_) - 1)        # leadfield stronger
        V["fsp.absdev_median"] = 100 * float(np.median([abs(x - 1) for x in r_]))
    p4 = jload(_p(d8, "fem4pad_summary.json"), "four-pad FEM check")
    sp_models = ("fem", "sp1", "sp2") if FS_V22_PARTS else ("fem",)
    if p4:
        for m in sp_models:
            k = p4.get(f"keep_{m}")
            if k:
                V[f"fs4.keep_{m}.median"], V[f"fs4.keep_{m}.min"] = k["median"], k["min"]
                V[f"fs4.keep_{m}.n_best"], V[f"fs4.keep_{m}.n"] = k["n_best"], k["n"]
        # leadfield F / four-pad F, by explicitly labelled population (review R4-C)
        for key, tag in (("F_leadfield_over_4pad", "Fratio"), ("F_leadfield_over_4pad_selected", "Fratio_sel"),
                         ("F_leadfield_over_4pad_all_rows", "Fratio_all")):
            fr = p4.get(key)
            if fr:
                V[f"fs4.{tag}.median"], V[f"fs4.{tag}.min"], V[f"fs4.{tag}.max"] = fr["median"], fr["min"], fr["max"]
                V[f"fs4.{tag}.n"] = fr.get("n")
        if p4.get("n_candidates"):
            V["fs4.n_candidates"] = p4["n_candidates"]["total"]
        if FS_V22_PARTS:
            V["fs4.n_disagree"] = p4["fem_vs_sp1_disagreements"]
            V["fs4.sp1_better"] = p4["sp1_choice_better_under_4pad"]
    # SimNIBS electrode-current diagnostics of the four-pad runs and the post hoc sensitivity
    qc = jload(_p(d8, "fem4pad_qc.json"), "four-pad FEM diagnostics")
    if qc:
        for k in ("n_logs", "n_events", "n_warn_messages", "n_warn_active_messages", "n_logs_any_warn",
                  "n_logs_active_warn", "n_geometries", "n_geometries_flagged", "max_pct",
                  "n_comparisons", "n_comparisons_flagged", "n_comparisons_clean"):
            V[f"fs4qc.{k}"] = qc[k]
        for m in sp_models:
            x = qc["sensitivity_clean_comparisons"].get(f"keep_{m}")
            if x:
                V[f"fs4qc.keep_{m}.median"], V[f"fs4qc.keep_{m}.min"] = x["median"], x["min"]
    summ = jload(_p(d, "fullsearch_summary.json"), "full-space FEM comparison summary")
    dec = rows(_p(d, "fullsearch_decisions.csv"), "full-space FEM comparison decisions")
    coh = rows(_p(d, "fullsearch_cohort.csv"), "full-space FEM comparison cohort table")
    tim = rows(_p(d8, "fullsearch_timing.csv"), "full-space FEM comparison timings (desktop)") if FS_V22_PARTS else []
    tim300 = rows(_p(d, "fullsearch_timing.csv"), "300-head timings") if d != d8 else []
    if not (summ and dec and coh and (tim or not FS_V22_PARTS)):
        return
    MP = [("fem", "sp2"), ("fem", "sp1"), ("sp2", "sp1")]
    NAME = {"fem": "FEM", "sp2": "SPICEhead 2 mm", "sp1": "SPICEhead 1 mm"}
    V["fs.n_heads"] = len(summ["heads_complete"])
    V["fs.n_rows"] = len(dec)
    V["fs.n_montages"] = int(f(dec[0]["n_montages"]))
    for a, b in MP:
        k = f"{a}_{b}"
        ag = [r[f"agree_{k}"] == "True" for r in dec]
        V[f"fs.agree_{k}"], V[f"fs.n_{k}"] = sum(ag), len(ag)
        rho = [f(r[f"rho_{k}"]) for r in dec]
        V[f"fs.rho_{k}.median"], V[f"fs.rho_{k}.min"] = float(np.median(rho)), min(rho)
        rho = [f(r[f"rho_top100_{k}"]) for r in dec]
        V[f"fs.rho100_{k}.median"] = float(np.median(rho))
        ra = [f(r[f"regret_{a}winner_under_{b}_pct"]) for r in dec]
        rb = [f(r[f"regret_{b}winner_under_{a}_pct"]) for r in dec]
        V[f"fs.regret_{a}_under_{b}.median"], V[f"fs.regret_{a}_under_{b}.max"] = float(np.median(ra)), max(ra)
        V[f"fs.regret_{b}_under_{a}.median"], V[f"fs.regret_{b}_under_{a}.max"] = float(np.median(rb)), max(rb)
        # the same, read as the share of the other model's optimum that the choice retains
        V[f"fs.keep_{a}_under_{b}.median"], V[f"fs.keep_{a}_under_{b}.min"] = 100 - float(np.median(ra)), 100 - max(ra)
        V[f"fs.keep_{b}_under_{a}.median"], V[f"fs.keep_{b}_under_{a}.min"] = 100 - float(np.median(rb)), 100 - max(rb)
        # over the combinations where the two models chose different montages only
        dis_rows = [r for r in dec if r[f"agree_{k}"] != "True"]
        if dis_rows:
            da = [f(r[f"regret_{a}winner_under_{b}_pct"]) for r in dis_rows]
            db = [f(r[f"regret_{b}winner_under_{a}_pct"]) for r in dis_rows]
            V[f"fs.dis_regret_{a}_under_{b}.median"] = float(np.median(da))
            V[f"fs.dis_regret_{b}_under_{a}.median"] = float(np.median(db))
            V[f"fs.dis_keep_{a}_under_{b}.median"] = 100 - float(np.median(da))
            V[f"fs.dis_keep_{b}_under_{a}.median"] = 100 - float(np.median(db))
            # how many disagreements cost less than 2% of the other model's optimum
            V[f"fs.dis_small_{a}_under_{b}"] = sum(x < 2 for x in da)
            V[f"fs.dis_small_{b}_under_{a}"] = sum(x < 2 for x in db)
            V[f"fs.dis_n_{k}"] = len(dis_rows)
        # 5th percentile of retention and the number of head-target rows keeping < 90 % (300 heads)
        V[f"fs.keep_{a}_under_{b}.p5"] = 100 - float(np.percentile(ra, 95))
        V[f"fs.keep_{b}_under_{a}.p5"] = 100 - float(np.percentile(rb, 95))
        V[f"fs.low90_{a}_under_{b}"] = int(sum(x > 10 for x in ra))
        V[f"fs.low90_{b}_under_{a}"] = int(sum(x > 10 for x in rb))
        V[f"fs.top10_{k}.median"] = float(np.median([f(r[f"top10_overlap_{k}"]) for r in dec]))
        V[f"fs.top25_{k}.median"] = float(np.median([f(r[f"top25_overlap_{k}"]) for r in dec]))
        dis = []
        for roi in ROIS:
            sub = [r for r in dec if r["roi"] == roi]
            n_dis = sum(r[f"agree_{k}"] != "True" for r in sub)
            V[f"fs.agree_{k}.{roi}"], V[f"fs.n_{k}.{roi}"] = len(sub) - n_dis, len(sub)
            if n_dis:
                dis.append(f"{PROSE[roi]} ({n_dis} of {len(sub)})")
        V[f"fs.disagree_{k}"] = ", ".join(dis) or "none"
    # cohort level (median F over the heads) per target
    for r in coh:
        roi = r["roi"]
        for m in NAME:
            V[f"fs.coh.{roi}.win_{m}"] = label(r[f"winner_{m}"])
            for q in "FTO":
                V[f"fs.coh.{roi}.{q}_{m}"] = f(r[f"{q}_{m}"])
        for a, b in MP:
            V[f"fs.coh.{roi}.agree_{a}_{b}"] = r[f"agree_{a}_{b}"] == "True"
            V[f"fs.coh.{roi}.regret_{a}_under_{b}"] = f(r[f"regret_{a}winner_under_{b}_pct"])
            V[f"fs.coh.{roi}.regret_{b}_under_{a}"] = f(r[f"regret_{b}winner_under_{a}_pct"])
    for a, b in MP:
        V[f"fs.coh.agree_{a}_{b}"] = sum(V.get(f"fs.coh.{r}.agree_{a}_{b}", False) for r in ROIS)
        V[f"fs.coh.regret_{a}_under_{b}.max"] = max(V.get(f"fs.coh.{r}.regret_{a}_under_{b}", 0) for r in ROIS)
        V[f"fs.coh.regret_{b}_under_{a}.max"] = max(V.get(f"fs.coh.{r}.regret_{b}_under_{a}", 0) for r in ROIS)
        V[f"fs.coh.keep_{a}_under_{b}.min"] = 100 - V[f"fs.coh.regret_{a}_under_{b}.max"]
        V[f"fs.coh.keep_{b}_under_{a}.min"] = 100 - V[f"fs.coh.regret_{b}_under_{a}.max"]
    # rows keeping < 90 % of the FEM optimum with the 1 mm choice, by target (300 heads)
    low = [r for r in dec if f(r["regret_sp1winner_under_fem_pct"]) > 10]
    V["fs.low90_rows"] = ", ".join(f"{PROSE[roi]} ({sum(r['roi'] == roi for r in low)})" for roi in ROIS
                                   if any(r["roi"] == roi for r in low)) or "none"
    V["fs.low90_keep_min"] = 100 - max((f(r["regret_sp1winner_under_fem_pct"]) for r in low), default=0)
    if tim300:
        chk = [f(r.get("basis_check_max")) for r in tim300 if r["model"] != "fem" and np.isfinite(f(r.get("basis_check_max")))]
        V["fs300.check_max"] = max(chk) if chk else float("nan")
        V["fs300.n_heads_timed"] = len({r["subject"] for r in tim300})
    # re-mesh check (18 October): outlier heads and matched controls re-meshed and rerun
    rm = jload(_p(d, "remesh_analysis.json"), "re-mesh check") if d != d8 else None
    if rm:
        ctl = [x for x in rm if x["role"] == "control"]
        outl = [x for x in rm if x["outlier_row"]]
        V["fsr.n_outlier_heads"] = len({x["code"] for x in rm if x["role"] == "outlier"})
        V["fsr.n_control_heads"] = len({x["code"] for x in ctl})
        V["fsr.n_outlier_rows"] = len(outl)
        V["fsr.outlier_still_low"] = sum(x["sp1_keeps_newfem"] < 90 for x in outl)
        V["fsr.outlier_keep_new_lo"] = min(x["sp1_keeps_newfem"] for x in outl)
        V["fsr.outlier_keep_new_hi"] = max(x["sp1_keeps_newfem"] for x in outl)
        dlt = [x["sp1_keeps_newfem"] - x["sp1_keeps_oldfem"] for x in ctl]
        V["fsr.ctrl_delta_absmax"] = max(abs(x) for x in dlt)
        V["fsr.ctrl_rho_oldnew.median"] = float(np.median([x["rho_fem_old_new"] for x in ctl]))
        V["fsr.ctrl_rho_oldnew.min"] = min(x["rho_fem_old_new"] for x in ctl)
        V["fsr.ctrl_fem_same"] = sum(x["fem_same_winner"] for x in ctl)
        V["fsr.ctrl_n_rows"] = len(ctl)
        V["fsr.ctrl_oldwin_keeps_min"] = min(x["oldwin_keeps_newfem"] for x in ctl)
        # v23: no SPICEhead rerun on the re-meshed heads (None); its network does not depend on the FEM mesh
        rr = [x["sp1_rerun_maxrel"] for x in rm if x["sp1_rerun_maxrel"] is not None]
        if rr:
            V["fsr.sp1_rerun_maxrel"] = max(rr)
    # the eight desktop heads inside the 300-head run (new meshes, GPU solver) against the desktop run
    c8 = jload(_p(d, "consistency_vs_8head.json"), "8-head consistency") if d != d8 and FS_V22_PARTS else None
    if c8:
        for m in ("fem", "sp2", "sp1"):
            V[f"fs8.{m}.same"], V[f"fs8.{m}.n"] = c8[m]["same_winner"], c8[m]["n"]
            V[f"fs8.{m}.dF.median"], V[f"fs8.{m}.dF.max"] = c8[m]["rel_change_F_median"], c8[m]["rel_change_F_max"]
            V[f"fs8.{m}.dF.median_pct"] = 100 * c8[m]["rel_change_F_median"]
        V["fs8.n_heads"] = c8["n_heads"]
    # stage times on one machine (the eight desktop heads), and the two numerical checks
    out_t = []
    for m in NAME:
        s = [r for r in tim if r["model"] == m]
        if not s:
            continue
        # a zero basis time marks a leadfield reused from the validation test; leave it out
        tb = [f(r["t_basis_s"]) for r in s if f(r["t_basis_s"]) > 0]
        V[f"fs.t.{m}.per_solve"] = float(np.median([f(r["t_per_solve_s"]) for r in s if f(r["t_basis_s"]) > 0]))
        V[f"fs.t.{m}.basis_min"] = float(np.median(tb)) / 60
        V[f"fs.t.{m}.score_min"] = float(np.median([f(r["t_score_s"]) for r in s])) / 60
        # FEM: exact by construction (one geometry), so no numerical check to report here
        chk = [f(r["basis_check_max"]) for r in s] if m != "fem" else []
        chk = [c for c in chk if np.isfinite(c)]
        V[f"fs.t.{m}.check_max"] = max(chk) if chk else float("nan")
        nn = [f(r["n_nodes"]) for r in s if np.isfinite(f(r["n_nodes"]))]
        V[f"fs.t.{m}.nodes_med_k"] = float(np.median(nn)) / 1e3 if nn else float("nan")
        out_t.append([NAME[m], f"{V[f'fs.t.{m}.nodes_med_k']:.0f}k" if nn else "tetrahedral mesh",
                      f"{V[f'fs.t.{m}.per_solve']:.0f}", f"{V[f'fs.t.{m}.basis_min']:.1f}",
                      f"{V[f'fs.t.{m}.score_min']:.1f}",
                      sci(V[f"fs.t.{m}.check_max"]) if chk else "exact (one geometry)"])
    if out_t:
        T["fs_timing"] = table(["Model", "Nodes per head, median", "Per basis solve (s)",
                                "18 basis solves (min)", "Scoring 11,628 montages (min)",
                                "Superposition check, largest relative difference"], out_t)
    # clean sequential benchmark on one head (idle machine), when it exists
    b = jload(_p(d8, "bench", "bench_SUBJ06.json"), "clean one-head benchmark") \
        if FS_V22_PARTS and os.path.exists(_p(d8, "bench", "bench_SUBJ06.json")) else None
    V["fsb.available"] = bool(b)
    if b:
        out_b = []
        for m in ("fem", "sp2", "sp1"):
            x = b.get(m)
            if not x:
                continue
            size = (f"{x['n_elements']/1e6:.2f}M elements" if m == "fem" else f"{x['n_nodes']/1e6:.2f}M nodes")
            prep = x.get("t_map_s") if m == "fem" else (x.get("t_build_s", 0) + x.get("t_setup_s", 0))
            V[f"fsb.{m}.basis_min"] = x["t_basis_s"] / 60
            V[f"fsb.{m}.per_solve"] = x["t_per_solve_s"]
            out_b.append([NAME[m], size, f"{x['t_per_solve_s']:.0f}", f"{x['t_basis_s']/60:.1f}",
                          f"{prep/60:.1f}" if prep is not None else "—"])
        V["fsb.score_min"] = b.get("score_s", float("nan")) / 60
        # contention at the start of the benchmark (other processes), in logical processors
        mj = _p(d8, "bench", "machine_at_start.json")
        if os.path.exists(mj):
            mm = json.load(open(mj, encoding="utf-8"))
            busy = [x for x in mm.get("busy_processes_at_start", []) if not x.startswith("System Idle")]
            V["fsb.busy_n"] = len(busy)
            V["fsb.busy_cpus"] = sum(float(x.rsplit(":", 1)[1].rstrip("%")) for x in busy) / 100
            V["fsb.logical"] = mm.get("logical_cores")
        T["fs_bench"] = table(["Model", "Size", "Per basis solve (s)", "18 basis solves (min)",
                               "Mapping to the grid or network build (min)"], out_b)
    out = []
    for roi in ROIS + ["all"]:
        sub = dec if roi == "all" else [r for r in dec if r["roi"] == roi]
        c = [NICE.get(roi, "All targets")]
        for a, b in MP:
            c.append(f"{sum(r[f'agree_{a}_{b}'] == 'True' for r in sub)}/{len(sub)}")
        rho1 = [f(r["rho_fem_sp1"]) for r in sub]
        c.append(f"{np.median(rho1):.3f} [{min(rho1):.3f}]")
        k1 = [100 - f(r["regret_sp1winner_under_fem_pct"]) for r in sub]
        k2 = [100 - f(r["regret_sp2winner_under_fem_pct"]) for r in sub]
        c += [f"{np.median(k1):.1f} / {np.percentile(k1, 5):.1f} / {min(k1):.1f}",
              f"{np.median(k2):.1f} / {np.percentile(k2, 5):.1f} / {min(k2):.1f}"]
        out.append(c)
    T["fs_decisions"] = table(["Target", "Same best montage: FEM vs 2 mm", "FEM vs 1 mm", "2 mm vs 1 mm",
                               "ρ over all montages, FEM vs 1 mm (median [lowest])",
                               "1 mm choice: share of the FEM optimum's F kept, % (median / 5th pct / min)",
                               "2 mm choice: share kept, % (median / 5th pct / min)"], out)
    out = []
    for r in coh:
        roi = r["roi"]
        out.append([NICE[roi], label(r["winner_fem"]),
                    f"{f(r['F_fem']):.3f} / {f(r['T_fem']):.3f} / {f(r['O_fem']):.3f}",
                    label(r["winner_sp2"]) if r["agree_fem_sp2"] != "True" else "same",
                    label(r["winner_sp1"]) if r["agree_fem_sp1"] != "True" else "same",
                    f"{f(r['F_sp1']):.3f} / {f(r['T_sp1']):.3f} / {f(r['O_sp1']):.3f}",
                    f"{f(r['regret_sp1winner_under_fem_pct']):.2f}%",
                    f"{f(r['regret_femwinner_under_sp1_pct']):.2f}%"])
    T["fs_cohort"] = table(["Target", "FEM optimum", "FEM F / T / O", "2 mm optimum", "1 mm optimum",
                            "1 mm F / T / O", "Loss of the 1 mm choice under FEM",
                            "Loss of the FEM choice under 1 mm"], out)


def contact_change(V, T):
    """v21 versus v22 at 2 mm against FEM (8 heads x 4 montages), for the supplement."""
    rs = rows(_p(RES0, "v21_contact_sensitivity", "contact_agreement.csv"), "contact sensitivity")
    if not rs:
        return
    for (rule, sup), tag in ((("arithmetic", "volume"), "v21"), (("harmonic", "surface"), "v22")):
        s = [r for r in rs if r["rule"] == rule and r["support"] == sup]
        for t in MM_ORDER:
            q = [r for r in s if r["montage"] == t]
            if q:
                V[f"csens.{tag}.{t}.relbias"] = mean(q, "rel_bias_pct")
                V[f"csens.{tag}.{t}.r"] = mean(q, "r")
    sup = rows(_p(RES0, "v21_contact_sensitivity", "contact_support.csv"), "contact supports")
    for kind in ("volume", "surface"):
        s = [r for r in sup if r["support"] == kind]
        if s:
            V[f"csens.{kind}.nodes_med"] = float(np.median([f(r["n_nodes"]) for r in s]))
            V[f"csens.{kind}.nonscalp_med"] = float(np.median([f(r["non_scalp_pct"]) for r in s]))
            V[f"csens.{kind}.nonscalp_max"] = max(f(r["non_scalp_pct"]) for r in s)


SOURCES = {"white_matter": "Wagner et al. 2004; Opitz et al. 2015",
           "gray_matter": "Wagner et al. 2004; Opitz et al. 2015", "csf": "Wagner et al. 2004",
           "bone": "Wagner et al. 2004", "scalp": "Wagner et al. 2004",
           "eyeballs": "SimNIBS default", "compact_bone": "Opitz et al. 2015",
           "spongy_bone": "Opitz et al. 2015", "blood": "SimNIBS default",
           "muscle": "SimNIBS default"}


def supplement(V, T):
    """Tables that only the supplement carries."""
    from spicehead.conductivity import CHARM_LABELS, CONDUCTIVITY
    T["conductivities"] = table(["CHARM label", "Tissue", "σ (S/m)", "Source"], [
        [lab, {"csf": "CSF"}.get(name, name.replace("_", " ").capitalize()),
         f"{CONDUCTIVITY[name]:.3f}", SOURCES[name]]
        for lab, name in sorted(CHARM_LABELS.items()) if lab > 0])
    d = rows(_p(RES0, "v21_validation", "disk_analytic_gradient.csv"), "disk benchmark")
    T["disk"] = table(["Grid", "Nodes", "Network vs exact gradient: nMAE (%)", "Slope",
                       "Network vs discrete reference: nMAE (%)",
                       "Discrete reference vs exact: nMAE (%)", "Discrete reference vs exact: p95 (%)"],
                      [[f"{f(r['dx_mm']):g} mm", f"{int(f(r['n_nodes'])):,}",
                        f"{f(r['model_vs_exact_nmae_pct']):.3f}",
                        f"{f(r['model_vs_exact_slope']):.4f}",
                        f"{f(r['model_vs_discrete_nmae_pct']):.3f}",
                        f"{f(r['discrete_vs_exact_nmae_pct']):.4f}",
                        f"{f(r['discrete_vs_exact_p95_abs_rel_pct']):.4f}"] for r in d])
    s = rows(_p(RES0, "v21_interface", "interface_rule.csv"), "layered slab")
    by = {}
    for r in s:
        by.setdefault(f(r["dx_mm"]), {})[r["rule"]] = f(r["R_ratio"])
    T["slab"] = table(["Grid", "Harmonic", "Arithmetic"],
                      [[f"{dx:g} mm", f"{v.get('harmonic', float('nan')):.4f}",
                        f"{v.get('arithmetic', float('nan')):.4f}"] for dx, v in sorted(by.items(), reverse=True)])
    # per-head FEM agreement, bilateral temporal montage, unadjusted (pseudonymous head numbers)
    raw = rows(_p(RES, "v21_simnibs_validation", "agreement_raw.csv"), "FEM raw agreement")
    heads = sorted({r["subject"] for r in raw})
    out = []
    for n, h in enumerate(heads, 1):
        cells = [f"Head {n}", next(r["site"] for r in raw if r["subject"] == h),
                 f"{int(float(next(r['age'] for r in raw if r['subject'] == h)) // 10 * 10)}s"]
        for dx in ("2.0", "1.0"):
            q = next((r for r in raw if r["subject"] == h and r["dx_mm"] == dx), None)
            cells += ([f"{f(q['r']):.3f}", f"{f(q['slope']):.3f}",
                       pct(100 * (f(q["mean_spicehead"]) / f(q["mean_fem"]) - 1))]
                      if q else ["—"] * 3)
        out.append(cells)
    T["fem_heads"] = table(["Head", "Site", "Age", "r, 2 mm", "Slope, 2 mm", "Bias, 2 mm",
                            "r, 1 mm", "Slope, 1 mm", "Bias, 1 mm"], out)
    T["qc"] = table(["Montage", "Runs", "Median diagnostic (%)", "Runs above 10%"],
                    [[MM_NICE[t], V.get(f"qc.{t}.n", "—"), f"{V.get(f'qc.{t}.median', float('nan')):.2f}",
                      V.get(f"qc.{t}.over10", "—")] for t in MM_ORDER])
    rk = {lab: rows(_p(RES, "v21_multimontage", fn), "ranks")
          for lab, fn in (("F", "rank_preservation_focality.csv"), ("T", "rank_preservation.csv"))}
    out = []
    for roi in ROIS + ["all"]:
        c = [NICE.get(roi, "All targets")]
        for lab, col in (("F", "agree"), ("F", "agree3"), ("T", "agree")):
            for dx in (2.0, 1.0):
                g = [r for r in rk[lab] if (roi == "all" or r["roi"] == roi) and f(r["dx_mm"]) == dx]
                c.append(f"{sum(r.get(col) == 'True' for r in g)}/{len(g)}"
                         if g and all(r.get(col) in ("True", "False") for r in g) else "—")
        out.append(c)
    T["decisions"] = table(["Target", "By F, best of 4, 2 mm", "By F, best of 4, 1 mm",
                            "By F, best of 3, 2 mm", "By F, best of 3, 1 mm",
                            "By T, best of 4, 2 mm", "By T, best of 4, 1 mm"], out)
    lo = rows(_p(RES, "v21_generalization", "loso.csv"), "leave-one-site-out")
    if lo:
        sites = sorted({r["held_out_site"] for r in lo})
        T["loso"] = table(["Target"] + [f"Held out: {x}" for x in sites], [
            [NICE[roi]] + [next((f"{f(r['transfer_loss_pct']):.2f}%" + (" (same)" if r["transfer_is_oracle"] == "True" else "")
                                 for r in lo if r["roi"] == roi and r["held_out_site"] == x), "—") for x in sites]
            for roi in ROIS])
    lc = rows(_p(RES, "v21_generalization", "learning_curve_regret.csv"), "learning curve")
    if lc:
        ns = sorted({int(r["n_train"]) for r in lc})
        T["learning"] = table(["Training heads"] + [SHORT_[r] for r in ROIS], [
            [str(n)] + [next((f"{100 * f(r['p_within_1pct']):.0f}% / {f(r['median_regret_pct']):.2f}%"
                              for r in lc if int(r["n_train"]) == n and r["roi"] == roi), "—") for roi in ROIS]
            for n in ns])
    se = rows(_p(RES, "v21_sensitivity_maxdir", "sensitivity.csv"), "focality sensitivity")
    if se:
        cols = [(20.0, 95.0), (20.0, 99.0), (20.0, 100.0), (10.0, 99.0), (30.0, 99.0)]
        T["sens"] = table(["Target"] + [f"{int(e)} mm, p{q:g}" for e, q in cols], [
            [NICE[roi]] + [next((f"{f(r['F_median']):.2f}" for r in se if r["roi"] == roi
                                 and f(r["exclusion_mm"]) == e and f(r["percentile"]) == q), "—")
                           for e, q in cols] for roi in ROIS])
        cnt = {}
        for r in se:
            cnt.setdefault((f(r["exclusion_mm"]), f(r["percentile"])), 0)
            cnt[(f(r["exclusion_mm"]), f(r["percentile"]))] += f(r["F_median"]) > 1
        V["sens.n_above1_lo"], V["sens.n_above1_hi"] = min(cnt.values()), max(cnt.values())
        V["sens.n_above1_pub"] = cnt.get((20.0, 99.0))
    rt = jload(_p(RES, "montage_optimization", "montage_ratio_n300_maxdir.json"), "current ratio")
    if rt:
        pr = rt["per_roi"]
        T["ratio"] = table(["Target", "Best ratio, median", "Gain over 1:1, median"],
                           [[NICE[r], f"{pr[r]['median_best_ratio']:g}", pct(pr[r]["median_gain_pct"])]
                            for r in ROIS if r in pr])
        V["ratio.max_gain"] = max(pr[r]["median_gain_pct"] for r in ROIS if r in pr)
    # (the 10-10 extension was dropped from the v22 paper on 2 October 2026)
    for tis in ("brain", "gm"):
        p = _p(RES, f"aging_n300_{tis}_maxdir", "cohort_roi_metrics.csv")
        if not os.path.exists(p):
            MISSING.append(f"age analysis ({tis}): {os.path.relpath(p, ROOT)}")
            continue
        rs = list(csv.DictReader(open(p, encoding="cp1252")))
        age = np.array([float(x["Age"]) for x in rs])
        out = []
        for roi in ROIS:
            y = np.array([float(x[f"thalamic_{roi}_mean"]) for x in rs])
            g = sstats.linregress(age, y)
            d = 100 * g.slope * 60 / (g.intercept + g.slope * 25)
            V[f"aging.{tis}.{roi}.pct"], V[f"aging.{tis}.{roi}.p"] = d, g.pvalue
            out.append([NICE[roi], sci(g.slope), pfmt(g.pvalue).replace("= ", "").replace("< ", "<"), pct(d)])
        V[f"aging.{tis}.n"] = len(rs)
        T[f"aging_{tis}"] = table(["Target", "Slope (V·m⁻¹·yr⁻¹)", "P", "Difference, age 25 to 85"], out)
    tc1 = rows(_p(RES, "v21_twochannel", "twochannel_exact_dx1.csv"), "co-simulation 1 mm")
    if tc1:
        out = []
        for r2 in sorted({f(r["r2_contact_ohm"]) for r in tc1}):
            k = f"tc1.{r2 / 1e3:g}k"
            if f"{k}.com1" in V:
                out.append([f"{r2 / 1e3:g} kΩ", f"{V[f'{k}.com2']:.2f} mm", f"{V[f'{k}.com1']:.2f} mm",
                            f"{V[f'{k}.diff']:+.2f} ± {V[f'{k}.diff_sd']:.2f} mm".replace("-", MINUS),
                            f"{V[f'{k}.r']:.3f}" if f"{k}.r" in V else "—"])
        T["cosim_dx1"] = table(["Channel-2 contact", "Static shift, 2 mm", "Static shift, 1 mm",
                                "Paired difference", "r between grids"], out)


SHORT_ = {"M1_handknob": "M1", "thalamus": "Thalamus", "insula": "Insula", "ACC": "ACC",
          "hippocampus_L": "Hipp. L", "hippocampus_R": "Hipp. R"}


def model_change(V, T):
    """The headline outcomes under v21 (published model) and v22, side by side (Supplement S9)."""
    old = {}
    t3 = jload(_p(RES0, "montage_optimization", "cohort_gm300_dx5_maxdir", "cohort_montage_optima.json"),
               "v21 optima")
    if t3:
        for roi in ROIS:
            old[f"opt.{roi}"] = (label(t3["top"][roi][0]["montage"]), t3["top"][roi][0]["median_score"])
    rr = rows(_p(RES0, "v21_multimontage", "agreement_by_montage.csv"), "v21 FEM agreement")
    for t in MM_ORDER:
        s = [r for r in rr if r["montage"] == t and f(r["dx_mm"]) == 1.0]
        if s:
            old[f"fem.{t}"] = float(np.mean([100 * (f(r["mean_spicehead"]) / f(r["mean_fem"]) - 1) for r in s]))
    sw = jload(_p(RES0, "v21_dx2_check_full", "dx2_full_summary.json"), "v21 full sweep")
    vs = rows(_p(RES0, "v21_dx2_check_full", "dx2_full_vs_dx5.csv"), "v21 full sweep table")
    if sw and vs:
        old["sweep.kept"] = sum(r["winner_kept"] == "True" for r in vs)
        old["sweep.loss_max"] = max(f(r["cohort_regret_5mm_winner_pct"]) for r in vs)
        old["sweep.own5_max"] = max(sw["per_target"][r]["regret_own5"]["max"] for r in ROIS)
    tc = rows(_p(RES0, "v21_twochannel", "twochannel_exact_dx2.csv"), "v21 co-simulation")
    if tc:
        s50 = [r for r in tc if f(r["r2_contact_ohm"]) == 50e3]
        old["tc.50k"] = float(np.mean([f(r["com_shift_td_mm"]) for r in s50]))
        old["tc.rhead"] = float(np.mean([f(r["R11"]) - 2e3 for r in tc if f(r["r2_contact_ohm"]) == 1e3]))
    out = []
    for roi in ROIS:
        if f"opt.{roi}" in old and f"opt.{roi}.montage" in V:
            out.append([f"Cohort optimum, {NICE[roi]}", f"{old[f'opt.{roi}'][0]} (F {old[f'opt.{roi}'][1]:.2f})",
                        f"{V[f'opt.{roi}.montage']} (F {V[f'opt.{roi}.F']:.2f})"])
    for t in MM_ORDER:
        if f"fem.{t}" in old and f"mm.1mm.{t}.relbias" in V:
            out.append([f"FEM bias at 1 mm, {MM_NICE[t].split(' (')[0].lower()}", pct(old[f"fem.{t}"]),
                        pct(V[f"mm.1mm.{t}.relbias"])])
    if "sweep.kept" in old and "sweep.kept" in V:
        out += [["5 mm cohort winner kept at 2 mm", f"{old['sweep.kept']} of 6", f"{V['sweep.kept']} of 6"],
                ["Largest cohort loss at 2 mm", f"{old['sweep.loss_max']:.2f}%", f"{V['sweep.loss_max']:.2f}%"],
                ["Worst loss, head's own 5 mm optimum", f"{old['sweep.own5_max']:.1f}%",
                 f"{V['sweep.own5.max_hi']:.1f}%"]]
    if "tc.50k" in old and "tc.50k.com" in V:
        out += [["Head resistance per channel path", f"{old['tc.rhead']:.0f} Ω", f"{V['tc.rhead_mean']:.0f} Ω"],
                ["Center-of-mass shift at 50 kΩ", f"{old['tc.50k']:.1f} mm", f"{V['tc.50k.com']:.1f} mm"]]
    T["model_change"] = table(["Outcome", "v21 (volumetric contacts, arithmetic rule)",
                               "v22 (skin contacts, harmonic rule)"], out)
    V["model_change.rows"] = len(out)


def criteria(V, T):
    """The proposed acceptance criteria (v22/ACCEPTANCE_CRITERIA_proposed_2026-10-01.md), evaluated.

    Computed, not judged: each criterion is met, not met, or not yet evaluable. The manuscript's
    <!--CHECK--> sentences are reworded by the authors from this table.
    """
    def ok(keys):
        return all(k in V and V[k] == V[k] for k in keys)

    rows_, out = [], {}
    stab = {r: V.get(f"conv.step21_absmax.{r}") for r in ROIS}
    if all(v is not None for v in stab.values()):
        for r in ROIS:
            out[f"grid.{r}"] = stab[r] < 5
        rows_.append(["Grid stability, per target (< 5% from 2 to 1 mm, every head)",
                      ", ".join(f"{SHORT_[r]} {stab[r]:.1f}%" for r in ROIS),
                      f"{sum(out[f'grid.{r}'] for r in ROIS)} of 6 met"])
        out["grid.all"] = all(out[f"grid.{r}"] for r in ROIS) and V.get("conv.r_2mm_lo", 0) >= 0.95
        rows_.append(["Grid independence at 2 mm (all targets, and r ≥ 0.95 in every head)",
                      f"lowest r {V.get('conv.r_2mm_lo', float('nan')):.3f}",
                      "met" if out["grid.all"] else "not met"])
    for t in MM_ORDER:
        if ok([f"mm.1mm.{t}.relbias", f"mm.1mm.{t}.ccc"]):
            b, c = V[f"mm.1mm.{t}.relbias"], V[f"mm.1mm.{t}.ccc"]
            out[f"fem.{t}"] = abs(b) < 10 and c >= 0.90
            # a mean-bias criterion over eight heads, not pointwise or per-head agreement (R4)
            flag = (f"; {V[f'qc.{t}.over10']} of {V[f'qc.{t}.n']} reference runs flagged"
                    if V.get(f"qc.{t}.over10") else "")
            # no literal pipes: they split Markdown table cells (review of 2 October, _03)
            rows_.append([f"FEM agreement, {MM_NICE[t].split(' (')[0].lower()} (absolute mean bias "
                          "< 10%, CCC ≥ 0.90; means over eight heads)",
                          f"mean bias {pct(b)}, CCC {c:.3f}{flag}",
                          "met" if out[f"fem.{t}"] else "not met"])
    if ok(["mm.1mm.rank_foc.agree"]):
        out["decisions"] = V["mm.1mm.rank_foc.agree"] >= 44
        rows_.append(["Best-of-four agreement by F at 1 mm (≥ 44 of 48)",
                      f"{V['mm.1mm.rank_foc.agree']} of {V['mm.1mm.rank_foc.n']}",
                      "met" if out["decisions"] else "not met"])
    if ok(["sweep.loss_max", "sweep.rho300_lo"]):
        out["cohort"] = V["sweep.loss_max"] < 2 and V["sweep.rho300_lo"] >= 0.90
        # evaluated as accepted (300-head 5 mm choice in the 20 heads at 2 mm); the paired
        # same-head loss, the primary resolution comparison since the 2 October review, beside it
        rows_.append(["Cohort coarse-to-fine robustness (loss < 2%, ρ ≥ 0.90)",
                      f"worst loss {V['sweep.loss_max']:.2f}% (300-head choice; same-head choice "
                      f"{V.get('sweep.same_loss_max', float('nan')):.2f}%), lowest ρ {V['sweep.rho300_lo']:.3f}",
                      "met" if out["cohort"] else "not met"])
        bad = sum(V[f"sweep.{r}.own5.median"] > 2 or V[f"sweep.{r}.own5.max"] > 10 for r in ROIS)
        out["individual"] = bad >= 3
        rows_.append(["Individual 5 mm selection unreliable (median > 2% or max > 10%, ≥ 3 targets)",
                      f"{bad} of 6 targets", "met" if out["individual"] else "not met"])
    if ok(["tc.unclamped_dev_pct", "tc.unclamped_com_max"]):
        out["circuit"] = V["tc.unclamped_dev_pct"] < 2 and V["tc.unclamped_com_max"] < 1
        rows_.append(["Ideal current reproduced below the boundary (< 2%, shift < 1 mm)",
                      f"{V['tc.unclamped_dev_pct']:.2f}%, {V['tc.unclamped_com_max']:.2f} mm",
                      "met" if out["circuit"] else "not met"])
    for k, v in out.items():
        V[f"crit.{k}"] = v
    T["criteria"] = table(["Criterion (accepted 1 October 2026)", "Value", "Result"], rows_)


def datasets(V, T):
    """Table 1: every analysis with its heads, grid and network size (after the other sections)."""
    def rng(lo, hi, unit="k"):
        return f"{V[lo]:.0f}–{V[hi]:.0f}{unit}" if lo in V and hi in V else "—"

    n5 = (f"{V['scale.nodes5_med_k']:.0f}k ({V['scale.nodes5_lo_k']:.0f}–{V['scale.nodes5_hi_k']:.0f}k)"
          if "scale.nodes5_med_k" in V else "—")
    n2 = (f"{V['sweep.nodes_med_k']:.0f}k ({V['sweep.nodes_lo_k']:.0f}–{V['sweep.nodes_hi_k']:.0f}k)"
          if "sweep.nodes_med_k" in V else "—")
    n1 = (f"{V['conv.nodes1_min_M']:.1f}–{V['conv.nodes1_max_M']:.1f}M at 1 mm"
          if "conv.nodes1_min_M" in V else "—")
    def size(n):                                  # 30k, 462k, 3.7M
        return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"

    def span(lo, hi):                             # 398–577k, 3.2–4.6M
        return (f"{lo / 1e6:.1f}–{hi / 1e6:.1f}M" if lo >= 1e6
                else f"{lo / 1e3:.0f}–{hi / 1e3:.0f}k")

    nb = (" / ".join(size(V[f"bench.{d}.n_nodes"]) for d in ("5mm", "2mm", "1mm")
                     if f"bench.{d}.n_nodes" in V) or "—")
    nm = f"{V['mni.nodes_k']:.0f}k" if "mni.nodes_k" in V else "—"
    nf = ("; ".join(f"{span(V[f'scale.fem8_{d}mm_lo'], V[f'scale.fem8_{d}mm_hi'])} at {d} mm"
                    for d in (2, 1) if f"scale.fem8_{d}mm_lo" in V) or "—")
    rows_ = [
        ["Analytic disk; layered slab", "—", "5–1 mm; 3–0.5 mm", "—", "—", "3.1"],
        ["Grid refinement", str(V.get("conv.n_heads", 4)), "5, 4, 3, 2, 1 mm", "1", n1, "3.1"],
        ["Comparison with FEM", "8", "2 and 1 mm", "4", nf, "3.2"],
        ["Full-space comparison with FEM", str(V.get("fs.n_heads", 8)), "2 and 1 mm", "11,628", nf, "3.2, S11"],
        ["Stage benchmark", "1", "5, 2, 1 mm", "11,628", nb, "3.3"],
        ["Coupling to non-ideal sources", "8", "2 and 1 mm", "1", nf, "3.3"],
        ["Cohort montage search", str(V.get("coh.n", 300)), "5 mm", "11,628", n5, "3.4, S2"],
        ["Full coarse-to-fine sweep", str(V.get("sweep.n", 20)), "2 mm", "11,628", n2, "3.4"],
        ["MNI152 exposure maps (display resampled to 1 mm)", "template", "2 mm", "11,628", nm, "3.4"],
    ]
    T["datasets"] = table(["Analysis", "Heads", "Grid", "Montages per head",
                           "Nodes per head, median (range)", "Section"], rows_)


TD_LEVEL = {"s1": "10-20, bilateral", "uni": "10-20, one-sided", "ratio": "10-20, unequal currents",
            "1010": "10-10", "1010_noinf": "10-10, no inferior row", "hdti": "steering",
            "hdti_noinf": "steering, no inferior row"}
TD_ORDER = ["insula", "insula_L", "insula_R", "insula_ant_L", "insula_post_L",
            "hippocampus", "hippocampus_L", "hippocampus_R"]


def template_design(V, T):
    """Template montage-design exploration for the insula and hippocampus (13-15 October 2026):
    one-sided targets, unequal channel currents, 10-10 positions, multi-electrode steering and
    insular subregions, collected by pipeline/template_design_collect.py. Template-only results
    with their own consistency checks; not cohort results and not FEM-compared."""
    d = _p(RES, TD_DIR)
    lv = rows(_p(d, "design_levels.csv"), "template design levels")
    sm = jload(_p(d, "summary.json"), "template design summary")
    ck = jload(_p(d, "checks.json"), "template design checks")
    sw = rows(_p(d, "current_ratio_sweep.csv"), "current-ratio sweep")
    sub = rows(_p(d, "subregions.csv"), "insular subregions")
    if not lv or not sm:
        return
    cap, hd = sm["cap_1010"], sm["hdti"]
    V["td.n_positions"], V["td.n_pairs"], V["td.n_montages"] = cap["n_positions"], cap["n_pairs"], cap["n_montages"]
    V["td.cross_min_mm"], V["td.screen_frac_pct"], V["td.top_k"] = cap["cross_min_mm"], 100 * cap["screen_fraction"], cap["top_k"]
    V["td.max_e"], V["td.min_pad_mm"], V["td.min_opp_mm"] = hd["max_electrodes_per_channel"], hd["min_pad_mm"], hd["min_opposite_mm"]
    V["td.deltas"] = ", ".join(f"{x:g}" for x in hd["deltas_mA"])
    inf_order = ["F9", "F10", "FT9", "FT10", "T9", "T10", "TP9", "TP10", "P9", "P10", "PO9", "PO10", "Iz", "I1", "I2"]
    inf = set(sm["inferior_row"])
    V["td.inferior_row"] = ", ".join([l for l in inf_order if l in inf] + sorted(inf - set(inf_order)))
    V["td.field_pct"], V["td.focus_frac_pct"] = sm["field_scale_percentile"], 100 * sm["focus_fraction"]
    V["td.total_mA"] = sm["total_current_mA"]
    if ck:
        V["td.n_checks"], V["td.n_checks_failed"] = ck["n_checks"], ck["n_failed"]
    ratios = sorted({f(r["ratio_I1_I2"]) for r in sw})

    def rl(r):                                       # 0.3333 -> "1:3", 1.5 -> "3:2"
        from fractions import Fraction
        fr = Fraction(r).limit_denominator(12)
        return f"{fr.numerator}:{fr.denominator}"
    V["td.n_ratios"], V["td.ratio_lo"], V["td.ratio_hi"] = len(ratios), rl(ratios[0]), rl(ratios[-1])

    by = {(r["target"], r["level"]): r for r in lv}
    for r in lv:
        k = f"td.{r['target']}.{r['level']}"
        V[f"{k}.F"], V[f"{k}.montage"] = f(r["F"]), r["montage"]
        if r["T_V_per_m"] != "":
            V[f"{k}.T"] = f(r["T_V_per_m"])
        if r["gain_pct"] != "":
            V[f"{k}.gain"], V[f"{k}.ref_F"] = f(r["gain_pct"]), f(r["ref_F"])
        for c, name in (("target_in_focus_pct", "focus_pct"), ("focus_in_target_pct", "focus_in_target"),
                        ("focus_cm3", "focus_cm3"), ("T_over_top", "T_over_top"), ("top_V_per_m", "top"),
                        ("F_currents_rounded_0p01mA", "F_rounded"), ("screen_rank", "screen_rank")):
            if r.get(c, "") != "":
                V[f"{k}.{name}"] = f(r[c])
        ref = {"ratio": "uni" if r["target"] not in ("insula", "hippocampus") else "s1",
               "1010": "uni" if r["target"] not in ("insula", "hippocampus") else "s1",
               "1010_noinf": "uni" if r["target"] not in ("insula", "hippocampus") else "s1",
               "hdti": "1010", "hdti_noinf": "1010_noinf"}.get(r["level"])
        rr = by.get((r["target"], ref)) if ref else None
        if rr and rr["T_V_per_m"] != "" and r["T_V_per_m"] != "":
            V[f"{k}.T_change"] = 100 * (f(r["T_V_per_m"]) / f(rr["T_V_per_m"]) - 1)
    for lvl in ("ratio", "1010", "1010_noinf", "hdti", "hdti_noinf"):
        g = [f(r["gain_pct"]) for r in lv if r["level"] == lvl and r["gain_pct"] != ""]
        V[f"td.{lvl}.gain_lo"], V[f"td.{lvl}.gain_hi"], V[f"td.{lvl}.n"] = min(g), max(g), len(g)
        V[f"td.{lvl}.n_F_above_1"] = sum(f(r["F"]) > 1 for r in lv if r["level"] == lvl)
    V["td.1010.screen_rank_max"] = max(int(f(r["screen_rank"])) for r in lv if r.get("screen_rank", "") != "")
    V["td.1020.focus_in_target_max"] = max(f(r["focus_in_target_pct"]) for r in lv
                                           if r["level"] in ("s1", "uni", "ratio") and r["focus_in_target_pct"] != "")
    V["td.hdti.rounded_dF_max"] = max(abs(f(r["F"]) - f(r["F_currents_rounded_0p01mA"])) for r in lv
                                      if r["level"].startswith("hdti"))
    hT = [f(r["T_V_per_m"]) for r in lv if r["level"].startswith("hdti")]
    V["td.hdti.T_lo"], V["td.hdti.T_hi"] = min(hT), max(hT)
    mapped = [r for r in lv if r["top_V_per_m"] != ""]
    V["td.focus_cm3_lo"], V["td.focus_cm3_hi"] = min(f(r["focus_cm3"]) for r in mapped), max(f(r["focus_cm3"]) for r in mapped)
    V["td.target_cm3_lo"], V["td.target_cm3_hi"] = min(f(r["target_cm3"]) for r in mapped), max(f(r["target_cm3"]) for r in mapped)
    V["td.focus_in_target_max"] = max(f(r["focus_in_target_pct"]) for r in mapped)
    hu = [f(r["target_in_focus_pct"]) for r in mapped if r["target"] in ("hippocampus_L", "hippocampus_R")]
    V["td.hippo_uni.focus_pct_lo"], V["td.hippo_uni.focus_pct_hi"] = min(hu), max(hu)

    def cell(t, lvl):
        r = by.get((t, lvl))
        if not r:
            return "–"
        return f"{f(r['F']):.3f}" + (f" ({pct(f(r['gain_pct']))})" if r["gain_pct"] != "" else "")

    targets = [t for t in TD_ORDER if any(r["target"] == t for r in lv)]
    title = {r["target"]: r["title"] for r in lv}
    T["td_F"] = table(["Target", "10-20, equal currents", "Unequal currents", "10-10", "10-10, no inferior row",
                       "Steering", "Steering, no inferior row"],
                      [[title[t], cell(t, "s1" if (t, "s1") in by else "uni"), cell(t, "ratio"), cell(t, "1010"),
                        cell(t, "1010_noinf"), cell(t, "hdti"), cell(t, "hdti_noinf")] for t in targets])
    T["td_focus"] = table(["Target", "Design level", "Scale top (V/m)", "T / top", "Focus (cm³)",
                           "Target inside focus (%)", "Focus inside target (%)"],
                          [[r["title"], TD_LEVEL[r["level"]], f"{f(r['top_V_per_m']):.3f}", f"{f(r['T_over_top']):.2f}",
                            f"{f(r['focus_cm3']):.0f}", f"{f(r['target_in_focus_pct']):.0f}", f"{f(r['focus_in_target_pct']):.0f}"]
                           for r in lv if r["top_V_per_m"] != ""])
    T["td_hdti"] = table(["Target", "Variant", "Channel 1 (mA)", "Channel 2 (mA)", "F", "F, currents rounded to 0.01 mA", "T (V/m)"],
                         [[r["title"], "all positions" if r["level"] == "hdti" else "no inferior row",
                           r["currents_ch1_mA"], r["currents_ch2_mA"], f"{f(r['F']):.3f}",
                           f"{f(r['F_currents_rounded_0p01mA']):.3f}", f"{f(r['T_V_per_m']):.3f}"]
                          for r in lv if r["level"].startswith("hdti")])
    T["td_montages"] = table(["Target", "Design level", "Montage", "F", "T (V/m)", "Gain vs reference"],
                             [[r["title"], TD_LEVEL[r["level"]], r["montage"], f"{f(r['F']):.3f}",
                               f"{f(r['T_V_per_m']):.3f}" if r["T_V_per_m"] != "" else "–",
                               pct(f(r["gain_pct"])) if r["gain_pct"] != "" else "–"] for r in lv])
    sweep = {}
    for r in sw:
        sweep.setdefault(r["target"], {})[f(r["ratio_I1_I2"])] = f(r["best_F"])
    T["td_ratio_sweep"] = table(["Target"] + [rl(x) for x in ratios],
                                [[title.get(t, t)] + [f"{sweep[t][x]:.3f}" for x in ratios] for t in TD_ORDER if t in sweep])
    T["td_subregions"] = table(["Target", "AAL parcels", "MNI y rule", "GM volume, 2 mm grid (cm³)", "GM volume, 1 mm (cm³)"],
                               [[r["title"], r["aal_parcels"], r["mni_y_rule"].replace(">=", "≥").replace("<=", "≤"),
                                 r["gm_cm3_2mm"], r["gm_cm3_1mm"]] for r in sub])


def values():
    del MISSING[:]
    V, T = {"model": config.MODEL, "model.describe": config.describe(),
            "res_root": os.path.relpath(RES, ROOT)}, {}
    for fn in (cohort, verification, fem_qc, fem, sweep, bench, scale, optima, mni, circuit,
               fullsearch, contact_change, datasets, supplement, model_change, criteria,
               template_design):
        try:
            fn(V, T)
        except Exception as ex:                       # report, do not hide, a broken section
            MISSING.append(f"{fn.__name__}: {type(ex).__name__}: {ex}")
    V["missing"] = list(MISSING)
    return V, T


if __name__ == "__main__":
    V, T = values()
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    for k in sorted(V):
        if pat in k and k != "missing":
            v = V[k]
            print(f"{k:42s} {v:.6g}" if isinstance(v, float) else f"{k:42s} {v}")
    print(f"\n{len(V)} values, {len(T)} tables ({', '.join(T)}), results root {V['res_root']}")
    if V["missing"]:
        print("MISSING:")
        for m in V["missing"]:
            print("  -", m)
