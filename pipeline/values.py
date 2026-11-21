# -*- coding: utf-8 -*-
"""Single source of truth for every reported number.

Third-round review, Major Comment 1. The previous design audited *selected* expected strings, so
anything nobody thought to register drifted silently: §3.6 aging, the §3.7 focality-sensitivity
table and the thalamic intensity were all still per-axis values months after the rotation-invariant
recompute, while the audit reported 85 passing checks. Checking a list you wrote by hand cannot
find the entry missing from that list.

The fix is generation, not verification. Every number and every table below is computed here from
the result files; `render_docs.py` substitutes them into the manuscript template, and
`gen_analyses_md.py` reads the same dict. A stale number is now impossible rather than detectable,
because there is no second place for a number to live.

Two rules for anything added here:
  * a value is computed, never typed;
  * prose that states a number takes it from this dict, so a table and the sentence describing it
    cannot disagree (they did: the ratio table said 1.0 for every target while the sentence below
    it claimed the thalamus preferred 0.79:1).

  python pipeline/values.py            # print every key and value
  python pipeline/values.py <substr>   # print matching keys
"""
import csv
import json
import os
import statistics as st
import sys

import numpy as np
from scipy import stats as sstats

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
R = os.path.join(ROOT, "results", "montage_optimization")
A = os.path.join(ROOT, "results")

ROIS = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]
NICE = {"M1_handknob": "M1 hand-knob", "thalamus": "Thalamus", "insula": "Insula",
        "ACC": "Anterior cingulate", "hippocampus_L": "Hippocampus (L)",
        "hippocampus_R": "Hippocampus (R)"}
MM_NICE = {"M1": "Fronto-parietal (F3+F4 / Pz+O1)", "THAL": "Bilateral temporal (T8+P8 / F7+T7)",
           "HIPP": "Fronto-occipital (Fp2+F7 / P8+O2)",
           "INFERIOR": "Inferior 9/10 chain (F9+T9 / F10+T10)"}
MM_ORDER = ["M1", "THAL", "HIPP", "INFERIOR"]
EXCL = [10.0, 15.0, 20.0, 25.0, 30.0]
PCTL = [95.0, 97.0, 99.0, 99.5, 100.0]


def jload(*p):
    return json.load(open(os.path.join(*p), encoding="utf-8"))


def rows_of(*parts):
    p = os.path.join(A, *parts)
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def _mean(rs, k):
    return float(np.mean([float(r[k]) for r in rs])) if rs else float("nan")


def table(header, rows, align=None):
    """Render a markdown pipe table. Rows are sequences of pre-formatted strings."""
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(align or ["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
MINUS = "−"          # the manuscript uses a true minus sign, not a hyphen


def sup(n):
    return str(int(n)).translate(_SUP)


def sci(x, sig=1):
    """Scientific notation with superscript exponent, e.g. −4.6×10⁻⁴."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    s = f"{abs(m):.{sig}f}".rstrip("0").rstrip(".")
    return f"{MINUS if x < 0 else ''}{s}×10{sup(e)}"


def pct(x, dp=1):
    """Signed percentage using a true minus sign."""
    return f"{MINUS if x < 0 else '+'}{abs(x):.{dp}f}%"


def num(x, dp=4):
    """Signed fixed-point using a true minus sign."""
    return f"{MINUS if x < 0 else '+'}{abs(x):.{dp}f}"


def pfmt(p):
    """P value for prose, in JAMA convention: written as "*P* {pfmt(x)}", so it carries its own
    relational operator and drops the leading zero."""
    if p < 0.001:
        return "< .001"
    return f"= {p:.3g}".replace("0.", ".", 1) if p < 1 else f"= {p:.3g}"


def pcell(p):
    """P value for a table cell, where the column header carries the label."""
    return pfmt(p).replace("= ", "").replace("< ", "<")


# ---------------------------------------------------------------- aging
def _aging(path):
    rows = list(csv.DictReader(open(path, encoding="cp1252")))
    age = np.array([float(x["Age"]) for x in rows])
    out = {}
    for k in ROIS:
        y = np.array([float(x[f"thalamic_{k}_mean"]) for x in rows])
        g = sstats.linregress(age, y)
        out[k] = dict(slope=g.slope, p=g.pvalue,
                      pct=100 * g.slope * 60 / (g.intercept + g.slope * 25))
    return len(rows), out


def _aging_subset(path, keep=None, drop_site=None):
    rows = list(csv.DictReader(open(path, encoding="cp1252")))
    if keep:
        rows = [r for r in rows if r.get("Site") == keep]
    if drop_site:
        rows = [r for r in rows if r.get("Site") != drop_site]
    age = np.array([float(x["Age"]) for x in rows])
    out = {}
    for k in ROIS:
        y = np.array([float(x[f"thalamic_{k}_mean"]) for x in rows])
        g = sstats.linregress(age, y)
        out[k] = dict(p=g.pvalue, pct=100 * g.slope * 60 / (g.intercept + g.slope * 25))
    return len(rows), out


def values():
    V = {}

    # ---- Table 3 optima -------------------------------------------------------
    t3 = jload(R, "cohort_gm300_dx5_maxdir", "cohort_montage_optima.json")
    V["envelope"] = t3.get("envelope")
    V["t3.n"] = t3["n_subjects"]
    for r in ROIS:
        top = t3["top"][r][0]
        V[f"t3.{r}.montage"] = top["montage"]
        V[f"t3.{r}.F"] = top["median_score"]

    # ---- Table 4 / steering ---------------------------------------------------
    d = os.path.join(R, "cohort_deep_gm300_maxdir")
    per = {r: {k: [] for k in ("A", "B", "C", "C4", "gB", "gC", "gC4")} for r in ROIS}
    c1 = {r: 0 for r in ROIS}
    bad = tot = 0
    for f in os.listdir(d):
        if not f.endswith(".json") or f == "SUMMARY.json":
            continue
        rr = (jload(d, f) or {}).get("rois") or {}
        for r in ROIS:
            e = rr.get(r)
            if not e or e.get("A") is None:
                continue
            for k in ("A", "B", "C", "C4"):
                per[r][k].append(e[k])
            for k, a, b in (("gB", "B", "A"), ("gC", "C", "A"), ("gC4", "C4", "A")):
                per[r][k].append(100 * (e[a] - e[b]) / e[b])
            c1[r] += e["C"] > 1.0
            tot += 1
            bad += e["C4"] < e["A"]
    for r in ROIS:
        for k in per[r]:
            V[f"t4.{r}.{k}"] = st.median(per[r][k])
        V[f"t4.{r}.n_F_above_1"] = c1[r]
    V["t4.c4_below_pct"] = 100 * bad / tot
    for k, lab in (("gC", "steer"), ("gC4", "trunc"), ("gB", "extset")):
        V[f"t4.{lab}.lo"] = min(V[f"t4.{r}.{k}"] for r in ROIS)
        V[f"t4.{lab}.hi"] = max(V[f"t4.{r}.{k}"] for r in ROIS)

    # ---- ratio and 10-10 ------------------------------------------------------
    rt = jload(R, "montage_ratio_n300_maxdir.json")["per_roi"]
    V["ratio.max_gain"] = max(x["median_gain_pct"] for x in rt.values())
    V["ratio.n_equal_optimal"] = sum(1 for x in rt.values() if x["median_best_ratio"] == 1.0)
    V["ratio.n_roi"] = len(rt)
    V["ratio.all_equal"] = V["ratio.n_equal_optimal"] == V["ratio.n_roi"]
    for r in ROIS:
        V[f"ratio.{r}.best"] = rt[r]["median_best_ratio"]
        V[f"ratio.{r}.gain"] = rt[r]["median_gain_pct"]
    tn = jload(R, "montage_1010_n300_maxdir.json")["per_roi"]
    g = {r: 100 * (tn[r]["best"][0]["score"] - tn[r]["best_1020_only_same_subset"]["score"])
         / tn[r]["best_1020_only_same_subset"]["score"] for r in ROIS}
    V["ten.max_gain"] = max(g.values())
    V["ten.median_gain"] = st.median(list(g.values()))

    # ---- aging (§3.6) ---------------------------------------------------------
    ab = os.path.join(A, "aging_n300_brain_maxdir", "cohort_roi_metrics.csv")
    ag = os.path.join(A, "aging_n300_gm_maxdir", "cohort_roi_metrics.csv")
    V["aging.n"], brain = _aging(ab)
    _, gm = _aging(ag)
    for r in ROIS:
        V[f"aging.brain.{r}.pct"] = brain[r]["pct"]
        V[f"aging.brain.{r}.p"] = brain[r]["p"]
        V[f"aging.brain.{r}.slope"] = brain[r]["slope"]
        V[f"aging.gm.{r}.pct"] = gm[r]["pct"]
        V[f"aging.gm.{r}.p"] = gm[r]["p"]
    V["aging.n_novet"], novet = _aging_subset(ab, drop_site="Vet_Home")
    V["aging.n_tami"], tami = _aging_subset(ab, keep="TAMI")
    for r in ROIS:
        V[f"aging.novet.{r}.pct"] = novet[r]["pct"]
        V[f"aging.novet.{r}.p"] = novet[r]["p"]
        V[f"aging.tami.{r}.pct"] = tami[r]["pct"]
        V[f"aging.tami.{r}.p"] = tami[r]["p"]

    # ---- focality-metric sensitivity (§3.7) -----------------------------------
    sens = rows_of("v21_sensitivity_maxdir", "sensitivity.csv")
    cell = {}
    for r in sens:
        cell.setdefault((float(r["exclusion_mm"]), float(r["percentile"])), []).append(
            float(r["F_median"]))
    V["sens.counts"] = {k: sum(1 for v in vs if v > 1.0) for k, vs in cell.items()}
    V["sens.published"] = V["sens.counts"][(20.0, 99.0)]
    V["sens.p95_20"] = V["sens.counts"][(20.0, 95.0)]
    V["sens.max_20"] = V["sens.counts"][(20.0, 100.0)]
    # Median in-ROI envelope at the published focality setting, per target. Needed for the merged
    # optimization table (Table 3), whose last column is absolute intensity — the quantity §3.5
    # argues matters more than F, which is definition-sensitive.
    for r in sens:
        if float(r["exclusion_mm"]) == 20.0 and float(r["percentile"]) == 99.0:
            V[f"sens.{r['roi']}_T"] = float(r["T_median_Vm"])
    V.setdefault("sens.thalamus_T", float("nan"))

    # ---- FEM agreement (§3.4) -------------------------------------------------
    fem = rows_of("v21_simnibs_validation", "agreement.csv")
    raw = rows_of("v21_simnibs_validation", "agreement_raw.csv")
    for dx, tag in (("1.0", "1mm"), ("2.0", "2mm")):
        s = [x for x in fem if x["dx_mm"] == dx]
        for k in ("r", "r2", "slope", "nmae", "dice", "peak_err_mm"):
            V[f"fem.{tag}.{k}"] = _mean(s, k)
        V[f"fem.{tag}.n"] = len({x["subject"] for x in s})
        q = [x for x in raw if x["dx_mm"] == dx]
        for k in ("raw_bias", "loa_lo", "loa_hi", "prop_bias_slope", "ccc", "slope",
                  "intercept", "mean_fem", "mean_spicehead"):
            V[f"raw.{tag}.{k}"] = _mean(q, k)
        V[f"raw.{tag}.bias_pct_of_mean"] = 100 * V[f"raw.{tag}.raw_bias"] / V[f"raw.{tag}.mean_fem"]

    # per-target intensity against FEM, at the finest resolution
    s1 = [x for x in fem if x["dx_mm"] == "1.0"]
    for r in ROIS:
        V[f"fem.1mm.{r}.sp"] = _mean(s1, f"roi_{r}_sp")
        V[f"fem.1mm.{r}.fem"] = _mean(s1, f"roi_{r}_fem")
        V[f"fem.1mm.{r}.relerr"] = _mean(s1, f"roi_{r}_relerr_pct")
    V["fem.1mm.relerr_lo"] = min(V[f"fem.1mm.{r}.relerr"] for r in ROIS)
    V["fem.1mm.relerr_hi"] = max(V[f"fem.1mm.{r}.relerr"] for r in ROIS)
    V["fem.1mm.relerr_worst_roi"] = NICE[max(ROIS, key=lambda r: V[f"fem.1mm.{r}.relerr"])]

    # ---- multi-montage, per resolution ----------------------------------------
    mm = rows_of("v21_multimontage", "agreement_by_montage.csv")
    fc = rows_of("v21_multimontage", "focality_by_montage.csv")
    rk_i = rows_of("v21_multimontage", "rank_preservation.csv")
    rk_f = rows_of("v21_multimontage", "rank_preservation_focality.csv")
    V["mm.resolutions"] = sorted({float(x["dx_mm"]) for x in mm}) if mm else []
    for dx in V["mm.resolutions"]:
        tag = f"{dx:g}mm"
        sub = [x for x in mm if float(x["dx_mm"]) == dx]
        tags = [t for t in MM_ORDER if any(x["montage"] == t for x in sub)]
        sl = {t: _mean([x for x in sub if x["montage"] == t], "slope") for t in tags}
        for t in tags:
            s = [x for x in sub if x["montage"] == t]
            V[f"mm.{tag}.{t}.slope"] = sl[t]
            for k in ("r", "ccc", "raw_bias"):
                V[f"mm.{tag}.{t}.{k}"] = _mean(s, k)
        V[f"mm.{tag}.n"] = len({x["subject"] for x in sub})
        V[f"mm.{tag}.slope_lo"] = min(sl.values())
        V[f"mm.{tag}.slope_hi"] = max(sl.values())
        V[f"mm.{tag}.slope_spread"] = max(sl.values()) - min(sl.values())
        V[f"mm.{tag}.slope_spread_pct"] = 100 * (max(sl.values()) - min(sl.values())) / float(
            np.mean(list(sl.values())))
        V[f"mm.{tag}.within_sd"] = float(np.mean(
            [np.std([float(x["slope"]) for x in sub if x["montage"] == t], ddof=1) for t in tags]))
        for lab, rk in (("int", rk_i), ("foc", rk_f)):
            g_ = [x for x in rk if float(x.get("dx_mm", 2.0)) == dx]
            if not g_:
                continue
            V[f"mm.{tag}.rank_{lab}.agree"] = sum(x["agree"] == "True" for x in g_)
            V[f"mm.{tag}.rank_{lab}.n"] = len(g_)
            V[f"mm.{tag}.rank_{lab}.rho"] = float(np.nanmean([float(x["spearman"]) for x in g_]))
            for r in ROIS:
                gr = [x for x in g_ if x["roi"] == r]
                if gr:
                    V[f"mm.{tag}.rank_{lab}.{r}.agree"] = sum(x["agree"] == "True" for x in gr)
                    V[f"mm.{tag}.rank_{lab}.{r}.n"] = len(gr)
        sf = [x for x in fc if float(x["dx_mm"]) == dx and x["F_ratio"]]
        if sf:
            fr = [float(x["F_ratio"]) for x in sf]
            V[f"mm.{tag}.fratio_mean"] = float(np.mean(fr))
            V[f"mm.{tag}.fratio_sd"] = float(np.std(fr, ddof=1))
            cm = {}
            for x in sf:
                cm.setdefault((x["montage"], x["roi"]), []).append(float(x["F_ratio"]))
            means = [float(np.mean(v)) for v in cm.values()]
            V[f"mm.{tag}.fratio_lo"] = min(means)
            V[f"mm.{tag}.fratio_hi"] = max(means)
            F = {}
            for x in sf:
                F.setdefault((x["subject"], x["roi"]), {})[x["montage"]] = (
                    float(x["F_spicehead"]), float(x["F_fem"]))
            for r in ROIS:
                ks = [k for k in F if k[1] == r]
                if not ks:
                    continue
                loss = []
                for k in ks:
                    dd = F[k]
                    pick = max(dd, key=lambda t: dd[t][0])
                    best = max(dd, key=lambda t: dd[t][1])
                    loss.append(100 * (dd[best][1] - dd[pick][1]) / dd[best][1])
                V[f"mm.{tag}.loss.{r}.median"] = float(np.median(loss))
                V[f"mm.{tag}.loss.{r}.worst"] = float(max(loss))
    V["mm.has_1mm"] = 1.0 in V["mm.resolutions"]

    # ---- cylinder/disk convergence (§3.1) and benchmark (§3.5) -----------------
    conv = list(csv.DictReader(open(os.path.join(A, "v21_validation", "convergence.csv"),
                                    encoding="utf-8")))
    for r in conv:
        d = f"{float(r['dx_mm']):g}mm"
        V[f"cyl.nmae_{d}"] = float(r["nmae"])
        V[f"cyl.slope_{d}"] = float(r["slope"])
        V[f"cyl.r_{d}"] = float(r["r"])
        V[f"cyl.dice_{d}"] = float(r["dice"])
    bench = list(csv.DictReader(open(os.path.join(A, "v21_benchmark", "stages.csv"),
                                     encoding="utf-8")))
    b79 = next(x for x in bench if x["nodes"] == "78600")
    V["bench.nodes_k"] = round(int(b79["nodes"]) / 1000)
    V["bench.t_fact_s"] = float(b79["t_fact"])
    V["bench.t_per_config_s"] = float(b79["t_per_config"])

    # ---- two-channel co-simulation (§3.3) ---------------------------------------
    # Prefer the exact formulation (contact network inside the Kron reduction, fifth-round
    # comment 2); fall back to the superseded fixed-injection run only if it has not been run.
    tc = rows_of("v21_twochannel", "twochannel_exact_dx2.csv") \
        or rows_of("v21_twochannel", "twochannel.csv")
    V["tc.exact"] = bool(rows_of("v21_twochannel", "twochannel_exact_dx2.csv"))
    if V["tc.exact"]:
        for k, col in (("recip", "reciprocity_err_pct"), ("contact", "chk_contact_rel"),
                       ("fullnet", "chk_fullnet_rel"), ("field", "chk_field_rel"),
                       ("driver", "chk_driver_rel"), ("uniform", "uniform_model_dI_rel")):
            vals = [float(x[col]) for x in tc if x.get(col) not in (None, "")
                    and not np.isnan(float(x[col]))]
            V[f"tc.chk_{k}"] = max(vals) if vals else float("nan")
    # 1 mm repeat (fifth-round comment 3), if present
    tc1 = rows_of("v21_twochannel", "twochannel_exact_dx1.csv")
    if tc1:
        V["tc1.n"] = len({x["subject"] for x in tc1})
        V["tc1.sweep"] = sorted({float(x["r2_contact_ohm"]) for x in tc1})
        for r2 in V["tc1.sweep"]:
            s_ = [x for x in tc1 if float(x["r2_contact_ohm"]) == r2]
            t = f"tc1.{r2/1e3:g}k"
            for k, col in (("I2_mA", "I2_mA"), ("dT_pct", "dT_pct"), ("dF_pct", "dF_pct"),
                           ("com_mm", "com_shift_mm"), ("comtop_mm", "com_top_shift_mm"),
                           ("peak_mm", "peak_shift_mm")):
                a = np.array([float(x[col]) for x in s_])
                V[f"{t}.{k}"] = float(a.mean())
                if k == "com_mm":
                    V[f"{t}.com_sd"] = float(a.std(ddof=1))
                    V[f"{t}.com_lo"], V[f"{t}.com_hi"] = float(a.min()), float(a.max())
        w1 = f"tc1.{max(V['tc1.sweep'])/1e3:g}k"
        for k in ("I2_mA", "dT_pct", "dF_pct", "com_mm", "comtop_mm", "peak_mm",
                  "com_sd", "com_lo", "com_hi"):
            V[f"tc1.worst.{k}"] = V[f"{w1}.{k}"]
        # paired 1 mm vs 2 mm, same subjects (fifth-round comment 3): a mean-vs-mean comparison
        # would hide whether the shift is systematic
        from scipy import stats as _st
        m2 = {(x["subject"], float(x["r2_contact_ohm"])): x for x in tc}
        for r2 in V["tc1.sweep"]:
            pr = [(float(x["com_shift_mm"]), float(m2[(x["subject"], r2)]["com_shift_mm"]))
                  for x in tc1 if float(x["r2_contact_ohm"]) == r2
                  and (x["subject"], r2) in m2]
            if len(pr) < 2:
                continue
            a = np.array([q[0] for q in pr]); b = np.array([q[1] for q in pr])
            d = a - b
            t = f"tc1.{r2/1e3:g}k"
            V[f"{t}.pair_diff"] = float(d.mean())
            V[f"{t}.pair_sd"] = float(d.std(ddof=1))
            V[f"{t}.pair_absmax"] = float(np.abs(d).max())
            V[f"{t}.pair_p"] = float(_st.ttest_rel(a, b).pvalue) if d.std() > 0 else float("nan")
            V[f"{t}.pair_r"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 else float("nan")
            V[f"{t}.pair_pct"] = 100 * float(d.mean()) / float(b.mean()) if b.mean() else 0.0
        for t in [f"tc1.{r2/1e3:g}k" for r2 in V["tc1.sweep"]]:
            if f"{t}.pair_diff" in V:      # prose says "smaller by X", so carry the magnitude
                V[f"{t}.pair_absdiff"] = abs(V[f"{t}.pair_diff"])
                V[f"{t}.pair_abspct"] = abs(V[f"{t}.pair_pct"])
        for k in ("pair_diff", "pair_sd", "pair_absmax", "pair_p", "pair_r", "pair_pct",
                  "pair_absdiff", "pair_abspct"):
            if f"{w1}.{k}" in V:
                V[f"tc1.worst.{k}"] = V[f"{w1}.{k}"]
    # The 1 mm repeat carries no two-carrier field reconstruction, so the resolution comparison is
    # static-against-static — which is the right way to isolate resolution. That the two-carrier
    # correction itself is resolution-invariant is established separately, from the stored port
    # matrices, with no field solve (v21_twocarrier_dx1_check.py).
    gk = rows_of("v21_twochannel", "twocarrier_dx1_gain.csv")
    if gk:
        V["tc1.gain_match_max"] = max(abs(float(x["gain_diff"])) for x in gk)
        V["tc1.gain_match_n"] = len(gk)
    # Temporal-sampling convergence of the recovered carrier fundamental. The Methods used to say
    # "recovered exactly"; a clipped waveform sampled at finite resolution is a discretization, so
    # the claim is replaced by this measured bound (final Brain Stimulation review, comment 2).
    ts = rows_of("v21_twochannel", "twocarrier_sampling.csv")
    if ts:
        for spc in (25, 50, 100, 200, 400):
            col = f"reldiff_spc{spc}"
            if col in ts[0]:
                V[f"tcs.rel_{spc}"] = max(float(x[col]) for x in ts)
        V["tcs.n"] = len(ts)
    if tc:
        V["tc.n"] = len({x["subject"] for x in tc})
        V["tc.sweep"] = sorted({float(x["r2_contact_ohm"]) for x in tc})
        for r2 in V["tc.sweep"]:
            s = [x for x in tc if float(x["r2_contact_ohm"]) == r2]
            t = f"tc.{r2/1e3:g}k"
            for k, col in (("I2_mA", "I2_mA"), ("imbal", "imbalance"), ("dT_pct", "dT_pct"),
                           ("dF_pct", "dF_pct"), ("com_mm", "com_shift_mm"),
                           ("comtop_mm", "com_top_shift_mm"), ("peak_mm", "peak_shift_mm")):
                V[f"{t}.{k}"] = _mean(s, col)
            # Two-carrier delivered amplitudes (Brain Stimulation comment 1). Compliance clipping
            # is instantaneous, and the TI envelope is built from the FUNDAMENTAL of each clipped
            # carrier, which exceeds the static .op value; these are the physically correct
            # amplitudes and the static ones are the worst-instant bound.
            if "com_shift_td_mm" in s[0]:
                for k, col in (("I2_mA", "A2_td_mA"), ("dT_pct", "dT_td_pct"),
                               ("dF_pct", "dF_td_pct"), ("com_mm", "com_shift_td_mm"),
                               ("comtop_mm", "com_top_shift_td_mm"),
                               ("peak_mm", "peak_shift_td_mm")):
                    V[f"{t}.td.{k}"] = _mean(s, col)
                for k, col in (("com", "com_shift_td_mm"), ("comtop", "com_top_shift_td_mm"),
                               ("peak", "peak_shift_td_mm")):
                    a = np.array([float(x[col]) for x in s])
                    V[f"{t}.td.{k}_sd"] = float(a.std(ddof=1))
                    V[f"{t}.td.{k}_lo"], V[f"{t}.td.{k}_hi"] = float(a.min()), float(a.max())
                V[f"{t}.td.gain"] = _mean(s, "td_over_static_I2")
            # subject-level spread, fifth-round comment 4: a mean alone cannot distinguish
            # "~20 mm in everyone" from "one anatomy dominating"
            for k, col in (("com", "com_shift_mm"), ("comtop", "com_top_shift_mm"),
                           ("peak", "peak_shift_mm"), ("dT", "dT_pct")):
                a = np.array([float(x[col]) for x in s])
                V[f"{t}.{k}_sd"] = float(a.std(ddof=1))
                V[f"{t}.{k}_lo"], V[f"{t}.{k}_hi"] = float(a.min()), float(a.max())
            V[f"{t}.n_clamped"] = sum(x["ch2_clamped"] == "True" for x in s)
        worst = f"tc.{max(V['tc.sweep'])/1e3:g}k"
        V["tc.worst_r2_k"] = max(V["tc.sweep"]) / 1e3
        for k in ("I2_mA", "imbal", "dT_pct", "dF_pct", "com_mm", "comtop_mm", "peak_mm",
                  "com_sd", "com_lo", "com_hi", "comtop_sd", "peak_sd", "dT_sd"):
            V[f"tc.worst.{k}"] = V[f"{worst}.{k}"]
        for k in ("I2_mA", "dT_pct", "dF_pct", "com_mm", "comtop_mm", "peak_mm",
                  "com_sd", "com_lo", "com_hi", "comtop_sd", "comtop_lo", "comtop_hi",
                  "peak_sd", "peak_lo", "peak_hi", "gain"):
            if f"{worst}.td.{k}" in V:
                V[f"tc.worst.td.{k}"] = V[f"{worst}.td.{k}"]
        V["tc.twocarrier"] = f"{worst}.td.com_mm" in V
        V["tc.recip_max_pct"] = max(float(x["reciprocity_err_pct"]) for x in tc)             if "reciprocity_err_pct" in tc[0] else float("nan")
        ngok = [x for x in tc if x.get("ng_I2_mA")]
        V["tc.ng_checked"] = len(ngok)
        if ngok:
            V["tc.ng_max_dI_mA"] = max(abs(float(x["ng_I2_mA"]) - float(x["I2_mA"]))
                                       for x in ngok)

    # ---- partial 2-mm cohort validation (R4 comment 2) --------------------------
    d2 = rows_of("v21_dx2_check", "dx2_vs_dx5.csv")
    if d2:
        V["d2.n_targets"] = len(d2)
        V["d2.n_subjects"] = int(d2[0]["n_subjects"])
        V["d2.winner_kept"] = sum(x["winner_kept"] == "True" for x in d2)
        V["d2.rho_mean"] = _mean(d2, "spearman_vs_300rank")
        V["d2.forfeit_med"] = float(np.median([float(x["forfeit_pct_at_2mm"]) for x in d2]))
        V["d2.forfeit_worst"] = max(float(x["forfeit_pct_at_2mm"]) for x in d2)
        V["d2.n_candidates"] = int(d2[0]["n_candidates"])
        # Brain Stimulation comment 7: keep the two questions apart. The resolution-only test uses
        # the SAME 20 subjects at both grids; the mixed statistic (300-subject 5-mm ranking vs
        # 20-subject 2-mm ranking) confounds cohort sampling with resolution and is reported only
        # as a secondary figure.
        V["d2.rho_lo"] = min(float(x["spearman_same_subjects"]) for x in d2)
        V["d2.rho_hi"] = max(float(x["spearman_same_subjects"]) for x in d2)
        V["d2.rho_mixed_lo"] = min(float(x["spearman_vs_300rank"]) for x in d2)
        V["d2.rho_mixed_hi"] = max(float(x["spearman_vs_300rank"]) for x in d2)
        V["d2.rank_worst"] = max(int(x["rank_of_5mm_winner_at_2mm"]) for x in d2)
        V["d2.jac_lo"] = min(float(x["top5_jaccard"]) for x in d2)
        V["d2.jac_hi"] = max(float(x["top5_jaccard"]) for x in d2)
        for x in d2:
            V[f"d2.{x['roi']}.kept"] = x["winner_kept"] == "True"
            V[f"d2.{x['roi']}.rank"] = int(x["rank_of_5mm_winner_at_2mm"])
            V[f"d2.{x['roi']}.forfeit"] = float(x["forfeit_pct_at_2mm"])
            V[f"d2.{x['roi']}.rho"] = float(x["spearman_same_subjects"])      # resolution-only
            V[f"d2.{x['roi']}.rho_mixed"] = float(x["spearman_vs_300rank"])   # mixed, secondary
            V[f"d2.{x['roi']}.jac"] = float(x["top5_jaccard"])

    # ---- circuit co-simulation (§3.5) -----------------------------------------
    cs = rows_of("v21_cosim", "cosim.csv")
    if cs:
        ez = [x for x in cs if x["sweep"] == "electrode_Z"]
        ro = [x for x in cs if x["sweep"] == "source_Rout"]
        V["cosim.v_cc"] = float(ez[0]["v_cc"])
        V["cosim.rows"] = [(float(x["param"]), float(x["v_drv"]), float(x["I_del"]),
                            float(x["E_med"])) for x in ez]
        V["cosim.I_lo_mA"] = 1e3 * min(float(x["I_del"]) for x in ez)
        V["cosim.I_hi_mA"] = 1e3 * max(float(x["I_del"]) for x in ez)
        V["cosim.field_fold"] = (max(float(x["E_med"]) for x in ez)
                                 / min(float(x["E_med"]) for x in ez))
        V["cosim.rout_hi_mA"] = 1e3 * max(float(x["I_del"]) for x in ro)
        V["cosim.rout_lo_mA"] = 1e3 * min(float(x["I_del"]) for x in ro)

    # ---- generalization (§3.7) ------------------------------------------------
    lo = rows_of("v21_generalization", "loso.csv")
    if lo:
        tl = [float(x["transfer_loss_pct"]) for x in lo]
        V["loso.n"] = len(lo)
        V["loso.same"] = sum(x["transfer_is_oracle"] == "True" for x in lo)
        V["loso.median"] = float(np.median(tl))
        V["loso.mean"] = float(np.mean(tl))
        V["loso.worst"] = max(tl)
        wi = int(np.argmax(tl))
        V["loso.worst_roi"] = NICE.get(lo[wi]["roi"], lo[wi]["roi"])
        V["loso.worst_site"] = lo[wi]["held_out_site"]
        V["loso.n_sites"] = len({x["held_out_site"] for x in lo})
    # Exact-match learning curve. Sourced from the regret run's exact_match_pct so that the
    # exact-identity and regret analyses use the SAME 40 random draws — the fourth-round review
    # flagged that Methods said 20 draws while the regret analysis used 40. The original 20-draw
    # learning_curve.csv is superseded and no longer read.
    reg = rows_of("v21_generalization", "learning_curve_regret.csv")
    V["lcurve"] = {}
    for x in reg:
        V["lcurve"].setdefault(int(x["n_train"]), {})[x["roi"]] = float(x["exact_match_pct"])
    V["lcurve.draws"] = int(float(reg[0]["n_draws"])) if reg else None
    if reg:
        V["regret.sizes"] = sorted({int(x["n_train"]) for x in reg})
        for x in reg:
            k = f"regret.{x['n_train']}.{x['roi']}"
            for f in ("median_regret_pct", "p_within_1pct", "p_within_2pct", "p_within_5pct",
                      "median_rank", "regret_ci_lo", "regret_ci_hi", "jaccard_top5",
                      "exact_match_pct"):
                if f in x and x[f] != "":
                    V[f"{k}.{f}"] = float(x[f])
        for n_ in V["regret.sizes"]:
            V[f"regret.{n_}.p1_mean_pct"] = 100 * float(np.mean(
                [V[f"regret.{n_}.{r}.p_within_1pct"] for r in ROIS]))
            V[f"regret.{n_}.regret_mean"] = float(np.mean(
                [V[f"regret.{n_}.{r}.median_regret_pct"] for r in ROIS]))
    return V


# ---------------------------------------------------------------- tables
def tables(V):
    """Markdown tables, built from V so a table and its surrounding prose cannot disagree."""
    T = {}

    # Tissue conductivities, read from the machine-generated parameter record rather than retyped
    # (reviewer item D). Air is omitted: it carries no current and is excluded from the network.
    _pp = os.path.join(A, "v21_params", "model_parameters.md")
    if os.path.exists(_pp):
        _txt = open(_pp, encoding="utf-8").read()
        _sec = _txt.split("## Tissue conductivities", 1)[-1].split("## ", 1)[0]
        _rows = [[c.strip() for c in ln.strip().strip("|").split("|")]
                 for ln in _sec.splitlines()
                 if ln.strip().startswith("|") and not set(ln.strip()) <= set("|- ")]
        _rows = [r for r in _rows[1:] if len(r) >= 4 and r[1].lower() != "air"]
        _nice_tissue = {"csf": "CSF"}          # don't sentence-case an acronym
        T["tissue_conductivities"] = table(
            ["CHARM label", "Tissue", "σ (S/m)", "Source"],
            [[r[0], _nice_tissue.get(r[1].lower(), r[1].capitalize()),
              r[2], r[3]] for r in _rows])
        # the remaining parameter blocks go to the supplement verbatim
        T["model_parameters_rest"] = "\n\n".join(
            ("### S8." + str(i + 1) + " " + blk.split("\n", 1)[0].strip() + "\n\n"
             + blk.split("\n", 1)[1].strip())
            for i, blk in enumerate(
                b for b in _txt.split("## ")[1:]
                if not b.startswith("Tissue conductivities")))

    # The reference list is part of the manuscript file, in the numbering renumber_refs.py
    # produced, so the committed manuscript cannot disagree with the bibliography.
    _van = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "references_vancouver.md")
    if os.path.exists(_van):
        _lines = [ln.strip() for ln in open(_van, encoding="utf-8").read().splitlines()
                  if ln.strip().startswith("[")]
        _lines.sort(key=lambda x: int(x[1:x.index("]")]))
        T["references"] = "\n\n".join(_lines)

    T["aging_main"] = table(
        ["Target", "Slope (V·m⁻¹·yr⁻¹)", "*P* value", "Difference, age 25 to 85"],
        [[NICE[r], sci(V[f"aging.brain.{r}.slope"]), pcell(V[f"aging.brain.{r}.p"]),
          pct(V[f"aging.brain.{r}.pct"])] for r in ROIS])

    T["aging_site"] = table(
        ["Target", f"all {V['aging.n']}", f"excluding Vet_Home (n = {V['aging.n_novet']})",
         f"within TAMI (n = {V['aging.n_tami']})"],
        [[NICE[r],
          f"{pct(V[f'aging.brain.{r}.pct'])}, *P* {pfmt(V[f'aging.brain.{r}.p'])}",
          f"{pct(V[f'aging.novet.{r}.pct'])}, *P* {pfmt(V[f'aging.novet.{r}.p'])}",
          f"{pct(V[f'aging.tami.{r}.pct'])}, *P* {pfmt(V[f'aging.tami.{r}.p'])}"] for r in ROIS])

    T["aging_gm"] = table(
        ["Target", "brain ROI", "gray-matter ROI"],
        [[NICE[r], f"{pct(V[f'aging.brain.{r}.pct'])} (*P* {pfmt(V[f'aging.brain.{r}.p'])})",
          f"{pct(V[f'aging.gm.{r}.pct'])} (*P* {pfmt(V[f'aging.gm.{r}.p'])})"] for r in ROIS])

    hdr = ["exclusion \\ percentile"] + [("**p99**" if p == 99.0 else
                                         ("max" if p == 100.0 else f"p{p:g}")) for p in PCTL]
    rows = []
    for e in EXCL:
        lab = f"**{e:g} mm**" if e == 20.0 else f"{e:g} mm"
        rows.append([lab] + [(f"**{V['sens.counts'][(e, p)]}**"
                              if (e, p) == (20.0, 99.0) else str(V["sens.counts"][(e, p)]))
                             for p in PCTL])
    T["focality_sensitivity"] = table(hdr, rows)

    # Merged optimization table — r20 Tables 3 and 4 in one, per the fourth-round review: the
    # optimal montage per target, its focality under each electrode set, and absolute intensity.
    # F is definition-sensitive (S2.4); the intensity column is the one §3.5 argues matters more.
    T["optimization"] = table(
        ["Target", "Optimal 10-20 montage", "F, 10-20", "F, +9/10 chain", "F, steered",
         "Median in-ROI envelope"],
        [[NICE[r], V[f"t3.{r}.montage"], f"{V[f't4.{r}.A']:.2f}",
          f"{V[f't4.{r}.B']:.2f}", f"{V[f't4.{r}.C']:.2f}",
          (f"{V[f'sens.{r}_T']:.3f} V/m" if f"sens.{r}_T" in V else "—")]
         for r in ROIS])

    T["fem_resolution"] = table(
        ["", "Δx = 2 mm", "Δx = 1 mm"],
        [["Pearson r", f"{V['fem.2mm.r']:.3f}", f"**{V['fem.1mm.r']:.3f}**"],
         ["R²", f"{V['fem.2mm.r2']:.3f}", f"**{V['fem.1mm.r2']:.3f}**"],
         ["Regression slope", f"{V['fem.2mm.slope']:.3f}", f"**{V['fem.1mm.slope']:.3f}**"],
         ["Intercept (V/m)", f"{V['raw.2mm.intercept']:.3f}", f"{V['raw.1mm.intercept']:.3f}"],
         ["Normalized MAE", f"{V['fem.2mm.nmae']:.1%}", f"**{V['fem.1mm.nmae']:.1%}**"],
         ["Dice, top-decile field", f"{V['fem.2mm.dice']:.3f}", f"**{V['fem.1mm.dice']:.3f}**"],
         ["Peak-location distance", f"{V['fem.2mm.peak_err_mm']:.0f} mm",
          f"{V['fem.1mm.peak_err_mm']:.0f} mm"]])

    T["fem_per_target"] = table(
        ["Target", "SPICEhead (V/m)", "FEM (V/m)", "Relative error"],
        [[NICE[r], f"{V[f'fem.1mm.{r}.sp']:.4f}", f"{V[f'fem.1mm.{r}.fem']:.4f}",
          (f"**{pct(V[f'fem.1mm.{r}.relerr'])}**"
           if V[f"fem.1mm.{r}.relerr"] == V["fem.1mm.relerr_hi"] else
           pct(V[f"fem.1mm.{r}.relerr"]))] for r in ROIS])

    T["raw_agreement"] = table(
        ["", "Δx = 2 mm", "Δx = 1 mm"],
        [["Bias (V/m)", f"{V['raw.2mm.raw_bias']:+.5f}", f"**{V['raw.1mm.raw_bias']:+.5f}**"],
         ["95% limits of agreement (V/m)",
          f"{num(V['raw.2mm.loa_lo'], 4)} to {num(V['raw.2mm.loa_hi'], 4)}",
          f"{num(V['raw.1mm.loa_lo'], 4)} to {num(V['raw.1mm.loa_hi'], 4)}"],
         ["Proportional bias", f"{V['raw.2mm.prop_bias_slope']:+.3f}",
          f"**{V['raw.1mm.prop_bias_slope']:+.3f}**"],
         ["Lin's CCC", f"{V['raw.2mm.ccc']:.3f}", f"**{V['raw.1mm.ccc']:.3f}**"],
         ["Slope / intercept", f"{V['raw.2mm.slope']:.3f} / {V['raw.2mm.intercept']:+.4f}",
          f"{V['raw.1mm.slope']:.3f} / {V['raw.1mm.intercept']:+.4f}"]])

    for dx in V["mm.resolutions"]:
        tag = f"{dx:g}mm"
        present = [t for t in MM_ORDER if f"mm.{tag}.{t}.slope" in V]
        T[f"mm_slope_{tag}"] = table(
            ["Montage", "Regression slope", "r", "CCC"],
            [[MM_NICE[t], f"{V[f'mm.{tag}.{t}.slope']:.3f}", f"{V[f'mm.{tag}.{t}.r']:.3f}",
              f"{V[f'mm.{tag}.{t}.ccc']:.3f}"]
             for t in sorted(present, key=lambda x: V[f"mm.{tag}.{x}.slope"])])
        if f"mm.{tag}.rank_foc.agree" in V:
            T[f"mm_rank_{tag}"] = table(
                ["Target", "Best montage agrees", "Median focality lost", "Worst case"],
                [[NICE[r], f"{V[f'mm.{tag}.rank_foc.{r}.agree']}/{V[f'mm.{tag}.rank_foc.{r}.n']}",
                  f"{V[f'mm.{tag}.loss.{r}.median']:.1f}%",
                  f"{V[f'mm.{tag}.loss.{r}.worst']:.1f}%"]
                 for r in ROIS if f"mm.{tag}.rank_foc.{r}.agree" in V])

    if V.get("tc.sweep"):
        # mean ± SD (range) per subject, fifth-round comment 4
        def _spread(t, k, dp=1):
            return (f"{V[f'{t}.{k}_mm' if k in ('com', 'comtop', 'peak') else f'{k}_pct']:.{dp}f} "
                    f"± {V[f'{t}.{k}_sd']:.{dp}f}")
        # Brain Stimulation comment 1: the two-carrier values are the reported result and the
        # static `.op` values are the worst-instant bound, so the table must lead with the former.
        # Reporting the static column here while the prose quotes the two-carrier figure was a
        # direct contradiction on the page.
        def _td(t, k, dp=1, unit=" mm"):
            """Two-carrier value where it exists, static otherwise (the sub-clamp rows agree)."""
            key = f"{t}.td.{k}" if f"{t}.td.{k}" in V else f"{t}.{k}"
            return f"{V[key]:.{dp}f}{unit}"

        has_td = any(f"tc.{r2/1e3:g}k.td.com_mm" in V for r2 in V["tc.sweep"])
        T["twochannel"] = table(
            ["Ch-2 contact", "I₂ delivered (two-carrier)", "Clamped", "Δ thalamic envelope",
             "Δ focality", "CoM shift (mean ± SD)", "range", "Top-decile CoM",
             "CoM, static bound"] if has_td else
            ["Ch-2 contact", "I₂ delivered", "Clamped", "Δ thalamic envelope",
             "Δ focality", "CoM shift (mean ± SD)", "range", "Top-decile CoM", "Peak shift"],
            [[f"{r2/1e3:g} kΩ",
              _td(f"tc.{r2/1e3:g}k", "I2_mA", 3, " mA"),
              f"{V[f'tc.{r2/1e3:g}k.n_clamped']}/{V['tc.n']}",
              pct(V.get(f"tc.{r2/1e3:g}k.td.dT_pct", V[f"tc.{r2/1e3:g}k.dT_pct"])),
              pct(V.get(f"tc.{r2/1e3:g}k.td.dF_pct", V[f"tc.{r2/1e3:g}k.dF_pct"])),
              (f"{_td(f'tc.{r2/1e3:g}k', 'com_mm', 1, '')} ± "
               f"{V.get(f'tc.{r2/1e3:g}k.td.com_sd', V[f'tc.{r2/1e3:g}k.com_sd']):.1f} mm"),
              (f"{V.get(f'tc.{r2/1e3:g}k.td.com_lo', V[f'tc.{r2/1e3:g}k.com_lo']):.1f}–"
               f"{V.get(f'tc.{r2/1e3:g}k.td.com_hi', V[f'tc.{r2/1e3:g}k.com_hi']):.1f} mm"),
              _td(f"tc.{r2/1e3:g}k", "comtop_mm"),
              (f"{V[f'tc.{r2/1e3:g}k.com_mm']:.1f} mm" if has_td else
               f"{V[f'tc.{r2/1e3:g}k.peak_mm']:.1f} ± {V[f'tc.{r2/1e3:g}k.peak_sd']:.1f} mm")]
             for r2 in V["tc.sweep"]])

    if V.get("tc1.sweep"):
        T["twochannel_1mm"] = table(
            # Both columns are static-model values; the main-text table leads with two-carrier,
            # so label these explicitly rather than let a reader assume they are comparable.
            ["Ch-2 contact", "I₂ (1 mm)", "Δ envelope (1 mm)", "CoM shift, 1 mm (static)",
             "CoM shift, 2 mm (static)", "Top-decile CoM (1 mm)"],
            [[f"{r2/1e3:g} kΩ", f"{V[f'tc1.{r2/1e3:g}k.I2_mA']:.3f} mA",
              pct(V[f"tc1.{r2/1e3:g}k.dT_pct"]),
              f"{V[f'tc1.{r2/1e3:g}k.com_mm']:.1f} ± {V[f'tc1.{r2/1e3:g}k.com_sd']:.1f} mm",
              (f"{V[f'tc.{r2/1e3:g}k.com_mm']:.1f} ± {V[f'tc.{r2/1e3:g}k.com_sd']:.1f} mm"
               if f"tc.{r2/1e3:g}k.com_mm" in V else "—"),
              f"{V[f'tc1.{r2/1e3:g}k.comtop_mm']:.1f} mm"]
             for r2 in V["tc1.sweep"]])

    if V.get("d2.n_targets"):
        T["dx2_check"] = table(
            ["Target", "**A.** Spearman, same 20 subjects", "Top-5 overlap",
             "**B.** 5-mm winner kept", "Its 2-mm rank", "Focality forfeited"],
            [[NICE[r], f"{V[f'd2.{r}.rho']:.2f}", f"{V[f'd2.{r}.jac']:.2f}",
              ("yes" if V[f"d2.{r}.kept"] else "**no**"),
              str(V[f"d2.{r}.rank"]), f"{V[f'd2.{r}.forfeit']:.2f}%"]
             for r in ROIS if f"d2.{r}.kept" in V])

    if V.get("cosim.rows"):
        T["cosim"] = table(
            ["Electrode impedance", "Drive voltage", "Delivered current",
             "Median field (a.u.)"],
            [[f"{p/1000:g} kΩ",
              (f"{v:.1f} V (at compliance)" if v >= V["cosim.v_cc"] - 0.1 else f"{v:.1f} V"),
              f"{1e3*i:.3f} mA", f"{e:.1f}"] for p, v, i, e in V["cosim.rows"]])

    if V.get("lcurve"):
        T["learning_curve"] = table(
            ["Training heads"] + [NICE[r] for r in ROIS],
            [[str(n)] + [f"{V['lcurve'][n][r]:.0f}%" for r in ROIS] for n in sorted(V["lcurve"])])

    if V.get("regret.sizes"):
        T["learning_regret"] = table(
            ["Training heads", "Median regret", "P(within 1%)", "P(within 2%)", "P(within 5%)",
             "Median rank"],
            [[str(n),
              f"{np.mean([V[f'regret.{n}.{r}.median_regret_pct'] for r in ROIS]):.2f}%",
              f"{100*np.mean([V[f'regret.{n}.{r}.p_within_1pct'] for r in ROIS]):.0f}%",
              f"{100*np.mean([V[f'regret.{n}.{r}.p_within_2pct'] for r in ROIS]):.0f}%",
              f"{100*np.mean([V[f'regret.{n}.{r}.p_within_5pct'] for r in ROIS]):.0f}%",
              f"{np.mean([V[f'regret.{n}.{r}.median_rank'] for r in ROIS]):.1f}"]
             for n in V["regret.sizes"]])
        T["learning_regret_m1"] = table(
            ["Training heads", "Exact-match recovery", "Median regret", "P(within 1%)",
             "Median rank"],
            [[str(n), f"{V['lcurve'][n]['M1_handknob']:.0f}%",
              f"{V[f'regret.{n}.M1_handknob.median_regret_pct']:.2f}%",
              f"{100*V[f'regret.{n}.M1_handknob.p_within_1pct']:.0f}%",
              f"{V[f'regret.{n}.M1_handknob.median_rank']:.0f}"]
             for n in V["regret.sizes"]])
    return T


if __name__ == "__main__":
    V = values()
    T = tables(V)
    pat = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    for k in sorted(V):
        if pat and pat not in k.lower():
            continue
        v = V[k]
        print(f"  {k:<44} {v if not isinstance(v, float) else f'{v:.6g}'}")
    print(f"\n{len(V)} values, {len(T)} tables: {', '.join(sorted(T))}")
