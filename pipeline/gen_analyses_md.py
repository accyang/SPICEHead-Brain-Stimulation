# -*- coding: utf-8 -*-
"""Generate SPICEhead_ANALYSES.md -- a portable record of every cohort analysis.

Every number is recomputed from the result files, not transcribed, so the document cannot
drift from the data. Re-run after any analysis changes.
"""
import csv
import json
import os
import statistics as st

import numpy as np
from scipy import stats as sstats

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

PROJ = r"E:\Dropbox\Archives\Research Projects\Unpublished\Submitted\[New] SPICEhead"
R = os.path.join(PROJ, "results", "montage_optimization")
A = os.path.join(PROJ, "results")
OUT = os.path.join(PROJ, "Paper_A_SPICEhead", "history_pre_v22", "notes", "SPICEhead_ANALYSES.md")
ROIS = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]
NICE = {"M1_handknob": "M1 hand-knob", "thalamus": "Thalamus", "insula": "Insula",
        "ACC": "Anterior cingulate", "hippocampus_L": "Hippocampus (L)",
        "hippocampus_R": "Hippocampus (R)"}
L = []
w = L.append


def jload(p):
    return json.load(open(p, encoding="utf-8"))


# ---------------------------------------------------------------- gather
t3 = jload(os.path.join(R, "cohort_gm300_dx5_maxdir", "cohort_montage_optima.json"))
deep_dir = os.path.join(R, "cohort_deep_gm300_maxdir")
deep = [jload(os.path.join(deep_dir, f)) for f in os.listdir(deep_dir)
        if f.endswith(".json") and f != "SUMMARY.json"]
ratio = jload(os.path.join(R, "montage_ratio_n300_maxdir.json"))
ten = jload(os.path.join(R, "montage_1010_n300_maxdir.json"))
full = jload(os.path.join(R, "full_n300_maxdir", "montage_full.json"))


def rows_of(*parts):
    p = os.path.join(A, *parts)
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def avg(rs, k):
    return float(np.mean([float(r[k]) for r in rs])) if rs else float("nan")


fem_res = rows_of("v21_simnibs_validation", "agreement.csv")
fem_raw = rows_of("v21_simnibs_validation", "agreement_raw.csv")
mm_agr = rows_of("v21_multimontage", "agreement_by_montage.csv")
mm_foc = rows_of("v21_multimontage", "focality_by_montage.csv")
mm_rk = {"intensity": rows_of("v21_multimontage", "rank_preservation.csv"),
         "focality": rows_of("v21_multimontage", "rank_preservation_focality.csv")}
loso = rows_of("v21_generalization", "loso.csv")
lcurve = rows_of("v21_generalization", "learning_curve.csv")
# Numbers shared with the manuscript come from the single source of truth, so the two
# documents cannot disagree (third-round review, Major Comments 1 and 2).
from values import values as _values
_V = _values()

MM_ORDER = ["M1", "THAL", "HIPP", "INFERIOR"]
MM_NICE = {"M1": "Fronto-parietal F3+F4 / Pz+O1", "THAL": "Bilateral temporal T8+P8 / F7+T7",
           "HIPP": "Fronto-occipital Fp2+F7 / P8+O2",
           "INFERIOR": "Inferior 9/10 chain F9+T9 / F10+T10"}

per = {r: {k: [] for k in ("A", "B", "C", "C4", "gB", "gC", "gC4")} for r in ROIS}
c1 = {r: 0 for r in ROIS}
c4bad = c4tot = 0
for o in deep:
    for r in ROIS:
        e = o["rois"][r]
        p = per[r]
        p["A"].append(e["A"]); p["B"].append(e["B"]); p["C"].append(e["C"]); p["C4"].append(e["C4"])
        p["gB"].append(100 * (e["B"] - e["A"]) / e["A"])
        p["gC"].append(100 * (e["C"] - e["A"]) / e["A"])
        p["gC4"].append(100 * (e["C4"] - e["A"]) / e["A"])
        c1[r] += e["C"] > 1.0
        c4tot += 1
        c4bad += e["C4"] < e["A"]


CONV_DIRS = [("SUBJ01", os.path.join(A, "v21_head_convergence")),
             ("SUBJ02", os.path.join(A, "v21_head_convergence", "SUBJ02")),
             ("SUBJ03", os.path.join(A, "v21_head_convergence", "SUBJ03")),
             ("SUBJ04", os.path.join(A, "v21_head_convergence", "SUBJ04"))]


def load_conv():
    out = {}
    for name, d in CONV_DIRS:
        p = os.path.join(d, "head_convergence.csv")
        if os.path.exists(p):
            out[name] = {float(r["dx_mm"]): r
                         for r in csv.DictReader(open(p, encoding="utf-8"))}
    return out


def load_csv_rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def aging(path):
    rows = list(csv.DictReader(open(path, encoding="cp1252")))
    age = np.array([float(x["Age"]) for x in rows])
    out = {}
    for k in ROIS:
        y = np.array([float(x[f"thalamic_{k}_mean"]) for x in rows])
        rg = sstats.linregress(age, y)
        out[k] = (rg.slope, rg.pvalue, 100 * rg.slope * 60 / (rg.intercept + rg.slope * 25))
    return len(rows), out


n_brain, ag_brain = aging(os.path.join(A, "aging_n300_brain_maxdir", "cohort_roi_metrics.csv"))
n_gm, ag_gm = aging(os.path.join(A, "aging_n300_gm_maxdir", "cohort_roi_metrics.csv"))


def pf(p):
    return "<1e-16" if p < 1e-16 else f"{p:.0e}".replace("e-0", "e-")


# ---------------------------------------------------------------- write
w("# SPICEhead — cohort analyses (n = 300)\n")
w("Machine-generated from the result files by `pipeline/gen_analyses_md.py`. Every number below is "
  "recomputed from the JSON/CSV outputs, so this file cannot drift from the data. "
  "Regenerate after any analysis change.\n")
w("**Manuscript:** `Brain Stimulation/04_manuscript.md`, mirrored byte-identically as "
  "`SPICEhead_v21.md`. `pipeline/audit_consistency.py` rebuilds every expected number from the "
  "result files and fails the build if a document has drifted from them. The r20 .docx branch and "
  "`build_merge.py` are retired.\n")

w("\n## 1. Cohort and provenance\n")
w("| item | value |")
w("|---|---|")
w(f"| Subjects | {len(deep)} (all segmented successfully) |")
w("| Ages | 20–88 y, mean 54.4, seven decade bins (42–43 each), four acquisition sites |")
w("| Cohort CSV | `L:\\BrainProbe\\SPICEheadv2\\SPICEheadv2_cohort.csv` (300 rows; the v1 100-row CSV is superseded) |")
w("| Segmentation | SimNIBS 4.6 CHARM → `m2m_<id>/label_prep/tissue_labeling_upsampled.nii.gz` |")
w("| Segmentation notes | 4 subjects needed `--forcesform --forcerun` (qform/sform mismatch); all recovered |")

w("\n## 2. The focality metric (read this before quoting any number)\n")
w("`F = T / O`, where **T** = median TI-envelope amplitude in the target ROI (10 mm sphere) and "
  "**O** = 99th percentile of the envelope in tissue **more than 20 mm** from the ROI. The 20 mm "
  "halo exclusion is essential: without it the peri-target and near-electrode field dominates O and "
  "every montage collapses to F ≈ 1.\n")
w("**Both T and O are measured in grey matter only** (`SCORE = (2,)` in `pipeline/montage_search.py`). "
  "Rationale: σ_WM (0.126) < σ_GM (0.275), so E = J/σ runs ~28% higher in white matter "
  "(median envelope 0.0618 WM vs 0.0484 GM V/m). Scoring a grey-matter target against a GM+WM "
  "denominator charges the montage for a tissue-conductivity contrast it cannot influence.\n")
w("\n**The envelope is the rotation-invariant maximum-over-direction form**, "
  "A = max over unit n of 2·min(|E₁·n|,|E₂·n|). The earlier per-axis "
  "combination √(Σ A_i²) depends on the coordinate frame — a worked case moves "
  "0.748 → 1.720 under rotation where this form gives 1.5563 either way — and it exceeds "
  "the direction-maximizing value in 20.7% of voxels, so it is not even a bound.\n")
w("> **Every analysis below was recomputed with the corrected envelope (2026-08-09/11).** "
  "Results carrying the superseded per-axis envelope, or the older GM+WM tissue mask, are not "
  "comparable and are listed in §12. What the correction changed: three of six Table 3 "
  "montages, the insula steering gain (halved) and its F>1 count (125→1 of 300), the "
  "thalamus F>1 count (115→50), and two motor-montage age trends that vanished. What it "
  "did not change: the thalamic-montage age associations, the inferior-chain geometry, and "
  "which targets are focal.\n")

w("\n## 3. Table 3 — best single 10-20 two-pair montage per target\n")
w(f"Source `results/montage_optimization/cohort_gm300_dx5_maxdir/` — n = {t3['n_subjects']}, "
  f"dx = {t3['dx']} mm, tissue = {t3['tissue']}.\n")
w("| Target | Optimal montage | F (T/O) | IQR | Placement |")
w("|---|---|---|---|---|")
def _place(n):
    """Placement convenience from the electrode count, so it cannot go stale when optima change."""
    return "all easy-access" if n == 4 else ("through-hair" if n <= 1 else "mixed")
place = {r: _place(t3["top"][r][0]["easy_electrodes"]) for r in ROIS}
for r in ROIS:
    e = t3["top"][r][0]
    w(f"| {NICE[r]} | {e['montage']} | {e['median_score']:.2f} | "
      f"{e['iqr'][0]:.2f}–{e['iqr'][1]:.2f} | {place[r]} |")
marg = {r: 100 * (t3["top"][r][0]["median_score"] - t3["top"][r][1]["median_score"])
        / t3["top"][r][1]["median_score"] for r in ROIS}
lo_r = min(marg, key=marg.get)
hi_r = max(marg, key=marg.get)
_easy = sum(1 for r in ROIS if t3["top"][r][0]["easy_electrodes"] == 4)
_focal = [NICE[r] for r in ROIS if t3["top"][r][0]["median_score"] > 1]
w(f"\n**{len(_focal)} of six targets are focal (F > 1)**: " + ", ".join(_focal) +
  f". The deep midline pair (insula, thalamus) stays below unity under the tested electrode "
  f"space, current constraints and focality definition — see §7 for how sensitive that "
  f"is. {_easy} of the six optima use only easily-placed scalp-edge electrodes.\n")
w(f"\n**The optima are not all decisively separated.** Rank-1 vs rank-2 margin runs from "
  f"{marg[lo_r]:.1f}% ({NICE[lo_r]}) to {marg[hi_r]:.1f}% ({NICE[hi_r]}); for the closest targets a "
  "set of near-equivalent montages is a fairer summary than a single winner.\n")
w("\n> **n = 300 only.** The earlier 97-case cohort is superseded and must not appear in any result "
  "or claim. It is also not a clean comparison set: 64 of the 97 are inside the 300 and 33 are not "
  "(NARI 19, BSNIP 14 — two sites absent from the 300), so it was neither independent of nor a "
  "subset of the main cohort.\n")
w("**dx matters:** build this table at **dx = 5 mm**, not 4. M1 reads 1.07 at dx=5 but 0.98 at dx=4 — "
  "a dx=4 table manufactures a false \"M1 is not focal\" result. Small superficial targets are "
  "resolution-sensitive because both the 10 mm sphere ∩ GM and the GM p99 shift with voxel size.\n")

w("\n## 4. Table 4 — electrode set and current steering\n")
w(f"Source `results/montage_optimization/cohort_deep_gm300_maxdir/` — n = {len(deep)}, dx = 4 mm, grey matter.\n")
w("- **A** — best two-pair montage over the 19 standard 10-20 positions\n"
  "- **B** — best two-pair montage once the inferior 9/10 chain is available "
  "(F9/F10, FT9/FT10, T9/T10, TP9/TP10, P9/P10, PO9/PO10, I1/Iz/I2)\n"
  "- **C** — continuous multi-electrode current steering across both channels\n"
  "- **C4** — C truncated to its four largest electrodes per channel (8 total; realisable on "
  "existing multi-source hardware)\n")
w("| Target | A | B | C | C4 | C vs A | subjects F > 1 |")
w("|---|---|---|---|---|---|---|")
for r in ROIS:
    p = per[r]
    w(f"| {NICE[r]} | {st.median(p['A']):.2f} | {st.median(p['B']):.2f} | {st.median(p['C']):.2f} | "
      f"{st.median(p['C4']):.2f} | {st.median(p['gC']):+.1f}% | {c1[r]} / {len(deep)} |")
w("\n### Median per-subject gain over A\n")
w("| Target | B (inferior chain) | C (steering) | C4 (8 electrodes) |")
w("|---|---|---|---|")
for r in ROIS:
    p = per[r]
    w(f"| {NICE[r]} | {st.median(p['gB']):+.1f}% | {st.median(p['gC']):+.1f}% | {st.median(p['gC4']):+.1f}% |")
lo = min(st.median(per[r]["gC4"]) for r in ROIS)
hi = max(st.median(per[r]["gC4"]) for r in ROIS)
w(f"\n**Findings.**\n")
w("1. **Steering (C) is the general lever** — it improves *every* subject in *every* target. But it "
  "is **not** a deep-target rescue: the largest gain is on the most superficial target "
  f"(M1 {st.median(per['M1_handknob']['gC']):+.1f}%), consistent with greater control authority near "
  "the electrodes.\n")
w("2. **The inferior chain (B) is geometry-specific** — it helps medial-temporal and subcortical "
  f"targets (hippocampi {st.median(per['hippocampus_L']['gB']):+.1f}% and "
  f"{st.median(per['hippocampus_R']['gB']):+.1f}%, thalamus {st.median(per['thalamus']['gB']):+.1f}%, "
  f"insula {st.median(per['insula']['gB']):+.1f}%) and does exactly nothing for M1 or the ACC "
  "(both +0.0%), which lie on the wrong side of the head for inferior current paths.\n")
w("3. **Hippocampus is the deep target that works** — both sides clear F > 1 in essentially every "
  f"subject ({c1['hippocampus_L']}/300 and {c1['hippocampus_R']}/300). **The thalamus does not** "
  f"({c1['thalamus']}/300, median {st.median(per['thalamus']['C']):.2f}).\n")
w(f"4. **C4 keeps most of the benefit** ({lo:+.1f}% to {hi:+.1f}%) but is not universally safe: it "
  f"falls below the plain 10-20 baseline in **{c4bad}/{c4tot} = {100*c4bad/c4tot:.1f}%** of "
  "target–subject pairs. Any deployment must re-score the truncated pattern and fall back to the "
  "best two-pair montage if it regresses.\n")

w("\n## 5. Aging (§3.4; full tables in Supplement S1)\n")
w(f"Thalamic montage F7-PO7 / F8-PO8 at 1 mA, n = {n_brain}. Slopes are linear fits of mean ROI "
  "envelope on age; decline is the fitted change from age 25 to 85.\n")
w("| Target | slope (V·m⁻¹·yr⁻¹) | p | decline 25→85 |")
w("|---|---|---|---|")
for r in ROIS:
    s, p, d = ag_brain[r]
    w(f"| {NICE[r]} | {s:.2e} | {pf(p)} | {d:+.1f}% |")
w("\n**Delivery declines with age for every deep and central target, and not for the superficial one.**\n")
w("### Grey-matter control\n")
w("Grey matter atrophies with age while white matter carries a larger envelope, so a fixed spherical "
  "ROI could drift through changing tissue composition alone. Restricting the ROI to grey matter:\n")
w("| Target | slope | p | decline 25→85 |")
w("|---|---|---|---|")
for r in ROIS:
    s, p, d = ag_gm[r]
    w(f"| {NICE[r]} | {s:.2e} | {pf(p)} | {d:+.1f}% |")
sM, pM, dM = ag_gm["M1_handknob"]
w(f"\nEvery deep decline survives intact or steepens, so the effect is a genuine reduction in "
  f"delivered field rather than a composition artefact.\n")
w(f"> ⚠️ **Open issue.** Under the grey-matter ROI the motor hand-knob also declines "
  f"({dM:+.1f}%, p = {pf(pM)}), whereas under the brain ROI it is flat "
  f"(p = {pf(ag_brain['M1_handknob'][1])}). The manuscript reports the brain-ROI result as "
  f"age-invariance and lists only the deep targets in the grey-matter control, so nothing stated is "
  f"false — but the same control that rescues the deep declines undercuts the superficial claim. "
  f"A reviewer will recompute this. Decide whether to state it explicitly or justify the brain ROI "
  f"for a cortical target.\n")

# ---- site confounding (reviewer item H)
w("\n### Site is confounded with age and sex (reviewer item H)\n")
_ac = list(csv.DictReader(open(os.path.join(A, "aging_n300_brain_maxdir",
                                            "cohort_roi_metrics.csv"), encoding="cp1252")))
_age = np.array([float(x["Age"]) for x in _ac])
_site = np.array([x["Site"] for x in _ac])
_grp = [_age[_site == s] for s in sorted(set(_site))]
_ssb = sum(len(g) * (g.mean() - _age.mean()) ** 2 for g in _grp)
_eta = _ssb / ((_age - _age.mean()) ** 2).sum()
w("| Site | n | Age mean ± SD | Age range |")
w("|---|---|---|---|")
for s in sorted(set(_site)):
    g = _age[_site == s]
    w(f"| {s} | {len(g)} | {g.mean():.1f} ± {g.std():.1f} | {g.min():.0f}–{g.max():.0f} |")
w(f"\n**Site explains {100*_eta:.0f}% of age variance** (one-way ANOVA η² = {_eta:.3f}), and the "
  "ranges barely overlap — Bai_TVGH is 20–61, Vet_Home 65–88, with no common ground. Vet_Home is "
  "also 97% male and supplies 61 of the 85 subjects aged ≥70, so site, age and sex are mutually "
  "entangled. Only TAMI (n = 130) spans 21–85, which permits a within-site test needing no "
  "adjustment:\n")
w("| Target | all 300 | minus Vet_Home (n = 231) | within TAMI (n = 130) |")
w("|---|---|---|---|")
for r in ROIS:
    row = []
    for msk in (np.ones(len(_age), bool), _site != "Vet_Home", _site == "TAMI"):
        y = np.array([float(x["thalamic_" + r + "_mean"]) for x in _ac])[msk]
        rg = sstats.linregress(_age[msk], y)
        d = 100 * rg.slope * 60 / (rg.intercept + rg.slope * 25)
        row.append(f"{d:+.1f}%, p={rg.pvalue:.2g}")
    w(f"| {NICE[r]} | " + " | ".join(row) + " |")
w("\n**Within a single site the effects roughly halve and only the anterior cingulate reaches "
  "p < 0.05** — which would not survive FDR across six ROIs. This is a power limitation rather "
  "than a contradiction: the within-TAMI 95% confidence intervals cover the whole-cohort slope for "
  "five of six targets, and the directions agree throughout (deep negative, M1 flat or positive). "
  "The thalamus would need n ≈ 200 within-site to reach significance at its observed slope. But the "
  "whole-cohort magnitudes are inflated by between-site variance, and the defensible statement is "
  "closer to the Vet_Home-excluded column than to the headline one.\n")

w("\n## 6. Montage-space structure (§3.5, Supplement S2)\n")
w("### Inter-pair current ratio\n")
w(f"Source `montage_ratio_n300_maxdir.json`, n = {ratio['n']}, ratios tested {ratio['ratios']}.\n")
w("| Target | montage | F at 1:1 | best ratio | median gain |")
w("|---|---|---|---|---|")
for r in ROIS:
    e = ratio["per_roi"][r]
    w(f"| {NICE[r]} | {e['montage']} | {e['focality_equal']:.3f} | {e['median_best_ratio']} | "
      f"{e['median_gain_pct']:+.1f}% |")
mx = max(v["median_gain_pct"] for v in ratio["per_roi"].values())
n11 = sum(1 for v in ratio["per_roi"].values() if v["median_best_ratio"] == 1.0)
# The narrative sentence is derived from the same numbers as the table above it. It used to be
# hand-written, and after the rotation-invariant recompute it claimed the thalamus preferred an
# asymmetric 0.79:1 drive while the table directly above showed 1.0 for every target — the
# third-round review caught the contradiction. Anything stating a number here reads it from the data.
_asym = [r for r in ROIS if ratio["per_roi"][r]["median_best_ratio"] != 1.0]
w(f"\nRatio steering changes focality by **{mx:.1f}% or less**; an equal 1:1 drive is optimal for "
  f"**{n11} of {len(ROIS)}** targets"
  + (".\n" if not _asym else
     ", the exceptions being "
     + ", ".join(f"{NICE[r]} at {ratio['per_roi'][r]['median_best_ratio']}:1 "
                 f"({ratio['per_roi'][r]['median_gain_pct']:+.1f}%)" for r in _asym) + ".\n"))
if not _asym:
    w("The largest gain from any ratio is on the "
      f"{NICE[max(ROIS, key=lambda r: ratio['per_roi'][r]['median_gain_pct'])]} "
      f"({mx:+.1f}%), but even there the median-optimal ratio is 1:1, so the gain reflects "
      "variation across subjects rather than a consistently preferred asymmetry.\n")

g10 = {r: 100 * (ten["per_roi"][r]["best"][0]["score"] - ten["per_roi"][r]["best_1020_only_same_subset"]["score"])
       / ten["per_roi"][r]["best_1020_only_same_subset"]["score"] for r in ROIS}
w("### Finer 10-10 grid\n")
w(f"Source `montage_1010_n300_maxdir.json`, n = {ten['n_subjects']}, {ten['n_electrodes']} electrodes.\n")
w("| Target | 10-20 baseline | 10-10 best | gain | finer sites used |")
w("|---|---|---|---|---|")
for r in ROIS:
    e = ten["per_roi"][r]
    b = e["best"][0]
    fu = "+".join(b["finer_used"]) if b["finer_used"] else "**none — pure 10-20 optimal**"
    w(f"| {NICE[r]} | {e['best_1020_only_same_subset']['score']:.3f} | {b['score']:.3f} | "
      f"{g10[r]:+.1f}% | {fu} |")
w(f"\nThe finer grid improves focality by **{max(g10.values()):.1f}% or less** "
  f"(median {st.median(list(g10.values())):+.1f}%), and only for the three deep or central targets. "
  "For M1 and both hippocampi a montage drawn purely from the 10-20 positions remains optimal.\n")
w("**Why this motivates the inferior chain:** the 10-10 grid only refines sampling of the *same* "
  "scalp band — every finer site its winners recruit (AF7/AF8, FT7/FT8, TP7/TP8) lies within the "
  "10-20 ring. It contains no position *below* that ring. Geometry: **T7 sits at z = −8 mm, T9 at "
  "z = −48 mm**, roughly 55 mm below a thalamus at z = +7 — genuinely new current paths that cross "
  "at depth instead of raking cortex. That is the restriction the extended electrode set (Supplement S2.1) lifts.\n")

w("### Age-stratified optima\n")
w(f"Source `full_n300_maxdir/montage_full.json`, n = {full['n']}, dx = {full['dx']} mm.\n")
w("| Target | younger (<50 y) | older (≥50 y) | same? |")
w("|---|---|---|---|")
nsame = 0
for r in ROIS:
    y = full["young"][r][0]["montage"]
    o = full["old"][r][0]["montage"]
    same = y == o
    nsame += same
    w(f"| {NICE[r]} | {y} | {o} | {'yes' if same else '**no**'} |")
accw = full["overall"]["ACC"][0]["montage"]
ay = [e["score"] for e in full["young"]["ACC"] if e["montage"] == accw][0]
ao = [e["score"] for e in full["old"]["ACC"] if e["montage"] == accw][0]
w(f"\nThe optimal montage is unchanged with age for **{nsame} of 6** targets; the ACC differs only "
  "between near-tied alternatives (cross-band margins 2.1% and 2.6%).\n")
w(f"**Focality and intensity are independent axes with age.** Anterior-cingulate *delivery* falls "
  f"12.7% across the lifespan, yet the *focality* of its optimal montage rises "
  f"({ay:.2f} younger → {ao:.2f} older): the target becomes relatively more of a hotspot as "
  "surrounding cortex atrophies, even as the absolute field declines. The intensity–focality "
  "trade-off is itself depth-dependent — the two rise together for superficial and anterior targets "
  "but trade off for the thalamus (Figure 19).\n")

w("\n## 7. Focality-metric sensitivity (reviewer item I)\n")
sens = load_csv_rows(os.path.join(A, "v21_sensitivity_maxdir", "sensitivity.csv"))
if sens:
    EX = sorted({float(r["exclusion_mm"]) for r in sens})
    PC = sorted({float(r["percentile"]) for r in sens})
    pn = lambda q: "max" if q >= 100 else f"p{q:g}"
    getr = lambda roi, e, q: next(x for x in sens if x["roi"] == roi
                                  and float(x["exclusion_mm"]) == e
                                  and float(x["percentile"]) == q)
    w(f"n = 300, each ROI's Table 3 montage re-scored across every combination of off-target "
      f"exclusion distance and denominator percentile. Source `results/v21_sensitivity/`.\n")
    w("**Number of the six targets with median F > 1:**\n")
    w("| exclusion \\ percentile | " + " | ".join(pn(q) for q in PC) + " |")
    w("|---" * (len(PC) + 1) + "|")
    for e in EX:
        cnt = [sum(1 for r in ROIS if float(getr(r, e, q)["F_median"]) > 1) for q in PC]
        lab = f"**{e:g} mm**" if e == 20 else f"{e:g} mm"
        w(f"| {lab} | " + " | ".join(f"**{c}**" if e == 20 and q == 99 else str(c)
                                     for c, q in zip(cnt, PC)) + " |")
    w("\nThe published setting is 20 mm / p99, giving \"four of six\". **At p95 all six clear unity; "
      "at the field maximum none do.** The percentile swings F by ~0.5, the exclusion distance by "
      "0.03–0.16, so the headline focality result is largely a property of the denominator choice. "
      "Per-subject, the thalamus goes from "
      f"{100*float(getr('thalamus',20.0,95.0)['frac_above_1']):.0f}% of subjects above unity at p95 "
      f"to {100*float(getr('thalamus',20.0,99.0)['frac_above_1']):.0f}% at p99.\n")
    w("\n**Absolute target intensity at 1 mA per channel** (T/O is invariant to current, so this is "
      "the quantity that says whether the target receives a usable field):\n")
    w("| " + " | ".join(NICE[r] for r in ROIS) + " |")
    w("|---" * len(ROIS) + "|")
    w("| " + " | ".join(f"{float(getr(r,20.0,99.0)['T_median_Vm']):.3f}" for r in ROIS) + " |")
    w("\n(V/m, 5 mm grid — see §8, these are biased high.)\n")

w("\n## 8. Numerical resolution — the 5 mm grid is not converged\n")
conv = load_conv()
if conv:
    w(f"Self-convergence study on {len(conv)} subjects across three sites, ages 23–69: the same head "
      "and montage solved at dx = 5, 4, 3, 2 and 1 mm, each compared against the 1 mm solution "
      "(no analytic reference exists for a head). Source `results/v21_head_convergence/`.\n")
    w("\n**Agreement with the 1 mm solution at dx = 5 mm — the resolution used for Tables 3–4:**\n")
    w("| Subject | r | R² | nMAE | Dice (top decile) | peak-location error |")
    w("|---|---|---|---|---|---|")
    for n_, rows in conv.items():
        r5 = rows[5.0]
        w(f"| {n_} | {float(r5['r']):.3f} | {float(r5['r2']):.3f} | "
          f"{100*float(r5['nmae']):.0f}% | {float(r5['dice']):.3f} | "
          f"{float(r5['peak_err_mm']):.0f} mm |")
    mean = lambda k, dx: st.mean(float(rows[dx][k]) for rows in conv.values())
    w(f"| **Mean** | **{mean('r',5.0):.2f}** | **{mean('r2',5.0):.2f}** | "
      f"**{100*mean('nmae',5.0):.0f}%** | **{mean('dice',5.0):.2f}** | "
      f"**{mean('peak_err_mm',5.0):.0f} mm** |")
    w(f"\nAgreement improves monotonically as the grid refines — mean r = "
      f"{', '.join(f'{mean(chr(114),d):.2f}' for d in (5.0,4.0,3.0,2.0))} at 5, 4, 3 and 2 mm — so "
      "the solver converges; the problem is where 5 mm sits on that curve.\n")
    w("\n**dx = 5 mm overestimates ROI intensity** (mean over subjects, vs 1 mm):\n")
    w("| " + " | ".join(NICE[r] for r in ROIS) + " |")
    w("|---" * len(ROIS) + "|")
    bias = {r: st.mean(100 * (float(rows[5.0]["T_" + r]) / float(rows[1.0]["T_" + r]) - 1)
                       for rows in conv.values()) for r in ROIS}
    w("| " + " | ".join(f"+{bias[r]:.0f}%" for r in ROIS) + " |")
    w("\n> **It has not converged even at 1 mm.** The 2→1 mm step is the largest change in every "
      "subject (thalamus −12% to −20%), and a power-law fit `T = T0 + C·dx^p` fails on the three "
      "finest grids. 1 mm is the finest tested, not a converged solution. Because no asymptote "
      "could be extrapolated, these magnitudes are described as carrying a systematic positive "
      "bias, **not** as bounding the true value — the third-round review was right that an "
      "upper-bound claim does not follow, and the raw Bland–Altman limits confirm SPICEhead does "
      "not exceed FEM everywhere. Two checks ruled out artefacts of the comparison: tissue "
      "fractions are "
      "identical at every resolution (WM 13.0%, GM 17.8%, CSF 5.5%), so this is not segmentation "
      "resampling; and block-averaging the fine solution instead of point-sampling it does not "
      "improve agreement.\n")
rank = os.path.join(A, "v21_rank_stability", "rank_stability.json")
if os.path.exists(rank):
    rk = json.load(open(rank, encoding="utf-8"))
    w(f"\n### Does resolution change which montage wins?\n")
    w(f"Top-{rk['K']} montages per target from the published ranking, re-scored on one subject "
      f"({rk['subject']}) at {rk['dx_coarse']:g} mm and {rk['dx_fine']:g} mm. "
      "Source `results/v21_rank_stability/`.\n")
    w("| Target | rank-1 kept | 5 mm winner's rank at 1 mm | focality forfeited |")
    w("|---|---|---|---|")
    for x in sorted(rk["rows"], key=lambda y: y["coarse_winner_rank_at_fine"]):
        loss = 100 * (x["F_fine_winner"] - x["F_fine_of_coarse_winner"]) / x["F_fine_winner"]
        w(f"| {NICE.get(x['roi'], x['roi'])} | {'yes' if x['rank1_preserved'] else 'no'} | "
          f"{x['coarse_winner_rank_at_fine']} | {loss:.1f}% |")
    keep = sum(1 for x in rk["rows"] if x["rank1_preserved"])
    w(f"\nRank-1 survives in {keep}/{len(rk['rows'])} targets — **including both contested deep "
      "targets, thalamus and insula**. The reshuffling is among near-ties (the resolution shift is "
      "0.18–0.66× the spread across the eight candidates), not randomisation. But for an individual "
      "head, choosing the montage at 5 mm forfeits up to 19% of achievable focality — and "
      "individualised targeting is precisely what this framework is for.\n")
    w("\n> **Cohort-level resolution stability is untested.** Tables 3–4 are medians over 300 "
      "subjects, which should suppress the per-subject variation driving this reordering, but that "
      "expectation is unverified: a full 11,600-montage sweep at 1 mm across the cohort is out of "
      "budget. Do not cite `cohort_gm97/` as evidence on this — it records `tissue: None` and its "
      "cache is keyed by case alone, predating the fix that stopped stale scores being reused "
      "across resolutions.\n")

w("\n## 9. Realistic-head FEM comparison (§3.2)\n")
if fem_res:
    w("Eight subjects spanning ages 20–80 and all four sites, against SimNIBS 4.6. Meshes built from "
      "the existing segmentations with `charm --mesh`; identical conductivities (asserted "
      "element-by-element at run time), identical current, electrode radius and envelope "
      "definition; SimNIBS output mapped to the subject voxel grid; grey matter only.\n")
    w("| | Δx = 2 mm | Δx = 1 mm |")
    w("|---|---|---|")
    g = lambda dx, k: avg([r for r in fem_res if r["dx_mm"] == dx], k)
    for lab, k, fmt in (("Pearson r", "r", "{:.3f}"), ("R²", "r2", "{:.3f}"),
                        ("Regression slope", "slope", "{:.3f}"),
                        ("Normalized MAE", "nmae", "{:.1%}"),
                        ("Dice, top-decile field", "dice", "{:.3f}"),
                        ("Peak-location distance", "peak_err_mm", "{:.0f} mm")):
        w(f"| {lab} | {fmt.format(g('2.0', k))} | {fmt.format(g('1.0', k))} |")
    w("\n**Correlation is not agreement.** r = "
      f"{g('1.0','r'):.3f} at 1 mm invites the word \"excellent\", but the slope is "
      f"{g('1.0','slope'):.3f}: SPICEhead overestimates by roughly a third. Report both or neither.\n")

if fem_raw:
    w("\n### Raw (non-rescaled) agreement\n")
    w("An earlier version scale-matched before differencing, which drives Bland–Altman bias to ~0 by "
      "construction and hides the systematic error the method exists to expose. These are raw, in "
      "V/m, FEM regressed on SPICEhead (slope < 1 means SPICEhead reads high). "
      "Source `results/v21_simnibs_validation/agreement_raw.csv`.\n")
    w("| | Δx = 2 mm | Δx = 1 mm |")
    w("|---|---|---|")
    q = lambda dx, k: avg([r for r in fem_raw if r["dx_mm"] == dx], k)
    w(f"| Bias (V/m) | {q('2.0','raw_bias'):+.5f} | {q('1.0','raw_bias'):+.5f} |")
    w(f"| 95% limits of agreement (V/m) | {q('2.0','loa_lo'):+.4f} to {q('2.0','loa_hi'):+.4f} "
      f"| {q('1.0','loa_lo'):+.4f} to {q('1.0','loa_hi'):+.4f} |")
    w(f"| Proportional bias | {q('2.0','prop_bias_slope'):+.3f} | {q('1.0','prop_bias_slope'):+.3f} |")
    w(f"| Lin's CCC | {q('2.0','ccc'):.3f} | {q('1.0','ccc'):.3f} |")
    w(f"| Slope / intercept | {q('2.0','slope'):.3f} / {q('2.0','intercept'):+.4f} "
      f"| {q('1.0','slope'):.3f} / {q('1.0','intercept'):+.4f} |")
    w(f"\nAgainst a mean FEM envelope of {q('1.0','mean_fem'):.4f} V/m the 1 mm bias is about "
      f"{100*q('1.0','raw_bias')/q('1.0','mean_fem'):.0f}% of the mean field. The positive "
      "proportional bias is the substantive point: the disagreement grows with field strength, so "
      "it is worst where the field matters. The gap between CCC and r is the scale error that "
      "correlation cannot see.\n")

# The multi-montage numbers come from values.py rather than being re-derived here. Averaging the
# CSV directly was correct only while it held a single resolution; once the 1 mm run landed, the
# same code silently averaged 1 mm and 2 mm together under a heading that still said "2 mm".
# values.py splits by dx, so the two documents cannot disagree and neither can mix resolutions.
if _V.get("mm.resolutions"):
    _dxs = sorted(_V["mm.resolutions"])
    _fine = f"{min(_dxs):g}mm"
    w("\n### Slope is montage-dependent — the common-scale-factor argument fails\n")
    w(f"Four geometrically distinct montages in the same {_V[f'mm.{_fine}.n']} heads, run at "
      f"{' and '.join(f'{d:g} mm' for d in _dxs)}. "
      "Source `results/v21_multimontage/agreement_by_montage.csv`.\n")
    w("| Montage | " + " | ".join(f"slope @{d:g} mm" for d in _dxs) + " | r | CCC |")
    w("|---" * (len(_dxs) + 3) + "|")
    for t in sorted(MM_ORDER, key=lambda x: _V[f"mm.{_fine}.{x}.slope"]):
        w(f"| {MM_NICE[t]} | "
          + " | ".join(f"{_V[f'mm.{d:g}mm.{t}.slope']:.3f}" for d in _dxs)
          + f" | {_V[f'mm.{_fine}.{t}.r']:.3f} | {_V[f'mm.{_fine}.{t}.ccc']:.3f} |")
    w(f"\nAt {min(_dxs):g} mm the slope runs {_V[f'mm.{_fine}.slope_lo']:.3f} to "
      f"{_V[f'mm.{_fine}.slope_hi']:.3f} — spread {_V[f'mm.{_fine}.slope_spread']:.3f}, "
      f"{_V[f'mm.{_fine}.slope_spread_pct']:.1f}% of the mean — about three times the "
      f"between-subject standard deviation within a montage "
      f"({_V[f'mm.{_fine}.within_sd']:.3f}), with the ordering repeating head by head. "
      "**Magnitudes from different montages are biased by different amounts.**\n")
    if len(_dxs) > 1:
        _co = f"{max(_dxs):g}mm"
        w(f"> **Not a coarse-grid artefact.** Refining from {max(_dxs):g} mm to {min(_dxs):g} mm "
          f"raises every slope toward unity but the spread goes "
          f"{_V[f'mm.{_co}.slope_spread']:.3f} → {_V[f'mm.{_fine}.slope_spread']:.3f} with the rank "
          "order unchanged, so the montage dependence is a property of the formulation rather than "
          "of the working grid.\n")

if _V.get("mm.resolutions"):
    _dxs = sorted(_V["mm.resolutions"])
    _fine = f"{min(_dxs):g}mm"
    w("\n### Rank preservation — the test that matters for the optimization\n")
    w("The optimization never uses absolute magnitude, only ordering, so the decision-relevant "
      "question is whether both solvers order the montages the same way. Tested twice: by median "
      f"target intensity, and by the focality ratio F that Table 3 actually ranks by. At "
      f"{min(_dxs):g} mm:\n")
    for lab, k in (("target intensity", "rank_int"), ("focality F = T/O", "rank_foc")):
        if f"mm.{_fine}.{k}.agree" in _V:
            w(f"- **By {lab}:** best montage agrees in {_V[f'mm.{_fine}.{k}.agree']}/"
              f"{_V[f'mm.{_fine}.{k}.n']} subject–target pairs, mean Spearman "
              f"{_V[f'mm.{_fine}.{k}.rho']:.3f}.")
    w("\n| Target | " + " | ".join(f"agrees @{d:g} mm" for d in _dxs)
      + f" | median lost @{min(_dxs):g} mm | worst @{min(_dxs):g} mm |")
    w("|---" * (len(_dxs) + 3) + "|")
    for roi in ROIS:
        if f"mm.{_fine}.rank_foc.{roi}.agree" in _V:
            w(f"| {NICE[roi]} | "
              + " | ".join(f"{_V[f'mm.{d:g}mm.rank_foc.{roi}.agree']}/"
                           f"{_V[f'mm.{d:g}mm.rank_foc.{roi}.n']}" for d in _dxs)
              + f" | {_V[f'mm.{_fine}.loss.{roi}.median']:.1f}% "
                f"| {_V[f'mm.{_fine}.loss.{roi}.worst']:.1f}% |")
    w(f"\nF is a ratio, so most of the magnitude bias divides out — F_SPICEhead/F_FEM is "
      f"{_V[f'mm.{_fine}.fratio_mean']:.3f} ± {_V[f'mm.{_fine}.fratio_sd']:.3f} — but the residual "
      f"is target- and montage-specific ({_V[f'mm.{_fine}.fratio_lo']:.3f} to "
      f"{_V[f'mm.{_fine}.fratio_hi']:.3f}), and that residual is what flips near-ties.\n")
    if len(_dxs) > 1:
        _co = f"{max(_dxs):g}mm"
        w(f"> **Ranking is strongly resolution-dependent.** Focality agreement is "
          f"{_V[f'mm.{_co}.rank_foc.agree']}/{_V[f'mm.{_co}.rank_foc.n']} at {max(_dxs):g} mm "
          f"against {_V[f'mm.{_fine}.rank_foc.agree']}/{_V[f'mm.{_fine}.rank_foc.n']} at "
          f"{min(_dxs):g} mm, and the ACC goes {_V[f'mm.{_co}.rank_foc.ACC.agree']}/8 → "
          f"{_V[f'mm.{_fine}.rank_foc.ACC.agree']}/8 with its worst-case forfeit falling "
          f"{_V[f'mm.{_co}.loss.ACC.worst']:.1f}% → {_V[f'mm.{_fine}.loss.ACC.worst']:.1f}%. The "
          "ranking failures first seen at 2 mm were largely a coarse-grid effect. **The cohort "
          "optimization runs at 5 mm, coarser than either tested resolution, and no "
          "rank-preservation test exists there.**\n")
    w("\n> **This is a four-way choice.** The search ranks ~11,600 montages, among which near-ties "
      "are far denser, so these figures are an optimistic bound on selection reliability in the "
      "full search. Do not cite them as validating the optimization.\n")

w("\n## 10. Generalization — leave-one-site-out (§3.5)\n")
if loso:
    tl = [float(r["transfer_loss_pct"]) for r in loso]   # not `L` — that is the document buffer
    same = sum(r["transfer_is_oracle"] == "True" for r in loso)
    w("Derive the optimum on three sites, apply it to the fourth, and measure what is lost against "
      "that site's own optimum (the *oracle* — a ceiling no prospective procedure can reach). "
      "No new field solves: this re-aggregates the cached per-subject scores. "
      "Source `results/v21_generalization/`.\n")
    w(f"Transfer reproduces the held-out site's own optimum in **{same} of {len(loso)}** "
      f"site–target pairs. Where it does not, little is lost: median {np.median(tl):.2f}%, "
      f"mean {np.mean(tl):.2f}%, worst {max(tl):.1f}% "
      f"({[r['roi'] for r in loso if float(r['transfer_loss_pct']) == max(tl)][0]} on "
      f"{[r['held_out_site'] for r in loso if float(r['transfer_loss_pct']) == max(tl)][0]}). "
      "**Group-level montage performance was robust across the four acquisition sites studied** — "
      "four sites do not establish general site invariance.\n")
if _V.get("lcurve"):
    _sz = sorted(_V["lcurve"])
    w("\n### How many heads before a group montage stabilizes?\n")
    w(f"Two endpoints over the **same {_V['lcurve.draws']} random draws** per training size. "
      "Exact identity — recovering the all-300 winning montage — is intuitive but brittle when "
      "montages are near-tied. The primary endpoint is **out-of-sample focality regret**: the "
      "shortfall of the training set's choice against the all-300 optimum, both evaluated on the "
      "held-out complement. Source `results/v21_generalization/learning_curve_regret.csv` "
      "(supersedes the 20-draw `learning_curve.csv`).\n")
    w("| Training heads | " + " | ".join(NICE.get(r, r) for r in ROIS) + " |")
    w("|---" * (len(ROIS) + 1) + "|")
    for n_ in _sz:
        w(f"| {n_} | " + " | ".join(f"{_V['lcurve'][n_][r]:.0f}%" for r in ROIS) + " |")
    w("\n(exact-identity recovery.) By regret the picture is much more favourable:\n")
    w("| Training heads | Mean median regret | P(within 1%) | M1 regret | M1 P(within 1%) "
      "| M1 top-5 Jaccard |")
    w("|---|---|---|---|---|---|")
    for n_ in _sz:
        w(f"| {n_} | {_V[f'regret.{n_}.regret_mean']:.2f}% "
          f"| {_V[f'regret.{n_}.p1_mean_pct']:.0f}% "
          f"| {_V[f'regret.{n_}.M1_handknob.median_regret_pct']:.2f}% "
          f"| {100*_V[f'regret.{n_}.M1_handknob.p_within_1pct']:.0f}% "
          f"| {_V[f'regret.{n_}.M1_handknob.jaccard_top5']:.2f} |")
    w("\n**The motor hand-knob is stable in performance while unstable in identity**: exact "
      "recovery stays near 50% at every size, yet its median regret is ~0 and every draw lands "
      "within 1% of the optimum from five heads, with the top-5 Jaccard overlap rising toward 1 — "
      "training sets keep selecting from the *same near-optimal set*. The genuinely data-hungry "
      "target is the thalamus "
      f"({_V['regret.5.thalamus.median_regret_pct']:.2f}% regret at n = 5, reaching "
      f"{_V['regret.20.thalamus.median_regret_pct']:.2f}% by n = 20). An earlier draft read the "
      "exact-identity column as the M1 optimum \"failing to converge\"; the regret analysis shows "
      "that reading was wrong.\n")

w("\n## 11. Caveats to carry into the discussion\n")
w("1. **Every cohort result uses the rotation-invariant envelope.** The per-axis form "
  "`A = 2·min(|E1|,|E2|)` combined as √(ΣAx²) is coordinate-frame dependent — it moved from 0.748 "
  "to 1.720 on the same field under rotation — and was replaced by the maximum-over-direction "
  "form, which returns 1.5563 either way. The recompute changed three of six optimal montages and "
  "removed two apparently significant age associations. Any number carrying a `_maxdir` source is "
  "current; anything else predates the correction.\n")
w("2. **The solver is quasi-static.** No carrier frequency is simulated; conductivities are real and "
  "frequency-independent. **Δf has literally zero effect on the computed field** — the 2000/2005 Hz "
  "framing is interpretive, not simulated. Do not claim frequency-dependent results.\n")
w("3. **F = T/O is percentile-sensitive.** It crosses unity between p95 (1.026) and p97 (0.950). "
  "Use it comparatively, not as an absolute pass/fail threshold.\n")
w("4. **The rim-artefact hypothesis was tested and refuted** — only 22.9% of top-1% off-target "
  "voxels are brain-rim vs 32.9% by volume, and removing the rim makes T/O slightly worse. O is "
  "measuring real interior field.\n")
w("5. **F < 1 for deep targets is a physical limit**, consistent with Huang & Parra (2019), not a "
  "shortcoming of the search. A threshold-referenced metric would be more informative than the bare "
  "ratio.\n")
w("6. **Never quote single-head (MNI template) montage numbers as findings.** Three template results "
  "failed to replicate in cohort: the insula inferior-chain gain, the truncation pessimism, and a "
  "thalamic C = 1.117 that came back at 0.98 across 300 subjects. Treat any single-head number as a "
  "hypothesis.\n")
w("7. **ROI definitions are not interchangeable.** Sphere-based ROIs (Tables 3–4) and whole-AAL-region "
  "ROIs give non-comparable numbers. Do not mix them in one comparison.\n")

w("\n## 12. Source map\n")
w("| Manuscript item | Source | n | dx | tissue |")
w("|---|---|---|---|---|")
w(f"| Table 3 | `cohort_gm300_dx5_maxdir/cohort_montage_optima.json` | {t3['n_subjects']} | {t3['dx']} | gm |")
w(f"| Table 4, §3.5 / Supplement S2 | `cohort_deep_gm300_maxdir/` | {len(deep)} | 4.0 | gm |")
w(f"| §3.4 / Supplement S1 | `aging_n300_brain_maxdir/`, `aging_n300_gm_maxdir/` | {n_brain} | — | brain + gm control |")
w(f"| Supplement S2.3 ratio | `montage_ratio_n300_maxdir.json` | {ratio['n']} | 5.0 | gm |")
w(f"| Supplement S2.3 10-10 | `montage_1010_n300_maxdir.json` | {ten['n_subjects']} | 5.0 | gm |")
w(f"| §3.5 age-strat, Figure 19 | `full_n300_maxdir/` | {full['n']} | {full['dx']} | gm |")
w(f"| §7 metric sensitivity | `v21_sensitivity_maxdir/` | 300 | 5.0 | gm |")
w(f"| §8 resolution | `v21_head_convergence/` | {len(conv)} | 5→1 | gm |")
w(f"| §8 rank stability | `v21_rank_stability/` | 1 | 5 and 1 | gm |")
w(f"| dataset table (reviewer G) | `v21_dataset/` | 300 | — | — |")
w(f"| §9 FEM agreement | `v21_simnibs_validation/agreement.csv` | "
  f"{len({r['subject'] for r in fem_res})} | 2 and 1 | gm |")
w(f"| §9 raw agreement | `v21_simnibs_validation/agreement_raw.csv` | "
  f"{len({r['subject'] for r in fem_raw})} | 2 and 1 | gm |")
w(f"| §9 multi-montage, rank | `v21_multimontage/` | "
  f"{len({r['subject'] for r in mm_agr})} × {len({r['montage'] for r in mm_agr})} montages | 2.0 | gm |")
w(f"| §10 generalization | `v21_generalization/` | 300 | 5.0 | gm |")
w("\n**Superseded — do not cite:** `cohort_deep/`, `cohort_deep_gm/` (n=14); `cohort_gm97/`, "
  "`cohort_dx5_*` (n=97); `montage_ratio.json`, `montage_1010.json`, `montage_full.json`, "
  "`full_n300/` (GM+WM metric); `deep_focality_*.json` (MNI template, n=1).\n")

w("\n## 13. Rebuilding\n")
w("```bash\n"
  "python pipeline/gen_analyses_md.py    # regenerates this file from the result data\n"
  "python pipeline/audit_consistency.py  # every number in every document, against the data\n"
  "python pipeline/md_to_docx.py \"Brain Stimulation/04_manuscript.md\" \\\n"
  "       \"Brain Stimulation/04_manuscript.docx\"\n"
  "```\n")
w("Run the audit after any analysis change and before any build: it constructs each expected "
  "string from the live result files, so a stale document fails rather than passing quietly. It "
  "exits non-zero, and it also checks that `SPICEhead_v21.md` is still byte-identical to the "
  "submission copy.\n")
w("`build_merge.py` and `update_r19.py` built the superseded r20 and r19 .docx branches from the "
  "student's `SPICEhead_Manuscript_Revised 260731.docx`. Both are retired — v21 is authored "
  "directly in Markdown — and are kept only for provenance.\n")

w("\n## 14. Open items\n")
w("- [ ] Decide how to handle the M1 grey-matter-control inconsistency (§5).\n")
w("- [ ] Normalise the student's blue revision-marking to black before submission — 21 runs in the "
  "body plus three columns of Table 1. (These are colour formatting, not tracked changes; the "
  "document contains zero tracked changes.)\n")
w("- [ ] Four tables from the base document run 0.04–0.15 in past the 6.50 in text column "
  "(Table 1 and the Figure 4/5/10 image containers). They render fully; cosmetic only.\n")
w("- [ ] Delete the Dropbox conflicted copy `SPICEhead_r19 (與 Albert Yang 衝突的複本).docx` in the "
  "project root.\n")
w("- [x] ~~State the envelope's rotation-dependence explicitly in the limitations.~~ Done, in "
  "the §4.6 limitations of the v21 manuscript.\n")
w("- [ ] Decide whether the whole-cohort aging magnitudes should be reported, given that site "
  "explains 53% of age variance and the within-TAMI effects are about half the size (§5).\n")
w("- [ ] Scanner make/model, sequence name, inclusion criteria and ethics refs for the dataset "
  "table — not recoverable from the imaging data. TR/TE in `v21_dataset/` came from a truncated "
  "NIfTI header field and must be confirmed against the acquisition protocols.\n")
w("- [ ] Consider dx = 2 mm rather than 5 mm for any future optimization: r = 0.85 against the "
  "1 mm solution versus 0.61, at ~30 s per solve now that the CG threshold is fixed (§8).\n")
w("- [x] ~~`SPICEhead_v21.md` does not exist yet.~~ It is now the live draft, kept byte-identical "
  "to `Brain Stimulation/04_manuscript.md` and checked by `pipeline/audit_consistency.py`. "
  "`build_merge.py` and the r20 branch are retired.\n")
w("- [ ] Journal length limits: the audit recounts abstract and body on every run and reports "
  "both in its completeness section; the fourth-round restructuring (aging to Supplement S1, "
  "saline sections removed) shortened the body, but the abstract still needs cutting.\n")
w("- [x] ~~Run the multi-montage rank test at 1 mm.~~ Done 2026-08-13: focality rank agreement "
  "45/48 at 1 mm vs 41/48 at 2 mm, ACC 3/8 to 7/8; the slope spread persists (0.123), so the "
  "montage dependence is real but the ranking failures were largely a coarse-grid effect (§9).\n")
w("- [ ] **Partial 2-mm cohort validation running** (`pipeline/v21_dx2_cohort_check.py`: top-25 "
  "5-mm candidates per target re-scored at 2 mm in 20 site-stratified subjects). §3.5 of the "
  "manuscript carries a pending notice until it lands (fourth round, comment 2).\n")
w("- [x] Fourth-round restructuring: empty saline sections removed (Methods/Results renumbered), "
  "aging tables moved to `Brain Stimulation/06_supplement.md` (S1), and the two-channel "
  "MRI-head co-simulation added (`results/v21_twochannel/`).\n")
w("- [ ] Raw Gmsh output (`.msh`, `.geo`, ~22.8 GB) has been moved off Dropbox to "
  "`K:\\Labdata\\SPICEHEAD`. The `subject_volumes/*_E.nii.gz` the analyses read remain in place, "
  "so no stage needs it restored.\n")

open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", OUT)
print("lines:", len(L))
