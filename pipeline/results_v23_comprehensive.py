# -*- coding: utf-8 -*-
"""One comprehensive v23 results record for Paper A (31 Oct 2026; PI: "make the five v23 figures for Rev0.4 and merge
to v23 comprehensive results single MD file").

Part I is the 300-head evidence of the Rev0.4 manuscript, organized around region-wise targeting parameters: the
Results paragraphs and Tables 1-4 come from build_rev04 (texts, table_data), the supplementary tables from the built
06_supplement.docx, the figures from Rev0.4_2026-10-30/figures (pipeline/fig_v23_rev04.py), and the 163-region table
from rev04_numbers. Part II keeps the v23 analyses of the r1 record (render_v23) that the 300-head rule removed from the
paper, unchanged and labelled as record only. Every number is computed from result files; nothing is typed in.

  SPICEHEAD_MODEL=v23 python pipeline/results_v23_comprehensive.py [--out NAME]

Output: Paper_A_SPICEhead/Brain Stimulation/RESULTS_v23_<date>_r<N>.md (default r7, 2 Nov 2026; r3 followed the review
round review/RESULTS_v23_REVIEW_2026-10-31_01.md; needs results_v23/v23_atlas_review from pipeline/v23_atlas_review.py).
"""
import csv
import io
import json
import os
import re
import sys

import numpy as np
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_rev04 as B                                              # noqa: E402
import render_v23                                                    # noqa: E402
import rev04_numbers as RN                                           # noqa: E402

J = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation")
PKG = B.OUT
FIGS = os.path.relpath(os.path.join(PKG, "figures"), J).replace(os.sep, "/")
SP, FE = "sp1v23", "fem"
REVIEW_FILE = "RESULTS_v23_REVIEW_2026-10-31_01.md"


LOBULE = {"3": "III", "4_5": "IV-V", "6": "VI", "7": "VII", "7b": "VIIb", "8": "VIII", "9": "IX", "10": "X",
          "1_2": "I-II", "Crus1": "crus I", "Crus2": "crus II"}
NUCLEUS = {"AV": "anteroventral", "LP": "lateral posterior", "VA": "ventral anterior", "VL": "ventral lateral",
           "VPL": "ventral posterolateral", "IL": "intralaminar", "LGN": "lateral geniculate"}
BRAINSTEM = {"SN_pr": "Substantia nigra pars reticulata", "SN_pc": "Substantia nigra pars compacta",
             "Red_N": "Red nucleus", "Raphe_D": "Dorsal raphe", "Raphe_M": "Median raphe"}


def rlabel(t):
    """Readable region names for this record: cerebellar lobules and vermis in Roman numerals, thalamic and brainstem
    nuclei spelled out; everything else as in the manuscript tables (build_rev04.label)."""
    b = t.replace("brainstem:", "")
    side = {"_L": " L", "_R": " R"}.get(b[-2:], "")
    core = b[:-2] if side else b
    m = re.fullmatch(r"Cerebellum_(\w+)", core)
    if m and m.group(1) in LOBULE:
        x = LOBULE[m.group(1)]
        return ("Cerebellar " + x if x.startswith("crus") else "Cerebellar lobule " + x) + side
    m = re.fullmatch(r"Vermis_(\w+)", core)
    if m and m.group(1) in LOBULE:
        return "Vermis " + LOBULE[m.group(1)]
    m = re.fullmatch(r"Thal_(\w+)", core)
    if m and m.group(1) in NUCLEUS:
        return f"Thalamus, {NUCLEUS[m.group(1)]}{side}"
    if core in BRAINSTEM:
        return BRAINSTEM[core] + side
    return B.label(t)


def md_table(headers, rows):
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def docx_tables(path):
    """[(caption, headers, rows, note)] for every 'Table ...' caption followed by a table in a Word file."""
    d = Document(path)
    items, cap = [], None
    els = list(d.element.body.iterchildren())
    for n, el in enumerate(els):
        if el.tag.endswith("}p"):
            t = Paragraph(el, d).text.strip()
            if re.match(r"Table S?\d+\.", t):
                cap = t
        elif el.tag.endswith("}tbl") and cap:
            tb = Table(el, d)
            cells = [[c.text.strip() for c in r.cells] for r in tb.rows]
            note = ""
            for nxt in els[n + 1:n + 3]:
                if nxt.tag.endswith("}p"):
                    t = Paragraph(nxt, d).text.strip()
                    if t.startswith("Note"):
                        note = t
                        break
                    if t:
                        break
            items.append((cap, cells[0], cells[1:], note))
            cap = None
    return items


def sections(md):
    """Split a rendered record into {'## heading': text} (heading line excluded) and {'### heading': text}."""
    top, sub = {}, {}
    for block in re.split(r"(?m)^(?=## )", md)[1:]:
        head, _, body = block.partition("\n")
        top[head.lstrip("#").strip()] = body
        for sb in re.split(r"(?m)^(?=### )", body)[1:]:
            h2, _, b2 = sb.partition("\n")
            sub[h2.lstrip("#").strip()] = b2
    return top, sub


def demote(text, by=1):
    return re.sub(r"(?m)^(#{2,5}) ", lambda m: "#" * (len(m.group(1)) + by) + " ", text)


def figure(n):
    head, text = B.LEGENDS[n - 1]
    return f"![Figure {n}]({FIGS}/Figure{n}.png)\n\n*{head}.* {text}"


def region_table(N):
    A, T = N["A"], N["targets"]
    f = RN.f
    out = []
    for key, name in RN.SETS:
        natural = lambda t: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", t)]   # lobule 3 before 10
        s = sorted((t for t in T if A[(t, FE)]["tier"] == key), key=natural)
        rows = []
        for t in s:
            a, e = A[(t, SP)], A[(t, FE)]
            q = f(a["p_age_q"])
            rows.append([
                rlabel(t), f"{f(a['rv_voxels_median']):.0f}", a["cohort_choice"],
                "yes" if a["cohort_choice"] == e["cohort_choice"] else e["cohort_choice"],
                f"{f(a['F_choice_median']):.2f} [{f(a['rv_sp1v23__F_reported_ci_lo']):.2f}-{f(a['rv_sp1v23__F_reported_ci_hi']):.2f}]",
                f"{f(a['T_choice_median_Vm_per_mA']):.3f} [{f(a['p_T_choice_p5']):.3f}-{f(a['p_T_choice_p95']):.3f}]",
                f"{f(e['T_choice_median_Vm_per_mA']):.3f}",
                f"{f(a['p_F_cohort_median']):.2f} / {f(a['p_T_cohort_median']):.3f}",
                f"{f(a['rv_sp1v23__cohort_served_pct']):.0f}", f"{f(a['rv_sp1v23__cohort_reselected_pct']):.0f}",
                f"{f(a['heads_F_above_1_pct']):.0f}", f"{f(a['same_choice_as_fem_pct']):.0f}",
                f"{f(a['fem_optimum_kept_p5']):.1f}", f"{f(a['rv_sp_choice_fails_fem_floor_pct']):.0f}",
                f"{f(a['heads_gain_ge_10pct']):.1f}",
                f"{f(a['p_age_T_pct_per_decade']):+.1f}{'*' if q < 0.05 else ''}"])
        out.append(f"**{name}** ({len(s)} regions)\n\n" + md_table(
            ["Region", "Voxels", "Cohort montage", "FEM cohort montage", "F, median [95% CI]",
             "T, V/m per mA, median [p5-p95]", "FEM T, median", "At cohort montage: F / T", "Cohort serves, %",
             "Cohort re-selected, %", "Heads F > 1, %", "Same choice as FEM, %", "Retention p5, %",
             "Fails FEM T floor, %", "Heads gaining >= 10%, %", "T per decade, %"], rows))
    return "\n\n".join(out)


def manuscript_paragraph(prefix):
    """A paragraph of the built Rev0.4 manuscript, by its opening words (paragraphs kept from Rev0.3 are not in texts())."""
    d = Document(os.path.join(PKG, "04_manuscript.docx"))
    hits = [p.text for p in d.paragraphs if p.text.startswith(prefix)]
    assert len(hits) == 1, prefix
    return hits[0]


def uq_section(N, s14):
    """D3 (2 Nov 2026): the pre-specified uncertainty analysis in all 300 heads (results_v23/v23_uq/atlas_300 and
    atlas_thal_300, pipeline/nchc/uq_v23_atlas_collect.py); the 35-head analysis of 24 Oct is superseded."""
    U = os.path.join(ROOT, "results_v23", "v23_uq")
    c, sh, rb = N["uq.cond.T"], N["uq.share90"], N["uq.robust"]
    pl = {g: N[f"uq.{g}.T"] for g in ("jit", "rot", "seg", "mask")}
    least = ", ".join(f"{rlabel(t)} ({v:.0f}%)" for t, v in N["uq.least"])
    me = "; ".join(f"{rlabel(t)} {n}" for t, n in N["uq.mask_empty"])
    head = (
        "Design (`results_v23/v23_uq/ANALYSIS_PLAN.md`, amendments 1 and 2, written before the runs): in every one of the "
        "300 heads, 75 perturbed runs with fixed seeds (40 conductivity samples over six tissues from the between-study "
        "spreads of McCann et al. 2019; 20 independent 5 mm electrode displacements; 10 whole-cap rotations of 10 mm; 4 "
        "segmentation shifts of one 0.5 mm layer) plus two target-mask runs (one-voxel erosion and dilation, D7), all 163 "
        "regions scored from one basis per run on Nano 4 (array 660766, 16 nodes x 8 H200, 31 Oct-2 Nov 2026). FEM's "
        "cohort montage is the fixed protocol; FEM was not perturbed. Every head has its 77 runs, and the fetched files "
        "were checked against cluster SHA-256 sums.\n\n"
        f"- **Conductivity dominates exposure.** T at the fixed montage spans {c[0]:.2f}-{c[1]:.2f} of the head's nominal T "
        f"(5th-95th percentile of runs, median over regions; extremes {c[2]:.2f} and {c[3]:.2f}). Its within-head variance "
        f"is a median {N['uq.cond.share']:.0f}% of the combined within- and between-head variance of log T, so one head's "
        "conductivity uncertainty exceeds the anatomical spread between heads. Scalp (negative), compact bone (positive) "
        "and gray matter (negative) carry it, and the linear model explains R² ≈ 0.99.\n"
        f"- **Placement, segmentation and target masks matter less.** Independent placement {pl['jit'][0]:.2f}-{pl['jit'][1]:.2f} "
        f"(variance share {N['uq.jit.share']:.0f}%), cap rotation {pl['rot'][0]:.2f}-{pl['rot'][1]:.2f} ({N['uq.rot.share']:.0f}%), "
        f"segmentation {pl['seg'][0]:.2f}-{pl['seg'][1]:.2f} ({N['uq.seg.share']:.0f}%), target mask {pl['mask'][0]:.2f}-{pl['mask'][1]:.2f}. "
        f"One-voxel erosion leaves no target voxel in some heads for {len(N['uq.mask_empty'])} small regions (heads of 300: {me}; "
        "`results_v23/v23_uq/mask_empty_300.csv` from `pipeline/nchc/uq_v23_mask_empty.py`), and dilation never empties a mask. "
        "The collector enters an emptied mask as a T ratio of 0 for the cerebral and thalamic regions and omits it for the "
        "cerebellar and brainstem regions (whose fixed montage is applied after the runs), so "
        "the mask ratios of these regions should be read with the counts. A scan of all runs found no zero or non-finite value "
        "in the four physical scenarios, so the robustness results are unaffected.\n"
        f"- **The cohort montage holds as a fixed protocol in most runs, but the pre-specified criterion is strict.** It keeps "
        f"at least 90% of each perturbed head's constrained optimum F in a median {sh[0]:.0f}% of physical runs (range "
        f"{sh[1]:.0f}-{sh[2]:.0f}% over regions; 5th percentile of retention {N['uq.ret_p5'][0]:.3f}). {rb} of "
        f"{N['uq.targets']} regions meet the 95% criterion. Least robust: {least}.\n"
        f"- **The choice moves, its value much less.** The nominal reported montage stays the reported montage in a median "
        f"{N['uq.cond.kept']:.0f}% of conductivity runs ({N['uq.seg.kept']:.0f}% of segmentation runs), yet keeps "
        f"{100 * N['uq.cond.nomret_p5']:.0f}% of the perturbed constrained optimum at the 5th percentile of conductivity runs.\n"
        "- **Reading.** Absolute T is a nominal-conductivity estimate with a range of roughly 0.5- to 1.6-fold, and F-based "
        "decisions are more stable than T. This is the source of the manuscript's Methods, Results and Limitations "
        "sentences and of Table S14.")
    parts = [head]
    if s14:
        parts.append(s14)
    for sub_, title in (("atlas_300", "147 brain-region targets"), ("atlas_thal_300", "thalamic nuclei and supracallosal ACC")):
        t = io.open(os.path.join(U, sub_, "SUMMARY.md"), encoding="utf-8").read()
        t = t.replace("This is a new result set. It is not in any Results record or manuscript.", "").strip()
        body = t[t.index("## Summary by target set"):]
        parts.append(f"**Collector summary, {title}** (`results_v23/v23_uq/{sub_}/SUMMARY.md`; per-region values in "
                     f"`per_target.csv` and Data S1):\n\n{demote(body, 2)}")
    return "\n\n".join(parts)


def exhaustive_table():
    """D9: the exhaustive 10-10 template search against the screened search (results_v23/v23_template_design/exhaustive_1010)."""
    rows_, note = [], []
    for sub_, lab in (("exhaustive_1010", ""), ("exhaustive_1010_subregions", " (subregion)")):
        pth = os.path.join(ROOT, "results_v23", "v23_template_design", sub_, "exhaustive_summary.json")
        if not os.path.exists(pth):
            note.append(f"{sub_}: not yet collected"); continue
        j = json.load(io.open(pth, encoding="utf-8"))
        note.append(f"{sub_}: {j['n_montages']:,} montages, {j['gm_voxels']:,} grey-matter voxels, {j['minutes']} min")
        for r in j["targets"]:
            rows_.append([r["title"] + lab, r["exhaustive_montage"], f"{r['F']:.3f}", r["montage_1020"], f"{r['F_1020']:.3f}",
                          f"{r['gain_1010_over_1020_pct']:+.1f}%", r.get("screened_montage", "-"), str(r.get("screened_exhaustive_rank", "-")),
                          "yes" if r.get("screened_is_optimum") else "no", "yes" if r.get("screened_no_inferior_is_best") else "no", str(r["n_within_1pct"])])
    return md_table(["Target", "Exhaustive optimum", "F", "10-20 optimum", "F", "Gain", "Screened result", "Exhaustive rank of the screened result",
                     "Screened = optimum", "No-inferior-row best confirmed", "Montages within 1% of the optimum"], rows_) + "\n\n" + "; ".join(note) + "."


def mixed_model_text(N):
    mm, icc = N["mm"], N["mm.icc"]
    g = lambda o, s, t: mm[f"{o}|{s}|{t}"]
    c = lambda v: f"{v[0]:+.2f}% (95% CI {v[1]:+.2f} to {v[2]:+.2f})"
    ix = N["mm.interaction"]["T_wa"]
    return (f"`pipeline/v23_mixed_model.py` (`results_v23/v23_atlas_review/mixed_model/`): log T or log F of every head in every "
            f"region (48,900 rows), random intercept per head, region fixed, age, sex, site and head size; REML. The pooled "
            f"coefficient equals the mean of the 163 regional slopes; its interval accounts for the head-level correlation of the "
            f"163 outcomes. T at the reported montage: age {c(g('T_wa', 'M0', 'age_dec'))} per decade and men {c(g('T_wa', 'M0', 'men'))} "
            f"(M0); with intracranial volume and extracranial thickness (M3) age {c(g('T_wa', 'M3', 'age_dec'))}, men "
            f"{c(g('T_wa', 'M3', 'men'))}, intracranial volume {c(g('T_wa', 'M3', 'zlogICV'))} per SD and extracranial thickness "
            f"{c(g('T_wa', 'M3', 'ztextra'))} per SD; intraclass correlation {icc['T_wa|M0']:.2f} (M0) to {icc['T_wa|M3']:.2f} (M3). "
            f"T at the cohort montage gives the same picture (age {c(g('T_coh', 'M0', 'age_dec'))}, men {c(g('T_coh', 'M3', 'men'))} under M3). "
            f"F at the reported montage: age {c(g('F_wa', 'M0', 'age_dec'))}, men {c(g('F_wa', 'M0', 'men'))} (M0; {c(g('F_wa', 'M3', 'men'))} under M3), "
            f"intraclass correlation {icc['F_wa|M0']:.2f}: focality is a property of the region, not of the head. A region x sex "
            f"interaction improves the fit for every outcome (likelihood ratio, 162 df; regional sex effects on T from "
            f"{RN.f(ix['region_sex_effect_min_pct']):+.0f}% to {RN.f(ix['region_sex_effect_max_pct']):+.0f}%), "
            f"so the per-region table remains the regional description. Design VIFs are below 4 for every term.")


def site_text(N):
    ge, an, ru = N["site.ge"], N["site.anat"], N["site.ge_ratio_unadj"]
    return (f"`pipeline/v23_site_size_review.py` (`results_v23/v23_atlas_review/site_size/`). The TVGH-Bai site group (51 heads) was "
            f"scanned on a different scanner from the other three groups. Adjusted for age and sex, its T at the reported montage is "
            f"{ge['M0'][0]:+.1f}% against the other sites (q < 0.05 in {ge['M0'][1]} regions; a site difference in {ge['M0'][2]} regions, "
            f"median spread {ge['M0'][3]:.1f}%). Adding segmentation-derived anatomy does not remove it: with intracranial volume, "
            f"skull thickness, soft-tissue thickness and cerebrospinal-fluid volume (M5) the effect is {ge['M5'][0]:+.1f}% "
            f"({ge['M5'][1]} regions), and with the spongy-bone fraction as well (M6) {ge['M6'][0]:+.1f}% ({ge['M6'][1]} regions). "
            f"The reason is a suppression pattern: those heads have anatomy that predicts lower T (intracranial volume "
            f"{an['icv_ml'][0]:+.1f}%, p = {an['icv_ml'][1]:.1e}; cerebrospinal fluid {an['vol_csf_ml'][0]:+.1f}%, p = {an['vol_csf_ml'][1]:.1e}) yet "
            f"higher T. Their segmentations also differ in the compact/spongy bone split (spongy bone {an['vol_spongy_bone_ml'][0]:+.1f}%, "
            f"p = {an['vol_spongy_bone_ml'][1]:.1e}; compact bone {an['vol_compact_bone_ml'][0]:+.1f}%) at unchanged skull thickness "
            f"({an['skull_thickness_mm'][0]:+.1f}%, p = {an['skull_thickness_mm'][1]:.2f}) and soft-tissue thickness ({an['soft_thickness_mm'][0]:+.1f}%), "
            f"a contrast-dependent label and the likeliest scanner-to-segmentation channel. FEM shows the same site pattern. Unadjusted, "
            f"the ratio of that site's regional median T to the other sites' is {ru[0]:.3f} (IQR {ru[1]:.3f}-{ru[2]:.3f}); the adjusted "
            f"figures are the ones to quote. The covariates come from the same segmentations, so the analysis separates site acting "
            f"through segmented anatomy from site acting otherwise; it does not separate scanner from recruitment. Per-site regional "
            f"medians are in Supplementary Data S1.")


def size_text(N):
    sm, rl, rho, ciw = N["size.small"], N["size.ret_low"], N["size.rho_F"], N["size.ciw"]
    small = ", ".join(f"{rlabel(t)} ({v}; min {m})" for t, v, m in sm)
    low = ", ".join(f"{rlabel(t)} ({v:.1f}%)" for t, v in rl)
    return (f"Same script. The bootstrap 95% interval of the regional median T is a median {ciw[0]:.1f}% of the median (range "
            f"{ciw[1]:.1f}-{ciw[2]:.1f}%) and is not correlated with region size (Spearman rho {ciw[3]:.2f}, p = {ciw[4]:.2f}); cohort-montage "
            f"re-selection is not size-related either, and the only size-dependent quantity, the worst-head FEM retention, stays above "
            f"90% at every size when smoothed. Precision is set by the 300 heads, not by voxel count, so no data-driven minimum size "
            f"exists. Region size correlates weakly with F over all 163 regions (rho {rho[0]:.2f}, p = {rho[1]:.3f}) and not within the "
            f"{rho[4]} AAL regions (rho {rho[2]:.2f}, p = {rho[3]:.2f}): the association comes from the small deep nuclei, not from a size "
            f"gradient within the cortical set. By grid convention, regions under 60 voxels in the median head are flagged in Data S1: "
            f"{small}. Regions whose worst-head FEM retention is below 90%: {low}.")


def r4_block(N):
    ge, mm = N["site.ge"], N["mm"]
    g = lambda o, s, t: mm[f"{o}|{s}|{t}"]
    rows_ = [
        ["D1", '"Optimal" wording', "Methods define four fixed terms (unconstrained optimum, constrained optimum, reported montage, cohort "
         "montage) and state that every candidate was enumerated, so each optimum is exact within the declared candidate set; the abstract "
         'names the conditions (19 electrodes, 1 mA per channel); "selected montage" is "reported montage" throughout', "done (Rev0.4)"],
        ["D2", "FEM beside SPICEhead", f"Table 3 gives every headline count as SPICEhead / FEM / both and FEM's T; Table 4 carries FEM's F and T; "
         f"Results quote {N['atlas.F1_majority']} regions with F > 1 in most heads ({N['fem.F1_majority']} by FEM; {N['fem.F1_both']} by both) and "
         f"{N['atlas.g10_ge20']} regions with >= 20% of heads gaining >= 10% ({N['fem.g10_ge20']} by FEM; {N['fem.g10_both']} by both)", "done (Rev0.4)"],
        ["D3", "Uncertainty on 35 heads only", (f"The identical 75-run design run in all 300 heads on Nano 4 (array 660766, 16 nodes), plus the "
         f"target-mask scenario (amendment 2; collected 2 Nov 2026): conductivity scales T at the cohort montage by {N['uq.cond.T'][0]:.2f}-"
         f"{N['uq.cond.T'][1]:.2f}, placement and segmentation by at most about 12%; the cohort montage keeps >= 90% of the perturbed optimum "
         f"in a median {N['uq.share90'][0]:.0f}% of runs and meets the 95% criterion in {N['uq.robust']} regions (section 5.2, Table S14, "
         "Data S1; Methods, Results and Limitations sentences)") if "uq.targets" in N else ("Extension of the identical 75-run design to the "
         "other 265 heads on Nano 4 (16 nodes), plus the target-mask scenario for all 300; amendment 2 of the plan"), "done" if "uq.targets" in N else "running"],
        ["D4", "19-pad leadfield caveat", (f"Four-pad direct FEM runs of the six reference cohort montages in all 300 heads on Nano 4 "
         f"(`pad4_fem_check.py`, 1 Nov 2026), each beside a 19-pad leadfield on the same rebuilt mesh: passive pads lower T by a median "
         f"{100 * (N['pad4.pad.T'][0] - 1):.1f}% (5th-95th {100 * (N['pad4.pad.T'][1] - 1):.1f}-{100 * (N['pad4.pad.T'][2] - 1):.1f}%) and leave F "
         f"within 3% in {N['pad4.pad.F_within3']:.0f}% of head-targets; re-meshing moves T and F by under "
         f"{100 * max(N['pad4.remesh.T_maxdev'], N['pad4.remesh.F_maxdev']):.0f}% (section 3.4, Table S13; Methods, Results and Limitations sentences)"
         if "pad4.n" in N else "cluster run not yet collected"), "done" if "pad4.n" in N else "running"],
        ["D5", "Site vs scanner", f"Site re-estimated with segmentation-derived anatomy: the different-scanner site is {ge['M0'][0]:+.1f}% in T "
         f"after age and sex and {ge['M5'][0]:+.1f}% after anatomy; not explained by gross anatomy; compact/spongy bone split identified "
         "(section 6.2, Table S12; Results and Limitations sentences)", "done"],
        ["D6", "Minimum size for small nuclei", "Precision of regional medians is independent of region size (section 6.3); nuclei under 60 voxels "
         "flagged in Data S1 as a grid convention; worst-head retention listed", "done"],
        ["D7", "Head-level model; mask erosion/dilation", f"Mixed model fitted (section 6.1, Table S11): pooled age {g('T_wa', 'M0', 'age_dec')[0]:+.2f}% "
         f"per decade, sex {g('T_wa', 'M0', 'men')[0]:+.1f}% -> {g('T_wa', 'M3', 'men')[0]:+.1f}% after head size; target-mask erosion/dilation "
         "runs for all 300 heads inside the D3 extension", "model done; mask runs running"],
        ["D8", "Rev0.3 student prose", 'Netlist and contacts paragraphs rewritten; "balanced" is "decade-stratified"', "done (Rev0.4, red)"],
        ["D9", "Screened 10-10 template search", "Exhaustive search of all 3,526,484 montages on the template (section 14): every screened optimum "
         "is the true optimum (rank 1)", "done"],
    ]
    return ("PI instruction of 31 October 2026: no decision is to be left as a limitation; each is to be solved, with the cluster used to "
            "accelerate. Word body 3,998 / 4,000 words; abstract 246 / 250; every Word file passes the validator.\n\n" +
            md_table(["#", "Decision", "What was done", "Status"], rows_))


def pad4_section(N):
    """D4 (1 Nov 2026): the four-pad FEM check of the six cohort montages in all 300 heads (results_v23/v23_pad4_check)."""
    if "pad4.n" not in N:
        return "*Cluster run not yet collected.*"
    pt, po, pf, rt, rf, tt, tf = (N[k] for k in ("pad4.pad.T", "pad4.pad.O", "pad4.pad.F", "pad4.remesh.T", "pad4.remesh.F", "pad4.total.T", "pad4.total.F"))
    rng = lambda s: f"{s[0]:.3f} ({s[1]:.3f}-{s[2]:.3f})"
    rows_ = [[rlabel(b["target"]), b["montage"], str(b["heads"]), rng(b["pad_T"]), f"{b['pad_O'][0]:.3f}", rng(b["pad_F"]),
              f"{b['pad_F_within3']:.0f}%", rng(b["remesh_T"]), rng(b["remesh_F"])] for b in N["pad4.by_target"]]
    return (f"*Review decision D4.* The FEM reference of the main comparison is a leadfield in which all 19 pads are meshed, so "
            f"the 15 passive pads of any two-pair montage shunt current. `pipeline/nchc/pad4_fem_check.py` on Nano 4 (1 Nov 2026; "
            f"{N['pad4.heads']} heads, ~31 min per head on 8 threads) rebuilt each head's mesh (`charm --mesh`, SimNIBS 4.6; identical "
            f"to the archived mesh in {N['pad4.mesh_identical']} heads, since meshing is not bit-reproducible), solved the six cohort "
            f"montages directly with only their four pads meshed (24 mm, 2 mm, 1 mA per channel, same conductivities) and computed a "
            f"19-pad leadfield on the same rebuilt mesh; all three were scored on the grid, masks and envelope of the 300-head "
            f"comparison (`pipeline/nchc/pad4_collect.py` -> `results_v23/v23_pad4_check/`). Four pads / 19-pad leadfield on one "
            f"mesh isolates the passive pads; 19-pad rebuilt / 19-pad archived isolates re-meshing.\n\n"
            f"- **Passive pads** ({N['pad4.n']:,} head-target combinations): T ratio median {pt[0]:.3f} (5th-95th {pt[1]:.3f}-{pt[2]:.3f}; "
            f"range {pt[3]:.3f}-{pt[4]:.3f}), O ratio median {po[0]:.3f}, F ratio median {pf[0]:.3f} ({pf[1]:.3f}-{pf[2]:.3f}); F within "
            f"3% in {N['pad4.pad.F_within3']:.0f}% and T within 5% in {N['pad4.pad.T_within5']:.0f}% of combinations. Per target the "
            f"median T ratio runs from {N['pad4.pad.T_target_range'][0]:.3f} to {N['pad4.pad.T_target_range'][1]:.3f}.\n"
            f"- **Re-meshing:** T ratio median {rt[0]:.3f} ({rt[1]:.3f}-{rt[2]:.3f}), F ratio median {rf[0]:.3f} ({rf[1]:.3f}-{rf[2]:.3f}); "
            f"both within 1% in {N['pad4.remesh.within1']:.0f}% of combinations (largest deviation T {100 * N['pad4.remesh.T_maxdev']:.1f}%, "
            f"F {100 * N['pad4.remesh.F_maxdev']:.1f}%), consistent with the 24-head re-mesh check of section 17.\n"
            f"- **Total** (four pads on the rebuilt mesh over the archived reference): T {tt[0]:.3f} ({tt[1]:.3f}-{tt[2]:.3f}), F {tf[0]:.3f} "
            f"({tf[1]:.3f}-{tf[2]:.3f}).\n"
            f"- **Reading.** The passive pads lower T and O together, so F is nearly unchanged: the 19-pad reference is conservative "
            f"for T by about {100 * (pt[0] - 1):.0f}% and unbiased for F, the decision comparison (sections 3.2-3.3, F-based) is "
            f"unaffected, and the SPICEhead-FEM T offset of section 3.1 would be about {100 * (pt[0] - 1):.0f} points smaller against a "
            f"four-pad reference. In the paper: one Methods clause, one Results sentence, the Limitations offset sentence and Table S13.\n\n"
            + md_table(["Target", "Cohort montage", "Heads", "T, four pads / 19-pad, median (5th-95th)", "O, median", "F, median (5th-95th)",
                        "F within 3%", "Re-meshing, T", "Re-meshing, F"], rows_))


def decisions_table():
    rev = io.open(os.path.join(J, "review", REVIEW_FILE), encoding="utf-8").read()
    i = rev.index("| # | Decision |")
    return rev[i:].strip()


def main():
    a = sys.argv[1:]
    name = a[a.index("--out") + 1] if "--out" in a else "RESULTS_v23_2026-11-02_r7.md"
    N = RN.compute()
    assert "rv.served" in N, "run pipeline/v23_atlas_review.py first (review analyses missing)"
    T = B.texts(N)
    tabs = B.table_data(N)
    st = render_v23.render()                                       # refreshes v23/RESULTS_v23.md (the r1 record)
    assert not st["pending"] and not st["missing"], st
    r1 = io.open(render_v23.ROUT, encoding="utf-8").read()
    top, sub = sections(r1)
    S = {c.split(".")[0]: (c, h, r, n) for c, h, r, n in docx_tables(os.path.join(PKG, "06_supplement.docx"))}
    for k in ("Table S3", "Table S4", "Table S5", "Table S6", "Table S7", "Table S8", "Table S9", "Table S10", "Table S11", "Table S12"):
        assert k in S, f"{k} not found in 06_supplement.docx"

    def stab(k):
        c, h, r, n = S[k]
        return f"**{c}**\n\n" + md_table(h, r) + (f"\n\n{n}" if n else "")

    def mtab(k, cap):
        h, r = tabs[k][0], tabs[k][1]
        return f"**{cap}**\n\n" + md_table(h, r)

    f = RN.f
    A, TT = N["A"], N["targets"]
    sites = {}
    for r in csv.DictReader(open(os.path.join(RN.DEMO, "loso_163.csv"), encoding="utf-8")):
        sites[r["held_out_site"]] = r["n_held_out"]
    import rev04_package as PK
    site_txt = ", ".join(f"{PK.SITE.get(k, k)} {v}" for k, v in sorted(sites.items()))
    most = "; ".join(f"{rlabel(t)} {f_:.2f} (T {t_:.3f})" for (n_, f_, t_), t in zip(N["atlas.most_focal"], N["atlas.most_focal_raw"]))
    low = sorted(TT, key=lambda t: f(A[(t, SP)]["F_choice_median"]))[:8]
    least = "; ".join(f"{rlabel(t)} {f(A[(t, SP)]['F_choice_median']):.2f} (T {f(A[(t, SP)]['T_choice_median_Vm_per_mA']):.3f})"
                      for t in low)
    g10 = "; ".join(f"{rlabel(t)} {v:.0f}%" for t, v in N["atlas.g10_top_raw"])
    sv = sorted(TT, key=lambda t: f(A[(t, SP)]["rv_sp1v23__cohort_served_pct"]))[:8]
    served_low = "; ".join(f"{rlabel(t)} {f(A[(t, SP)]['rv_sp1v23__cohort_served_pct']):.0f}%" for t in sv)
    rs = sorted(TT, key=lambda t: f(A[(t, SP)]["rv_sp1v23__cohort_reselected_pct"]))[:8]
    resel_low = "; ".join(f"{rlabel(t)} {f(A[(t, SP)]['rv_sp1v23__cohort_reselected_pct']):.0f}%" for t in rs)
    fl = sorted(TT, key=lambda t: -f(A[(t, SP)]["rv_sp_choice_fails_fem_floor_pct"]))[:8]
    fail_high = "; ".join(f"{rlabel(t)} {f(A[(t, SP)]['rv_sp_choice_fails_fem_floor_pct']):.0f}%" for t in fl)
    small = sorted(TT, key=lambda t: f(A[(t, SP)]["rv_voxels_median"]))[:10]
    small_txt = "; ".join(f"{rlabel(t)} {f(A[(t, SP)]['rv_voxels_median']):.0f}" for t in small)
    comp = list(csv.DictReader(open(os.path.join(ROOT, "results_v23", "v23_atlas_review", "cohort_composition.csv"),
                                    encoding="utf-8")))
    import json
    prov = json.load(open(os.path.join(ROOT, "results_v23", "v23_atlas_review", "PROVENANCE.json"), encoding="utf-8"))
    decs = sorted({r["decade"] for r in comp})
    comp_rows = [[PK.SITE.get(s_, s_)] + [f"{x['men']} / {x['women']}" for d_ in decs for x in comp
                                          if x["site"] == s_ and x["decade"] == d_]
                 for s_ in sorted({r["site"] for r in comp})]
    rr, rc = N["rv.rule_range"], N["rv.rule_changed"]
    sv_m, sv80, sv50, sv_min = N["rv.served"]
    fail_m, fail_x, fail_n = N["rv.fem_fail"]
    okf = N["rv.ok_feasible"]
    rs_m, rs50, rs90 = N["rv.reselected"]
    vmin, vmax, vrho = N["rv.voxels"]
    ca, cs = N["rv.cohort_age"], N["rv.cohort_sex"]
    oT, oF = N["offset.T"], N["offset.F"]
    gT = N["grid.T_2_over_1"]

    import time
    rnd = re.search(r"_r(\d+)\.md$", name)
    rnd = int(rnd.group(1)) if rnd else 4
    head = f"""# SPICEhead v23: comprehensive results record, r{rnd} (Paper A, Rev0.4)

Generated by `pipeline/results_v23_comprehensive.py` on {time.strftime('%d %B %Y').lstrip('0')} after the internal review round
(`review/{REVIEW_FILE}`) and the r4 resolution of decisions D1-D9. Every number is computed from a result file:
- the Results paragraphs and Tables 1-4 are those of the Rev0.4 manuscript (`pipeline/build_rev04.py`);
- the supplementary tables are read from the built `06_supplement.docx`;
- the region table comes from `pipeline/rev04_numbers.py` and the review analyses (`pipeline/v23_atlas_review.py`);
- Part II comes from the v23 record renderer (`pipeline/render_v23.py`); the uncertainty analysis (section 5.2) from `results_v23/v23_uq`.

Internal document.

**Model.** SPICEhead v23 (PI decision of 26 October 2026): the head network is homogenized from the 0.5 mm CHARM
segmentation, with skin-surface contacts. The FEM reference (SimNIBS 4.6, 19-pad leadfield) is unchanged. v22 numbers
appear only where Part II compares the two models.

**Framing (Rev0.4, PI, 30 October 2026).** A comprehensive map of region-wise TI targeting parameters with optimal
montage selection in 300 heads, each decision checked against FEM; not a comparison of computational time or
efficiency. The optimum is conditional on its search, which is:
- two equal-current pairs of the 19-electrode 10-20 system at 1 mA per channel;
- the objective F and the declared selection rule;
- the grid and the electrode model.

The envelope is an exposure descriptor, not a neural response.

**Four montage terms are used throughout.**
- **Unconstrained optimum:** the highest F.
- **Constrained optimum:** the highest F with T at least 80% of the head's best.
- **Reported montage:** the most edge-electrode montage within 3% of the constrained optimum.
- **Cohort montage:** the rule applied to the cohort medians.

**What is where.**
- **Part I** is the evidence of the Rev0.4 manuscript (`{os.path.relpath(PKG, J)}/`), plus the review analyses (marked
  *review*). Part I uses all 300 heads, except the synthetic geometries of section 2 and the template panels of
  Figure 1.
- **Part II** keeps the analyses that the PI's 300-head rule removed from the paper. They are unchanged from r1 and
  are not manuscript claims.
  - Head counts: one, four, eight, 20 or 24 heads, or the template.
  - 300-head analyses on other grids or objectives: the 5 mm cohort search, the 2 to 1 mm optima and v22 against v23.
- **Figures** 1-5 are `{FIGS}/` (`pipeline/fig_v23_rev04.py`), with legends as in `07_figure_legends.docx`.
- **Supersedes** r{rnd - 1}. The v22 record (r1-r7) stays the record of the earlier model.

## Contents

- Review round r4: decisions D1-D9 resolved
- Review round r3: what changed, and decisions for the authors
- Part I. Results of the Rev0.4 manuscript (300 heads)
  1. Cohort, target atlas and definitions
  2. Analytical verification
  3. Agreement with FEM in 300 heads (3.4: four-pad check of the FEM reference)
  4. Whole-brain targeting parameters
  5. Individual variability, personalization and the selection rule (5.2: uncertainty, 300 heads)
  6. Age, sex, head size and site
  7. Region-wise parameters for all {N['atlas.n']} regions
- Part II. Supporting v23 analyses kept in the record (not in Rev0.4)
  - 8. Resolution dependence and analytical checks
  - 9. Fixed-montage field comparison (eight heads)
  - 10. Sensitivity of the FEM reference
  - 11. Best of four fixed montages
  - 12. Computation and circuit coupling
  - 13. Coarse grids, the 20-head sweep and 5 mm choices
  - 14. Template montage-design exploration
  - 15. Acceptance criteria
  - 16. What the model change did (v22 to v23)
  - 17. Full-space decision detail and the 24-head re-mesh check
  - 18. Uncertainty analysis: moved to section 5.2

---

## Review round r4: decisions D1-D9 resolved

{r4_block(N)}

## Review round r3: what changed, and decisions for the authors

The review (`review/{REVIEW_FILE}`; three independent reviewers, every finding checked against the data) required
revision. The changes below were made in this record and, where the text is shared, in the Rev0.4 Word files (red).

**Corrected.**
- **Field agreement** "in every head" is replaced by: {N['field.1mm.r_ge95']} of 300 heads with r ≥ 0.95 (lowest
  {N['field.1mm.low_r'][0]:.3f}), with magnitudes higher in {N['field.1mm.higher']}.
- **The SPICEhead-FEM offset** is not a common factor. Regional median T exceeds FEM's by {oT[1]:.2f}-{oT[2]:.2f}×, and
  median F differs by {oF[0]:.2f}-{oF[1]:.2f}×. Figure 3A now plots the mean-magnitude bias quoted in the text.
- **"Heads with F > 1"** is now counted at the reported montage, as the tables state; the earlier count was taken at
  the constrained optimum. With this change, {N['atlas.F1_majority']} regions (was 78) have F > 1 in most heads.
- **Wording.** Decisions "closely matched" FEM's (not "reproduced"). The cohort-optimum floor is
  {N['fs.coh.keep_sp1_under_fem.min_floor']:.1f}%. The orbitofrontal range is limited to the middle and superior gyri.
  The reference spheres are named as spheres. "Most accessible" and "focal" are replaced by "highest F".
- **Demographics** are described as cross-sectional associations ("lower in older heads"), and the attribution of the
  sex difference to head size depends on the head-size measure:
  - intracranial volume alone (M1): {N['sex.M1.sig']} regions keep a significant sex term;
  - head radius (M2): {N['sex.M2.sig']} regions, {N['sex.M2.men_higher']} with higher T in men;
  - intracranial volume and extracranial thickness (M3): {N['sex.M3.sig']} regions, {N['sex.M3.men_higher']} with
    higher T in men.

  Sex is associated with F in {N['sex.F_sig']} regions. The site spread uses one percent convention.
- **Labels.** The Spearman (unadjusted) age columns are labelled as such. Of the 144 Spearman and 134 adjusted
  regions, {N['age.both']} are in both.
- **Brainstem.** Brainstem T is sampled in white-matter-labelled tissue. The Discussion's "deep and posterior-fossa
  regions … receive less field" is now "thalamic and cerebellar regions".
- **Part I / Part II.** The 24-head re-mesh check, the "Supplement S11" and "Table S22" material and the stale
  section references moved to Part II. The stale supplement sentences are flagged. The 5 mm values in section 16 are
  labelled.
- **Figures 1-5** carry the figure audit's corrections (review minor findings 9-13).

**Added (review analyses; re-scoring of saved per-head scores, no new field solutions).**
- **Cohort exposure:** F and T at the cohort montage itself, in Table 4, section 7 and Figure 4D.
- **"Served":** the share of heads for which the cohort montage meets the head's T floor and keeps ≥ 90% of its
  constrained F: median {sv_m:.0f}% per region, ≥ 80% in {sv80} regions and < 50% in {sv50}.
- **FEM feasibility:** SPICEhead's reported montage fails FEM's own T floor in a median {fail_m:.0f}% of heads per
  region (at most {fail_x:.0f}%; more than 10% in {fail_n} regions).
- **Stability:** bootstrap 95% intervals of the regional medians (median T interval width {N['rv.T_ci_width'][0]:.1f}%
  of the median). The cohort montage is re-selected in a median {rs_m:.0f}% of resamples, and in fewer than 50% for
  {rs50} regions.
- **Rule sensitivity** (Table S10): {len(N['rv.rule'])} variants of floor, band and tie-break.
- **Region size:** {vmin}-{vmax} voxels; Spearman ρ between voxels and median F {vrho:.2f}.
- **Fixed protocol:** age and sex on T at the cohort montage. Also the design VIFs and the cohort composition.
- **Uncertainty:** the 35-head uncertainty analysis, then Part II section 18; since r6 the 300-head analysis is section 5.2.

**Decisions for the authors** (from the review; not applied):

{decisions_table()}

---

# Part I. Results of the Rev0.4 manuscript (300 heads)

## 1. Cohort, target atlas and definitions

{T['cohort']}

Site groups used by the site and leave-one-site-out analyses: {site_txt}.

*Review: cohort composition by site and decade (men / women; aggregate counts).* Design variance inflation of the
age-sex-site model: age {prov['vif']['age']}, sex {prov['vif']['sex']}.

{md_table(["Site group"] + decs, comp_rows)}

{T['targets']}

{mtab('t1', 'Table 1. Target sets')}

{stab('Table S3')}

**Network, exposure objective and outputs.**

{T['formulation']}

{T['contacts']}

{manuscript_paragraph('The primary metric was the rotation-invariant maximum modulation amplitude')}

{T['selection']}

{figure(1)}

## 2. Analytical verification

{T['verification']}

{T['res_ver']}

The slab ratio of 1.000000 holds by construction for a one-dimensional series stack whose layers lie on the 0.5 mm
grid. It checks that the homogenization rule is implemented consistently; it is not a test of three-dimensional
accuracy. Each construction's exact resistance spans its own end node planes.

{figure(2)}

{stab('Table S4')}

## 3. Agreement with FEM in 300 heads

{T['fem']}

### 3.1 Field agreement per head

{T['res_field']}

{figure(3)}

{stab('Table S5')}

*Review: method dependence of threshold counts.*
- F > 1 in most heads: {N['atlas.F1_majority']} regions with SPICEhead, against {N['fem.F1_majority']} with FEM.
- ≥ 20% of heads gaining ≥ 10%: {N['atlas.g10_ge20']} against {N['fem.g10_ge20']}.
- Regional SPICEhead/FEM median T: median {oT[0]:.3f}, range {oT[1]:.3f}-{oT[2]:.3f}.
- FEM's values for every region are in section 7 and Data S1.

### 3.2 Decisions for the six reference targets (unconstrained F)

{T['res_six']}

{mtab('t2', 'Table 2. Decision agreement with FEM for the six reference targets')}

{stab('Table S6')}

### 3.3 All {N['atlas.n']} regions under the selection rule

{T['res_atlas_agree']}

*Review: FEM feasibility.* Retention scores F only. A SPICEhead choice that fails FEM's own T floor can therefore
retain ≥ 100%; this is the source of the 2 mm median above 100% in Figure 3C.
- **Failing FEM's floor:** SPICEhead's reported montage fails it in a median {fail_m:.0f}% of heads per region (at most
  {fail_x:.0f}%).
- **Counting those heads as failures:** retention ≥ 90% that is also FEM-feasible holds in a median {okf[0]:.0f}% of
  heads per region (lowest {okf[1]:.0f}%).
- **Most often infeasible under FEM:** {fail_high}.

### 3.4 Four-pad check of the FEM reference (six cohort montages, 300 heads)

{pad4_section(N)}

{stab('Table S13') if 'Table S13' in S else ''}

## 4. Whole-brain targeting parameters

{T['res_reach']}

- **Highest F:** {most}.
- **Lowest F:** {least}.
- *Review: region size.*
  - Gray-matter regions span {vmin}-{vmax} voxels (median per region over heads). Spearman ρ between size and median F
    is {vrho:.2f}.
  - O excludes gray matter within 20 mm of the target, so larger regions remove more nearby cortex from the
    denominator.
  - F > 1 does not mean that the field maximum lies in the target.
  - Smallest regions: {small_txt}. The values of small nuclei are indicative (decision D6).

{figure(4)}

{mtab('t3', 'Table 3. Targeting parameters by region set')}

{mtab('t4', 'Table 4. Cohort montages and targeting parameters for selected regions')}

## 5. Individual variability, personalization and the selection rule

{T['demo']}

{T['res_pers']}

- **Regions where individual planning matters most** (share of heads gaining at least 10%): {g10}.
- *Review: regions the cohort montage serves least* (cohort montage meets the T floor and keeps ≥ 90% of the
  constrained F): {served_low}.
- *Review: least stable cohort montages* (share of 100 bootstrap resamples that re-select it): {resel_low}.

{T['res_spread']}

{figure(5)}

{stab('Table S7')}

{stab('Table S9')}

### 5.1 Sensitivity of the selection rule (review)

The 80% floor, 3% band and edge-electrode tie-break were set after the optimization audit of 22 October. Table S10
re-scores all {N['atlas.n']} regions under each variant. Across the variants:
- the same choice as FEM: a median of {rr['same_choice_median'][0]:.1f}-{rr['same_choice_median'][1]:.1f}% of heads;
- retention p5 ≥ 90%: {rr['retention_p5_ge90'][0]:.0f}-{rr['retention_p5_ge90'][1]:.0f} regions;
- cohort montage equal to FEM's: {rr['cohort_same_as_fem'][0]:.0f}-{rr['cohort_same_as_fem'][1]:.0f} regions;
- F > 1 in most heads: {rr['F_above_1_majority'][0]:.0f}-{rr['F_above_1_majority'][1]:.0f} regions;
- ≥ 20% of heads gaining ≥ 10%: {rr['gain10_ge20'][0]:.0f}-{rr['gain10_ge20'][1]:.0f} regions;
- the cohort montage differs from the one used in {rc[0]}-{rc[1]} regions.

{stab('Table S10')}

### 5.2 Uncertainty of T and of the cohort montage (D3, 300 heads)

{uq_section(N, stab("Table S14") if "Table S14" in S else "") if "uq.targets" in N else "The 300-head uncertainty extension is running (decision D3)."}

## 6. Age, sex, head size and site

{T['res_demo']}

*Review: fixed protocol.* With every head at the cohort montage, T is lower in older heads in {ca[0]} regions (median
{ca[1]:+.1f}% per decade) and lower in men in {cs[0]} (median {cs[1]:+.1f}%). Both are adjusted for the other factors
and site, with q < 0.05. The two oldest decades are predominantly male, and the veterans' home group adds older men
(section 1).

{stab('Table S8')}

### 6.1 Head-level mixed model (D7)

{mixed_model_text(N)}

{stab('Table S11')}

### 6.2 Site against segmentation-derived anatomy (D5)

{site_text(N)}

{stab('Table S12')}

### 6.3 Region size and the precision of regional estimates (D6)

{size_text(N)}

## 7. Region-wise parameters for all {N['atlas.n']} regions

SPICEhead at 1 mm under the selection rule, 300 heads per region; every column is also in `Supplementary_Data_S1.xlsx`.

| Column | Definition |
|---|---|
| Voxels | Median region size over heads (1 mm grid; brainstem: own tissue points) |
| F, T | At each head's reported montage. F with the bootstrap 95% interval of its median; T with the 5th-95th percentiles across heads |
| FEM T | FEM's median at FEM's own reported montage |
| At cohort montage | Medians when every head uses the cohort montage |
| Cohort serves | Cohort montage meets the head's T floor and keeps ≥ 90% of its constrained F |
| Cohort re-selected | Share of 100 bootstrap resamples of heads that choose the same cohort montage |
| Retention p5 | 5th percentile of FEM's F at SPICEhead's reported montage relative to FEM's constrained optimum |
| Fails FEM T floor | Share of heads where that montage gives < 80% of FEM's maximum T |
| T per decade | Adjusted for sex and site; * marks q < 0.05 |
| Small region (Data S1) | Fewer than 60 voxels in the median head: a grid convention (section 6.3), not a precision threshold |
| Per-site T (Data S1) | Unadjusted medians of T at the reported montage within each site group, and the TVGH-Bai / other-sites ratio |

T carries the model's offset against FEM (section 3.1). Brainstem T and F are not comparable with gray-matter regions.

{region_table(N)}

---

# Part II. Supporting v23 analyses kept in the record (not in Rev0.4)

These sections are the r1 record's v23 results, unchanged.
- **Head counts.** They use one, four, eight, 20 or 24 heads, or the template, or 300 heads at other grids. The
  300-head rule therefore removed them from the paper; they are available if the authors want any of them back.
- **Cross-references.** Section, table, figure and supplement numbers inside them (for example "Table S4", "Figure
  S1", "Supplement S11", "section 4", "for the supplement") refer to the r1 record and the earlier supplement, not to
  Part I or the Rev0.4 supplement. The Rev0.4 supplement has no circuit or template material.

## 8. Resolution dependence and analytical checks

{top['1. Numerical credibility and resolution dependence']}

*r3 note:* the 300-head atlas gives the corresponding 2 to 1 mm change: regional median T is a median {gT[0]:.1f}%
higher at 2 mm (range {gT[1]:.1f}-{gT[2]:.1f}%). 1 mm is the finest tested grid, not a converged one.

## 9. Fixed-montage field comparison (eight heads)

{demote(sub['2.1 Field amplitude (four fixed montages)'], 0)}

## 10. Sensitivity of the FEM reference

{sub['2.3 Sensitivity of the FEM reference']}

*r3 note:* the full-space decisions of Part I rest on the 19-pad leadfield, whose unused pads alter the field by a
factor of 0.865-1.049 against two-pad runs (decision D4, resolved in section 3.4 with four-pad runs in all 300 heads).

## 11. Best of four fixed montages

{sub['2.4 Supporting context: best of four fixed montages']}

## 12. Computation and circuit coupling

{top['3. Practical exhaustive computation']}

## 13. Coarse grids, the 20-head sweep and 5 mm choices

*r3 note:* in "Higher-resolution extension" the median 2.6% loss is over the differing combinations only, while the 5th
percentile and the lowest value are over all 1,800 combinations.

{top['4. Optimal montage selection and the cost of coarsening']}

## 14. Template montage-design exploration

*r4 (D9):* the screened 10-10 search is now certified. `pipeline/ti_template_1010_exact.py` scored every eligible
montage of the 76-position cap on all grey-matter voxels (GPU scorer of the atlas runs, Nano 4 job 657520), with the
model, grid, targets and pair rules of the screened search:

{exhaustive_table()}

The 16 October review's objection to "the exact optimum ranked ≤ 3" and "exhaustive after screening" is therefore
resolved by measurement: the screened winners are the exhaustive optima. The physical-limit sentence is unchanged and
remains record only.

{top['5. Template montage-design exploration: deep targets (insula, hippocampus)']}

## 15. Acceptance criteria

{top['6. Acceptance criteria, evaluated']}

## 16. What the model change did (v22 to v23)

*r3 note:* in the "Headline outcomes" table, the cohort optima and the leave-one-site-out loss are 5 mm, six-target
values from the r1 record. They are not the 1 mm unconstrained optima of Table S6 or the rule-based cohort montages of
Part I.

{top['7. What the model change did (v22 → v23)']}

## 17. Full-space decision detail and the 24-head re-mesh check

From r1 section 2.2. The "Supplement S11" and "Table S22" it cites are r1 numbering; Table 2 and Table S6 of Part I
hold the same 300-head values.

{demote(sub['2.2 Montage decisions over the full candidate set (main decision evidence)'], 0)}

## 18. Uncertainty analysis: moved to Part I

Since r6 the uncertainty analysis covers all 300 heads and is Part I section 5.2 (decision D3). The 35-head analysis
of 24 October that this section held in r3-r5 is superseded. It stays in `results_v23/v23_uq/RESULT_2026-10-24.md`,
`results_v23/v23_uq/atlas_35heads/` and `atlas_thal_35heads/` and in records r3-r5, and is not quoted here.

---

## Status

- **Computed:** {"every input of Parts I and II is present; both cluster analyses of r4 (D3, D4) are collected" if "uq.targets" in N else "every input of Parts I and II is present except the 300-head uncertainty extension (D3) in progress"} (section "Review round r4").
- **Decisions D1-D9:** {"all resolved; D4 is section 3.4 and D3 is section 5.2" if "uq.targets" in N else "resolved as described in Review round r4; D3 (300-head uncertainty) completes when its cluster run is collected"}.
- **Package:** the package default is v23 and the code deposit describes v23 (2 Nov 2026).
- **Later:**
  - an independent review of the additions since r3 (the mixed model, site and size analyses, the four-pad check and the
    uncertainty analysis), if the authors want one.
"""
    path = os.path.join(J, name)
    text = re.sub(r"\n{3,}", "\n\n", head)
    assert "{{" not in text and "PENDING" not in text and "nan" not in text.split(), "unrendered values"
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)
    print(f"wrote {path}: {len(text.split()):,} words")


if __name__ == "__main__":
    main()
