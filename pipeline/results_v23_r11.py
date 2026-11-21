# -*- coding: utf-8 -*-
"""One comprehensive v23 results record for Paper A, r11 (4 Nov 2026; PI: "Fix three gaps").

r7 is the only full record and has no joint-success results; r10 has them but is a short reporting record. This script
combines them without new analysis:
  Part A  r10, verbatim: the current Results.
  Part B  the run-level uncertainty results in full, computed from results_v23/v23_uq_recollect_2026-11-04.
  Part C  every r7 statement known to have changed, quoted beside its current wording.
  Part D  r7 Parts I and II, verbatim except demoted headings, repointed figure links and marked r11 notes.

Part C holds no typed results. Each row quotes r7 and a current source exactly, and the build fails if a quote is not
found. A second check flags every r7 passage whose Rev0.4 text the Rev0.5 change log records as revised.

  python pipeline/results_v23_r11.py

Output: Paper_A_SPICEhead/Brain Stimulation/02_Results_Records/v23/Dated_Records/RESULTS_v23_2026-11-04_r11.md
Refuses to overwrite it without --rebuild. Run build_code_deposit.py afterwards, so the deposit carries this script.
"""
import json
import os
import re
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJ = os.path.dirname(ROOT)
J = os.path.join(PROJ, "Paper_A_SPICEhead", "Brain Stimulation")
REC = os.path.join(J, "02_Results_Records", "v23", "Dated_Records")
SUB = os.path.join(J, "01_Submission")
PKG = os.path.join(SUB, "Brain Stimulation_2026-11-04")
RV = os.path.join(ROOT, "results_v23", "v23_uq_recollect_2026-11-04")
R7 = os.path.join(REC, "RESULTS_v23_2026-11-02_r7.md")
R10 = os.path.join(REC, "RESULTS_v23_2026-11-04_r10.md")
OUT = os.path.join(REC, "RESULTS_v23_2026-11-04_r11.md")
DEPOSIT = "07_Code_and_Tools/Code_Deposit_2026-11-04"
OLD_FIGS = "Rev0.4_2026-10-30/figures/"
NEW_FIGS = "../../../01_Submission/Rev0.4_2026-10-30/figures/"
NOTE = "> **r11 note"


def docx_text(name):
    x = zipfile.ZipFile(os.path.join(PKG, name)).read("word/document.xml").decode("utf8")
    out = []
    for p in x.split("</w:p>"):
        t = "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p))
        out.append(t.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
    return "\n".join(out)


def read(path):
    return open(path, encoding="utf-8").read()


r7, r10 = read(R7), read(R10)
SOURCES = {
    "r10": r10,
    "FINDINGS.md": read(os.path.join(RV, "FINDINGS.md")),
    "manuscript": docx_text("04_manuscript.docx"),
    "supplement": docx_text("06_supplement.docx"),
    "tables": docx_text("05_tables.docx"),
    "legends": docx_text("07_figure_legends.docx"),
}

# (r7 quote, current source, current quote, what changed); numbers appear only inside the checked quotes
CHANGES = [
    ("in every one of the 300 heads, 75 perturbed runs with fixed seeds", "supplement",
     "One nominal solution and two mask evaluations give 77 evaluations; mask evaluation reuses nominal fields.",
     "Run accounting: the earlier count included the nominal solution, which is not a perturbation."),
    ("FEM's cohort montage is the fixed protocol; FEM was not perturbed.", "supplement",
     "The fixed montage is selected by FEM and evaluated using perturbed SPICEHead fields; FEM itself was not perturbed.",
     "Protocol identity: r7's \"cohort montage\" in this analysis is the FEM-selected montage, not the SPICEHead-selected one."),
    ("The cohort montage holds as a fixed protocol in most runs, but the pre-specified criterion is strict.", "r10",
     "A post-review joint analysis also required the T floor in the same perturbed run: the protocol met both criteria in a regional median 62.3% of runs (95% CI 61.0-63.7%), below 50% in 49 regions and at least 95% in nine (Table S14).",
     "r7's rates test F retention only. The joint endpoint did not exist in r7."),
    ("Least robust targets for a fixed protocol", "supplement",
     "This criterion does not test the T floor.",
     "\"Robust\" in r7's collector summaries and Table S14 means the F-only criterion."),
    ("target mask 0.98-1.03", "r10",
     "Among eligible heads, regional median p5-p95 T ratios were 0.99-1.04 after erosion and 0.98-1.00 after dilation, which left every mask eligible.",
     "Corrected: the earlier summary treated emptied masks differently between target groups."),
    ("One-voxel erosion leaves no target voxel in some heads for 10 small regions", "r10",
     "One-voxel erosion left fewer than five voxels in at least one head for 14 targets, and in all heads for both locus coeruleus nuclei and the left lateral geniculate (Table S15).",
     "Eligibility now uses the five-voxel rule; ten of the targets also have an empty mask."),
    ("The collector enters an emptied mask as a T ratio of 0", "supplement",
     "An empty mask is a localization failure, not a measured zero field.",
     "Empty masks are now undefined and counted as unsuccessful in applicability."),
    ("(variance share 10%)", "supplement",
     "Variance ratios compare within-head scenario variance with that variance plus between-head variance; they are not additive components.",
     "r7's \"variance share\" values do not sum across scenarios."),
    ("the 19-pad reference is conservative for T by about 5% and unbiased for F", "supplement",
     "TI target-envelope ratios cannot be subtracted from basis-field magnitude bias.",
     "Withdrawn: r7's reading that the offset would shrink by this amount. The check covers six fixed montages."),
    ("in 15 with intracranial volume and extracranial thickness (M3, 10 with higher T in men)", "supplement",
     "After head-size adjustment, nine regional sex associations remain (three negative, six positive).",
     "Regional demographic counts now use reported-montage T, with q values recomputed over 163 regions."),
    ("removed the pooled sex difference", "r10",
     "attenuated the pooled sex association to +0.1% (95% CI -1.6% to +1.9%)",
     "Wording: an association attenuated under one model, not an explained difference."),
    ("halved the between-head variance (intraclass correlation 0.53 to 0.24)", "r10",
     "Between-head random-intercept variance decreased by 72.7% (SD 0.09561 to 0.05000; ICC 0.531 to 0.237).",
     "Corrected: r7's \"halved\" described the change in intraclass correlation, not in variance."),
    ("whereas the age effect remained (-1.4% per decade)", "r10",
     "Pooled age remained associated at -1.4% per decade in this model, but the median regional association was -0.2% in a separate all-anatomy model",
     "The age association depends on the anatomical adjustment."),
    ("the age decrease was significant in 134 of 163 regions", "r10",
     "Region-wise age associations were negative in 134 and positive in two regions at q < 0.05",
     "Positive associations are now reported as well."),
    ("T declines with age (Spearman)", "tables",
     "Negative age association at reported montage (q < 0.05)",
     "Table 3's age column changed outcome and test, so its counts changed."),
    ("| Heads gaining >=10%, % | Same as FEM, % | T per decade of age |", "tables",
     "Reported T per decade of age",
     "Table 4's age column, and the \"T per decade\" column of section 7, are replaced by reported-montage coefficients; current values for all regions are in Data S1."),
    ("Age, sex, head size and site by region set", "supplement",
     "Table S8. Regional age and sex associations for T at the reported montage (SPICEHead, 1 mm)",
     "Table S8 was rebuilt on reported-montage T."),
    ("the comparison separates site acting through segmented anatomy from site acting otherwise", "supplement",
     "They do not identify causal mediation or separate scanner from recruitment.",
     "Withdrawn: the causal reading of the anatomy adjustment."),
    ("Individual planning mattered in a minority of regions (Figure 5).", "r10",
     "Individual planning improved focality substantially in selected regions (Figure 4).",
     "Framing of the same counts, from the author revision."),
    ("*Figure 2. Analytical verification.*", "supplement",
     "Figure S1. Analytical verification.",
     "Renumbered: this figure moved to the supplement."),
    ("*Figure 3. Agreement with FEM in 300 heads.*", "legends",
     "Figure 2. Numerical agreement and nominal targeting feasibility",
     "Renumbered and revised."),
    ("*Figure 4. Whole-brain targeting parameters.*", "legends",
     "Figure 3. Whole-brain targeting parameters",
     "Renumbered."),
    ("*Figure 5. Individual variability and demographic determinants.*", "legends",
     "Figure 4. Individual coverage and fixed-protocol sensitivity",
     "Split: coverage and uncertainty are the six panels of the current Figure 4; associations are the current Figure 5."),
    ("The study was approved by the Institutional Review Board of Taipei Veterans General Hospital (2023-02-008CCF).", "manuscript",
     "[AUTHOR INPUT: specify informed consent or waiver and confirm data-use authorization for all contributing sources.]",
     "Still open: the approval sentence stands, but consent and data-use coverage await the authors."),
    ("model done; mask runs running", "supplement",
     "Classes come from the stored masked voxel counts; there were no numerical failures or missing runs.",
     "Status: the mask runs finished."),
]
for quote, src, cur, _ in CHANGES:
    assert r7.count(quote) >= 1, f"r7 quote not found: {quote[:70]}"
    assert SOURCES[src].count(cur) >= 1, f"{src} quote not found: {cur[:70]}"


def md_table(headers, rows):
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def annotate_r7():
    """r7 without its title, headings demoted, figures repointed, notes after the passage each change touches."""
    lines = r7.split("\n")
    assert lines[0].startswith("# ")
    lines = lines[1:]
    notes = {}

    def add(i, text):
        j = i                                                # a note must not split a table or a paragraph
        while j + 1 < len(lines) and lines[j + 1].strip():
            j += 1
        notes.setdefault(j, [])
        if text not in notes[j]:
            notes[j].append(text)

    for n, (quote, *_rest) in enumerate(CHANGES, 1):
        for i, ln in enumerate(lines):
            if quote in ln:
                add(i, f"{NOTE} C{n}:** changed since r7; see Part C, row {n}.")
    log = json.load(open(os.path.join(SUB, "Rev0.5_2026-11-02", "evidence_revision", "document_changes.json"), encoding="utf-8"))
    logged = sorted({c["before"] for c in log if len(c.get("before", "")) >= 80})
    flagged = 0
    for before in logged:
        for i, ln in enumerate(lines):
            if before in ln:
                add(i, f"{NOTE}:** the Rev0.5 change log records this passage as revised; Part A and the current package hold the current wording.")
                flagged += 1
    out = []
    for i, ln in enumerate(lines):
        if re.match(r"#{1,5} ", ln):
            ln = "#" + ln
        out.append(ln.replace(OLD_FIGS, NEW_FIGS))
        for t in notes.get(i, []):
            out += ["", t]
    return "\n".join(out), flagged, len(logged)


def part_b():
    S = json.load(open(os.path.join(RV, "final_summary.json"), encoding="utf-8"))
    F, SP, M = S["fem_protocol"], S["spicehead_protocol"], S["masks"]
    E = json.load(open(os.path.join(RV, "endpoint_definition.json"), encoding="utf-8"))
    Q = json.load(open(os.path.join(RV, "qc_assertions.json"), encoding="utf-8"))
    run = json.load(open(os.path.join(RV, "sp_protocol_run.json"), encoding="utf-8"))
    fem = pd.read_csv(os.path.join(RV, "uq_fem_protocol_pooled.csv")).set_index("target")
    spc = pd.read_csv(os.path.join(RV, "uq_spicehead_protocol_pooled.csv")).set_index("target")
    pair = pd.read_csv(os.path.join(RV, "uq_protocol_paired_comparison.csv"))
    rec = pd.read_csv(os.path.join(RV, "protocol_score_recoverability.csv")).set_index("target")
    mask = pd.read_csv(os.path.join(RV, "mask_validity_by_target.csv"))
    b1 = pd.read_csv(os.path.join(RV, "b1_pad4_paired.csv"))
    assert len(fem) == len(spc) == len(rec) == S["targets"] == 163
    assert not Q["failed"] and len(Q["checks"]) == 37
    for tab, summ in ((fem, F), (spc, SP)):                  # the tables below must reproduce the summary
        assert round(tab.joint_pooled74_pct.median(), 2) == summ["joint_median"]
        assert round(tab.F_only_pooled74_pct.median(), 2) == summ["F_only_median"]
        assert round(tab.floor_pooled74_pct.median(), 2) == summ["floor_median"]
    p74 = pair[pair.scenario == "pooled74"].set_index("target")
    diff = p74[~p74.identical_protocol]
    assert len(diff) == SP["differing_34"]["regions"] == 34
    assert (p74[p74.identical_protocol].joint_diff_pct_points == 0).all()

    ci = lambda c: f"{c[0]}-{c[1]}"
    summary = md_table(
        ["Across 163 regions", "FEM-selected protocol", "SPICEHead-selected protocol"],
        [["F-only success, regional median, % (95% CI)", f"{F['F_only_median']} ({ci(F['F_only_median_ci'])})", f"{SP['F_only_median']} ({ci(SP['F_only_median_ci'])})"],
         ["T floor met, regional median, % (95% CI)", f"{F['floor_median']} ({ci(F['floor_median_ci'])})", f"{SP['floor_median']} ({ci(SP['floor_median_ci'])})"],
         ["Joint success, regional median, % (95% CI)", f"{F['joint_median']} ({ci(F['joint_median_ci'])})", f"{SP['joint_median']} ({ci(SP['joint_median_ci'])})"],
         ["Regions meeting the prespecified F-only criterion (95% CI)", f"{F['F_only_95_regions']} ({ci(F['F_only_95_ci'])})", f"{SP['F_only_95_regions']} ({ci(SP['F_only_95_ci'])})"],
         ["Regions with joint success >= 95% (95% CI)", f"{F['joint_95_regions']} ({ci(F['joint_95_ci'])})", f"{SP['joint_95_regions']} ({ci(SP['joint_95_ci'])})"],
         ["Regions with joint success < 50%", F["joint_lt50_regions"], SP["joint_lt50_regions"]],
         ["Nominal joint success, regional median, %", F["nominal_joint_median"], SP["nominal_joint_median"]],
         ["Regions with nominal joint success < 50%", F["nominal_joint_lt50"], SP["nominal_joint_lt50"]]])

    SCEN = (("conductivity", "Conductivity (40 runs)"), ("placement_independent", "Electrode placement (20)"),
            ("cap_rotation", "Cap rotation (10)"), ("segmentation", "Segmentation (4)"))
    per = pd.read_csv(os.path.join(RV, "uq_spicehead_protocol_per_target_scenario.csv"))
    rows = []
    for key, name in SCEN:
        a, b = F["by_scenario"][key], per[per.scenario == key]
        rows.append([name, a["F_only_median"], a["floor_median"], a["joint_median"],
                     round(b.F_only_pooled_pct.median(), 2), round(b.floor_pooled_pct.median(), 2), round(b.joint_pooled_pct.median(), 2)])
    scenario = md_table(["Scenario (regional medians, %)", "FEM: F-only", "FEM: T floor", "FEM: joint", "SPICEHead: F-only", "SPICEHead: T floor", "SPICEHead: joint"], rows)

    TIER = (("tier1", "Cerebral (AAL)"), ("tier2", "Psychiatric"), ("ref6", "Reference spheres"), ("cerebellum", "Cerebellum"),
            ("brainstem", "Brainstem nuclei"), ("thalamus", "Thalamic nuclei"), ("cingulate", "Supracallosal ACC"))
    rows = []
    for key, name in TIER:
        a, b = fem[fem.tier == key], spc[spc.tier == key]
        rows.append([name, len(a), f"{a.F_only_pooled74_pct.median():.1f}", f"{a.floor_pooled74_pct.median():.1f}", f"{a.joint_pooled74_pct.median():.1f}",
                     f"{b.F_only_pooled74_pct.median():.1f}", f"{b.floor_pooled74_pct.median():.1f}", f"{b.joint_pooled74_pct.median():.1f}"])
    tiers = md_table(["Set (regional medians, %)", "Regions", "FEM: F-only", "FEM: T floor", "FEM: joint", "SPICEHead: F-only", "SPICEHead: T floor", "SPICEHead: joint"], rows)

    rows = []
    for t in diff.sort_values("joint_diff_pct_points").index:
        d = diff.loc[t]
        route = {"new_basis_solves": "new solves", "float16_full_scores": "saved scores (float16)",
                 "saved_fem_cohort_unconstrained_column": "saved scores (float32)"}[d.score_route]
        rows.append([t, d.fem_montage, d.spicehead_montage, f"{fem.loc[t, 'joint_pooled74_pct']:.1f}", f"{spc.loc[t, 'joint_pooled74_pct']:.1f}",
                     f"{d.joint_diff_pct_points:+.1f} ({d.joint_diff_ci_lo:+.1f} to {d.joint_diff_ci_hi:+.1f})",
                     f"{d.floor_diff_pct_points:+.1f}", f"{d.F_only_diff_pct_points:+.1f}", route])
    differing = md_table(["Target", "FEM montage", "SPICEHead montage", "FEM: joint, %", "SPICEHead: joint, %", "Joint difference, points (95% CI)", "T floor difference", "F-only difference", "Source of scores"], rows)

    rows = []
    for t in fem.index:
        a, b, same = fem.loc[t], spc.loc[t], bool(p74.loc[t, "identical_protocol"])
        rows.append([t, a.tier, a.montage_label, "same" if same else b.montage_label,
                     f"{a.F_only_pooled74_pct:.1f}", f"{a.floor_pooled74_pct:.1f}", f"{a.joint_pooled74_pct:.1f}",
                     "yes" if a.meets_F_only_95pct_prespecified else "no",
                     "" if same else f"{b.F_only_pooled74_pct:.1f}", "" if same else f"{b.floor_pooled74_pct:.1f}", "" if same else f"{b.joint_pooled74_pct:.1f}",
                     "" if same else ("yes" if b.meets_F_only_95pct_prespecified else "no")])
    regions = md_table(["Target", "Set", "FEM montage", "SPICEHead montage", "FEM: F-only, %", "FEM: T floor, %", "FEM: joint, %", "FEM: F-only criterion", "SPICEHead: F-only, %", "SPICEHead: T floor, %", "SPICEHead: joint, %", "SPICEHead: F-only criterion"], rows)

    er = mask[(mask["mask"] == "erosion") & (mask.complete_ge5 < mask.heads_assessed)].sort_values("complete_ge5")
    assert len(er) == M["erosion_lt5_regions"] == 14
    masks = md_table(["Target", "Heads assessed", "At least 5 voxels", "1-4 voxels", "Empty"],
                     [[r.target, r.heads_assessed, r.complete_ge5, r.nonempty_1_to_4_voxels, r.empty] for r in er.itertuples()])

    qc = md_table(["Check", "Result"], [[k, "passed" if v["passed"] else "FAILED"] for k, v in Q["checks"].items()])
    four = md_table(["Target", "Ratio", "Heads", "Median", "5th-95th percentile"],
                    [[r.target, r.ratio, r.heads, f"{r.median:.3f}", f"{r.p5:.3f}-{r.p95:.3f}"] for r in b1.itertuples()])
    defs = md_table(["Term", "Definition"], [[k, E[k]] for k in ("floor_ratio", "F_retention", "valid", "F_only_success", "joint_success", "pooled74", "equal4", "bootstrap", "mask_policy")])
    ver = run["verification"]
    return f"""## Part B. Run-level uncertainty results in full

All from `01_Shared/results_v23/v23_uq_recollect_2026-11-04/`. Every table is computed from those files, and the medians are checked against `final_summary.json` when this record is built. Both protocols are evaluated with perturbed SPICEHead fields; perturbed FEM fields were not run. These are sensitivity results under assumed scenario sampling, not calibrated targeting probabilities. Confidence intervals come from a head-cluster bootstrap ({S['bootstrap']['n']:,} resamples, seed {S['bootstrap']['seed']}) and are not adjusted for multiple comparisons.

### B1. Endpoints

{defs}

### B2. Overall results, 74 physical runs per head

{summary}

### B3. By scenario

{scenario}

Over all 74 runs each scenario counts by its number of runs. With the four scenarios weighted equally, the joint medians are {F['joint_equal4_median']}% (FEM-selected) and {spc.joint_equal4_pct.median():.2f}% (SPICEHead-selected); this is a labelled sensitivity, not a real-world probability.

### B4. By target set

{tiers}

### B5. The 34 regions where the two cohort montages differ

Differences are SPICEHead minus FEM protocol over all 74 runs, ordered by the joint difference. The 95% CI of the joint difference excludes zero in favor of the SPICEHead protocol in {SP['differing_34']['spicehead_higher_joint_ci_excludes_0']} regions and of the FEM protocol in {SP['differing_34']['fem_higher_joint_ci_excludes_0']}. In the other 129 regions the montages are the same and every difference is exactly zero. Scoring with SPICEHead fields may favor the SPICEHead selection.

{differing}

### B6. Target masks after one-voxel erosion

The 14 targets with any eroded mask below five voxels. Dilation left all targets eligible, and there were no numerical failures. Exposure and joint applicability for these targets are in Table S15 and `mask_exposure_conditional.csv`, `mask_joint_applicability.csv`.

{masks}

### B7. New solves for the 28 protocols

{run['system']}; {run['state']}; {run['heads_complete']} heads; {run['run_files']:,} run files; {run['su']:,.0f} SU. Jobs: {'; '.join(f'{v}' for v in run['jobs'].values())}. Checksums matched for {ver['local_checksums_ok']:,} of {ver['cluster_checksums']:,} listed files. The FEM montage rescored in those runs equals the archive in all {ver['fem_control_evaluations']:,} evaluations (largest absolute gap {ver['fem_control_max_abs_gap']}). {run['timing_note']}.

### B8. Four-pad comparison of SPICEHead T and F (B1 of the request; record only, not in the manuscript)

Six fixed montages only, so this does not test atlas-wide ranking.

{four}

### B9. Quality checks

{len(Q['checks'])} checks, {len(Q['failed'])} failed.

{qc}

### B10. All 163 regions

Blank SPICEHead cells mean the montage is the same as FEM's, so the values are identical. The F-only criterion is at least 90% F retention in at least 95% of runs.

{regions}
"""


def main():
    if os.path.exists(OUT) and "--rebuild" not in sys.argv:
        raise SystemExit(f"{OUT} exists; pass --rebuild to replace it")
    body10 = r10.split("\n", 1)[1].strip()
    d, flagged, logged = annotate_r7()
    rows = [[n, q, s, c, w] for n, (q, s, c, w) in enumerate(CHANGES, 1)]
    part_c = md_table(["Row", "r7 said", "Current source", "Current wording", "What changed"], rows)
    text = f"""# SPICEHead v23: comprehensive results record, r11 (Paper A)

Date: 4 November 2026. Generated by `pipeline/results_v23_r11.py`. Status: **author-review record; not submission-ready.** Internal document: it keeps exact model IDs and source paths, which the paper-facing documents omit.

This is the one full record. It combines r7 (detailed evidence, 2 November) with r10 (current Results, 4 November) and contains no new analysis. Parts B and C are new; Parts A and D are the earlier records. It uses v23 exclusively and the 300-head cohort. r7 and r10 stay unchanged as dated records.

**Which part governs.** Part A states the current Results, and Part B gives the same run-level results in full. Where Part D differs, Parts A and B govern, and Part C lists the known differences. Part D keeps r7's wording, figure numbers and section references as they stood on 2 November.

**Paths.** Paths in Part D predate the reorganization of 2-3 November: code and results are now under `01_Shared/`, packages under `01_Submission/`, and records under `02_Results_Records/`. The move map is `00_Program/reorg_2026-11-02/file_moves.csv`.

**Current materials.**
- Package: `01_Submission/Brain Stimulation_2026-11-04/`, unchanged by this record. Its figures are `figures/Figure1-5` and `FigureS1`.
- Evidence: `01_Shared/results_v23/v23_uq_recollect_2026-11-04/`.
- Code: `{DEPOSIT}/`, which adds the run-level scripts. `07_Code_and_Tools/Frozen_Code_Deposit/` stays as the deposit of 2 November.

## Contents

- Part A. Current Results (r10)
- Part B. Run-level uncertainty results in full
- Part C. r7 statements that have changed
- Part D. Detailed record (r7): review rounds, Part I (manuscript evidence, Rev0.4 numbering), Part II (supporting analyses, not manuscript claims)
- Status

---

## Part A. Current Results (r10)

Verbatim from `RESULTS_v23_2026-11-04_r10.md`.

{body10}

---

{part_b()}
---

## Part C. r7 statements that have changed

Quotes are exact and checked when this record is built. The table lists the changes identified by comparing r7 with r10, the current package and the Rev0.5 change log. It may not be exhaustive, so Parts A and B govern in any case. In Part D, `C<n>` notes point to these rows.

{part_c}

In addition, {flagged} occurrences in Part D match passages the Rev0.5 change log records as revised ({logged} logged passages of at least 80 characters were searched); they carry a general note. Table cells changed by the log are covered by the rows for Tables 3, 4 and S8.

---

## Part D. Detailed record (r7)

Verbatim from `RESULTS_v23_2026-11-02_r7.md`, except for demoted headings, repointed figure links and the marked r11 notes. The status list at its end describes 2 November.

{d}

---

## Status

- **Computed:** A0-A4 of `10_ANALYSIS_REQUEST_v23_2026-11-02.md`, and all 163 protocols under both selection models. B1 is a record only.
- **Not run:** B2 (needs new solves) and B3 (depends on author metadata); perturbed FEM fields.
- **Open for the authors:** declarations, ethics and consent coverage, the acquisition discrepancy, release details, licence and repository, title and current journal requirements.
- **Review:** the additions since r3 have had no independent review.
- This record does not certify submission readiness. The paused monitoring automation was not restarted or changed.
"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    check(text)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {OUT}: {len(text.split()):,} words; {len(CHANGES)} change rows; {flagged} log flags")


def check(text):
    """Parts A and D must equal their sources once this record's own additions are removed."""
    sys.path.insert(0, HERE)
    from build_code_deposit import ID_LIKE
    assert body_of(text, "## Part A.", "\n---\n\n## Part B.").split("\n\n", 2)[2].strip() == re.sub(r"\n{3,}", "\n\n", r10.split("\n", 1)[1]).strip(), "Part A differs from r10"
    part_d = body_of(text, "## Part D.", "\n---\n\n## Status").split("\n\n", 2)[2].strip()
    kept = "\n".join(ln for ln in part_d.split("\n") if not ln.startswith(NOTE))
    kept = re.sub(r"(?m)^#(#{1,5} )", r"\1", kept).replace(NEW_FIGS, OLD_FIGS)
    squeeze = lambda s: re.sub(r"\n{2,}", "\n", s).strip()
    assert squeeze(kept) == squeeze(r7.split("\n", 1)[1]), "Part D differs from r7"
    assert not re.search(r"(?<![A-Za-z0-9])h\d{3}(?![A-Za-z0-9])", text), "a head code is in the record"
    assert not ID_LIKE.search(text), "a study ID is in the record"
    for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        assert os.path.exists(os.path.normpath(os.path.join(REC, m))), f"broken figure link: {m}"


def body_of(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


if __name__ == "__main__":
    main()
