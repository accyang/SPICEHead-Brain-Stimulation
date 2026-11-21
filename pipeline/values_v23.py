# -*- coding: utf-8 -*-
"""Every number in the v23 Results record, computed from result files and never typed.

Paper A's final results are v23, the homogenized network (PI, 26 October 2026). This module reuses
values_v22.py section by section with the v23 sources and replaces the sections whose v22 content has no v23
counterpart. values_v22.py keeps producing the v22 record unchanged (its v23 switches default to v22).

Where the inputs live
  model-dependent    RES = RES0 = results_v23/ (the analytic disk, the cylinder benchmark and the contact
                     experiment were rerun under v23 too, so nothing model-dependent is read from results/)
  FEM reference      results/v21_simnibs_validation (SimNIBS fields and logs; unchanged by the model switch)
  MNI-template maps  results_v23/figures/TI_targets_MNI
  full-space FEM     results_v23/v23_fem_fullsearch_300 (pipeline/v23_fullsearch_collect.py), plus the FEM-only
                     checks of the eight-head desktop run in results_v22/v22_fem_fullsearch: leadfield validation,
                     passive pads, the FEM side of the four-pad check and its SimNIBS diagnostics. Not carried over
                     (PI, 30 October): the four-pad comparison of the v22 SPICEhead optima, the desktop stage
                     timings (the benchmark is the Forerunner run) and the desktop-vs-Nano 4 consistency of v22
  layered slab       results_v23/v23_layered_slab: nearest-label (v22) and homogenized (v23) networks against the
                     exact series resistance. Replaces the interface-rule slab: v23 has no interface rule
                     (PI decision 2, 28 October)
  contacts           results_v23/v21_contact_sensitivity: skin-surface vs volumetric contacts under v23 (its
                     "rule" column is a leftover label; v23 has no rule)
  model change       v22 -> v23: the v22 record's results (results_v22/) against these, and the 300-head
                     nearest-label vs homogenized FEM comparison of 23 October (results_v22/v22_fem_fullsearch_300_homog)
  template design    results_v23/v23_template_design

  SPICEHEAD_MODEL=v23 python pipeline/values_v23.py [substr]
"""
import csv
import json
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import values_v22 as V2                                          # noqa: E402
from values_v22 import (ROOT, MISSING, ROIS, NICE, PROSE, MM_ORDER, MM_NICE, SHORT_, _p, rows, jload, f,  # noqa: E402
                        mean, label, table, pct, sci, config)

if config.MODEL != "v23":
    raise SystemExit("run with SPICEHEAD_MODEL=v23")

RES = config.RESULTS
V22 = _p(ROOT, "results_v22")
HOMOG = _p(V22, "v22_fem_fullsearch_300_homog")
V2.RES0 = RES
V2.MNI = _p(RES, "figures", "TI_targets_MNI")
V2.FS8_DIR = _p(V22, "v22_fem_fullsearch")
V2.FS300_DIR = _p(RES, "v23_fem_fullsearch_300")
V2.FS_V22_PARTS = False
V2.TD_DIR = "v23_template_design"
STACK_NICE = {"aligned": "6 / 6 / 3 / 15 mm (whole 1 mm cells)", "thin": "5.5 / 4.5 / 2.5 / 15 mm"}


def _drop_missing(prefix):
    MISSING[:] = [m for m in MISSING if not m.startswith(prefix)]


# ---------------------------------------------------------------- sections replaced or extended
def verification(V, T):
    """values_v22.verification (disk and grid study from results_v23) with the v23 layered slab."""
    V2.verification(V, T)
    _drop_missing("layered slab:")                     # the interface-rule slab does not exist under v23
    s = rows(_p(RES, "v23_layered_slab", "layered_slab.csv"), "layered slab (v23)")
    for r in s:
        k = f"slab.{r['stack']}.{r['model']}_{f(r['dx_mm']):g}mm"
        V[k], V[k.replace("mm", "mm_err", 1)] = f(r["R_ratio"]), f(r["err_pct"])
    if s:
        for m in ("v22", "v23"):
            for st in ("aligned", "thin"):
                e = [abs(f(r["err_pct"])) for r in s if r["model"] == m and r["stack"] == st]
                V[f"slab.{st}.{m}_abserr_max"] = max(e)
        V["slab.v23_abserr_max"] = max(abs(f(r["err_pct"])) for r in s if r["model"] == "v23")
        thin = [f(r["err_pct"]) for r in s if r["model"] == "v22" and r["stack"] == "thin"]
        V["slab.thin.v22_err_lo"], V["slab.thin.v22_err_hi"] = min(thin), max(thin)


def supplement(V, T):
    V2.supplement(V, T)
    _drop_missing("layered slab:")
    s = rows(_p(RES, "v23_layered_slab", "layered_slab.csv"), "layered slab (v23)")
    by = {(r["stack"], f(r["dx_mm"]), r["model"]): f(r["R_ratio"]) for r in s}
    T["slab"] = table(["Layers (scalp / bone / CSF / grey matter)", "Grid", "Nearest-label network (v22)",
                       "Homogenized network (v23)"],
                      [[STACK_NICE[st], f"{dx:g} mm", f"{by[(st, dx, 'v22')]:.4f}", f"{by[(st, dx, 'v23')]:.6f}"]
                       for st in ("aligned", "thin") for dx in (3.0, 2.0, 1.0) if (st, dx, "v23") in by])


def bench(V, T):
    """values_v22.bench on the Forerunner run; v23's network is built while loading, so the load time is kept."""
    V2.bench(V, T)
    out = []
    for r in rows(_p(RES, "v22_benchmark", "heads.csv"), "head benchmark"):
        tag = f"{float(r['dx_mm']):g}mm"
        V[f"bench.{tag}.t_load_s"] = f(r["t_load_s"])
        V[f"bench.{tag}.load_includes_build"] = r.get("t_load_includes_network_build") == "True"
        out.append([f"{float(r['dx_mm']):g} mm", f"{int(f(r['n_nodes'])):,}", r["solver"],
                    f"{f(r['t_load_s']):.1f}", f"{f(r['t_build_s']) + f(r['t_setup_s']):.1f}",
                    f"{f(r['t_per_solve_s']):.1f}", f"{f(r['t_score_s']) / 60:.1f}",
                    f"{f(r['t_total_s']) / 60:.1f}", f"{f(r['peak_gb']):.1f}"])
    if out:
        # v23 builds the network while loading the segmentation (homogenization); its "build" stage then places
        # the electrode contacts only (v22_benchmark_heads.py), so the columns are named for what they time
        T["bench"] = table(["Grid", "Nodes", "Solver", "Load and network build (s)", "Contacts and solver setup (s)",
                            "Per basis solve incl. field extraction (s)", "Scoring 11,628 montages (min)",
                            "Sum of stages (min)", "Peak working set (GB)"], out)
    node = jload(_p(RES, "v22_benchmark", "node.json"), "benchmark node record (lscpu)") \
        if os.path.exists(_p(RES, "v22_benchmark", "node.json")) else None
    if node:
        V["bench.cpu"] = node["cpu_model"]
        V["bench.node"] = node.get("node")


def contact_change(V, T):
    """Skin-surface against volumetric contacts under v23, at 2 mm against FEM (8 heads x 4 montages)."""
    d = _p(RES, "v21_contact_sensitivity")
    rs = rows(_p(d, "contact_agreement.csv"), "contact sensitivity (v23)")
    if not rs:
        return
    for sup in ("volume", "surface"):
        for t in MM_ORDER:
            q = [r for r in rs if r["support"] == sup and r["montage"] == t]
            if q:
                V[f"csens.{sup}.{t}.relbias"] = mean(q, "rel_bias_pct")
                V[f"csens.{sup}.{t}.r"] = mean(q, "r")
                V[f"csens.{sup}.{t}.ccc"] = mean(q, "ccc")
    # paired, per head and montage: surface minus volume, in points of the FEM mean
    vol = {(r["subject"], r["montage"]): f(r["rel_bias_pct"]) for r in rs if r["support"] == "volume"}
    dif = [f(r["rel_bias_pct"]) - vol[(r["subject"], r["montage"])] for r in rs
           if r["support"] == "surface" and (r["subject"], r["montage"]) in vol]
    if dif:
        mt = {t: float(np.mean([f(r["rel_bias_pct"]) - vol[(r["subject"], t)] for r in rs
                                if r["support"] == "surface" and r["montage"] == t])) for t in MM_ORDER}
        V["csens.diff_lo"], V["csens.diff_hi"] = min(mt.values()), max(mt.values())
        V["csens.n_heads"] = len({r["subject"] for r in rs})
    # best of the four montages by F, per head and target, against FEM's best
    roi = rows(_p(d, "contact_roi.csv"), "contact sensitivity ROI table (v23)")
    for sup in ("volume", "surface"):
        g = {}
        for r in roi:
            if r["support"] == sup:
                g.setdefault((r["subject"], r["roi"]), []).append(r)
        agree = sum(max(x, key=lambda r: f(r["F_sp"]))["montage"] == max(x, key=lambda r: f(r["F_fem"]))["montage"]
                    for x in g.values())
        V[f"csens.{sup}.best4_agree"], V[f"csens.{sup}.best4_n"] = agree, len(g)
    sup = rows(_p(d, "contact_support.csv"), "contact supports (v23)")
    for kind in ("volume", "surface"):
        s = [r for r in sup if r["support"] == kind]
        if s:
            V[f"csens.{kind}.nodes_med"] = float(np.median([f(r["n_nodes"]) for r in s]))
            V[f"csens.{kind}.nonscalp_med"] = float(np.median([f(r["non_scalp_pct"]) for r in s]))
            V[f"csens.{kind}.nonscalp_max"] = max(f(r["non_scalp_pct"]) for r in s)
    T["contacts"] = table(["Montage", "Volumetric contacts: bias, r", "Skin-surface contacts: bias, r"],
                          [[MM_NICE[t], f"{pct(V[f'csens.volume.{t}.relbias'])}, {V[f'csens.volume.{t}.r']:.3f}",
                            f"{pct(V[f'csens.surface.{t}.relbias'])}, {V[f'csens.surface.{t}.r']:.3f}"]
                           for t in MM_ORDER if f"csens.surface.{t}.relbias" in V])


def network_change(V, T):
    """The 23 October comparison of the two network constructions against FEM over all 300 heads and 11,628
    montages (collect_homog300.py): nearest-label v22 and homogenized (= v23) at 2 and 1 mm, same FEM scores."""
    s = jload(_p(HOMOG, "summary.json"), "nearest-label vs homogenized comparison")
    if not s:
        return
    n = s["n_rows"]
    NM = {"sp2": ("v22", "2 mm"), "sp1": ("v22", "1 mm"), "sp2hom": ("v23", "2 mm"), "sp1hom": ("v23", "1 mm")}
    out = []
    for m in ("sp2", "sp2hom", "sp1", "sp1hom"):
        x, (mod, g) = s["models"][m], NM[m]
        k = f"net.{mod}.{g.replace(' ', '')}"
        V[f"{k}.same"], V[f"{k}.same_pct"] = x["same"], 100 * x["same"] / n
        for q in ("keep_median", "keep_p5", "keep_min", "below90", "below95", "rho_median", "field_rel",
                  "field_ratio", "field_bias", "field_r", "field_ang"):
            V[f"{k}.{q}"] = x[q]
        out.append([f"{'Nearest-label (v22)' if mod == 'v22' else 'Homogenized (v23)'}, {g}",
                    f"{x['same']}/{n} ({100 * x['same'] / n:.0f}%)",
                    f"{x['keep_median']:.1f} / {x['keep_p5']:.1f} / {x['keep_min']:.1f}", x["below90"],
                    f"{x['rho_median']:.3f}", f"{x['field_rel']:.3f}", pct(x["field_bias"]), f"{x['field_r']:.3f}"])
    V["net.n_rows"], V["net.n_heads"] = n, s["n_heads"]
    for hom in ("sp2hom", "sp1hom"):
        p, g = s["paired"][hom], NM[hom][1].replace(" ", "")
        for q in ("better", "worse", "same", "median_change", "heads_rel_lower", "heads_r_higher", "heads_abs_bias_lower"):
            V[f"net.paired.{g}.{q}"] = p[q]
        # rows below 90% under the v23 choice (either model below 90% is what the file lists)
        low = [x for x in p["rows_below90"] if x["homogenized"] < 90]
        V[f"net.paired.{g}.v23_low"] = "; ".join(f"{PROSE[x['roi']]} {x['homogenized']:.1f}%" for x in low) or "none"
    for m in ("sp2hom", "sp1hom"):
        nt = s["network"][m]
        g = NM[m][1].replace(" ", "")
        V[f"net.v23.{g}.nodes_med_M"] = nt["nodes_median"] / 1e6
        V[f"net.v23.{g}.dropped_med"], V[f"net.v23.{g}.dropped_max"] = nt["dropped_median"], nt["dropped_max"]
        V[f"net.v23.{g}.t_total_med_s"] = nt["t_total_median_s"]
    cp = s["check_published"]
    V["net.check_n"], V["net.check_mismatch"] = cp["n_compared"], len(cp["mismatches"])
    T["network_change"] = table(["Network, grid", "Same best montage as FEM", "Share of the FEM optimum kept, % "
                                 "(median / 5th pct / min)", "Below 90%", "ρ over all montages, median",
                                 "Field relative error", "Field bias", "Field r"], out)


def model_change(V, T):
    """The headline outcomes under v22 (the earlier record) and v23, side by side."""
    old = {}
    t3 = jload(_p(V22, "montage_optimization", "cohort_gm300_dx5_maxdir", "cohort_montage_optima.json"), "v22 optima")
    if t3:
        for roi in ROIS:
            old[f"opt.{roi}"] = (label(t3["top"][roi][0]["montage"]), t3["top"][roi][0]["median_score"])
    rr = rows(_p(V22, "v21_multimontage", "agreement_by_montage.csv"), "v22 FEM agreement")
    for t in MM_ORDER:
        s = [r for r in rr if r["montage"] == t and f(r["dx_mm"]) == 1.0]
        if s:
            old[f"fem.{t}"] = float(np.mean([100 * (f(r["mean_spicehead"]) / f(r["mean_fem"]) - 1) for r in s]))
            old[f"ccc.{t}"] = mean(s, "ccc")
    if all(f"fem.{t}" in old for t in MM_ORDER):
        # the accepted FEM criterion (values_v22.criteria) on the v22 record's numbers
        V["mc.crit_fem_v22"] = sum(abs(old[f"fem.{t}"]) < 10 and old[f"ccc.{t}"] >= 0.90 for t in MM_ORDER)
    rk = rows(_p(V22, "v21_multimontage", "rank_preservation_focality.csv"), "v22 decisions")
    q = [r for r in rk if f(r["dx_mm"]) == 1.0]
    if q:
        old["rank"] = (sum(r["agree"] == "True" for r in q), len(q))
    sw = jload(_p(V22, "v21_dx2_check_full", "dx2_full_summary.json"), "v22 full sweep")
    vs = rows(_p(V22, "v21_dx2_check_full", "dx2_full_vs_dx5.csv"), "v22 full sweep table")
    if sw and vs:
        old["sweep.kept"] = sum(r["winner_kept"] == "True" for r in vs)
        old["sweep.loss_max"] = max(f(r["cohort_regret_5mm_winner_pct"]) for r in vs)
        old["sweep.own5_max"] = max(sw["per_target"][r]["regret_own5"]["max"] for r in ROIS)
        V["mc.own5_max_v22"] = old["sweep.own5_max"]
    # target T at 5 mm against each head's 1 mm solution, means over the four grid-study heads (as verification())
    per = {}
    for h in V2.HEADS4:
        p = _p(V22, "v21_head_convergence", "head_convergence.csv") if h == "SUBJ01" else \
            _p(V22, "v21_head_convergence", h, "head_convergence.csv")
        rs = rows(p, f"v22 grid study {h}")
        if rs:
            per[h] = {float(r["dx_mm"]): r for r in rs}
    if len(per) == len(V2.HEADS4):
        exc = [float(np.mean([100 * (f(per[h][5.0][f"T_{roi}"]) / f(per[h][1.0][f"T_{roi}"]) - 1) for h in per]))
               for roi in ROIS]
        V["mc.exc5_v22_lo"], V["mc.exc5_v22_hi"] = min(exc), max(exc)
    fs = rows(_p(V22, "v22_fem_fullsearch_300", "fullsearch_decisions.csv"), "v22 full-space decisions")
    if fs:
        k1 = [100 - f(r["regret_sp1winner_under_fem_pct"]) for r in fs]
        old["fs"] = (sum(r["agree_fem_sp1"] == "True" for r in fs), len(fs), float(np.percentile(k1, 5)),
                     sum(x < 90 for x in k1))
    lo = rows(_p(V22, "v21_generalization", "loso.csv"), "v22 leave-one-site-out")
    if lo:
        old["loso"] = max(f(r["transfer_loss_pct"]) for r in lo)
    tc = rows(_p(V22, "v21_twochannel", "twochannel_exact_dx2.csv"), "v22 co-simulation")
    if tc:
        old["tc.50k"] = float(np.mean([f(r["com_shift_td_mm"]) for r in tc if f(r["r2_contact_ohm"]) == 50e3]))
        old["tc.rhead"] = float(np.mean([f(r["R11"]) - 2e3 for r in tc if f(r["r2_contact_ohm"]) == 1e3]))
    out = []
    for roi in ROIS:
        if f"opt.{roi}" in old and f"opt.{roi}.montage" in V:
            out.append([f"Cohort optimum, {NICE[roi]}", f"{old[f'opt.{roi}'][0]} (F {old[f'opt.{roi}'][1]:.2f})",
                        f"{V[f'opt.{roi}.montage']} (F {V[f'opt.{roi}.F']:.2f})"])
    V["mc.n_optimum_same"] = sum(old.get(f"opt.{r}", ("",))[0] == V.get(f"opt.{r}.montage") for r in ROIS)
    for t in MM_ORDER:
        if f"fem.{t}" in old and f"mm.1mm.{t}.relbias" in V:
            V[f"mc.fem.{t}.v22"] = old[f"fem.{t}"]
            out.append([f"FEM bias at 1 mm, {MM_NICE[t].split(' (')[0].lower()}", pct(old[f"fem.{t}"]),
                        pct(V[f"mm.1mm.{t}.relbias"])])
    if "rank" in old and "mm.1mm.rank_foc.agree" in V:
        V["mc.rank.v22"] = old["rank"][0]
        out.append(["Best of four by F at 1 mm, same as FEM", f"{old['rank'][0]} of {old['rank'][1]}",
                    f"{V['mm.1mm.rank_foc.agree']} of {V['mm.1mm.rank_foc.n']}"])
    if "fs" in old and "fs.agree_fem_sp1" in V:
        V["mc.fs.agree_v22"], V["mc.fs.n"], V["mc.fs.p5_v22"], V["mc.fs.low90_v22"] = old["fs"]
        out += [["Full space, 300 heads: 1 mm choice = FEM's", f"{old['fs'][0]} of {old['fs'][1]}",
                 f"{V['fs.agree_fem_sp1']} of {V['fs.n_fem_sp1']}"],
                ["Full space: share of the FEM optimum kept, 5th percentile", f"{old['fs'][2]:.1f}%",
                 f"{V['fs.keep_sp1_under_fem.p5']:.1f}%"],
                ["Full space: head-target rows below 90%", str(old["fs"][3]), str(V["fs.low90_sp1_under_fem"])]]
    if "net.v22.1mm.field_r" in V:
        out += [["Field against FEM, 1 mm: r (median over heads)", f"{V['net.v22.1mm.field_r']:.3f}",
                 f"{V['net.v23.1mm.field_r']:.3f}"],
                ["Field against FEM, 1 mm: bias", pct(V["net.v22.1mm.field_bias"]), pct(V["net.v23.1mm.field_bias"])]]
    if "sweep.kept" in old and "sweep.kept" in V:
        out += [["5 mm cohort winner kept at 2 mm", f"{old['sweep.kept']} of 6", f"{V['sweep.kept']} of 6"],
                ["Largest cohort loss at 2 mm", f"{old['sweep.loss_max']:.2f}%", f"{V['sweep.loss_max']:.2f}%"],
                ["Worst loss, head's own 5 mm optimum", f"{old['sweep.own5_max']:.1f}%",
                 f"{V['sweep.own5.max_hi']:.1f}%"]]
    if "loso" in old and "loso.worst" in V:
        V["mc.loso.v22"] = old["loso"]
        out.append(["Leave-one-site-out, largest loss", f"{old['loso']:.1f}%", f"{V['loso.worst']:.1f}%"])
    if "tc.50k" in old and "tc.50k.com" in V:
        out += [["Head resistance per channel path", f"{old['tc.rhead']:.0f} Ω", f"{V['tc.rhead_mean']:.0f} Ω"],
                ["Center-of-mass shift at 50 kΩ", f"{old['tc.50k']:.1f} mm", f"{V['tc.50k.com']:.1f} mm"]]
    T["model_change"] = table(["Outcome", "v22 (nearest-label network)", "v23 (homogenized network)"], out)
    V["model_change.rows"] = len(out)


def fullsearch_v23(V, T):
    """values_v22.fullsearch on the v23 300-head files, then what only v23 has: run times on Nano 4 and the
    collector's checks."""
    V2.fullsearch(V, T)
    d = V2.FS300_DIR
    tim = rows(_p(d, "fullsearch_timing.csv"), "300-head timings (v23)")
    for m in ("sp2", "sp1"):
        s = [r for r in tim if r["model"] == m]
        if s:
            V[f"fs300.t.{m}.total_med_s"] = float(np.median([f(r["t_total_s"]) for r in s]))
            nn = [f(r["n_nodes"]) for r in s]
            V[f"fs300.t.{m}.nodes_med"], V[f"fs300.t.{m}.nodes_lo"], V[f"fs300.t.{m}.nodes_hi"] = \
                float(np.median(nn)), min(nn), max(nn)
            V[f"fs300.t.{m}.dropped_max"] = max(int(f(r["dropped_nodes"])) for r in s)
    ck = jload(_p(d, "checks.json"), "v23 full-space checks")
    if ck:
        c = ck["vs_homog300_collection"]
        V["fs300.check_rows"], V["fs300.check_mismatch"] = c["n_compared"], len(c["mismatches"])
        V["fs300.check_rho_diff"], V["fs300.check_regret_diff"] = c["max_abs_rho_diff"], c["max_abs_regret_diff_pct"]
        V["fs300.check_cohort_same"] = ck["cohort_vs_homog300_collection"]["winners_identical"]
        V["fsr.fem_only_diff_vs_v22"] = ck["remesh"]["fem_only_columns_max_abs_diff_vs_v22"]
    rm = jload(_p(d, "remesh_analysis.json"), "re-mesh check (v23)")
    if rm:
        outl = [x for x in rm if x["outlier_row"]]
        V["fsr.outlier_keep_old_lo"] = min(x["sp1_keeps_oldfem"] for x in outl)
        V["fsr.outlier_keep_old_hi"] = max(x["sp1_keeps_oldfem"] for x in outl)
        V["fsr.outlier_low_old"] = sum(x["sp1_keeps_oldfem"] < 90 for x in outl)
        V["fsr.outlier_high_old"] = sum(x["sp1_keeps_oldfem"] >= 95 for x in outl)
        # every re-meshed row (selected outlier or control) where the v23 choice is below 90% under either mesh
        low = sorted((x for x in rm if min(x["sp1_keeps_oldfem"], x["sp1_keeps_newfem"]) < 90),
                     key=lambda x: x["sp1_keeps_oldfem"])
        V["fsr.v23_low_rows_n"] = len(low)
        V["fsr.v23_low_still"] = sum(x["sp1_keeps_newfem"] < 90 for x in low)
        # "the insula 41.3 → 41.3%": the v23 choice's retention with the original → re-meshed FEM
        V["fsr.v23_low_rows"] = "; ".join(f"{PROSE[x['roi']]} {x['sp1_keeps_oldfem']:.1f} → {x['sp1_keeps_newfem']:.1f}%"
                                          for x in low) or "none"
        # which of the v23 1 mm rows below 90% (all 300 heads) lie in a re-meshed head
        remeshed = {x["code"] for x in rm}
        dec = rows(_p(d, "fullsearch_decisions.csv"), "v23 full-space decisions")
        lo90 = [r for r in dec if f(r["regret_sp1winner_under_fem_pct"]) > 10]
        V["fsr.v23_low_n"] = len(lo90)
        V["fsr.v23_low_remeshed"] = sum(r["subject"] in remeshed for r in lo90)
        V["fsr.v23_low_not_remeshed"] = len(lo90) - V["fsr.v23_low_remeshed"]


def template_design(V, T):
    """values_v22.template_design on results_v23/v23_template_design, plus each target's best current split for
    its own equal-current montage. The checks expected 1:1 there (true under v22); under v23 the bilateral
    hippocampus prefers 2:3, which is the one failing check, recorded here as a result."""
    V2.template_design(V, T)
    ck = jload(_p(RES, V2.TD_DIR, "checks.json"), "template design checks")
    if not ck:
        return
    off = []
    for c in ck["checks"]:
        if c["name"].startswith("same montage, best split = 1:1 ("):
            t = c["name"].rsplit("(", 1)[1].rstrip(")")
            sp, gain = c["detail"].replace("best split ", "").split(", gain ")
            V[f"td.{t}.same_split"], V[f"td.{t}.same_split_gain"] = float(sp), float(gain.rstrip("%"))
            from fractions import Fraction                   # 0.667 -> "2:3", as values_v22's ratio labels
            fr = Fraction(float(sp)).limit_denominator(12)
            V[f"td.{t}.same_split_label"] = f"{fr.numerator}:{fr.denominator}"
            if not c["passed"]:
                off.append(t)
    V["td.split_not_1to1"] = ", ".join(off) or "none"
    V["td.failed_checks"] = "; ".join(f"{c['name']}: {c['detail']}" for c in ck["checks"] if not c["passed"]) or "none"
    # steering against the two-pair 10-10 montage of the same variant, over the steered targets
    tc = [V[k] for k in V if k.startswith("td.") and k.endswith((".hdti.T_change", ".hdti_noinf.T_change"))]
    if tc:
        V["td.steer.T_change_lo"], V["td.steer.T_change_hi"] = min(tc), max(tc)
    st = [k.split(".")[1] for k in V if k.startswith("td.") and k.endswith(".hdti.focus_pct")
          and f"td.{k.split('.')[1]}.1010.focus_pct" in V]
    V["td.steer.n"] = len(st)
    V["td.steer.share_up"] = sum(V[f"td.{t}.hdti.focus_pct"] > V[f"td.{t}.1010.focus_pct"] for t in st)
    V["td.steer.focus_smaller"] = sum(V[f"td.{t}.hdti.focus_cm3"] < V[f"td.{t}.1010.focus_cm3"] for t in st)


def derived(V, T):
    """Counts behind sentences of the record, so no direction is typed by hand."""
    b = {t: (V.get(f"mm.2mm.{t}.relbias"), V.get(f"mm.1mm.{t}.relbias")) for t in MM_ORDER}
    if all(x is not None and y is not None for x, y in b.values()):
        V["mm.n_bias_closer_1mm"] = sum(abs(y) < abs(x) for x, y in b.values())
        V["mm.bias_not_closer"] = ", ".join(MM_NICE[t].split(" (")[0].lower() for t, (x, y) in b.items()
                                            if abs(y) >= abs(x)) or "none"
        V["mm.n_ccc_higher_1mm"] = sum(V[f"mm.1mm.{t}.ccc"] > V[f"mm.2mm.{t}.ccc"] for t in MM_ORDER)
        V["mm.n_r_higher_1mm"] = sum(V[f"mm.1mm.{t}.r"] > V[f"mm.2mm.{t}.r"] for t in MM_ORDER)
    V["crit.fem_met"] = sum(bool(V.get(f"crit.fem.{t}")) for t in MM_ORDER)
    # range of the mean bias over the three montages other than the inferior chain
    for tag in ("2mm", "1mm"):
        up = [V[f"mm.{tag}.{t}.relbias"] for t in MM_ORDER if t != "INFERIOR" and f"mm.{tag}.{t}.relbias" in V]
        if up:
            V[f"mm.{tag}.upper_lo"], V[f"mm.{tag}.upper_hi"] = min(up), max(up)
    V["conv.step21_all_negative"] = all(V.get(f"conv.step21.{r}", 0) < 0 for r in ROIS)
    # the accepted criterion "individual 5 mm selection unreliable" (values_v22.criteria), per target
    V["sweep.own5_unreliable"] = sum(V[f"sweep.{r}.own5.median"] > 2 or V[f"sweep.{r}.own5.max"] > 10 for r in ROIS)
    # target T at 5 mm against each head's 1 mm solution (means over the four heads)
    V["conv.exc5_lo"] = min(V[f"conv.exc_5mm.{r}"] for r in ROIS)
    V["conv.exc5_hi"] = max(V[f"conv.exc_5mm.{r}"] for r in ROIS)


def datasets(V, T):
    """Table 1: values_v22.datasets with the v23 slab grids and the 300-head network sizes of the full-space
    comparison (the v22 table showed the eight FEM heads' sizes in that row)."""
    V2.datasets(V, T)

    def size(n):
        return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"
    fs = ("; ".join(f"{size(V[f'fs300.t.{m}.nodes_med'])} ({size(V[f'fs300.t.{m}.nodes_lo'])}–"
                    f"{size(V[f'fs300.t.{m}.nodes_hi'])}) at {g}" for m, g in (("sp2", "2 mm"), ("sp1", "1 mm")))
          if "fs300.t.sp1.nodes_med" in V else "—")
    new = {"| Analytic disk; layered slab |": ["Analytic disk; layered slab", "—", "5–1 mm; 3–1 mm", "—", "—", "3.1"],
           "| Full-space comparison with FEM |": ["Full-space comparison with FEM", str(V.get("fs.n_heads", "—")),
                                                  "2 and 1 mm", "11,628", fs, "3.2, S11"]}
    out = []
    for r in T["datasets"].split("\n"):
        k = next((k for k in new if r.startswith(k)), None)
        out.append("| " + " | ".join(new.pop(k)) + " |" if k else r)
    if new:
        MISSING.append(f"datasets: rows not found in the v22 table: {list(new)}")
    T["datasets"] = "\n".join(out)


def values():
    del MISSING[:]
    V, T = {"model": config.MODEL, "model.describe": config.describe(),
            "res_root": os.path.relpath(RES, ROOT)}, {}
    for fn in (V2.cohort, verification, V2.fem_qc, V2.fem, V2.sweep, bench, V2.scale, V2.optima, V2.mni,
               V2.circuit, network_change, fullsearch_v23, contact_change, datasets, supplement, model_change,
               V2.criteria, template_design, derived):
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
