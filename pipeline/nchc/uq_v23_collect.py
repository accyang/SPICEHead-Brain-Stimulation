# -*- coding: utf-8 -*-
"""Collect the v23 uncertainty analysis (plan: results_v23/v23_uq/ANALYSIS_PLAN.md), 24 Oct 2026.

Input: L:/BrainProbe/SPICEhead/nchc_fullsearch/uq_v23/runs/<code>/*.npz (fetched and verified from Nano 4).
Analyses, as pre-specified:
  1 nominal check against the 300-head homogenized run (L:/.../homog300/<code>__sp1hom.json)
  2 exposure uncertainty at the cohort montage: within-head spread of T per scenario group; share of the total
    variance of log T (within-head uncertainty vs between-head anatomy)
  3 decision robustness: nominal winner kept; retention of the nominal winner and of the cohort montage
  4 conductivity attribution: standardized regression coefficients of log T and log F at the cohort montage on the
    six log factors (per head and target, then medians)
  5 safety outputs: peak and 99.9th-percentile GM envelope, peak single-channel |E| per mA
  6 age: within-head conductivity spread by age decade (aggregates only; demographics read in memory)
Output: results_v23/v23_uq/ (h-codes only; no demographics per head).
    python pipeline/nchc/uq_v23_collect.py [--runs DIR] [--out DIR]
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\uq_v23\runs"
HOM = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\homog300"
OUT = os.path.join(ROOT, "results_v23", "v23_uq")
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read in memory only
COHORT = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"          # private; read in memory only
ROI = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]
GROUPS = {"conductivity": "cond_", "placement (independent 5 mm)": "jit_", "placement (cap rotation 10 mm)": "rot_", "segmentation (0.5 mm layer)": "seg_"}
TISSUES = ["scalp", "compact_bone", "spongy_bone", "csf", "gray_matter", "white_matter"]


def load_head(d):
    keys = json.load(open(os.path.join(d, "montages.json")))
    runs = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz") and not f.endswith(".part.npz"):
            with np.load(os.path.join(d, f)) as z:
                runs[f[:-4]] = dict(F=z["F"].astype(np.float64), T=z["T"].astype(np.float64), O=z["O"].astype(np.float64),
                                    peaks=z["peaks"].astype(np.float64), fixed=list(z["fixed"]), params=json.loads(str(z["params"])))
    return keys, runs


def q(a, p):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else float("nan")


def main():
    runs_dir = sys.argv[sys.argv.index("--runs") + 1] if "--runs" in sys.argv else RUNS
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUT      # tables and summary
    fixed_json = json.load(open(os.path.join(OUT, "uq_fixed_montages.json")))["montages"]
    heads = sorted(c for c in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, c)))
    t0 = time.time()
    rows, attr, safety, checks = [], [], [], []
    var_parts = {(g, r): dict(within=[], nominal=[]) for g in GROUPS for r in ROI}
    spread_by_head = {}
    for code in heads:
        keys, R = load_head(os.path.join(runs_dir, code))
        if "nominal" not in R:
            continue
        nom = R["nominal"]; idx = {k: i for i, k in enumerate(keys)}
        # 1 nominal check
        p = os.path.join(HOM, f"{code}__sp1hom.json")
        if os.path.exists(p):
            ref = json.load(open(p))["rois"]
            Fr = np.array([[ref[r][k][0] for k in keys] for r in ROI])
            checks.append(dict(code=code, F_max_rel_diff=float(np.abs(nom["F"] - Fr).max() / np.abs(Fr).max()),
                               same_winner=int(sum(nom["F"][t].argmax() == Fr[t].argmax() for t in range(6)))))
        cm = [idx[fixed_json[r]] for r in ROI]                     # cohort montage per target
        wn = [int(nom["F"][t].argmax()) for t in range(6)]          # nominal winner per target
        for g, pre in GROUPS.items():
            S = [v for n, v in R.items() if n.startswith(pre)]
            if not S:
                continue
            for t, r in enumerate(ROI):
                Tc = np.array([s["T"][t][cm[t]] for s in S]); Tn = nom["T"][t][cm[t]]
                ratio = Tc / Tn
                kept = np.mean([s["F"][t].argmax() == wn[t] for s in S])
                ret_w = np.array([s["F"][t][wn[t]] / s["F"][t].max() for s in S])
                ret_c = np.array([s["F"][t][cm[t]] / s["F"][t].max() for s in S])
                rows.append(dict(code=code, scenario=g, target=r, n_runs=len(S), T_nominal_cohort_Vm_per_mA=round(Tn, 5),
                                 T_ratio_p5=round(q(ratio, 5), 4), T_ratio_median=round(q(ratio, 50), 4), T_ratio_p95=round(q(ratio, 95), 4),
                                 logT_sd=round(float(np.std(np.log(Tc), ddof=1)) if len(S) > 1 else float("nan"), 4),
                                 F_ratio_median=round(q(np.array([s["F"][t][cm[t]] for s in S]) / nom["F"][t][cm[t]], 50), 4),
                                 winner_kept_pct=round(100 * kept, 1), nominal_winner_retention_median=round(q(ret_w, 50), 4),
                                 nominal_winner_retention_min=round(float(ret_w.min()), 4), cohort_retention_nominal=round(float(nom["F"][t][cm[t]] / nom["F"][t].max()), 4),
                                 cohort_retention_median=round(q(ret_c, 50), 4), cohort_retention_min=round(float(ret_c.min()), 4)))
                if len(S) > 1:
                    var_parts[(g, r)]["within"].append(float(np.var(np.log(Tc), ddof=1)))
                var_parts[(g, r)]["nominal"].append(float(np.log(Tn)))
                if g == "conductivity":
                    spread_by_head.setdefault(code, []).append(float(np.std(np.log(Tc), ddof=1)) if len(S) > 1 else np.nan)
            # 4 attribution
            if g == "conductivity" and len(S) >= 10:
                Z = np.array([[s["params"]["log_factors"][k] for k in TISSUES] for s in S])
                Zs = (Z - Z.mean(0)) / Z.std(0, ddof=1)
                for t, r in enumerate(ROI):
                    for qty in ("T", "F"):
                        y = np.log(np.array([s[qty][t][cm[t]] for s in S])); ys = (y - y.mean()) / y.std(ddof=1)
                        X = np.c_[np.ones(len(S)), Zs]; beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
                        r2 = 1 - np.sum((ys - X @ beta) ** 2) / np.sum(ys ** 2)
                        attr.append(dict(code=code, target=r, quantity=qty, R2=round(float(r2), 4), **{k: round(float(b), 4) for k, b in zip(TISSUES, beta[1:])}))
        # 5 safety (cohort montages: first 6 fixed entries)
        allr = [v for n, v in R.items()]
        for t, r in enumerate(ROI):
            pk = np.array([v["peaks"][t] for v in allr])          # rows: runs; cols: env_max, env_p999, ch1, ch2
            safety.append(dict(code=code, target=r, montage=fixed_json[r], env_max_nominal=round(nom["peaks"][t][0], 4),
                               env_max_p95_all_runs=round(q(pk[:, 0], 95), 4), env_p999_nominal=round(nom["peaks"][t][1], 4),
                               env_p999_p95_all_runs=round(q(pk[:, 1], 95), 4), channel_absE_max_nominal=round(float(max(nom["peaks"][t][2:4])), 4),
                               channel_absE_max_p95_all_runs=round(q(pk[:, 2:4].max(1), 95), 4)))
    os.makedirs(out, exist_ok=True)

    def write(name, rr):
        if not rr:
            return
        ks = list(rr[0]); [ks.append(k) for r in rr for k in r if k not in ks]
        with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ks); w.writeheader(); w.writerows(rr)
    write("per_head_target.csv", rows); write("attribution_conductivity.csv", attr); write("safety_peaks.csv", safety); write("nominal_check.csv", checks)
    # 2 variance decomposition: within-head uncertainty vs between-head anatomy (log T at the cohort montage)
    var = []
    for (g, r), d in var_parts.items():
        if len(d["within"]) >= 2 and len(d["nominal"]) >= 2:
            w, b = float(np.mean(d["within"])), float(np.var(d["nominal"], ddof=1))
            var.append(dict(scenario=g, target=r, heads=len(d["within"]), sd_within=round(np.sqrt(w), 4), sd_between_heads=round(np.sqrt(b), 4),
                            uncertainty_share_pct=round(100 * w / (w + b), 1)))
    write("variance_decomposition.csv", var)
    # 6 age (aggregates)
    age_rows = []
    if os.path.exists(CODEMAP) and os.path.exists(COHORT) and spread_by_head:
        cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
        coh = {r["CaseID"]: r for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))}
        dec = {}
        for c, v in spread_by_head.items():
            dec.setdefault(coh[cmap[c]]["Decade"], []).append(float(np.nanmedian(v)))
        age_rows = [dict(decade=k, heads=len(v), median_logT_sd_conductivity=round(float(np.median(v)), 4)) for k, v in sorted(dec.items())]
        write("age_conductivity_spread.csv", age_rows)

    # ---------- summary
    nh = len({r["code"] for r in rows})
    L = [f"# SPICEhead v23 uncertainty analysis: {nh} heads", "",
         f"Collected {time.strftime('%Y-%m-%d %H:%M')} by `pipeline/nchc/uq_v23_collect.py`. The plan, written before the runs, is `ANALYSIS_PLAN.md`.",
         "Model v23 at 1 mm, currents 1 mA per channel, six targets, all 11,628 montages. The cohort montage is the v23 cohort winner of the 300-head comparison.",
         "T and the peaks are in V/m per mA per channel. Ratios are to the head's nominal run.", "",
         "This is a new result set. It is not in any Results record or manuscript.", ""]
    if checks:
        L += ["## 1. Nominal check", "",
              f"The nominal run matches the 300-head homogenized run in {sum(c['F_max_rel_diff'] < 1e-6 and c['same_winner'] == 6 for c in checks)} of {len(checks)} heads: "
              f"largest F difference {max(c['F_max_rel_diff'] for c in checks):.1e}, same winner in {sum(c['same_winner'] for c in checks)}/{6*len(checks)} head-targets.", ""]
    L += ["## 2. Exposure at the cohort montage", "",
          "| Scenario | Target | T ratio 5th–95th pct (median over heads) | SD of log T within head | SD of log T between heads | Uncertainty share of variance |", "|---|---|---|---|---|---|"]
    for v in var:
        sub = [r for r in rows if r["scenario"] == v["scenario"] and r["target"] == v["target"]]
        L.append(f"| {v['scenario']} | {v['target']} | {q([r['T_ratio_p5'] for r in sub], 50):.2f}–{q([r['T_ratio_p95'] for r in sub], 50):.2f} | "
                 f"{v['sd_within']:.3f} | {v['sd_between_heads']:.3f} | {v['uncertainty_share_pct']:.0f} % |")
    L += ["", "## 3. Decision robustness", "",
          "| Scenario | Nominal winner kept (median over head-targets) | Nominal winner's retention: median / 5th pct / min | Cohort montage's retention: nominal → median under perturbation (5th pct of head-targets) |", "|---|---|---|---|"]
    for g in GROUPS:
        sub = [r for r in rows if r["scenario"] == g]
        if sub:
            L.append(f"| {g} | {q([r['winner_kept_pct'] for r in sub], 50):.0f} % | {q([r['nominal_winner_retention_median'] for r in sub], 50):.3f} / "
                     f"{q([r['nominal_winner_retention_median'] for r in sub], 5):.3f} / {min(r['nominal_winner_retention_min'] for r in sub):.3f} | "
                     f"{q([r['cohort_retention_nominal'] for r in sub], 50):.3f} → {q([r['cohort_retention_median'] for r in sub], 50):.3f} ({q([r['cohort_retention_median'] for r in sub], 5):.3f}) |")
    if attr:
        L += ["", "## 4. Conductivity attribution (standardized regression coefficients, median over heads and targets)", "",
              "| Quantity | " + " | ".join(TISSUES) + " | R² |", "|---|" + "---|" * (len(TISSUES) + 1)]
        for qty in ("T", "F"):
            sub = [a for a in attr if a["quantity"] == qty]
            L.append(f"| log {qty} | " + " | ".join(f"{q([a[k] for a in sub], 50):+.2f}" for k in TISSUES) + f" | {q([a['R2'] for a in sub], 50):.2f} |")
    if safety:
        L += ["", "## 5. Safety outputs at the cohort montages (V/m per mA per channel)", "",
              "| Target | Envelope peak in GM: median nominal / 95th pct over all runs and heads | Envelope 99.9th pct | Peak single-channel abs(E) |", "|---|---|---|---|"]
        for r in ROI:
            sub = [s for s in safety if s["target"] == r]
            L.append(f"| {r} | {q([s['env_max_nominal'] for s in sub], 50):.3f} / {q([s['env_max_p95_all_runs'] for s in sub], 95):.3f} | "
                     f"{q([s['env_p999_nominal'] for s in sub], 50):.3f} / {q([s['env_p999_p95_all_runs'] for s in sub], 95):.3f} | "
                     f"{q([s['channel_absE_max_nominal'] for s in sub], 50):.3f} / {q([s['channel_absE_max_p95_all_runs'] for s in sub], 95):.3f} |")
    if age_rows:
        L += ["", "## 6. Age (conductivity uncertainty, SD of log T within head, median over targets and heads)", "",
              "| Decade | Heads | SD of log T |", "|---|---|---|"] + [f"| {a['decade']} | {a['heads']} | {a['median_logT_sd_conductivity']:.3f} |" for a in age_rows]
    open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    json.dump(dict(collected=time.strftime("%Y-%m-%d %H:%M"), runs=runs_dir, heads=heads, n_heads_with_rows=nh, script="pipeline/nchc/uq_v23_collect.py"),
              open(os.path.join(out, "PROVENANCE.json"), "w"), indent=1)
    print("\n".join(L)); print(f"\n({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
