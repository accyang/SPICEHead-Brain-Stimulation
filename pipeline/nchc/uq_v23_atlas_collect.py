# -*- coding: utf-8 -*-
"""Collect the atlas extension of the v23 uncertainty analysis (amendment 1 of results_v23/v23_uq/ANALYSIS_PLAN.md).

Input: L:/BrainProbe/SPICEhead/nchc_fullsearch/uq_v23/runs_atlas/<code>/ (fetched and verified), optionally the
six-target runs (uq_v23/runs) for the ref6 consistency check.
Per target, over the heads present (35 on 24 Oct; 300 after the 31 Oct extension) and the perturbed runs per head:
  exposure at the FEM cohort choice (fixed protocol): T ratio to nominal (5th-95th pct), SD of log T within head,
  share of the variance (uncertainty vs between-head anatomy);
  fixed-protocol retention: F at the FEM cohort choice / the perturbed head's constrained optimum F; share of runs >= 0.90;
  decision robustness: reported choice kept, retention of the nominal reported choice;
  conductivity attribution: standardized regression coefficients of log T at the FEM cohort choice;
  peaks at the FEM cohort choice.
Robust-for-a-fixed-protocol criterion (amendment 1): retention >= 0.90 in >= 95 % of all perturbed runs of all heads.
Output: results_v23/v23_uq/atlas/ (aggregates per target; h-codes only in per_head files).
    python pipeline/nchc/uq_v23_atlas_collect.py [--runs DIR] [--six DIR] [--out DIR]
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\uq_v23\runs_atlas"
SIX = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\uq_v23\runs"
OUT = os.path.join(ROOT, "results_v23", "v23_uq", "atlas")
# cerebellum and brainstem (25 Oct): the FEM cohort choices of the 300-head extension, evaluated after the runs from
# their stored float16 F_new/T_new; montage labels map to run indices through the extension's montage list
FIXEXT = os.path.join(ROOT, "results_v23", "v23_uq", "uq_atlas_fixed_montages_ext.json")
EXTMONT = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\atlas_ext\out\h001__ext.npz"
GROUPS = {"conductivity": "cond_", "placement, independent 5 mm": "jit_", "placement, cap rotation 10 mm": "rot_", "segmentation, 0.5 mm layer": "seg_",
          "target mask, 1 voxel": "mask_"}                 # 31 Oct 2026 (amendment 2): erosion/dilation of the target masks
ROBUST_GROUPS = [g for g in GROUPS if g != "target mask, 1 voxel"]   # the pre-specified robustness criterion uses the four physical scenarios
TISSUES = ["scalp", "compact_bone", "spongy_bone", "csf", "gray_matter", "white_matter"]
RET = 0.90
TAG = {"conductivity": "cond", "placement, independent 5 mm": "jit", "placement, cap rotation 10 mm": "rot", "target mask, 1 voxel": "mask",
       "segmentation, 0.5 mm layer": "seg"}


def q(a, p):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else float("nan")


def main():
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    runs_dir, six_dir, out = arg("--runs", RUNS), arg("--six", SIX), arg("--out", OUT)
    heads = sorted(c for c in os.listdir(runs_dir) if os.path.exists(os.path.join(runs_dir, c, "nominal.npz")))
    t0 = time.time()
    meta0 = json.load(open(os.path.join(runs_dir, heads[0], "targets.json")))
    names, tiers = meta0["targets"], meta0["tiers"]
    NT = len(names)
    newset = set(meta0["new_targets"])
    ext = {}                                                  # target index -> run montage index of its FEM cohort choice
    if "--no-ext" not in sys.argv and os.path.exists(FIXEXT):
        with np.load(EXTMONT) as z:
            mi = {m: i for i, m in enumerate(z["montages"])}
        with np.load(os.path.join(runs_dir, heads[0], "nominal.npz")) as z:
            fidx = z["fixed_idx"][:, 0]
        for t, lab in enumerate(meta0["fem_choice"]):         # same montage order as the runs
            if lab and t not in newset:
                assert mi[lab] == fidx[t], (names[t], lab)
        fx = json.load(open(FIXEXT, encoding="utf-8"))["targets"]
        ext = {t: mi[fx[names[t]]["cohort_choice"]] for t in newset if names[t] in fx}
    acc = {t: dict(logT_nom=[], within_var={g: [] for g in GROUPS}, ratio={g: [] for g in GROUPS}, ret={g: [] for g in GROUPS},
                   ret_nom=[], kept={g: [] for g in GROUPS}, ret_rep={g: [] for g in GROUPS}, src=[], peak_nom=[], peak_all=[], n_heads=0)
           for t in range(NT)}
    ref6 = []
    for code in heads:
        d = os.path.join(runs_dir, code)
        meta = json.load(open(os.path.join(d, "targets.json")))
        assert meta["targets"] == names, code
        R = {}
        for f in os.listdir(d):
            if f.endswith(".npz") and not f.endswith(".part.npz"):
                with np.load(os.path.join(d, f)) as z:
                    R[f[:-4]] = dict(idx=z["idx"], vals=z["vals"].astype(np.float64), fv=z["fixed_vals"].astype(np.float64),
                                     peaks=z["peaks"].astype(np.float64), params=json.loads(str(z["params"])))
                    if "peaks_new" in z.files and len(z["new_targets"]):
                        R[f[:-4]]["peaks"][z["new_targets"]] = z["peaks_new"]
                    if ext and "F_new" in z.files:
                        Fn, Tn_ = z["F_new"].astype(np.float64), z["T_new"].astype(np.float64)
                        for j, t in enumerate(z["new_targets"]):
                            m = ext.get(int(t))
                            if m is not None and Fn[j, m] > 0:
                                R[f[:-4]]["fv"][t, 0] = [Fn[j, m], Tn_[j, m], Tn_[j, m] / Fn[j, m]]
        nom = R["nominal"]
        # ref6 consistency with the six-target run (same perturbations): max F per ref6 target
        sd = os.path.join(six_dir, code)
        if os.path.isdir(sd):
            ROI6 = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]
            for n, r in R.items():
                p6 = os.path.join(sd, f"{n}.npz")
                if os.path.exists(p6):
                    with np.load(p6) as z6:
                        F6 = z6["F"]
                    for j, roi in enumerate(ROI6):
                        t = names.index(f"ref6:{roi}")
                        ref6.append(abs(r["vals"][t, 0, 0] - F6[j].max()) / F6[j].max())
        for t in range(NT):
            # fixed montage: the FEM cohort choice of the 300-head atlas; regions without one yet (cerebellum and
            # brainstem, added 24 Oct) use the head's own nominal reported choice (column 3) and are marked as such
            fc = 0 if np.isfinite(nom["fv"][t, 0, 1]) else 3
            acc[t]["fixed"] = ("FEM cohort choice (300-head extension)" if t in ext else "FEM cohort choice") if fc == 0 else "own nominal choice"
            if meta["n_roi"][t] == 0 or not np.isfinite(nom["fv"][t, fc, 1]) or nom["fv"][t, fc, 1] <= 0:
                continue
            A = acc[t]; A["n_heads"] += 1
            Tn = nom["fv"][t, fc, 1]; A["logT_nom"].append(np.log(Tn))
            A["ret_nom"].append(nom["fv"][t, fc, 0] / nom["vals"][t, 1, 0])
            A["peak_nom"].append(nom["peaks"][t, 0])
            for g, pre in GROUPS.items():
                S = [v for n, v in R.items() if n.startswith(pre)]
                if not S:
                    continue
                Tc = np.array([s["fv"][t, fc, 1] for s in S])
                A["ratio"][g] += list(Tc / Tn)
                if len(S) > 1:
                    A["within_var"][g].append(float(np.var(np.log(Tc), ddof=1)))
                A["ret"][g] += [s["fv"][t, fc, 0] / s["vals"][t, 1, 0] for s in S]
                A["kept"][g] += [float(s["idx"][t, 2] == nom["idx"][t, 2]) for s in S]
                A["ret_rep"][g] += [s["fv"][t, 3, 0] / s["vals"][t, 1, 0] for s in S]
                A["peak_all"] += [s["peaks"][t, 0] for s in S]
                if g == "conductivity" and len(S) >= 10:
                    Z = np.array([[s["params"]["log_factors"][k] for k in TISSUES] for s in S]); Zs = (Z - Z.mean(0)) / Z.std(0, ddof=1)
                    y = np.log(Tc); ys = (y - y.mean()) / y.std(ddof=1)
                    X = np.c_[np.ones(len(S)), Zs]; beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
                    A["src"].append(dict(zip(TISSUES, beta[1:]), R2=1 - np.sum((ys - X @ beta) ** 2) / np.sum(ys ** 2)))
    rows, srows = [], []
    for t in range(NT):
        A = acc[t]
        if A["n_heads"] < 2:
            continue
        b = float(np.var(A["logT_nom"], ddof=1))
        allret = np.concatenate([np.asarray(A["ret"][g]) for g in ROBUST_GROUPS if A["ret"][g]])
        row = dict(target=names[t], tier=tiers[t], fixed_montage=A.get("fixed", "FEM cohort choice"), heads=A["n_heads"], T_nominal_median_Vm_per_mA=round(float(np.exp(np.median(A["logT_nom"]))), 5),
                   sd_logT_between_heads=round(np.sqrt(b), 4), fixed_retention_nominal_median=round(q(A["ret_nom"], 50), 4),
                   fixed_retention_all_runs_p5=round(q(allret, 5), 4), share_runs_retention_ge_90pct=round(100 * float(np.mean(allret >= RET)), 1),
                   fixed_protocol_robust=bool(np.mean(allret >= RET) >= 0.95),
                   peak_env_nominal_median=round(q(A["peak_nom"], 50), 4), peak_env_all_runs_p95=round(q(A["peak_all"], 95), 4),
                   peaks_at="own nominal choice" if t in newset else "fixed montage")
        for g in GROUPS:
            if not A["ratio"][g]:
                continue
            w = float(np.mean(A["within_var"][g])) if A["within_var"][g] else float("nan")
            tag = TAG[g]
            row.update({f"{tag}_T_ratio_p5": round(q(A["ratio"][g], 5), 4), f"{tag}_T_ratio_p95": round(q(A["ratio"][g], 95), 4),
                        f"{tag}_sd_logT_within": round(np.sqrt(w), 4), f"{tag}_uncertainty_share_pct": round(100 * w / (w + b), 1),
                        f"{tag}_reported_kept_pct": round(100 * float(np.mean(A["kept"][g])), 1),
                        f"{tag}_nominal_choice_retention_p5": round(q(A["ret_rep"][g], 5), 4),
                        f"{tag}_fixed_retention_p5": round(q(A["ret"][g], 5), 4)})
        rows.append(row)
        if A["src"]:
            srows.append(dict(target=names[t], tier=tiers[t], heads=len(A["src"]), **{k: round(float(np.median([s[k] for s in A["src"]])), 3) for k in TISSUES + ["R2"]}))
    os.makedirs(out, exist_ok=True)

    def write(name, rr):
        ks = list(rr[0]); [ks.append(k) for r in rr for k in r if k not in ks]
        with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ks); w.writeheader(); w.writerows(rr)
    write("per_target.csv", rows)
    if srows:
        write("attribution_by_target.csv", srows)

    # ---------- summary
    tag_new = "" if ext else "; own nominal choice"
    tierset = [("all regions", lambda r: True), ("AAL90", lambda r: r["tier"] == "tier1"), ("tier 2", lambda r: r["tier"] == "tier2"), ("Paper A spheres", lambda r: r["tier"] == "ref6"),
               (f"cerebellum (AAL3{tag_new})", lambda r: r["tier"] == "cerebellum"), (f"brainstem nuclei (AAL3{tag_new})", lambda r: r["tier"] == "brainstem"),
               ("thalamic nuclei (AAL3)", lambda r: r["tier"] == "thalamus"), ("ACC_sup (AAL3)", lambda r: r["tier"] == "cingulate")]
    new_text = (f"Cerebellar and brainstem regions ({len(ext)}) use the FEM cohort choice of the 300-head extension (`../uq_atlas_fixed_montages_ext.json`), "
                "evaluated after the runs from their stored float16 scores (about 3 significant digits); their envelope peaks are at each head's own nominal choice."
                if ext else "Cerebellar and brainstem regions have no FEM cohort choice yet: their rows use each head's own nominal choice, so they measure decision robustness, not a fixed protocol."
                if newset else "The thalamic-nucleus and ACC_sup targets use the FEM cohort choice of the 300-head thalamic extension (`uq_atlas_fixed_montages_thal.json`), applied during the runs."
                if any(tr in ("thalamus", "cingulate") for tr in tiers) else "")
    scope = "over all brain-region targets" if "tier1" in tiers else "for the AAL3 thalamic nuclei and ACC_sup (with the six Paper A spheres)"
    L = [f"# SPICEhead v23 uncertainty analysis {scope}: {len(heads)} heads, {len(rows)} targets",
         "", f"Collected {time.strftime('%Y-%m-%d %H:%M')} by `pipeline/nchc/uq_v23_atlas_collect.py` (amendment 1 of `../ANALYSIS_PLAN.md`).",
         "The fixed protocol is the FEM cohort choice of the 300-head atlas (constrained optimum, top band, accessibility). Retention = its F as a share of the perturbed head's constrained optimum. " + new_text,
         "T is in V/m per mA per channel.", "", "This is a new result set. It is not in any Results record or manuscript.", ""]
    if ref6:
        L += [f"Consistency: the six Paper A spheres scored here match the six-target run in {len(ref6)} run-targets (largest relative difference in max F {max(ref6):.1e}).", ""]
    L += ["## Summary by target set (medians over targets)", "",
          "| Set | Targets | Conductivity: T ratio 5th–95th pct | Conductivity share of variance | Placement (5 mm) share | Segmentation share | Fixed-protocol retention, 5th pct of runs | Robust for a fixed protocol |", "|---|---|---|---|---|---|---|---|"]
    for lab, f in tierset:
        sub = [r for r in rows if f(r)]
        if not sub:
            continue
        L.append(f"| {lab} | {len(sub)} | {q([r.get('cond_T_ratio_p5') for r in sub], 50):.2f}–{q([r.get('cond_T_ratio_p95') for r in sub], 50):.2f} | "
                 f"{q([r.get('cond_uncertainty_share_pct') for r in sub], 50):.0f} % | {q([r.get('jit_uncertainty_share_pct') for r in sub], 50):.0f} % | "
                 f"{q([r.get('seg_uncertainty_share_pct') for r in sub], 50):.0f} % | {q([r['fixed_retention_all_runs_p5'] for r in sub], 50):.3f} | "
                 f"{sum(r['fixed_protocol_robust'] for r in sub)}/{len(sub)} |")
    rob = sorted(rows, key=lambda r: r["share_runs_retention_ge_90pct"])
    L += ["", "## Least robust targets for a fixed protocol (share of runs keeping ≥ 90 % of the head's constrained optimum)", "",
          "| Target | Tier | Share of runs ≥ 90 % | Retention 5th pct | Nominal retention (anatomy alone), median | Conductivity T ratio 5th–95th pct |", "|---|---|---|---|---|---|"]
    for r in rob[:12]:
        L.append(f"| {r['target']} | {r['tier']} | {r['share_runs_retention_ge_90pct']:.1f} % | {r['fixed_retention_all_runs_p5']:.3f} | {r['fixed_retention_nominal_median']:.3f} | "
                 f"{r.get('cond_T_ratio_p5', float('nan')):.2f}–{r.get('cond_T_ratio_p95', float('nan')):.2f} |")
    if srows:
        L += ["", "## Conductivity attribution for log T at the fixed protocol (median standardized coefficient over targets)", "",
              "| Set | " + " | ".join(TISSUES) + " | R² |", "|---|" + "---|" * (len(TISSUES) + 1)]
        for lab, f in tierset:
            sub = [s for s in srows if f(s)]
            if sub:
                L.append(f"| {lab} | " + " | ".join(f"{q([s[k] for s in sub], 50):+.2f}" for k in TISSUES) + f" | {q([s['R2'] for s in sub], 50):.2f} |")
    L += ["", "## Peak grey-matter envelope at the fixed protocol (V/m per mA per channel)", "",
          f"Median over targets: nominal {q([r['peak_env_nominal_median'] for r in rows], 50):.2f}; 95th percentile over all runs and heads {q([r['peak_env_all_runs_p95'] for r in rows], 50):.2f}; "
          f"highest target {max(rows, key=lambda r: r['peak_env_all_runs_p95'])['target']} ({max(r['peak_env_all_runs_p95'] for r in rows):.2f}).",
          "", "Per-target values for every region and scenario: `per_target.csv`; conductivity coefficients per region: `attribution_by_target.csv`."]
    open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    json.dump(dict(collected=time.strftime("%Y-%m-%d %H:%M"), runs=runs_dir, heads=heads, targets=len(rows), script="pipeline/nchc/uq_v23_atlas_collect.py"),
              open(os.path.join(out, "PROVENANCE.json"), "w"), indent=1)
    print("\n".join(L)); print(f"\n({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
