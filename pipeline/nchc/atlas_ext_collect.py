# -*- coding: utf-8 -*-
"""Collect the cerebellum and brainstem extension of the 300-head atlas (24 Oct 2026).

Input: L:/BrainProbe/SPICEhead/nchc_fullsearch/atlas_ext/out/<code>__ext.npz (fetched and verified from Nano 4):
F and T for 6 models (fem, sp2, sp1, sp5 = v22; sp2v23, sp1v23 = v23) x 44 targets (6 ref6 spheres, 26 AAL3 cerebellar
regions, 12 AAL3 brainstem nuclei) x 11,628 montages.
1. Regression check: the ref6 maxima per head reproduce the published scores (fem/sp2/sp1: heads/<code>/<code>__<m>.json;
   sp5: targets_out_sp5; v23: homog300/<code>__sp{2,1}hom.json).
2. Per target and model, the rule of the v2 atlas (collect_targets_v2.choose: constrained optimum T >= 80 % of the
   head's best, 3 % top band, most easy electrodes): per-head choice, cohort choice on the cohort medians, personalization
   gain, agreement with FEM (same choice, FEM optimum kept by the model's choice, rank correlation), age (Spearman, BH-FDR).
3. FEM fixed-protocol choices of the new targets, for the uncertainty run (results_v23/v23_uq/uq_atlas_fixed_montages_ext.json).
Outputs (aggregates; demographics in memory only):
  results_v22/v22_target_atlas_300/extension_cerebellum_brainstem/  FEM and v22 models
  results_v23/v23_target_atlas_300_ext/                              FEM and v23 models
    python pipeline/nchc/atlas_ext_collect.py [--in DIR]
--set thal (25 Oct 2026; PI: "add the thalamic nuclei and ACC_sup on nano4") collects atlas_thal/out/<code>__thal.npz
(atlas_thal_score.py: ref6 + 7 AAL3 thalamic nuclei and ACC_sup, left and right, same 6 models) the same way, into
results_v22/v22_target_atlas_300/extension_thalamic_nuclei_acc_sup/, results_v23/v23_target_atlas_300_thal/ and
results_v23/v23_uq/uq_atlas_fixed_montages_thal.json.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline"), HERE]
from collect_targets_v2 import choose, bh, EASY            # noqa: E402  (the approved v2 rule, unchanged)

L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
IN = os.path.join(L, "atlas_ext", "out")
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only
COHORT = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"          # private; read only
OUT22 = os.path.join(ROOT, "results_v22", "v22_target_atlas_300", "extension_cerebellum_brainstem")
OUT23 = os.path.join(ROOT, "results_v23", "v23_target_atlas_300_ext")
ROI6 = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]
SETS = {
    "ext": dict(inp=IN, suffix="__ext.npz", out22=OUT22, out23=OUT23, fixed="uq_atlas_fixed_montages_ext.json",
                tiers=("cerebellum", "brainstem"), title="cerebellum and brainstem",
                text=["Cerebellum: 26 AAL3 regions on grey matter. Brainstem: 12 AAL3 nuclei on brain-tissue points (T over the nucleus, O over grey matter beyond 20 mm).",
                      "FEM: SimNIBS leadfield rerun with white and grey matter stored; its grey-matter mapping reproduces the published FEM basis.",
                      "The FEM regression difference (up to 4e-4 in 21 heads, no winner changed) is examined in `FEM_RERUN_CHECK.md` beside the v22 summary."]),
    "thal": dict(inp=os.path.join(L, "atlas_thal", "out"), suffix="__thal.npz",
                 out22=os.path.join(ROOT, "results_v22", "v22_target_atlas_300", "extension_thalamic_nuclei_acc_sup"),
                 out23=os.path.join(ROOT, "results_v23", "v23_target_atlas_300_thal"), fixed="uq_atlas_fixed_montages_thal.json",
                 tiers=("thalamus", "cingulate"), title="AAL3 thalamic nuclei and supracallosal ACC",
                 text=["Thalamus: 7 AAL3 nuclei (AV, LP, VA, VL, VPL, IL, LGN), left and right, on grey matter; Re and MGN are too small. "
                       "Cingulate: AAL3 ACC_sup, left and right.",
                       "FEM: the published FEM grey-matter basis of the 300-head comparison."]),
    # 26 Oct 2026 (PI: "score v23 on the 109 targets on nano4"): v23 scores of the 22 Oct atlas targets, joined per head
    # with that atlas's own FEM scores (targets_out) by target name
    "v23main": dict(inp=os.path.join(L, "atlas_v23", "out"), suffix="__v23atlas.npz", out22=None,
                    out23=os.path.join(ROOT, "results_v23", "v23_target_atlas_300"), fixed=None, fem_from=os.path.join(L, "targets_out"),
                    tiers=("tier1", "tier2", "ref6"), title="SPICEhead v23 on the targets of the 22 Oct atlas (AAL90, tier 2, Paper A spheres)",
                    text=["Targets: those of results_v22/v22_target_atlas_300 (score_targets_v2.build_targets); VTA_L/R have no grey matter.",
                          "FEM: that atlas's own FEM scores (targets_out), joined by target name; the target masks agree voxel for voxel in count."]),
}


def load_head(S, src, f):
    """One head's arrays; for a set with fem_from, the FEM scores are joined from the atlas by target name."""
    with np.load(os.path.join(src, f)) as z:
        d = dict(F=z["F"], T=z["T"], n_roi=z["n_roi"], models=list(z["models"]), targets=list(z["targets"]),
                 tiers=list(z["tiers"]), montages=list(z["montages"]))
    if S.get("fem_from"):
        with np.load(os.path.join(S["fem_from"], f"{f.split('__')[0]}__targets.npz")) as e:
            assert list(e["montages"]) == d["montages"], f
            en = list(e["targets"]); m = list(e["models"]).index("fem"); ix = [en.index(n) for n in d["targets"]]
            assert np.array_equal(e["n_roi"][ix], d["n_roi"]), f                # same target masks
            d["F"] = np.concatenate([e["F"][m][ix][None], d["F"]]); d["T"] = np.concatenate([e["T"][m][ix][None], d["T"]])
            d["models"] = ["fem"] + d["models"]
    return d


def ref6_reference(code, model, t_names, keys):
    """published max F per ref6 target for a model, or None"""
    out = {}
    if model in ("fem", "sp2", "sp1", "sp2v23", "sp1v23"):
        p = (os.path.join(L, "heads", code, f"{code}__{model}.json") if model in ("fem", "sp2", "sp1")
             else os.path.join(L, "homog300", f"{code}__{model[:3]}hom.json"))
        if not os.path.exists(p):
            return None
        r = json.load(open(p))["rois"]
        return {f"ref6:{roi}": max(v[0] for v in r[roi].values()) for roi in ROI6}
    if model == "sp5":
        p = os.path.join(L, "targets_out_sp5", f"{code}__targets.npz")
        if not os.path.exists(p):
            return None
        with np.load(p) as z:
            names = list(z["targets"]); m = list(z["models"]).index("sp5")
            return {n: float(z["F"][m][names.index(n)].max()) for n in names if n.startswith("ref6:")}
    return out


def main():
    S = SETS[sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "ext"]
    src = sys.argv[sys.argv.index("--in") + 1] if "--in" in sys.argv else S["inp"]
    files = sorted(f for f in os.listdir(src) if f.endswith(S["suffix"]))
    codes = [f.split("__")[0] for f in files]
    t0 = time.time()
    d0 = load_head(S, src, files[0])
    models, names, tiers, mont = d0["models"], d0["targets"], d0["tiers"], d0["montages"]
    NM, NT, H = len(mont), len(names), len(codes)
    easy = np.array([sum(e in EASY for p in m.split(" / ") for e in p.split("+")) for m in mont], np.float32)
    F = np.zeros((len(models), H, NT, NM), np.float32); T = np.zeros_like(F)
    nroi = np.zeros((H, NT), np.int32)
    reg = {m: [] for m in models}
    for hi, f in enumerate(files):
        d = load_head(S, src, f)
        assert d["targets"] == names and d["models"] == models and d["montages"] == mont, f
        F[:, hi], T[:, hi] = d["F"], d["T"]; nroi[hi] = d["n_roi"]
        for mi, m in enumerate(models):
            ref = ref6_reference(codes[hi], m, names, mont)
            if ref:
                for n, v in ref.items():
                    reg[m].append(abs(float(F[mi, hi, names.index(n)].max()) - v) / v)
        if (hi + 1) % 50 == 0:
            print(f"  loaded {hi+1}/{H} ({time.time()-t0:.0f} s)", flush=True)
    valid = nroi >= 5
    # per head choices under the v2 rule
    W = {}
    for mi, m in enumerate(models):
        W[m] = [choose(F[mi, hi], T[mi, hi], easy) for hi in range(H)]
    fe = models.index("fem")
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    crow = {r["CaseID"]: r for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))}
    age = np.array([float(crow[cmap[c]]["Age_Scan"]) for c in codes])
    rows, fixed = [], {}
    q = lambda a, p: float(np.nanpercentile(a, p)) if np.isfinite(np.asarray(a, float)).any() else float("nan")
    for t in range(NT):
        v = valid[:, t]
        if v.sum() < 10:
            continue
        for mi, m in enumerate(models):
            medF = np.median(F[mi, v, t], axis=0); medT = np.median(T[mi, v, t], axis=0)
            co = choose(medF[None], medT[None], easy)
            cc = int(co["wa"][0])
            wa = np.array([W[m][hi]["wa"][t] for hi in range(H)]); wc = np.array([W[m][hi]["wc"][t] for hi in range(H)])
            FC = F[mi, np.arange(H), t, wc]; Fcoh = F[mi, :, t, cc]
            gain = np.where(Fcoh > 0, FC / Fcoh, np.nan)
            r = dict(target=names[t], tier=tiers[t], model=m, heads=int(v.sum()), cohort_choice=mont[cc],
                     F_choice_median=round(q(F[mi, np.arange(H), t, wa][v], 50), 4), T_choice_median_Vm_per_mA=round(q(T[mi, np.arange(H), t, wa][v], 50), 5),
                     F_cohort_median=round(q(Fcoh[v], 50), 4), T_cohort_median_Vm_per_mA=round(q(T[mi, :, t, cc][v], 50), 5),
                     gain_median=round(q(gain[v], 50), 4), heads_gain_ge_10pct=round(100 * float(np.nanmean(gain[v] >= 1.10)), 1),
                     heads_F_above_1_pct=round(100 * float(np.mean(FC[v] > 1)), 1))
            if m != "fem":
                wf = np.array([W["fem"][hi]["wa"][t] for hi in range(H)]); wfc = np.array([W["fem"][hi]["wc"][t] for hi in range(H)])
                kept = 100 * F[fe, np.arange(H), t, wa] / F[fe, np.arange(H), t, wfc]
                rho = [spearmanr(F[fe, hi, t], F[mi, hi, t]).correlation for hi in np.flatnonzero(v)[:: max(1, int(v.sum()) // 60)]]
                r.update(same_choice_as_fem_pct=round(100 * float(np.mean((wa == wf)[v])), 1), fem_optimum_kept_median=round(q(kept[v], 50), 2),
                         fem_optimum_kept_p5=round(q(kept[v], 5), 2), rho_vs_fem_median=round(float(np.median(rho)), 4))
            rf = spearmanr(age[v], FC[v]).correlation; rt = spearmanr(age[v], T[mi, np.arange(H), t, wc][v]).correlation
            r.update(rho_age_F=round(float(rf), 3), rho_age_T=round(float(rt), 3))
            rows.append(r)
            if m == "fem":
                fixed[names[t]] = dict(cohort_choice=mont[cc], cohort_unconstrained=mont[int(medF.argmax())], tier=tiers[t])
    # BH-FDR on age, per model
    from scipy.stats import t as tdist
    for m in models:
        sub = [r for r in rows if r["model"] == m]
        n = int(np.median([r["heads"] for r in sub]))
        for key in ("rho_age_F", "rho_age_T"):
            rr = np.array([r[key] for r in sub]); tt = rr * np.sqrt((n - 2) / np.maximum(1 - rr ** 2, 1e-12))
            p = 2 * tdist.sf(np.abs(tt), n - 2); qv = bh(p)
            for r, qq in zip(sub, qv):
                r["q" + key[3:]] = round(float(qq), 4)
    for out, keep in ((S["out22"], ("fem", "sp2", "sp1", "sp5")), (S["out23"], ("fem", "sp2v23", "sp1v23"))):
        if out is None:
            continue
        os.makedirs(out, exist_ok=True)
        rr = [r for r in rows if r["model"] in keep]
        ks = list(rr[0]); [ks.append(k) for r in rr for k in r if k not in ks]
        with open(os.path.join(out, "target_summary_ext.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ks); w.writeheader(); w.writerows(rr)
        lines = [f"# 300-head atlas extension: {S['title']} ({H} heads, {len({r['target'] for r in rr if r['tier'] != 'ref6'})} new targets)", "",
                 f"Collected {time.strftime('%Y-%m-%d %H:%M')} by `pipeline/nchc/atlas_ext_collect.py`. Rule of the v2 atlas (constrained optimum, 3 % band, accessibility).",
                 *S["text"], "",
                 "Regression check (ref6 maxima vs the published scores; largest relative difference per model): " +
                 ", ".join(f"{m} {max(reg[m]):.1e} ({len(reg[m])})" for m in keep if reg[m]), "",
                 "| Set | Model | Targets | F of the reported choice, median | T, median (V/m per mA) | Same choice as FEM | FEM optimum kept, median [5th pct] | Heads gaining ≥ 10 % from personalization | Age: T declines (q < 0.05) |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for tier in S["tiers"]:
            for m in keep:
                sub = [r for r in rr if r["tier"] == tier and r["model"] == m]
                if not sub:
                    continue
                lines.append(f"| {tier} | {m} | {len(sub)} | {q([r['F_choice_median'] for r in sub], 50):.3f} | {q([r['T_choice_median_Vm_per_mA'] for r in sub], 50):.3f} | "
                             + (f"{q([r['same_choice_as_fem_pct'] for r in sub], 50):.0f} % | {q([r['fem_optimum_kept_median'] for r in sub], 50):.1f} % [{q([r['fem_optimum_kept_p5'] for r in sub], 50):.1f} %]" if m != "fem" else "– | –")
                             + f" | {q([r['heads_gain_ge_10pct'] for r in sub], 50):.1f} % | {sum(r['q_age_T'] < 0.05 and r['rho_age_T'] < 0 for r in sub)}/{len(sub)} |")
        lines += ["", "Per target and model: `target_summary_ext.csv`. Not in any record."]
        open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
        json.dump(dict(collected=time.strftime("%Y-%m-%d %H:%M"), heads=codes, models=list(keep), regression={m: [len(reg[m]), max(reg[m]) if reg[m] else None] for m in keep},
                       source=src), open(os.path.join(out, "PROVENANCE.json"), "w"), indent=1)
        print("\n".join(lines))
    if S["fixed"]:
        json.dump(dict(source=f"atlas_ext_collect.py, model fem (300-head extension: {S['title']})", targets=fixed),
                  open(os.path.join(ROOT, "results_v23", "v23_uq", S["fixed"]), "w"), indent=1)
    print(f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
