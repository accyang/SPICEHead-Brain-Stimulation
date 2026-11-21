# -*- coding: utf-8 -*-
"""Collect the region-target TI atlas over the 300-head cohort (Paper C; 18 Oct 2026).

Input: the verified score arrays on L: (pulled from Nano 4):
  - targets_out/<code>__targets.npz: F, T, O for FEM, SP2, SP1;
  - targets_out_sp5/<code>__targets.npz: the same for SP5.
Each holds 111 targets (ref6, tier 2, AAL90) × 11,628 two-pair 10-20 montages.

Per target and model:
  - Individual optimum per head: F_ind = max over montages of T/O. Its T is the target envelope in V/m at
    1 mA per channel.
  - Cohort montage: the montage with the highest median F across heads, as in
    v22_fem_fullsearch.stage_collect.
  - Personalization gain per head: F_ind / F at the cohort montage (1 = no gain).
  - Agreement of each SPICEhead model with FEM:
    - same individual winner;
    - share of the FEM optimum kept by the SPICEhead winner (judged by FEM);
    - Spearman rho over all montages;
    - cohort winners.
  - Age (Spearman, BH-FDR across targets within a model), sex and site. These are computed in memory from
    the private cohort table and written only as aggregates.

Checks:
  - For the six ref6 targets, the cohort winners must equal those of the collected 300-head comparison
    (results_v22/v22_fem_fullsearch_300/fullsearch_cohort.csv).
  - Target lists and montage lists must be identical across all heads and both runs.

Outputs, aggregates only, no head codes or study IDs:
  - results_v22/v22_target_atlas_300/ with target_summary.csv, agreement_vs_fem.csv, demographics.csv,
    SUMMARY.md and PROVENANCE.json.
  - Per-head arrays (h-codes) stay on L: in nchc_fullsearch/target_atlas_perhead.npz.

    python pipeline/nchc/collect_targets.py
"""
import csv
import json
import os
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
MAIN, SP5 = os.path.join(L, "targets_out"), os.path.join(L, "targets_out_sp5")
OUT = os.path.join(ROOT, "results_v22", "v22_target_atlas_300")
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only
COHORT = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"          # private; read only
REF300 = os.path.join(ROOT, "results_v22", "v22_fem_fullsearch_300", "fullsearch_cohort.csv")
MODELS = ["fem", "sp2", "sp1", "sp5"]


def bh(p):
    p = np.asarray(p, float); n = np.isfinite(p).sum(); q = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]; order = idx[np.argsort(p[idx])]
    ranked = p[order] * n / np.arange(1, n + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return q


def main():
    t0 = time.time()
    codes = sorted(f.split("__")[0] for f in os.listdir(MAIN) if f.endswith("__targets.npz"))
    codes5 = sorted(f.split("__")[0] for f in os.listdir(SP5) if f.endswith("__targets.npz"))
    assert codes == codes5 and len(codes) == 300, (len(codes), len(codes5))
    with np.load(os.path.join(MAIN, f"{codes[0]}__targets.npz")) as z:
        names, tiers, mont = list(z["targets"]), list(z["tiers"]), list(z["montages"])
        assert list(z["models"]) == ["fem", "sp2", "sp1"], list(z["models"])
    H, NT, NM = len(codes), len(names), len(mont)
    F = np.zeros((4, H, NT, NM), np.float32)
    nroi = np.zeros((H, NT), np.int32)
    T_ind = np.zeros((4, H, NT), np.float32); W_ind = np.zeros((4, H, NT), np.int32)
    for hi, c in enumerate(codes):
        with np.load(os.path.join(MAIN, f"{c}__targets.npz")) as a, np.load(os.path.join(SP5, f"{c}__targets.npz")) as b:
            for z, ms in ((a, ["fem", "sp2", "sp1"]), (b, ["sp5"])):
                assert list(z["targets"]) == names and list(z["montages"]) == mont and list(z["models"]) == ms, c
                Fz, Tz = z["F"], z["T"]
                for k, m in enumerate(ms):
                    mi = MODELS.index(m)
                    F[mi, hi] = Fz[k]
                    W_ind[mi, hi] = Fz[k].argmax(1)
                    T_ind[mi, hi] = Tz[k][np.arange(NT), W_ind[mi, hi]]
            nroi[hi] = a["n_roi"]
        if (hi + 1) % 50 == 0:
            print(f"  loaded {hi+1}/{H} heads ({time.time()-t0:.0f} s)", flush=True)
    valid = nroi >= 5                                                    # (H, NT) target resolvable in the head
    F_ind = F.max(3)                                                     # (4, H, NT)

    # cohort montage (median F across heads with the target resolvable), F at it, gain
    coh = np.zeros((4, NT), np.int32); F_coh = np.zeros((4, H, NT), np.float32)
    for mi in range(4):
        for t in range(NT):
            v = valid[:, t]
            if v.sum() == 0:
                coh[mi, t] = -1; continue
            coh[mi, t] = int(np.median(F[mi, v, t, :], axis=0).argmax())
            F_coh[mi, :, t] = F[mi, :, t, coh[mi, t]]
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(F_coh > 0, F_ind / F_coh, np.nan)
    T_coh = np.zeros((4, H, NT), np.float32)                             # second pass: T at the cohort montage
    for hi, c in enumerate(codes):
        with np.load(os.path.join(MAIN, f"{c}__targets.npz")) as a, np.load(os.path.join(SP5, f"{c}__targets.npz")) as b:
            for z, ms in ((a, ["fem", "sp2", "sp1"]), (b, ["sp5"])):
                Tz = z["T"]
                for k, m in enumerate(ms):
                    mi = MODELS.index(m); ok = coh[mi] >= 0
                    T_coh[mi, hi, ok] = Tz[k][np.where(ok)[0], coh[mi, ok]]
    print(f"cohort montages done ({time.time()-t0:.0f} s)", flush=True)

    # agreement with FEM per head and target
    same = np.zeros((3, H, NT), bool); keep = np.full((3, H, NT), np.nan, np.float32); rho = np.full((3, H, NT), np.nan, np.float32)
    for hi in range(H):
        Rf = rankdata(F[0, hi], axis=1)
        Rf = (Rf - Rf.mean(1, keepdims=True)); Rf /= np.linalg.norm(Rf, axis=1, keepdims=True) + 1e-30
        for j, mi in enumerate((1, 2, 3)):
            same[j, hi] = W_ind[mi, hi] == W_ind[0, hi]
            with np.errstate(divide="ignore", invalid="ignore"):
                keep[j, hi] = 100 * F[0, hi][np.arange(NT), W_ind[mi, hi]] / F_ind[0, hi]
            Rm = rankdata(F[mi, hi], axis=1); Rm = (Rm - Rm.mean(1, keepdims=True)); Rm /= np.linalg.norm(Rm, axis=1, keepdims=True) + 1e-30
            rho[j, hi] = (Rf * Rm).sum(1)
    print(f"agreement done ({time.time()-t0:.0f} s)", flush=True)

    # demographics (in memory only)
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    coh_rows = {r["CaseID"]: r for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))}
    age = np.array([float(coh_rows[cmap[c]]["Age_Scan"]) for c in codes])
    sex = np.array([coh_rows[cmap[c]]["Sex_1M0F"] for c in codes])
    site = np.array([coh_rows[cmap[c]]["Site"] for c in codes])
    sites = sorted(set(site))

    # ---- tables
    os.makedirs(OUT, exist_ok=True)
    q = lambda a, p: float(np.nanpercentile(a, p)) if np.isfinite(a).any() else float("nan")
    rows, arows, drows = [], [], []
    for t in range(NT):
        v = valid[:, t]
        for mi, m in enumerate(MODELS):
            fi, gi = F_ind[mi, v, t], gain[mi, v, t]
            rows.append(dict(target=names[t], tier=tiers[t], model=m, heads_resolvable=int(v.sum()),
                             roi_voxels_median=int(np.median(nroi[v, t])) if v.any() else 0,
                             F_ind_median=round(q(fi, 50), 4), F_ind_p25=round(q(fi, 25), 4), F_ind_p75=round(q(fi, 75), 4),
                             heads_F_ind_above_1_pct=round(100 * float(np.mean(fi > 1)), 1) if v.any() else float("nan"),
                             T_ind_median_Vm_per_mA=round(q(T_ind[mi, v, t], 50), 5),
                             cohort_montage=mont[coh[mi, t]] if coh[mi, t] >= 0 else "",
                             F_cohort_median=round(q(F_coh[mi, v, t], 50), 4),
                             T_cohort_median_Vm_per_mA=round(q(T_coh[mi, v, t], 50), 5),
                             gain_median=round(q(gi, 50), 4), gain_p90=round(q(gi, 90), 4),
                             heads_gain_ge_10pct=round(100 * float(np.nanmean(gi >= 1.10)), 1) if v.any() else float("nan")))
        for j, m in enumerate(("sp2", "sp1", "sp5")):
            mi = MODELS.index(m)
            if not v.any():
                arows.append(dict(target=names[t], tier=tiers[t], model=m)); continue
            cf = F[0][:, t, :][v]                                       # FEM per head (heads, NM)
            fem_coh_med = np.median(cf[:, coh[0, t]]); m_coh_under_fem = np.median(cf[:, coh[mi, t]])
            arows.append(dict(target=names[t], tier=tiers[t], model=m,
                              same_individual_winner_pct=round(100 * float(same[j, v, t].mean()), 1),
                              fem_optimum_kept_median=round(q(keep[j, v, t], 50), 2),
                              fem_optimum_kept_p5=round(q(keep[j, v, t], 5), 2),
                              fem_optimum_kept_min=round(float(np.nanmin(keep[j, v, t])), 2),
                              rows_kept_below_90_pct=int(np.sum(keep[j, v, t] < 90)),
                              rho_median=round(q(rho[j, v, t], 50), 4), rho_min=round(float(np.nanmin(rho[j, v, t])), 4),
                              cohort_winner_same_as_fem=bool(coh[mi, t] == coh[0, t]),
                              cohort_winner_kept_under_fem_pct=round(100 * float(m_coh_under_fem / fem_coh_med), 2)))
        for mi, m in enumerate(MODELS):
            if v.sum() < 10:
                continue
            ra, pa = spearmanr(age[v], F_ind[mi, v, t]); rt, pt = spearmanr(age[v], T_ind[mi, v, t])
            rg, pg = spearmanr(age[v], gain[mi, v, t])
            d = dict(target=names[t], tier=tiers[t], model=m, n=int(v.sum()),
                     rho_age_F=round(float(ra), 3), p_age_F=float(pa), rho_age_T=round(float(rt), 3), p_age_T=float(pt),
                     rho_age_gain=round(float(rg), 3), p_age_gain=float(pg),
                     F_ind_median_men=round(q(F_ind[mi, v & (sex == "1"), t], 50), 4),
                     F_ind_median_women=round(q(F_ind[mi, v & (sex == "0"), t], 50), 4))
            for s in sites:
                d[f"F_ind_median_{s}"] = round(q(F_ind[mi, v & (site == s), t], 50), 4)
            drows.append(d)
    for m in MODELS:                                                     # BH-FDR across targets within a model
        sub = [d for d in drows if d["model"] == m]
        for key in ("F", "T", "gain"):
            qs = bh([d[f"p_age_{key}"] for d in sub])
            for d, qq in zip(sub, qs):
                d[f"q_age_{key}"] = float(qq)

    def write(name, rr):
        keys = []
        for r in rr:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rr)
    write("target_summary.csv", rows); write("agreement_vs_fem.csv", arows); write("demographics.csv", drows)
    np.savez(os.path.join(L, "target_atlas_perhead.npz"), codes=np.array(codes), targets=np.array(names), tiers=np.array(tiers),
             models=np.array(MODELS), montages=np.array(mont), F_ind=F_ind, W_ind=W_ind, T_ind=T_ind, F_coh=F_coh, T_coh=T_coh,
             gain=gain, cohort_montage=coh, same_winner_vs_fem=same, fem_kept_pct=keep, rho_vs_fem=rho, n_roi=nroi)

    # ---- check: ref6 cohort winners equal the collected 300-head comparison
    chk = []
    if os.path.exists(REF300):
        ref = {r["roi"]: r for r in csv.DictReader(open(REF300, encoding="utf-8"))}
        for t, n in enumerate(names):
            if n.startswith("ref6:"):
                r = n.split(":", 1)[1]
                for m in ("fem", "sp2", "sp1"):
                    chk.append((r, m, mont[coh[MODELS.index(m), t]] == ref[r][f"winner_{m}"]))
    json.dump(dict(collected=time.strftime("%Y-%m-%d %H:%M:%S"), n_heads=H, n_targets=NT, n_montages=NM, models=MODELS,
                   compute="NCHC Nano 4 H200: jobs 565002 (FEM/SP2/SP1, 2-4 nodes) and 565061 (SP5, 2 nodes), 18 Oct 2026",
                   scorer="pipeline/nchc/score_targets.py (v1) and score_targets_v2.py (SP5); ref6 rescoring matched the "
                          "collected 300-head scores within 6.6e-7 with all 5,400 winners identical",
                   targets="ref6 (Paper A), tier 2 (AAL3v1 regions + Fox 2012 DLPFC spheres, PI-approved 18 Oct), tier 1 (AAL90)",
                   atlases="SPM12 AAL (nilearn copy, the GUI's file); AAL3v1_1mm (oxcns.org zip, sha256 45a89d28...)",
                   fallback_warp="one head (CHARM folder without MNI2Conform_nonl): atlas labels via warp_fallback.pull_labels (identical to warp_volume on 3 test heads); points within 0.4 mm",
                   ref6_cohort_winners_match_collected=f"{sum(x[2] for x in chk)}/{len(chk)}",
                   inputs_verified="every score file checked against a cluster-side SHA-256 after download"),
              open(os.path.join(OUT, "PROVENANCE.json"), "w", encoding="utf-8"), indent=1)
    print(f"ref6 cohort winners vs the collected 300-head comparison: {sum(x[2] for x in chk)}/{len(chk)} identical")
    for x in chk:
        if not x[2]:
            print("   differs:", x)
    print(f"wrote {OUT} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
