# -*- coding: utf-8 -*-
"""Region-target TI atlas, version 2 (22 Oct 2026): constrained optimum and top-band reporting.

The PI approved two changes after the logic audit (OPTIMIZATION_LOGIC_AUDIT_2026-10-22.md):
  1. Constrained optimum. Among the 11,628 montages, keep those whose target field T is at least
     TMIN_FRAC (0.80) of the highest T any montage gives that target in that head; among them take the
     highest F = T/O. The unconstrained F-winner is kept beside it for reference.
  2. Top band and accessibility. The band is every admissible montage with F within BAND (3 %) of the
     constrained optimum. Within the band the montage with the most "easy" electrodes (hairline and edge
     sites: Fp1 Fp2 F7 F8 T7 T8 P7 P8 O1 O2), then the highest F, is the reported choice. The band size and its
     F spread are reported; so is the easy-electrode count.
Cohort level: medians of F and T across resolvable heads per montage; the same constraint and band on the
medians; personalization gain = the head's constrained optimum F / F at the cohort choice in that head.
Agreement with FEM: same constrained choice; FEM optimum kept by the SPICEhead choice; Spearman rho over all
montages (unchanged); and whether the SPICEhead choice lies inside FEM's band.
Check: the unconstrained ref6 cohort winners must equal the collected 300-head comparison.

Inputs: L:/BrainProbe/SPICEhead/nchc_fullsearch/targets_out[_sp5]/<code>__targets.npz (verified).
Model version (24 Oct 2026): SPICEHEAD_MODEL selects the score folders and outputs (unset = v23 since 2 Nov 2026). v22 is unchanged;
v23 reads targets_out_v23[/targets_out_sp5_v23 when present], whose bundles carry the v23 bases under the same
keys (prep_target_bundles_v23.py), and writes results_v23/v23_target_atlas_300 and target_atlas_perhead_v2_v23.npz.
Outputs (aggregates only): results_v22/v22_target_atlas_300/{target_summary.csv, agreement_vs_fem.csv,
demographics.csv, PROVENANCE.json, SUMMARY.md}; previous tables moved to unconstrained_2026-10-18/.
Per-head arrays (h-codes) to L:/.../target_atlas_perhead_v2.npz.

    python pipeline/nchc/collect_targets_v2.py
"""
import csv
import json
import os
import shutil
import time

import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
sys.path.insert(0, ROOT)
from spicehead import config                                          # noqa: E402
TAG = "" if config.MODEL == "v22" else f"_{config.MODEL}"
# score folders and the models each holds (the bundles of every version use the keys fem / sp2 / sp1)
SOURCES = [(os.path.join(L, f"targets_out{TAG}"), ["fem", "sp2", "sp1"])]
if config.MODEL == "v22" or os.path.isdir(os.path.join(L, f"targets_out_sp5{TAG}")):
    SOURCES.append((os.path.join(L, f"targets_out_sp5{TAG}"), ["sp5"]))
OUT = os.path.join(config.RESULTS, f"{config.MODEL}_target_atlas_300")
PERHEAD = os.environ.get("SPICEHEAD_ATLAS_PERHEAD") or os.path.join(L, f"target_atlas_perhead_v2{TAG}.npz")
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only
COHORT = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"          # private; read only
# check of the unconstrained ref6 cohort winners against the same model's 300-head full-space comparison
REF300, REFCOL = ((os.path.join(ROOT, "results_v22", "v22_fem_fullsearch_300", "fullsearch_cohort.csv"),
                   {"fem": "winner_fem", "sp2": "winner_sp2", "sp1": "winner_sp1"}) if config.MODEL == "v22" else
                  (os.path.join(ROOT, "results_v22", "v22_fem_fullsearch_300_homog", "cohort.csv"),
                   {"fem": "winner_fem", "sp2": "winner_sp2hom", "sp1": "winner_sp1hom"}))
MODELS = [m for _, ms in SOURCES for m in ms]                         # v22: fem, sp2, sp1, sp5
NMOD = len(MODELS)
TMIN_FRAC, BAND = 0.80, 0.03
EASY = {"Fp1", "Fp2", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2"}
NEG = np.float32(-1e9)


def bh(p):
    p = np.asarray(p, float); n = int(np.isfinite(p).sum()); q = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]; order = idx[np.argsort(p[idx])]
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1].clip(max=1)
    return q


def choose(F, T, easy):
    """F, T: (nt, nm). Returns dict of per-target indices and band statistics under the approved rule."""
    tmax = T.max(1, keepdims=True)
    adm = T >= TMIN_FRAC * tmax                                    # admissible montages
    Fa = np.where(adm, F, NEG)
    wc = Fa.argmax(1); Fc = Fa[np.arange(len(F)), wc]
    band = Fa >= (1 - BAND) * Fc[:, None]
    nband = band.sum(1)
    fmin = np.where(band, Fa, np.inf).min(1)
    wa = np.where(band, easy[None, :] * 10.0 + F, NEG).argmax(1)   # most accessible, then highest F
    w0 = F.argmax(1)
    return dict(w0=w0, wc=wc, wa=wa, nband=nband, spread=100 * (1 - fmin / np.maximum(Fc, 1e-12)), band=band)


def main():
    t0 = time.time()
    lst = lambda d: sorted(f.split("__")[0] for f in os.listdir(d) if f.endswith("__targets.npz"))
    codes = lst(SOURCES[0][0])
    assert len(codes) == 300 and all(lst(d) == codes for d, _ in SOURCES[1:])
    with np.load(os.path.join(SOURCES[0][0], f"{codes[0]}__targets.npz")) as z:
        names, tiers, mont = list(z["targets"]), list(z["tiers"]), list(z["montages"])
    H, NT, NM = len(codes), len(names), len(mont)
    easy = np.array([sum(e in EASY for p in m.split(" / ") for e in p.split("+")) for m in mont], np.float32)
    F = np.zeros((NMOD, H, NT, NM), np.float32); T = np.zeros((NMOD, H, NT, NM), np.float32)
    nroi = np.zeros((H, NT), np.int32)
    for hi, c in enumerate(codes):
        for si, (d, ms) in enumerate(SOURCES):
            with np.load(os.path.join(d, f"{c}__targets.npz")) as z:
                assert list(z["targets"]) == names and list(z["montages"]) == mont and list(z["models"]) == ms, c
                for k, m in enumerate(ms):
                    F[MODELS.index(m), hi] = z["F"][k]; T[MODELS.index(m), hi] = z["T"][k]
                if si == 0:
                    nroi[hi] = z["n_roi"]
        if (hi + 1) % 50 == 0:
            print(f"  loaded {hi+1}/{H} ({time.time()-t0:.0f} s)", flush=True)
    valid = nroi >= 5
    ar = np.arange(NT)

    # ---- per head: unconstrained, constrained and band choices
    W0 = np.zeros((NMOD, H, NT), np.int32); WC = np.zeros_like(W0); WA = np.zeros_like(W0)
    NB = np.zeros((NMOD, H, NT), np.int32); SPR = np.zeros((NMOD, H, NT), np.float32)
    BANDS = np.zeros((NMOD, H, NT, NM), bool)
    for mi in range(NMOD):
        for hi in range(H):
            ch = choose(F[mi, hi], T[mi, hi], easy)
            W0[mi, hi], WC[mi, hi], WA[mi, hi], NB[mi, hi], SPR[mi, hi] = ch["w0"], ch["wc"], ch["wa"], ch["nband"], ch["spread"]
            BANDS[mi, hi] = ch["band"]
    pick = lambda A, W: A[np.arange(NMOD)[:, None, None], np.arange(H)[None, :, None], ar[None, None, :], W]
    F0, T0 = pick(F, W0), pick(T, W0)
    FC, TC = pick(F, WC), pick(T, WC)
    FA, TA = pick(F, WA), pick(T, WA)
    Tmax = T.max(3)
    EA = easy[WA]; EC = easy[WC]; E0 = easy[W0]
    print(f"per-head choices done ({time.time()-t0:.0f} s)", flush=True)

    # ---- cohort level (medians over resolvable heads)
    coh0 = np.full((NMOD, NT), -1, np.int32); cohA = np.full((NMOD, NT), -1, np.int32); cohNB = np.zeros((NMOD, NT), np.int32); cohSPR = np.zeros((NMOD, NT), np.float32)
    medF = np.zeros((NMOD, NT, NM), np.float32); medT = np.zeros((NMOD, NT, NM), np.float32)
    for mi in range(NMOD):
        for t in range(NT):
            v = valid[:, t]
            if not v.any():
                continue
            medF[mi, t] = np.median(F[mi, v, t, :], axis=0); medT[mi, t] = np.median(T[mi, v, t, :], axis=0)
            coh0[mi, t] = int(medF[mi, t].argmax())
            ch = choose(medF[mi, t][None], medT[mi, t][None], easy)
            cohA[mi, t], cohNB[mi, t], cohSPR[mi, t] = ch["wa"][0], ch["nband"][0], ch["spread"][0]
    F_coh = np.zeros((NMOD, H, NT), np.float32); T_coh = np.zeros((NMOD, H, NT), np.float32)
    for mi in range(NMOD):
        ok = cohA[mi] >= 0
        F_coh[mi][:, ok] = F[mi][:, ar[ok], cohA[mi, ok]]; T_coh[mi][:, ok] = T[mi][:, ar[ok], cohA[mi, ok]]
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(F_coh > 0, FC / F_coh, np.nan)
    print(f"cohort choices done ({time.time()-t0:.0f} s)", flush=True)

    # ---- agreement with FEM (constrained choice with tie-break)
    same = np.zeros((NMOD - 1, H, NT), bool); inband = np.zeros((NMOD - 1, H, NT), bool)
    keep = np.full((NMOD - 1, H, NT), np.nan, np.float32); rho = np.full((NMOD - 1, H, NT), np.nan, np.float32)
    for hi in range(H):
        Rf = rankdata(F[0, hi], axis=1); Rf -= Rf.mean(1, keepdims=True); Rf /= np.linalg.norm(Rf, axis=1, keepdims=True) + 1e-30
        for j, mi in enumerate(range(1, NMOD)):
            same[j, hi] = WA[mi, hi] == WA[0, hi]
            inband[j, hi] = BANDS[0, hi][ar, WA[mi, hi]]
            with np.errstate(divide="ignore", invalid="ignore"):
                keep[j, hi] = 100 * F[0, hi][ar, WA[mi, hi]] / FC[0, hi]
            Rm = rankdata(F[mi, hi], axis=1); Rm -= Rm.mean(1, keepdims=True); Rm /= np.linalg.norm(Rm, axis=1, keepdims=True) + 1e-30
            rho[j, hi] = (Rf * Rm).sum(1)
    print(f"agreement done ({time.time()-t0:.0f} s)", flush=True)

    # ---- demographics (in memory only)
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    crow = {r["CaseID"]: r for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))}
    age = np.array([float(crow[cmap[c]]["Age_Scan"]) for c in codes]); sex = np.array([crow[cmap[c]]["Sex_1M0F"] for c in codes])
    site = np.array([crow[cmap[c]]["Site"] for c in codes]); sites = sorted(set(site))

    # ---- move the earlier (v1, unconstrained) tables aside, write the new ones; only v22 had v1 tables
    os.makedirs(OUT, exist_ok=True)
    old = os.path.join(OUT, "unconstrained_2026-10-18")
    if config.MODEL == "v22":
        os.makedirs(old, exist_ok=True)
        for f in ("target_summary.csv", "agreement_vs_fem.csv", "demographics.csv", "PROVENANCE.json", "SUMMARY.md"):
            if os.path.exists(os.path.join(OUT, f)) and not os.path.exists(os.path.join(old, f)):
                shutil.move(os.path.join(OUT, f), os.path.join(old, f))
    q = lambda a, p: float(np.nanpercentile(a, p)) if np.isfinite(a).any() else float("nan")
    rows, arows, drows = [], [], []
    for t in range(NT):
        v = valid[:, t]
        for mi, m in enumerate(MODELS):
            r = dict(target=names[t], tier=tiers[t], model=m, heads_resolvable=int(v.sum()),
                     roi_voxels_median=int(np.median(nroi[v, t])) if v.any() else 0)
            if v.any():
                r.update(
                    F_unconstrained_median=round(q(F0[mi, v, t], 50), 4), T_unconstrained_median_Vm_per_mA=round(q(T0[mi, v, t], 50), 5),
                    T_max_median_Vm_per_mA=round(q(Tmax[mi, v, t], 50), 5),
                    F_constrained_median=round(q(FC[mi, v, t], 50), 4), F_constrained_p25=round(q(FC[mi, v, t], 25), 4), F_constrained_p75=round(q(FC[mi, v, t], 75), 4),
                    T_constrained_median_Vm_per_mA=round(q(TC[mi, v, t], 50), 5),
                    heads_F_constrained_above_1_pct=round(100 * float(np.mean(FC[mi, v, t] > 1)), 1),
                    F_cost_of_constraint_pct_median=round(q(100 * (1 - FC[mi, v, t] / np.maximum(F0[mi, v, t], 1e-12)), 50), 2),
                    T_gain_of_constraint_median=round(q(TC[mi, v, t] / np.maximum(T0[mi, v, t], 1e-12), 50), 3),
                    band_size_median=int(np.median(NB[mi, v, t])), band_F_spread_pct_median=round(q(SPR[mi, v, t], 50), 2),
                    choice_F_median=round(q(FA[mi, v, t], 50), 4), choice_T_median_Vm_per_mA=round(q(TA[mi, v, t], 50), 5),
                    choice_easy_electrodes_mean=round(float(EA[mi, v, t].mean()), 2), unconstrained_winner_easy_electrodes_mean=round(float(E0[mi, v, t].mean()), 2),
                    cohort_choice=mont[cohA[mi, t]], cohort_choice_easy_electrodes=int(easy[cohA[mi, t]]),
                    cohort_band_size=int(cohNB[mi, t]), cohort_band_F_spread_pct=round(float(cohSPR[mi, t]), 2),
                    cohort_unconstrained_winner=mont[coh0[mi, t]],
                    F_cohort_median=round(q(F_coh[mi, v, t], 50), 4), T_cohort_median_Vm_per_mA=round(q(T_coh[mi, v, t], 50), 5),
                    gain_median=round(q(gain[mi, v, t], 50), 4), gain_p90=round(q(gain[mi, v, t], 90), 4),
                    heads_gain_ge_10pct=round(100 * float(np.nanmean(gain[mi, v, t] >= 1.10)), 1))
            rows.append(r)
        for j, m in enumerate(MODELS[1:]):                            # v22: sp2, sp1, sp5
            mi = MODELS.index(m)
            r = dict(target=names[t], tier=tiers[t], model=m)
            if v.any():
                cf = F[0][:, t, :][v]
                r.update(same_choice_pct=round(100 * float(same[j, v, t].mean()), 1),
                         choice_inside_fem_band_pct=round(100 * float(inband[j, v, t].mean()), 1),
                         fem_optimum_kept_median=round(q(keep[j, v, t], 50), 2), fem_optimum_kept_p5=round(q(keep[j, v, t], 5), 2),
                         fem_optimum_kept_min=round(float(np.nanmin(keep[j, v, t])), 2), rows_kept_below_90_pct=int(np.sum(keep[j, v, t] < 90)),
                         rho_median=round(q(rho[j, v, t], 50), 4), rho_min=round(float(np.nanmin(rho[j, v, t])), 4),
                         cohort_choice_same_as_fem=bool(cohA[mi, t] == cohA[0, t]),
                         cohort_choice_kept_under_fem_pct=round(100 * float(np.median(cf[:, cohA[mi, t]]) / np.median(cf[:, cohA[0, t]])), 2))
            arows.append(r)
        for mi, m in enumerate(MODELS):
            if v.sum() < 10:
                continue
            ra, pa = spearmanr(age[v], FC[mi, v, t]); rt, pt = spearmanr(age[v], TC[mi, v, t]); rg, pg = spearmanr(age[v], gain[mi, v, t])
            d = dict(target=names[t], tier=tiers[t], model=m, n=int(v.sum()), rho_age_F=round(float(ra), 3), p_age_F=float(pa),
                     rho_age_T=round(float(rt), 3), p_age_T=float(pt), rho_age_gain=round(float(rg), 3), p_age_gain=float(pg),
                     F_median_men=round(q(FC[mi, v & (sex == "1"), t], 50), 4), F_median_women=round(q(FC[mi, v & (sex == "0"), t], 50), 4))
            for s in sites:
                d[f"F_median_{s}"] = round(q(FC[mi, v & (site == s), t], 50), 4)
            drows.append(d)
    for m in MODELS:
        sub = [d for d in drows if d["model"] == m]
        for key in ("F", "T", "gain"):
            for d, qq in zip(sub, bh([d[f"p_age_{key}"] for d in sub])):
                d[f"q_age_{key}"] = float(qq)

    def write(name, rr):
        keys = []
        for r in rr:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rr)
    write("target_summary.csv", rows); write("agreement_vs_fem.csv", arows); write("demographics.csv", drows)
    np.savez(PERHEAD, codes=np.array(codes), targets=np.array(names), tiers=np.array(tiers),
             models=np.array(MODELS), montages=np.array(mont), easy=easy, W_unconstrained=W0, W_constrained=WC, W_choice=WA,
             F_unconstrained=F0, T_unconstrained=T0, F_constrained=FC, T_constrained=TC, F_choice=FA, T_choice=TA, T_max=Tmax,
             band_size=NB, band_spread_pct=SPR, cohort_choice=cohA, cohort_unconstrained=coh0, F_coh=F_coh, T_coh=T_coh, gain=gain,
             same_choice_vs_fem=same, choice_in_fem_band=inband, fem_kept_pct=keep, rho_vs_fem=rho, n_roi=nroi)

    chk = []
    if os.path.exists(REF300):
        ref = {r["roi"]: r for r in csv.DictReader(open(REF300, encoding="utf-8"))}
        for t, n in enumerate(names):
            if n.startswith("ref6:"):
                for m in ("fem", "sp2", "sp1"):
                    chk.append((n, m, mont[coh0[MODELS.index(m), t]] == ref[n.split(":", 1)[1]][REFCOL[m]]))
    pold = os.path.join(old, "PROVENANCE.json")
    prov = json.load(open(pold, encoding="utf-8")) if os.path.exists(pold) else dict(
        collected=time.strftime("%Y-%m-%d %H:%M:%S"), n_heads=len(codes), n_targets=NT, n_montages=NM, models=MODELS)
    prov["model_version"] = config.MODEL
    prov.update(version="v2 (22 Oct 2026): constrained optimum and top-band reporting", collected_v2=time.strftime("%Y-%m-%d %H:%M:%S"),
                rule=dict(T_min_fraction_of_achievable=TMIN_FRAC, band_fraction_of_best_F=BAND, tie_break="most easy electrodes, then highest F",
                          easy_electrodes=sorted(EASY)),
                ref6_unconstrained_cohort_winners_match_collected=f"{sum(x[2] for x in chk)}/{len(chk)}",
                previous_tables="unconstrained_2026-10-18/" if config.MODEL == "v22" else None)
    json.dump(prov, open(os.path.join(OUT, "PROVENANCE.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"ref6 unconstrained cohort winners vs collected comparison: {sum(x[2] for x in chk)}/{len(chk)}")
    print(f"wrote {OUT} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
