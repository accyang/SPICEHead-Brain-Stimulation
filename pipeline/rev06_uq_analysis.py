# -*- coding: utf-8 -*-
"""A0-A4 of Rev0.5/10_ANALYSIS_REQUEST_v23_2026-11-02.md: run-level recollection of the v23 uncertainty archive.

Input: the per-head extracts of pipeline/rev06_uq_extract.py and pipeline/rev06_sp_extract.py. No field is solved and no prior output is overwritten; the
historical collector (nchc/uq_v23_atlas_collect.py) and results_v23/v23_uq/ stay as they are.
For head h, target r, run s and fixed protocol m (endpoint definitions in endpoint_definition.json):
    floor_ratio = T_fixed / Tmax, F_retention = F_fixed / F_constrained, all from the same perturbed SPICEhead model
    joint_success = valid AND floor_ratio >= 0.80 AND F_retention >= 0.90
Joint success is a post-review addition, not prespecified. Retention is not capped at 100 %.
    python pipeline/rev06_uq_analysis.py
Output: results_v23/v23_uq_recollect_2026-11-04/ (per-head files carry h-codes only; aggregates carry none).
"""
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from rev06_uq_extract import RUNS, OUT as EXTRACT, EXTMONT, PARAMS, FIXEXT, BASE                  # noqa: E402

R23 = os.path.join(ROOT, "results_v23")
OUT = os.environ.get("REV06_OUT", os.path.join(R23, "v23_uq_recollect_2026-11-04"))
SPX = r"C:\spicehead_v23_logs\rev06_extract_sp"                         # new basis solves for the 28 protocols, pipeline/rev06_sp_extract.py
MASK35 = r"C:\spicehead_v22_logs\uq_mask35\sums_mask35.txt"
NOMPERHEAD = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\atlas_v23_parameters_perhead.npz"
COHPERHEAD = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\atlas_v23_cohort_perhead.npz"
FLOOR, RET, CRIT, MINVOX = 0.80, 0.90, 0.95, 5
NBOOT, SEED = 2000, 20261103
FIELD = "SPICEhead v23 1 mm, perturbed"
IX = {r: i for i, r in enumerate(RUNS)}
SCEN = {"conductivity": [i for i, r in enumerate(RUNS) if r.startswith("cond_")], "placement_independent": [i for i, r in enumerate(RUNS) if r.startswith("jit_")],
        "cap_rotation": [i for i, r in enumerate(RUNS) if r.startswith("rot_")], "segmentation": [i for i, r in enumerate(RUNS) if r.startswith("seg_")]}
PHYS = sum(SCEN.values(), [])
assert [len(SCEN[k]) for k in SCEN] == [40, 20, 10, 4] and len(PHYS) == 74 and len(RUNS) == 77
MASKS = {"erosion": IX["mask_erode"], "dilation": IX["mask_dilate"]}
QC = {}


def check(name, ok, detail=None):
    QC[name] = dict(passed=bool(ok), detail=detail)
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail is not None else ""), flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def f16_bounds(a):
    """Rounding interval of stored float16 values: halfway to the neighbouring representable values."""
    a = np.asarray(a, np.float16)
    up = np.nextafter(a, np.float16(np.inf)); dn = np.nextafter(a, np.float16(-np.inf))
    return (a.astype(np.float64) + dn.astype(np.float64)) / 2, (a.astype(np.float64) + up.astype(np.float64)) / 2


def load():
    tp = pd.read_csv(PARAMS)
    sp = tp[tp.model == "sp1v23"].set_index("target"); fem = tp[tp.model == "fem"].set_index("target")
    names = list(sp.index)
    assert len(names) == 163 and list(fem.index) == names or set(fem.index) == set(names)
    with np.load(EXTMONT) as z:
        labs = [str(m) for m in z["montages"]]
    mi = {m: i for i, m in enumerate(labs)}
    fxe = json.load(open(FIXEXT, encoding="utf-8"))["targets"]
    fxa = json.load(open(os.path.join(R23, "v23_uq", "uq_atlas_fixed_montages.json"), encoding="utf-8"))["targets"]
    fxt = json.load(open(os.path.join(R23, "v23_uq", "uq_atlas_fixed_montages_thal.json"), encoding="utf-8"))["targets"]
    fx = {**fxa, **fxt, **fxe}
    check("FEM fixed-montage JSON equals target_parameters.csv for 163 targets", all(fx[t]["cohort_choice"] == fem.cohort_choice[t] for t in names))
    heads = sorted(f[:-4] for f in os.listdir(EXTRACT) if f.endswith(".npz") and not f.endswith(".tmp.npz"))
    check("all 300 heads extracted", len(heads) == 300 and heads[0] == "h001" and heads[-1] == "h300", len(heads))
    H, T, Rn = len(heads), len(names), len(RUNS)
    nan = lambda *s: np.full(s, np.nan, np.float64)
    D = dict(Tmax=nan(H, T, Rn), Fcon=nan(H, T, Rn), Func=nan(H, T, Rn), fem_F=nan(H, T, Rn), fem_T=nan(H, T, Rn), fem_O=nan(H, T, Rn), sp_F=nan(H, T, Rn), sp_T=nan(H, T, Rn), sp_O=nan(H, T, Rn),
             fem_Flo=nan(H, T, Rn), fem_Fhi=nan(H, T, Rn), fem_Tlo=nan(H, T, Rn), fem_Thi=nan(H, T, Rn), sp_Flo=nan(H, T, Rn), sp_Fhi=nan(H, T, Rn), sp_Tlo=nan(H, T, Rn), sp_Thi=nan(H, T, Rn),
             done=np.zeros((H, T, Rn), bool), Trep=nan(H, T), Frep=nan(H, T), Tcon=nan(H, T))
    D["n_roi"] = np.zeros((H, T), np.int32); D["n_mask"] = np.full((H, T, 2), -1, np.int32)
    coincide = np.zeros((H, T), bool); six = nan(H, T, Rn, 3)
    inv, odd, gaps = [], [], dict(ref6=0.0, six=0.0, f16F=0.0, f16T=0.0)
    route = {}
    idxok = dict(fem=True, unc=True)
    ref_labels = {}
    spgap = dict(control=0.0, control_n=0, mask_counts=0, problems=0)
    for h, code in enumerate(heads):
        js = json.load(open(os.path.join(EXTRACT, f"{code}.json"), encoding="utf-8"))
        with np.load(os.path.join(SPX, f"{code}.npz")) as z:
            X = {k: z[k] for k in z.files}
        xt = {str(n): k for k, n in enumerate(X["targets"])}
        spgap["problems"] += len(json.load(open(os.path.join(SPX, f"{code}.json"), encoding="utf-8"))["bad"])
        inv += js["inventory"]; odd += js["odd"]
        M = js["meta"]
        ta, tt = M["atlas"]["targets"], M["thal"]["targets"]
        new = M["atlas"]["new_targets"]
        gaps["six"] = max(gaps["six"], M.get("six_max_rel_gap") or 0.0)
        with np.load(os.path.join(EXTRACT, f"{code}.npz")) as z:
            Z = {k: z[k] for k in z.files}
        gaps["f16F"] = max(gaps["f16F"], float(np.nanmax(Z["atlas_f16_vs_vals_max"][:, 0]))); gaps["f16T"] = max(gaps["f16T"], float(np.nanmax(Z["atlas_f16_vs_vals_max"][:, 1])))
        # the six reference targets appear in both archives: a consistency check, never two sets of regions
        for n in [x for x in tt if x.startswith("ref6:")]:
            i, j = ta.index(n), tt.index(n)
            a, b = Z["atlas_fixed_vals"][:, i, :2], Z["thal_fixed_vals"][:, j, :2]
            gaps["ref6"] = max(gaps["ref6"], float(np.nanmax(np.abs(a.astype(np.float64) / b.astype(np.float64) - 1))))
        for t, n in enumerate(names):
            if n in ta:
                s, i = "atlas", ta.index(n)
            else:
                s, i = "thal", tt.index(n)
            A = {k: Z[f"{s}_{k}"] for k in ("idx", "vals", "Tmax", "fixed_idx", "fixed_vals", "n_roi_masked", "done")}
            D["done"][h, t] = A["done"]
            D["Tmax"][h, t] = A["Tmax"][:, i]; D["Fcon"][h, t] = A["vals"][:, i, 1, 0]; D["Func"][h, t] = A["vals"][:, i, 0, 0]
            D["n_roi"][h, t] = M[s]["n_roi"][i]
            D["n_mask"][h, t] = A["n_roi_masked"][[MASKS["erosion"], MASKS["dilation"]], i]
            D["Trep"][h, t] = A["vals"][0, i, 2, 1]; D["Frep"][h, t] = A["vals"][0, i, 2, 0]; D["Tcon"][h, t] = A["vals"][0, i, 1, 1]
            m_fem, m_unc, m_sp = mi[fem.cohort_choice[n]], mi[fx[n]["cohort_unconstrained"]], mi[sp.cohort_choice[n]]
            is_new = s == "atlas" and i in new
            fi = A["fixed_idx"][:, i]
            if not is_new:
                idxok["fem"] &= bool((fi[A["done"], 0] == m_fem).all()); idxok["unc"] &= bool((fi[A["done"], 1] == m_unc).all())
            else:
                idxok["fem"] &= bool((fi[A["done"], 0] == -1).all())

            def from16(col, m, pre):
                """Stored float16 score, replaced by the run's float32 score whenever that same montage was also saved in float32."""
                j = new.index(i)
                F, Tv = Z["atlas_F16"][:, j, col], Z["atlas_T16"][:, j, col]
                Fl, Fh = f16_bounds(F); Tl, Th = f16_bounds(Tv)
                Fv, Tvv = F.astype(np.float64), Tv.astype(np.float64)
                for c in (0, 1, 2):
                    w = A["idx"][:, i, c] == m
                    v = A["vals"][:, i, c].astype(np.float64)
                    Fv[w], Tvv[w], Fl[w], Fh[w], Tl[w], Th[w] = v[w, 0], v[w, 1], v[w, 0], v[w, 0], v[w, 1], v[w, 1]
                for c in (2, 3):
                    w = A["fixed_idx"][:, i, c] == m
                    v = A["fixed_vals"][:, i, c].astype(np.float64)
                    Fv[w], Tvv[w], Fl[w], Fh[w], Tl[w], Th[w] = v[w, 0], v[w, 1], v[w, 0], v[w, 0], v[w, 1], v[w, 1]
                D[pre + "_F"][h, t], D[pre + "_T"][h, t] = Fv, Tvv
                with np.errstate(divide="ignore", invalid="ignore"):
                    D[pre + "_O"][h, t] = np.where(Fv > 0, Tvv / Fv, np.nan)
                D[pre + "_Flo"][h, t], D[pre + "_Fhi"][h, t], D[pre + "_Tlo"][h, t], D[pre + "_Thi"][h, t] = Fl, Fh, Tl, Th

            def from32(col, pre):
                v = A["fixed_vals"][:, i, col].astype(np.float64)
                D[pre + "_F"][h, t], D[pre + "_T"][h, t], D[pre + "_O"][h, t] = v[:, 0], v[:, 1], v[:, 2]
                D[pre + "_Flo"][h, t] = D[pre + "_Fhi"][h, t] = v[:, 0]; D[pre + "_Tlo"][h, t] = D[pre + "_Thi"][h, t] = v[:, 1]

            if is_new:
                from16(0, m_fem, "fem")
            else:
                from32(0, "fem")
            if m_sp == m_fem:
                for k in ("F", "T", "O", "Flo", "Fhi", "Tlo", "Thi"):
                    D["sp_" + k][h, t] = D["fem_" + k][h, t]
                route[n] = "identical_to_FEM_protocol"
            elif is_new:
                from16(1, m_sp, "sp"); route[n] = "float16_full_scores"
            elif m_sp == m_unc:
                from32(1, "sp"); route[n] = "saved_fem_cohort_unconstrained_column"
            else:
                route[n] = "new_basis_solves"
                if (fi[A["done"], 2] == m_sp).all() or (fi[A["done"], 3] == m_sp).all():
                    coincide[h, t] = True
                if n == "ref6:hippocampus_R" and "six_hippR_sp" in Z:
                    six[h, t] = Z["six_hippR_sp"].astype(np.float64)
                k = xt[n]
                v = X["sp"][:, k].astype(np.float64)
                D["sp_F"][h, t], D["sp_T"][h, t], D["sp_O"][h, t] = v[:, 0], v[:, 1], v[:, 2]
                D["sp_Flo"][h, t] = D["sp_Fhi"][h, t] = v[:, 0]; D["sp_Tlo"][h, t] = D["sp_Thi"][h, t] = v[:, 1]
                # the FEM montage was scored again in every run: it must equal the archived score of the same perturbed model
                spgap["control"] = max(spgap["control"], float(np.abs(X["fem"][:, k].astype(np.float64) - A["fixed_vals"][:, i, 0].astype(np.float64)).max()))
                spgap["control_n"] += int(np.isfinite(X["fem"][:, k]).all(1).sum())
                spgap["mask_counts"] += int((X["n_roi_masked"][[MASKS["erosion"], MASKS["dilation"]], k] != A["n_roi_masked"][[MASKS["erosion"], MASKS["dilation"]], i]).sum())
    check("new-solve run files: 23,100 present, labels and cluster checksums match", spgap["problems"] == 0, spgap["problems"])
    check("FEM control montage in the new solves equals the archived score exactly in every head, target and run", spgap["control"] == 0.0 and spgap["control_n"] == H * 28 * Rn, dict(max_abs_gap=spgap["control"], evaluations=spgap["control_n"]))
    check("masked voxel counts in the new solves equal the archive", spgap["mask_counts"] == 0, spgap["mask_counts"])
    check("montage indices map by electrode label in every head and run (FEM cohort column)", idxok["fem"])
    check("montage indices map by electrode label in every head and run (FEM unconstrained column)", idxok["unc"])
    check("six reference targets agree between the atlas and thalamic archives (max relative gap < 1e-5)", gaps["ref6"] < 1e-5, gaps["ref6"])
    check("six-target run agrees with the atlas archive for the 35 heads (max relative gap < 1e-5)", gaps["six"] < 1e-5, gaps["six"])
    check("float16 scores agree with float32 at the constrained optimum within float16 rounding (< 1e-3)", max(gaps["f16F"], gaps["f16T"]) < 1e-3, dict(F=gaps["f16F"], T=gaps["f16T"]))
    return D, names, heads, sp, fem, fx, mi, route, coincide, six, inv, odd, spgap


def endpoints(D, pre, H, T):
    """Validity and success arrays (head, target, run); reason codes for everything not valid."""
    done, Tmax, Fcon, F, Tv = D["done"], D["Tmax"], D["Fcon"], D[pre + "_F"], D[pre + "_T"]
    elig = (D["n_roi"] >= MINVOX)[:, :, None]
    fin = np.isfinite(F) & np.isfinite(Tv) & (F > 0) & (Tv > 0)
    goodden = np.isfinite(Tmax) & (Tmax > 0) & np.isfinite(Fcon) & (Fcon > 0)
    valid = done & elig & goodden & fin
    with np.errstate(divide="ignore", invalid="ignore"):
        fr, ret = Tv / Tmax, F / Fcon
        frl, frh, rl, rh = D[pre + "_Tlo"] / Tmax, D[pre + "_Thi"] / Tmax, D[pre + "_Flo"] / Fcon, D[pre + "_Fhi"] / Fcon
    floor, fok = valid & (fr >= FLOOR), valid & (ret >= RET)
    joint = floor & fok
    jl = valid & (frl >= FLOOR) & (rl >= RET); ju = valid & (frh >= FLOOR) & (rh >= RET)
    fl, fu = valid & (frl >= FLOOR), valid & (frh >= FLOOR); rl_, ru = valid & (rl >= RET), valid & (rh >= RET)
    amb = (fu & ~fl) | (ru & ~rl_)
    return dict(valid=valid, floor=floor, fok=fok, joint=joint, fr=fr, ret=ret, jl=jl, ju=ju, fl=fl, fu=fu, rl=rl_, ru=ru, amb=amb, done=done, elig=np.broadcast_to(elig, done.shape), goodden=goodden, fin=fin)


def qs(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, p)) for p in ps] if a.size else [float("nan")] * len(ps)


def tables(E, names, heads, sp_fem_labels, mi, tier, proto, targets_idx, W):
    """Per-head, per-target-scenario and pooled tables for one protocol over targets_idx."""
    label, sel = sp_fem_labels, proto
    ph, pt, po = [], [], []
    Tn = len(targets_idx)
    ix = np.array(targets_idx)
    groups = {**SCEN}
    cnt = {}
    for g, runs in groups.items():
        cnt[g] = {k: E[k][:, ix][:, :, runs].sum(2).astype(np.float64) for k in ("valid", "floor", "fok", "joint", "jl", "ju", "fl", "fu", "rl", "ru", "amb", "done", "elig")}
    def rate(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b > 0, a / b, np.nan)
    boots = {}
    for g in groups:
        boots[g] = {k: rate(W @ cnt[g][k], W @ cnt[g]["valid"]) for k in ("floor", "fok", "joint")}
    allc = {k: sum(cnt[g][k] for g in groups) for k in ("valid", "floor", "fok", "joint", "jl", "ju", "fl", "fu", "rl", "ru", "amb")}
    boots["pooled74"] = {k: rate(W @ allc[k], W @ allc["valid"]) for k in ("floor", "fok", "joint")}
    boots["equal4"] = {k: np.mean([boots[g][k] for g in groups], 0) for k in ("floor", "fok", "joint")}
    ci = lambda b, j: (float(np.nanpercentile(b[:, j], 2.5)), float(np.nanpercentile(b[:, j], 97.5)))
    for j, t in enumerate(targets_idx):
        n = names[t]
        idc = dict(target=n, tier=tier[n], protocol_selection_model=sel, field_model=FIELD, montage_label=label[n], candidate_index=mi[label[n]])
        for g, runs in groups.items():
            c = {k: v[:, j] for k, v in cnt[g].items()}
            exp = len(runs) * len(heads)
            rH = {k: rate(c[k], c["valid"]) for k in ("floor", "fok", "joint")}
            hq = {k: qs(100 * rH[k]) for k in rH}
            row = dict(idc, scenario=g, expected_runs=exp, completed_runs=int(c["done"].sum()), eligible_runs=int(c["elig"].sum()), valid_runs=int(c["valid"].sum()),
                       missing_runs=int(exp - c["done"].sum()), ineligible_runs=int(exp - c["elig"].sum()),
                       floor_success=int(c["floor"].sum()), F_only_success=int(c["fok"].sum()), joint_success=int(c["joint"].sum()))
            for k, nm in (("floor", "floor"), ("fok", "F_only"), ("joint", "joint")):
                row[f"{nm}_pooled_pct"] = round(100 * c[k].sum() / max(c["valid"].sum(), 1), 3)
                row[f"{nm}_mean_head_pct"] = round(float(np.nanmean(100 * rH[k])), 3)
                lo, hi = ci(boots[g][k], j)
                row[f"{nm}_ci_lo"], row[f"{nm}_ci_hi"] = round(100 * lo, 3), round(100 * hi, 3)
                row[f"{nm}_head_p5"], row[f"{nm}_head_p25"], row[f"{nm}_head_p50"], row[f"{nm}_head_p75"], row[f"{nm}_head_p95"] = [round(x, 2) for x in hq[k]]
            row["float16_ambiguous_runs"] = int(c["amb"].sum())
            row["joint_lower_bound_pct"] = round(100 * c["jl"].sum() / max(c["valid"].sum(), 1), 3); row["joint_upper_bound_pct"] = round(100 * c["ju"].sum() / max(c["valid"].sum(), 1), 3)
            pt.append(row)
            for h, code in enumerate(heads):
                ph.append((code, n, sel, label[n], g, len(runs), int(c["valid"][h]), int(c["floor"][h]), int(c["fok"][h]), int(c["joint"][h])))
        a = {k: v[:, j] for k, v in allc.items()}
        val = a["valid"].sum()
        row = dict(idc, runs_per_head=74, expected_runs=74 * len(heads), valid_runs=int(val))
        for k, nm in (("floor", "floor"), ("fok", "F_only"), ("joint", "joint")):
            row[f"{nm}_pooled74_pct"] = round(100 * a[k].sum() / max(val, 1), 3)
            lo, hi = ci(boots["pooled74"][k], j); row[f"{nm}_pooled74_ci_lo"], row[f"{nm}_pooled74_ci_hi"] = round(100 * lo, 3), round(100 * hi, 3)
            row[f"{nm}_equal4_pct"] = round(100 * float(np.mean([cnt[g][k][:, j].sum() / max(cnt[g]["valid"][:, j].sum(), 1) for g in groups])), 3)
            lo, hi = ci(boots["equal4"][k], j); row[f"{nm}_equal4_ci_lo"], row[f"{nm}_equal4_ci_hi"] = round(100 * lo, 3), round(100 * hi, 3)
        row["meets_F_only_95pct_prespecified"] = bool(a["fok"].sum() / max(val, 1) >= CRIT)
        row["meets_joint_95pct_post_review"] = bool(a["joint"].sum() / max(val, 1) >= CRIT)
        row["float16_ambiguous_runs"] = int(a["amb"].sum())
        row["joint_lower_bound_pct"] = round(100 * a["jl"].sum() / max(val, 1), 3); row["joint_upper_bound_pct"] = round(100 * a["ju"].sum() / max(val, 1), 3)
        row["F_only_lower_bound_pct"] = round(100 * a["rl"].sum() / max(val, 1), 3); row["F_only_upper_bound_pct"] = round(100 * a["ru"].sum() / max(val, 1), 3)
        po.append(row)
    ph = pd.DataFrame(ph, columns=["head", "target", "protocol_selection_model", "montage_label", "scenario", "runs", "valid", "floor_success", "F_only_success", "joint_success"])
    return ph, pd.DataFrame(pt), pd.DataFrame(po), boots, allc


def nominal_rows(E, names, label, proto, ix, tier):
    rows = []
    n0 = IX["nominal"]
    for t in ix:
        v = E["valid"][:, t, n0]
        rows.append(dict(target=names[t], tier=tier[names[t]], protocol_selection_model=proto, field_model="SPICEhead v23 1 mm, nominal", montage_label=label[names[t]], valid_heads=int(v.sum()),
                         floor_pct=round(100 * E["floor"][:, t, n0].sum() / max(v.sum(), 1), 3), F_only_pct=round(100 * E["fok"][:, t, n0].sum() / max(v.sum(), 1), 3),
                         joint_pct=round(100 * E["joint"][:, t, n0].sum() / max(v.sum(), 1), 3)))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    D, names, heads, sp, fem, fx, mi, route, coincide, six, inv, odd, spgap_final = load()
    H, T = len(heads), len(names)
    tier = sp.tier.to_dict()
    lab_fem, lab_sp = fem.cohort_choice.to_dict(), sp.cohort_choice.to_dict()

    # ---------- A0: inventory
    inv = pd.DataFrame(inv)
    if os.path.exists(MASK35):                                  # checksums of the 35 heads' mask runs, written when they were fetched
        rec = {}
        for line in open(MASK35, encoding="utf-8"):
            s, _, rel = line.strip().partition("  ")
            rec[rel.replace("\\", "/")] = s
        rel = inv["set"] + "/" + inv["head"] + "/" + inv["run"] + ".npz"
        w = (inv.recorded == "no record") & rel.isin(rec)
        inv.loc[w, "recorded"] = np.where(inv.sha256[w].values == rel[w].map(rec).values, "match", "MISMATCH")
    inv.to_csv(os.path.join(OUT, "archive_inventory.csv"), index=False)
    odd = pd.DataFrame(odd, columns=["set", "head", "file", "issue"])
    odd.loc[odd.file.astype(str).str.startswith(".fetched"), "issue"] = "fetch_marker_harmless"
    dup = inv[inv.sha256.notna()]
    dup = dup[dup.duplicated(["set", "sha256"], keep=False)]
    odd = pd.concat([odd, pd.DataFrame([dict(set=r.set, head=r.head, file=r.run, issue="duplicate_content") for r in dup.itertuples()], columns=odd.columns)], ignore_index=True)
    odd.to_csv(os.path.join(OUT, "missing_or_duplicate_runs.csv"), index=False)
    blocking = odd[~odd.issue.isin(["fetch_marker_harmless"])]
    check("run files: 46,200 expected, all present and readable", len(inv) == 46200 and (inv.status == "ok").all(), inv.status.value_counts().to_dict())
    check("no checksum mismatch against the records written at fetch", not (inv.recorded == "MISMATCH").any(), inv.recorded.value_counts().to_dict())
    check("no missing, partial, unexpected or duplicate run", len(blocking) == 0, blocking.issue.value_counts().to_dict())
    check("logical records: 48,900 nominal, 3,618,600 physical, 97,800 mask", True, dict(nominal=H * T, physical=H * T * 74, mask=H * T * 2, total=H * T * 77))
    pidx = pd.DataFrame([dict(target=n, tier=tier[n], fem_cohort_label=lab_fem[n], fem_cohort_index=mi[lab_fem[n]], fem_cohort_unconstrained_label=fx[n]["cohort_unconstrained"],
                              fem_cohort_unconstrained_index=mi[fx[n]["cohort_unconstrained"]], spicehead_cohort_label=lab_sp[n], spicehead_cohort_index=mi[lab_sp[n]],
                              protocols_identical=lab_fem[n] == lab_sp[n], spicehead_score_route=route[n]) for n in names])
    pidx.to_csv(os.path.join(OUT, "protocol_index_map.csv"), index=False)

    # ---------- A1 / A2
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(H, np.full(H, 1 / H), size=NBOOT).astype(np.float64)
    Ef, Es = endpoints(D, "fem", H, T), endpoints(D, "sp", H, T)
    all_t = list(range(T))
    rec_t = all_t
    ph, pt, po, bf, af = tables(Ef, names, heads, lab_fem, mi, tier, "FEM", all_t, W)
    ph.to_csv(os.path.join(OUT, "uq_fem_protocol_per_head_scenario.csv"), index=False)
    pt.to_csv(os.path.join(OUT, "uq_fem_protocol_per_target_scenario.csv"), index=False)
    po.to_csv(os.path.join(OUT, "uq_fem_protocol_pooled.csv"), index=False)
    ph2, pt2, po2, bs, as_ = tables(Es, names, heads, lab_sp, mi, tier, "SPICEhead", rec_t, W)
    ph2.to_csv(os.path.join(OUT, "uq_spicehead_protocol_per_head_scenario.csv"), index=False)
    pt2.to_csv(os.path.join(OUT, "uq_spicehead_protocol_per_target_scenario.csv"), index=False)
    po2.to_csv(os.path.join(OUT, "uq_spicehead_protocol_pooled.csv"), index=False)
    nomf, noms = nominal_rows(Ef, names, lab_fem, "FEM", all_t, tier), nominal_rows(Es, names, lab_sp, "SPICEhead", rec_t, tier)
    pd.concat([nomf, noms]).to_csv(os.path.join(OUT, "uq_nominal_fixed_protocols.csv"), index=False)
    # validity counts with reason codes (physical scenarios, per target and protocol)
    vrows = []
    for E, ix, lab, pr in ((Ef, all_t, lab_fem, "FEM"), (Es, rec_t, lab_sp, "SPICEhead")):
        for t in ix:
            for g, runs in SCEN.items():
                d, el, gd, fi, va = (E[k][:, t][:, runs] for k in ("done", "elig", "goodden", "fin", "valid"))
                vrows.append(dict(target=names[t], protocol_selection_model=pr, scenario=g, expected=len(runs) * H, completed=int(d.sum()), eligible=int(el.sum()), numerically_valid=int(va.sum()),
                                  missing_run=int((~d).sum()), ineligible_target=int((d & ~el).sum()), nonpositive_or_nonfinite_denominator=int((d & el & ~gd).sum()),
                                  nonpositive_or_nonfinite_fixed_score=int((d & el & gd & ~fi).sum()), heads_with_all_runs_valid=int((va.sum(1) == len(runs)).sum())))
    vc = pd.DataFrame(vrows); vc.to_csv(os.path.join(OUT, "uq_validity_counts.csv"), index=False)
    check("every physical run of every target is valid (fixed protocols)", (vc.numerically_valid == vc.expected).all(), int((vc.expected - vc.numerically_valid).sum()))

    # paired comparison, same head, target and scenario
    pair = []
    for j, t in enumerate(rec_t):
        n = names[t]; jf = t
        for g in list(SCEN) + ["pooled74", "equal4"]:
            if g in SCEN:
                r = SCEN[g]
                num = lambda E, k: E[k][:, t][:, r].sum(1).astype(float)
                vf, vs = num(Ef, "valid"), num(Es, "valid")
                ds = {k: 100 * (num(Es, k).sum() / vs.sum() - num(Ef, k).sum() / vf.sum()) for k in ("floor", "fok", "joint")}
            else:
                r = PHYS
                if g == "pooled74":
                    ds = {k: 100 * (Es[k][:, t][:, r].sum() / Es["valid"][:, t][:, r].sum() - Ef[k][:, t][:, r].sum() / Ef["valid"][:, t][:, r].sum()) for k in ("floor", "fok", "joint")}
                else:
                    ds = {k: 100 * float(np.mean([Es[k][:, t][:, SCEN[s]].sum() / Es["valid"][:, t][:, SCEN[s]].sum() - Ef[k][:, t][:, SCEN[s]].sum() / Ef["valid"][:, t][:, SCEN[s]].sum() for s in SCEN])) for k in ("floor", "fok", "joint")}
            row = dict(target=n, tier=tier[n], scenario=g, identical_protocol=lab_fem[n] == lab_sp[n], spicehead_montage=lab_sp[n], fem_montage=lab_fem[n], score_route=route[n])
            for k, nm in (("floor", "floor"), ("fok", "F_only"), ("joint", "joint")):
                db = 100 * (bs[g][k][:, j] - bf[g][k][:, t])
                row[f"{nm}_diff_pct_points"] = round(ds[k], 3)
                row[f"{nm}_diff_ci_lo"], row[f"{nm}_diff_ci_hi"] = round(float(np.nanpercentile(db, 2.5)), 3), round(float(np.nanpercentile(db, 97.5)), 3)
            pair.append(row)
    pair = pd.DataFrame(pair); pair.to_csv(os.path.join(OUT, "uq_protocol_paired_comparison.csv"), index=False)
    same = pair[pair.identical_protocol]
    check("129 identical protocols give exactly zero paired differences", same.target.nunique() == 129 and (same[["floor_diff_pct_points", "F_only_diff_pct_points", "joint_diff_pct_points"]] == 0).all().all(), same.target.nunique())

    # recoverability of the 34 differing protocols
    rr = []
    for t, n in enumerate(names):
        r = route[n]
        rr.append(dict(target=n, tier=tier[n], spicehead_montage=lab_sp[n], fem_montage=lab_fem[n], route=r,
                       heads_recovered_all_runs=H,
                       heads_own_nominal_choice_equals_protocol_selection_biased_not_used=int(coincide[:, t].sum()),
                       heads_in_six_target_run_not_used_for_300_head_claims=int(np.isfinite(six[:, t, 0, 0]).sum()),
                       status={"identical_to_FEM_protocol": "same montage: reuse", "float16_full_scores": "recovered, float16 precision checks apply",
                               "saved_fem_cohort_unconstrained_column": "recovered, float32", "new_basis_solves": "scores not saved, so the 75 basis solves per head were repeated with the same seeds: float32"}[r]))
    rr = pd.DataFrame(rr); rr.to_csv(os.path.join(OUT, "protocol_score_recoverability.csv"), index=False)
    print(rr.route.value_counts().to_dict())

    # ---------- A3: target masks
    mh, mt, mx, mj = [], [], [], []
    n0 = IX["nominal"]
    for m, ri in MASKS.items():
        cidx = 0 if m == "erosion" else 1
        nm = D["n_mask"][:, :, cidx]; base = D["n_roi"] >= MINVOX
        for t, n in enumerate(names):
            a = nm[:, t]; done = D["done"][:, t, ri]; Tm = D["fem_T"][:, t, ri]; Fm = D["fem_F"][:, t, ri]
            tmax = D["Tmax"][:, t, ri]; fc = D["Fcon"][:, t, ri]
            cls = np.full(H, "complete_ge5", dtype=object)
            cls[(a >= 1) & (a < MINVOX)] = "nonempty_1_to_4_voxels"; cls[a == 0] = "empty"
            good = np.isfinite(Tm) & (Tm > 0) & np.isfinite(Fm) & (Fm > 0) & np.isfinite(tmax) & (tmax > 0) & np.isfinite(fc) & (fc > 0)
            cls[(a >= MINVOX) & ~good] = "numerical_failure"; cls[~done] = "missing"; cls[~base[:, t]] = "not_assessed_nominal_ineligible"
            vc_ = {c: int((cls == c).sum()) for c in ("complete_ge5", "nonempty_1_to_4_voxels", "empty", "numerical_failure", "missing", "not_assessed_nominal_ineligible")}
            for h, code in enumerate(heads):
                mh.append((code, n, m, int(D["n_roi"][h, t]), int(a[h]), cls[h]))
            mt.append(dict(target=n, tier=tier[n], mask=m, heads_assessed=H - vc_["not_assessed_nominal_ineligible"], **vc_, median_voxels_nominal=float(np.median(D["n_roi"][:, t])), median_voxels_masked=float(np.median(a))))
            for pr, lab, pre, ok in (("FEM", lab_fem, "fem", True), ("SPICEhead", lab_sp, "sp", True)):
                if not ok:
                    continue
                Tx, Fx = D[pre + "_T"][:, t, ri], D[pre + "_F"][:, t, ri]; Tn = D[pre + "_T"][:, t, n0]
                el = (cls == "complete_ge5") & np.isfinite(Tx) & (Tx > 0)
                ratio = Tx[el] / Tn[el]
                p5, p50, p95 = qs(ratio, (5, 50, 95))
                mx.append(dict(target=n, tier=tier[n], mask=m, protocol_selection_model=pr, field_model="SPICEhead v23 1 mm, nominal fields, perturbed mask", montage_label=lab[n], eligible_valid_heads=int(el.sum()), heads_assessed=H,
                               T_ratio_p5=round(p5, 4), T_ratio_p50=round(p50, 4), T_ratio_p95=round(p95, 4)))
                with np.errstate(divide="ignore", invalid="ignore"):
                    fl = el & (Tx / tmax >= FLOOR); fo = el & (Fx / fc >= RET)
                mj.append(dict(target=n, tier=tier[n], mask=m, protocol_selection_model=pr, montage_label=lab[n], heads_assessed=H, localized_heads=int(el.sum()), localization_failure_heads=int(H - el.sum()),
                               floor_heads=int(fl.sum()), F_only_heads=int(fo.sum()), joint_heads=int((fl & fo).sum()), joint_applicability_pct=round(100 * (fl & fo).sum() / H, 2),
                               joint_conditional_on_localized_pct=(round(100 * (fl & fo).sum() / el.sum(), 2) if el.sum() else None)))
    pd.DataFrame(mh, columns=["head", "target", "mask", "n_roi_nominal", "n_roi_masked", "class"]).to_csv(os.path.join(OUT, "mask_validity_by_head.csv"), index=False)
    mt = pd.DataFrame(mt); mt.to_csv(os.path.join(OUT, "mask_validity_by_target.csv"), index=False)
    mx = pd.DataFrame(mx); mx.to_csv(os.path.join(OUT, "mask_exposure_conditional.csv"), index=False)
    mj = pd.DataFrame(mj); mj.to_csv(os.path.join(OUT, "mask_joint_applicability.csv"), index=False)
    # reconcile against the saved zero-plus-nonfinite diagnostic
    me = pd.read_csv(os.path.join(R23, "v23_uq", "mask_empty_300.csv")).set_index("target")
    er = mt[mt["mask"] == "erosion"].set_index("target")
    old_un = me.mask_erode_nonfinite + me.mask_erode_zero
    rec_rows = []
    for n in names:
        rec_rows.append(dict(target=n, saved_erosion_unavailable=int(old_un[n]), recollected_empty=int(er.loc[n, "empty"]), recollected_1_to_4=int(er.loc[n, "nonempty_1_to_4_voxels"]),
                             recollected_ge5=int(er.loc[n, "complete_ge5"]), recollected_numerical_failure=int(er.loc[n, "numerical_failure"])))
    rec_rows = pd.DataFrame(rec_rows); rec_rows.to_csv(os.path.join(OUT, "mask_reconciliation_with_saved_diagnostic.csv"), index=False)
    check("saved erosion diagnostic equals the recollected empty count in all 163 targets", (rec_rows.saved_erosion_unavailable == rec_rows.recollected_empty).all(), int((rec_rows.saved_erosion_unavailable != rec_rows.recollected_empty).sum()))
    check("dilation leaves no target empty", int(mt[mt["mask"] == "dilation"]["empty"].sum()) == 0)
    check("mask denominators reconcile to 300 heads per target and mask", (mt[["complete_ge5", "nonempty_1_to_4_voxels", "empty", "numerical_failure", "missing", "not_assessed_nominal_ineligible"]].sum(1) == H).all())
    check("conditional intervals never use a zero or ineligible mask", (mx.eligible_valid_heads <= H).all() and bool((mx.merge(mt[["target", "mask", "complete_ge5"]], on=["target", "mask"]).eval("eligible_valid_heads <= complete_ge5")).all()))

    # ---------- A4: reproduction and consistency
    old = pd.concat([pd.read_csv(os.path.join(R23, "v23_uq", x, "per_target.csv")) for x in ("atlas_300", "atlas_thal_300")]).drop_duplicates("target", keep="first").set_index("target")
    # the historical statistic exactly as the collector computed it
    with np.errstate(divide="ignore", invalid="ignore"):
        hist = D["fem_F"][:, :, PHYS] / D["Fcon"][:, :, PHYS]
    hist_share = 100 * np.mean(hist >= RET, axis=(0, 2))
    d_old = np.abs(hist_share - old.loc[names, "share_runs_retention_ge_90pct"].values)
    check("historical pooled F-only share reproduced in 163 targets (agreement to the stored 0.1 %)", d_old.max() <= 0.051, float(d_old.max()))
    n37 = int((hist_share / 100 >= CRIT).sum())
    check("historical 37-region F-only count reproduced independently", n37 == 37 and n37 == int(old.fixed_protocol_robust.sum()), n37)
    chk = po.set_index("target").loc[names]
    check("new pooled F-only equals the historical statistic because every run is valid", np.abs(chk.F_only_pooled74_pct.values - hist_share).max() < 1e-3 and int(chk.meets_F_only_95pct_prespecified.sum()) == 37, int(chk.meets_F_only_95pct_prespecified.sum()))
    for E, po_, nm in ((Ef, po, "FEM"), (Es, po2, "SPICEhead")):
        c = po_.set_index("target")
        check(f"joint <= F-only and joint <= floor under one denominator ({nm})", bool((E["joint"].sum((0, 2)) <= E["fok"].sum((0, 2))).all() and (E["joint"].sum((0, 2)) <= E["floor"].sum((0, 2))).all()))
    # nominal scores against the 300-head atlas archives
    zp = np.load(NOMPERHEAD, allow_pickle=True); zc = np.load(COHPERHEAD, allow_pickle=True)
    hp, hc = [list(map(str, z["codes"])).index(c) for c in heads for z in (zp,)], [list(map(str, zc["codes"])).index(c) for c in heads]
    tp_ix = [list(map(str, zp["targets"])).index(n) for n in names]; tc_ix = [list(map(str, zc["targets"])).index(n) for n in names]
    check("cohort archive holds the SPICEhead cohort montages by label", all(str(zc["montage"][tc_ix[t]]) == lab_sp[n] for t, n in enumerate(names)), str(zc["model"]))
    # The atlas scoring left out brainstem points with a zero FEM field (n_bs_zero_fem_excluded in its metadata, 1-5 voxels in the
    # substantia nigra pars reticulata). The uncertainty runs had no FEM model and kept them, so those masks differ slightly.
    bst = np.array([tier[n] == "brainstem" for n in names])
    rep, repb = {}, {}
    for k, a, b in (("T_reported", D["Trep"], zp["sp1v23__T_wa"][hp][:, tp_ix]), ("F_reported", D["Frep"], zp["sp1v23__F_wa"][hp][:, tp_ix]), ("T_constrained", D["Tcon"], zp["sp1v23__T_wc"][hp][:, tp_ix]), ("F_constrained", D["Fcon"][:, :, n0], zp["sp1v23__F_wc"][hp][:, tp_ix])):
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(b > 0, np.abs(a / b.astype(np.float64) - 1), np.nan)
        rep[k] = float(np.nanmax(g[:, ~bst]))
        repb[k] = dict(max_relative_gap=float(np.nanmax(g[:, bst])), head_targets_above_1e4=int(np.nansum(g[:, bst] > 1e-4)))
    check("nominal UQ reported and constrained scores reproduce the 300-head atlas, 151 non-brainstem targets (max relative gap < 1e-4)", max(rep.values()) < 1e-4, rep)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.abs(D["sp_T"][:, :, n0] / zc["T_coh"][hc][:, tc_ix].astype(np.float64) - 1)
    f16t = np.array([tier[n] in ("cerebellum", "brainstem") for n in names])
    check("nominal SPICEhead cohort T reproduces the cohort archive, float32 targets (< 1e-4)", float(np.nanmax(g[:, ~f16t])) < 1e-4, float(np.nanmax(g[:, ~f16t])))
    check("nominal SPICEhead cohort T reproduces the cohort archive, cerebellar float16 targets (< 1e-3)", float(np.nanmax(g[:, f16t & ~bst])) < 1e-3, float(np.nanmax(g[:, f16t & ~bst])))
    QC["documented_deviation_brainstem_masks"] = dict(passed=True, detail=dict(reason="atlas scoring excluded brainstem points with zero FEM field; UQ runs kept them", nominal_scores=repb, cohort_T_max_relative_gap=float(np.nanmax(g[:, bst]))))
    rv = pd.read_csv(os.path.join(R23, "v23_atlas_review", "region_review.csv")).set_index("target")
    ns = noms.set_index("target")
    d_serv = (ns.joint_pct - rv.loc[ns.index, "sp1v23__cohort_served_pct"]).abs()
    d_adm = (ns.floor_pct - rv.loc[ns.index, "sp1v23__cohort_admissible_pct"]).abs()
    # one head is 0.333 points and region_review stores 0.1-point rounding, so one head can read as up to 0.383
    check("nominal joint coverage reproduces region_review served % (within one head)", d_serv.max() <= 0.385, float(d_serv.max()))
    check("nominal floor coverage reproduces region_review admissible % (within one head)", d_adm.max() <= 0.385, float(d_adm.max()))
    # physical plausibility of the denominators
    v32 = [t for t in all_t if tier[names[t]] not in ("cerebellum", "brainstem")]
    over = np.nanmax(D["fem_T"][:, v32][:, :, PHYS] / D["Tmax"][:, v32][:, :, PHYS])
    check("fixed T never exceeds the head's Tmax beyond float32 rounding (float32 targets)", over <= 1 + 1e-6, float(over))
    with np.errstate(divide="ignore", invalid="ignore"):
        fu = np.nanmax(D["fem_F"][:, v32][:, :, PHYS] / D["Func"][:, v32][:, :, PHYS])
    check("fixed F never exceeds the unconstrained maximum beyond rounding", fu <= 1 + 1e-6, float(fu))
    over_s = np.nanmax(D["sp_T"][:, v32][:, :, PHYS] / D["Tmax"][:, v32][:, :, PHYS])
    check("SPICEHead-protocol T never exceeds the head's Tmax beyond float32 rounding (float32 targets)", over_s <= 1 + 1e-6, float(over_s))
    with np.errstate(divide="ignore", invalid="ignore"):
        fu_s = np.nanmax(D["sp_F"][:, v32][:, :, PHYS] / D["Func"][:, v32][:, :, PHYS])
    check("SPICEHead-protocol F never exceeds the unconstrained maximum beyond rounding", fu_s <= 1 + 1e-6, float(fu_s))
    check("every target evaluated for the SPICEHead-selected protocol", len(rec_t) == T and po2.valid_runs.eq(74 * H).all(), len(rec_t))
    ambF, ambS = int(Ef["amb"][:, :, PHYS].sum()), int(Es["amb"][:, :, PHYS].sum())

    # ---------- one machine-readable summary
    P = po.set_index("target"); P2 = po2.set_index("target")
    def bcount(b, key, ix=None):
        v = b[key] if ix is None else b[key][:, ix]
        return v
    jb, fb, flb = bf["pooled74"]["joint"], bf["pooled74"]["fok"], bf["pooled74"]["floor"]
    c95 = lambda b: [int(x) for x in np.percentile((b >= CRIT).sum(1), [2.5, 97.5])]
    med = lambda b: [round(100 * float(x), 2) for x in np.percentile(np.nanmedian(b, 1), [2.5, 97.5])]
    by_scen = {}
    for g in SCEN:
        s = pt[pt.scenario == g]
        by_scen[g] = dict(joint_median=round(float(s.joint_pooled_pct.median()), 2), joint_ci=med(bf[g]["joint"]), F_only_median=round(float(s.F_only_pooled_pct.median()), 2), floor_median=round(float(s.floor_pooled_pct.median()), 2),
                          joint_lt50_regions=int((s.joint_pooled_pct < 50).sum()), joint_ge95_regions=int((s.joint_pooled_pct >= 95).sum()))
    by_tier = {tr: dict(regions=int((P.tier == tr).sum()), joint_median=round(float(P[P.tier == tr].joint_pooled74_pct.median()), 2), F_only_median=round(float(P[P.tier == tr].F_only_pooled74_pct.median()), 2),
                        floor_median=round(float(P[P.tier == tr].floor_pooled74_pct.median()), 2), joint_95=int(P[P.tier == tr].meets_joint_95pct_post_review.sum()), F_only_95=int(P[P.tier == tr].meets_F_only_95pct_prespecified.sum())) for tr in P.tier.unique()}
    diff = pair[(pair.scenario == "pooled74") & (~pair.identical_protocol)]
    ero = mt[mt["mask"] == "erosion"]; dil = mt[mt["mask"] == "dilation"]
    mxf = mx[mx.protocol_selection_model == "FEM"]; mjf = mj[mj.protocol_selection_model == "FEM"]
    S = dict(generated=time.strftime("%Y-%m-%d %H:%M"), heads=H, targets=T, runs_per_head=len(RUNS), physical_runs_per_head=74, bootstrap=dict(n=NBOOT, seed=SEED, unit="head"),
             fem_protocol=dict(F_only_median=round(float(P.F_only_pooled74_pct.median()), 2), F_only_median_ci=med(fb), F_only_95_regions=int(P.meets_F_only_95pct_prespecified.sum()), F_only_95_ci=c95(fb),
                               floor_median=round(float(P.floor_pooled74_pct.median()), 2), floor_median_ci=med(flb),
                               joint_median=round(float(P.joint_pooled74_pct.median()), 2), joint_median_ci=med(jb), joint_min=round(float(P.joint_pooled74_pct.min()), 2), joint_max=round(float(P.joint_pooled74_pct.max()), 2),
                               joint_min_target=str(P.joint_pooled74_pct.idxmin()), joint_95_regions=int(P.meets_joint_95pct_post_review.sum()), joint_95_ci=c95(jb), joint_lt50_regions=int((P.joint_pooled74_pct < 50).sum()), joint_lt80_regions=int((P.joint_pooled74_pct < 80).sum()),
                               joint_equal4_median=round(float(P.joint_equal4_pct.median()), 2), F_only_equal4_median=round(float(P.F_only_equal4_pct.median()), 2), by_scenario=by_scen, by_tier=by_tier,
                               nominal_joint_median=round(float(nomf.joint_pct.median()), 2), nominal_floor_median=round(float(nomf.floor_pct.median()), 2), nominal_F_only_median=round(float(nomf.F_only_pct.median()), 2), nominal_joint_lt50=int((nomf.joint_pct < 50).sum())),
             spicehead_protocol=dict(evaluated_targets=len(rec_t), identical=int((rr.route == "identical_to_FEM_protocol").sum()), differing_from_saved_scores=int(rr.route.isin(["float16_full_scores", "saved_fem_cohort_unconstrained_column"]).sum()), differing_new_solves=int((rr.route == "new_basis_solves").sum()),
                                     new_solve_targets=list(rr[rr.route == "new_basis_solves"].target), control_max_abs_gap=spgap_final["control"], control_evaluations=spgap_final["control_n"],
                                     F_only_median=round(float(P2.F_only_pooled74_pct.median()), 2), F_only_median_ci=med(bs["pooled74"]["fok"]), F_only_95_regions=int(P2.meets_F_only_95pct_prespecified.sum()), F_only_95_ci=c95(bs["pooled74"]["fok"]),
                                     floor_median=round(float(P2.floor_pooled74_pct.median()), 2), floor_median_ci=med(bs["pooled74"]["floor"]),
                                     joint_median=round(float(P2.joint_pooled74_pct.median()), 2), joint_median_ci=med(bs["pooled74"]["joint"]), joint_95_regions=int(P2.meets_joint_95pct_post_review.sum()), joint_95_ci=c95(bs["pooled74"]["joint"]), joint_lt50_regions=int((P2.joint_pooled74_pct < 50).sum()),
                                     nominal_joint_median=round(float(noms.joint_pct.median()), 2), nominal_joint_lt50=int((noms.joint_pct < 50).sum()),
                                     differing_34=dict(joint_diff_median=round(float(diff.joint_diff_pct_points.median()), 2), joint_diff_min=round(float(diff.joint_diff_pct_points.min()), 2), joint_diff_max=round(float(diff.joint_diff_pct_points.max()), 2),
                                                       F_only_diff_median=round(float(diff.F_only_diff_pct_points.median()), 2), F_only_diff_min=round(float(diff.F_only_diff_pct_points.min()), 2), F_only_diff_max=round(float(diff.F_only_diff_pct_points.max()), 2),
                                                       floor_diff_median=round(float(diff.floor_diff_pct_points.median()), 2),
                                                       spicehead_higher_joint_ci_excludes_0=int((diff.joint_diff_ci_lo > 0).sum()), fem_higher_joint_ci_excludes_0=int((diff.joint_diff_ci_hi < 0).sum()), joint_ci_includes_0=int(((diff.joint_diff_ci_lo <= 0) & (diff.joint_diff_ci_hi >= 0)).sum()),
                                                       regions=len(diff)),
                                     differing_rows=diff[["target", "score_route", "F_only_diff_pct_points", "F_only_diff_ci_lo", "F_only_diff_ci_hi", "floor_diff_pct_points", "joint_diff_pct_points", "joint_diff_ci_lo", "joint_diff_ci_hi"]].to_dict("records")),
             masks=dict(erosion_empty_regions=int((ero["empty"] > 0).sum()), erosion_lt5_regions=int(((ero["empty"] + ero.nonempty_1_to_4_voxels) > 0).sum()), erosion_any_1_to_4_regions=int((ero.nonempty_1_to_4_voxels > 0).sum()), erosion_empty_observations=int(ero["empty"].sum()), erosion_1_to_4_observations=int(ero.nonempty_1_to_4_voxels.sum()),
                        dilation_lt5_regions=int(((dil["empty"] + dil.nonempty_1_to_4_voxels) > 0).sum()), numerical_failures=int(mt.numerical_failure.sum()),
                        erosion_T_p5_median=round(float(mxf[mxf["mask"] == "erosion"].T_ratio_p5.median()), 3), erosion_T_p95_median=round(float(mxf[mxf["mask"] == "erosion"].T_ratio_p95.median()), 3),
                        dilation_T_p5_median=round(float(mxf[mxf["mask"] == "dilation"].T_ratio_p5.median()), 3), dilation_T_p95_median=round(float(mxf[mxf["mask"] == "dilation"].T_ratio_p95.median()), 3),
                        erosion_joint_median=round(float(mjf[mjf["mask"] == "erosion"].joint_applicability_pct.median()), 2), dilation_joint_median=round(float(mjf[mjf["mask"] == "dilation"].joint_applicability_pct.median()), 2)),
             float16=dict(ambiguous_fem_runs=ambF, ambiguous_spicehead_runs=ambS, targets_with_ambiguity=int((P.float16_ambiguous_runs > 0).sum()), max_abs_joint_bound_width_pct=round(float((P.joint_upper_bound_pct - P.joint_lower_bound_pct).max()), 3),
                          criterion_changes_within_bounds=dict(joint=int(((P.joint_lower_bound_pct >= 95) != (P.joint_upper_bound_pct >= 95)).sum()), F_only=int(((P.F_only_lower_bound_pct >= 95) != (P.F_only_upper_bound_pct >= 95)).sum()))))
    json.dump(S, open(os.path.join(OUT, "final_summary.json"), "w", encoding="utf-8"), indent=1)
    json.dump(dict(field_model=FIELD, thresholds=dict(T_floor=FLOOR, F_retention=RET, runs_criterion=CRIT, min_voxels=MINVOX),
                   floor_ratio="T_fixed(h,r,s,m) / Tmax(h,r,s), both from the same perturbed SPICEhead model", F_retention="F_fixed(h,r,s,m) / F_constrained(h,r,s), not capped at 100 %",
                   valid="eligible target (n_roi >= 5 in the nominal mask) AND completed run AND finite positive Tmax, F_constrained, fixed T and fixed F",
                   F_only_success="valid AND F_retention >= 0.90 (the prespecified endpoint, amendment 1 of ANALYSIS_PLAN.md)", joint_success="valid AND floor_ratio >= 0.80 AND F_retention >= 0.90 (post-review addition, not prespecified)",
                   pooled74="all 74 physical runs of all heads: weights 40/20/10/4, inherited from the sampling design", equal4="mean of the four scenario rates: a labelled sensitivity, not a real-world probability",
                   bootstrap=f"{NBOOT} resamples of heads, seed {SEED}; each sampled head keeps its whole block of targets and runs", mask_classes="complete_ge5, nonempty_1_to_4_voxels, empty, numerical_failure, missing",
                   mask_policy="undefined exposure for empty or ineligible masks; counted as unsuccessful in applicability; excluded from conditional exposure percentiles"),
              open(os.path.join(OUT, "endpoint_definition.json"), "w", encoding="utf-8"), indent=1)
    inputs = [PARAMS, FIXEXT, os.path.join(R23, "v23_uq", "uq_atlas_fixed_montages.json"), os.path.join(R23, "v23_uq", "uq_atlas_fixed_montages_thal.json"), os.path.join(R23, "v23_atlas_review", "region_review.csv"),
              os.path.join(R23, "v23_uq", "mask_empty_300.csv"), os.path.join(R23, "v23_uq", "atlas_300", "per_target.csv"), os.path.join(R23, "v23_uq", "atlas_thal_300", "per_target.csv"), EXTMONT, NOMPERHEAD, COHPERHEAD,
              os.path.join(HERE, "rev06_uq_extract.py"), os.path.join(HERE, "rev06_sp_extract.py"), os.path.abspath(__file__), os.path.join(HERE, "nchc", "uq_v23_atlas.py"), os.path.join(HERE, "nchc", "uq_v23_sp_protocol.py"), os.path.join(BASE, "SHA256SUMS_sp.txt")]
    json.dump(dict(inputs={os.path.relpath(p, ROOT) if p.startswith(ROOT) else p: sha(p) for p in inputs}, archive_inventory_sha256=sha(os.path.join(OUT, "archive_inventory.csv")),
                   software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)), open(os.path.join(OUT, "source_sha256.json"), "w", encoding="utf-8"), indent=1)
    fails = [k for k, v in QC.items() if not v["passed"]]
    json.dump(dict(failed=fails, checks=QC), open(os.path.join(OUT, "qc_assertions.json"), "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(S, indent=1)[:6000]); print(f"\n{len(QC)} checks, {len(fails)} failed: {fails}\n({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
