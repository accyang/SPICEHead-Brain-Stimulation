"""
Cohort validation of the deep-focality levers on the decade-spread subset.

Per subject, for each of the 4 sphere ROIs (roi_coords.json), compare:
  A  best two-pair montage over the 19 10/20 sites
  B  best two-pair montage over 10/20 + the inferior 9/10 chain (electrodes_1020ext.json)
  C  multi-electrode current steering over a candidate set drawn from B's top montages
  C4 C truncated to 4 electrodes/channel (the practically implementable configuration)

Same 14 subjects as place_1010_subset.py, so this is directly comparable to the earlier 10-10
result (+1-2%) on identical heads. Resumable: subjects with an existing output JSON are skipped.

  python montage_deep_cohort.py <charm_outroot> [dx] [outdir]
"""
import os
import sys
import csv
import json
import time
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)
from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components
from montage_search import (make_multiground_solver, snap, roi_mask, ELEC_RADIUS_MM,
                            MIN_DIST_MM, ROI_RADIUS_MM, EXCLUDE_MM, SCALP, BRAIN)
from montage_deep_focality import TEN20, INFERIOR, env_from, MAX_CURRENT_MA, PER_ELEC_CAP_MA

SUBSET = ["SUBJ01", "SUBJ02", "SUBJ03", "SUBJ04", "SUBJ05", "SUBJ06",
          "SUBJ07", "SUBJ08", "SUBJ09", "SUBJ10", "SUBJ11", "SUBJ12",
          "SUBJ13", "SUBJ14"]
# Scored ROIs. Hippocampus is split L/R (not a bilateral union like thalamus/insula/M1) so a
# lateralised optimum is not averaged away; keys must exist in each m2m/roi_coords.json.
ROI_ORDER = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]

REF = "Cz"                 # global reference; carries the balance current
SCORE = (2,)               # grey matter only, in BOTH T and O (see montage_deep_focality.SCORE)
N_BRAIN = 15000            # GM voxels sampled for O (scoring)
N_OPT = 8000               # smaller subsample used INSIDE the optimiser (final score uses N_BRAIN)
N_STARTS = 4               # seeded starts only - random starts were uniformly poor on the template


def resolve(outroot, cid):
    return os.path.join(outroot, "m2m_SUBJ15asc") if cid.startswith("SUBJ15") \
        else os.path.join(outroot, "m2m_" + cid)


def run_subject(m2m, dx, rng, rois):
    t0 = time.time()
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, affine = mri.head_model(seg, dx)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    brain = np.isin(labels, SCORE)      # scoring mask = grey matter, not anatomical extent
    bidx = np.argwhere(brain)
    bw = (affine[:3, :3] @ bidx.T).T + affine[:3, 3]

    rj = json.load(open(os.path.join(m2m, "roi_coords.json")))
    rmask = {r: roi_mask(labels.shape, bidx, bw, rj[r], ROI_RADIUS_MM) for r in rois}
    outside = {r: (brain & ~roi_mask(labels.shape, bidx, bw, rj[r], EXCLUDE_MM)) for r in rois}

    # one probe set: every ROI's voxels, then a shared brain sample used for O
    roi_vox = {r: np.argwhere(rmask[r]) for r in rois}
    if any(v.shape[0] == 0 for v in roi_vox.values()):
        raise RuntimeError("empty ROI at this dx")
    # GM-only shrinks the sphere ROIs; a median over very few voxels is noisy, so make it visible
    sizes = {r: int(v.shape[0]) for r, v in roi_vox.items()}
    small = [f"{r}={n}" for r, n in sizes.items() if n < 15]
    print(f"    GM ROI sizes: {sizes}" + (f"   <-- SMALL: {', '.join(small)}" if small else ""),
          flush=True)
    bs = bidx[rng.choice(len(bidx), size=min(N_BRAIN, len(bidx)), replace=False)]
    P, sl, o = [], {}, 0
    for r in rois:
        P.append(roi_vox[r]); sl[r] = slice(o, o + len(roi_vox[r])); o += len(roi_vox[r])
    P.append(bs); bs_sl = slice(o, o + len(bs))
    P = np.vstack(P)
    # per-ROI: which sampled brain voxels count as off-target
    off_m = {r: outside[r][bs[:, 0], bs[:, 1], bs[:, 2]] for r in rois}
    opt_m = {}
    for r in rois:
        idx = np.flatnonzero(off_m[r])
        opt_m[r] = idx if len(idx) <= N_OPT else rng.choice(idx, size=N_OPT, replace=False)

    def take(F):
        return np.stack([f[P[:, 0], P[:, 1], P[:, 2]] for f in F]).astype(np.float32)

    # electrodes
    scalp = np.argwhere(labels == SCALP)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020ext.json")))
    evox, elecs, seen = {}, {}, {}
    for lab in TEN20 + INFERIOR:
        if lab not in ej:
            continue
        v = snap(scalp, mri.world_to_voxel(affine, ej[lab]))
        k = tuple(v.astype(int))
        if k in seen:
            continue
        seen[k] = lab
        evox[lab] = v.astype(float)
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                               radius_nodes=ELEC_RADIUS_MM / dx), lab)
    labsX = [l for l in TEN20 + INFERIOR if l in elecs]
    labs20 = [l for l in TEN20 if l in elecs]

    solver, _ = make_multiground_solver(model.G)
    pairs = [(a, b) for a, b in combinations(labsX, 2)
             if np.linalg.norm(evox[a] - evox[b]) * dx >= MIN_DIST_MM]
    PE = {}
    for (a, b) in pairs:
        F = efield_components(model, solver(pair_injection(model, elecs[a], elecs[b], 1e-3)))
        PE[(a, b)] = take(tuple(np.nan_to_num(f) for f in F))
    # single-electrode basis vs REF (superposition: any zero-sum pattern is a combination of these)
    basis_all = {}
    for l in labsX:
        if l == REF:
            continue
        F = efield_components(model, solver(pair_injection(model, elecs[l], elecs[REF], 1e-3)))
        basis_all[l] = take(tuple(np.nan_to_num(f) for f in F))
    print(f"    {len(labsX)} electrodes, {len(pairs)} pairs, setup {time.time()-t0:.0f}s", flush=True)

    # ---- A / B sweeps: one envelope pass, all ROIs scored from it ----
    def sweep(allowed):
        ps = [p for p in pairs if p[0] in allowed and p[1] in allowed]
        best = {r: [] for r in rois}
        for pa, pb in combinations(ps, 2):
            if len(set(pa) | set(pb)) < 4:
                continue
            e = env_from(PE[pa], PE[pb])
            eb = e[bs_sl]
            for r in rois:
                T = float(np.median(e[sl[r]]))
                O = float(np.percentile(eb[off_m[r]], 99))
                if O > 0:
                    best[r].append((T / O, T, O, pa, pb))
        for r in rois:
            best[r].sort(key=lambda x: -x[0])
            best[r] = best[r][:25]
        return best

    t1 = time.time()
    A = sweep(set(labs20))
    B = sweep(set(labsX))
    print(f"    sweeps {time.time()-t1:.0f}s", flush=True)

    # ---- C: current steering, per ROI ----
    out = {}
    for r in rois:
        t2 = time.time()
        freq = {}
        for s_, T_, O_, pa, pb in B[r][:25]:
            for e in set(pa) | set(pb):
                freq[e] = freq.get(e, 0) + 1
        cand = [e for e, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))][:16]
        for e in sorted(set(B[r][0][3]) | set(B[r][0][4])):
            if e not in cand:
                cand.append(e)
        if REF not in cand:
            cand.append(REF)
        free = [e for e in cand if e != REF and e in basis_all]
        nb = len(free)
        Bm = np.stack([basis_all[e] for e in free])          # (nb, 3, nP)

        roi_sl, offi, opti = sl[r], np.flatnonzero(off_m[r]), opt_m[r]
        Br = Bm[:, :, roi_sl]
        Bo_full = Bm[:, :, bs_sl][:, :, offi]
        Bo_opt = Bm[:, :, bs_sl][:, :, opti]

        def currents(v):
            v = np.asarray(v, float)
            inj = 0.5 * (np.abs(v).sum() + abs(v.sum()))
            return v if inj < 1e-9 else v * (MAX_CURRENT_MA / inj)

        def raw(x, y, Boff):
            T = float(np.median(env_from(np.tensordot(x, Br, axes=(0, 0)),
                                         np.tensordot(y, Br, axes=(0, 0)))))
            O = float(np.percentile(env_from(np.tensordot(x, Boff, axes=(0, 0)),
                                             np.tensordot(y, Boff, axes=(0, 0))), 99))
            return (T / O if O > 0 else 0.0), T, O

        def pen_of(x, y):
            p = 0.0
            for c in (x, y):
                peak = max(float(np.abs(c).max()), abs(float(c.sum())))
                if peak > PER_ELEC_CAP_MA:
                    p += 20.0 * (peak - PER_ELEC_CAP_MA)
            return p

        def obj(z):
            x, y = currents(z[:nb]), currents(z[nb:])
            s, _, _ = raw(x, y, Bo_opt)
            return -s + pen_of(x, y)

        def seed(pa, pb):
            z = np.zeros(2 * nb)
            for j, pr in enumerate((pa, pb)):
                for lab, sg in ((pr[0], +1.0), (pr[1], -1.0)):
                    if lab in free:
                        z[j * nb + free.index(lab)] = sg
            return z

        chk, _, _ = raw(*[currents(v) for v in np.split(seed(*B[r][0][3:5]), 2)], Bo_full)
        best_z, best_f = None, np.inf
        for i in range(min(N_STARTS, len(B[r]))):
            res = minimize(obj, seed(*B[r][i][3:5]), method="Powell",
                           options={"maxfev": 25000, "xtol": 1e-3, "ftol": 1e-4})
            if res.fun < best_f:
                best_f, best_z = res.fun, res.x
        xs, ys = currents(best_z[:nb]), currents(best_z[nb:])
        cS, cT, cO = raw(xs, ys, Bo_full)          # scored on the FULL sample, not the opt subsample

        # truncate to 4 electrodes/channel
        def full(v):
            return np.append(v, -v.sum())

        def trunc(t, k):
            z = np.zeros_like(t)
            z[np.argsort(-np.abs(t))[:k]] = t[np.argsort(-np.abs(t))[:k]]
            p, n = z > 0, z < 0
            if not p.any() or not n.any():
                return None
            z[p] *= MAX_CURRENT_MA / z[p].sum()
            z[n] *= MAX_CURRENT_MA / -z[n].sum()
            return z
        tx, ty = trunc(full(xs), 4), trunc(full(ys), 4)
        c4 = raw(tx[:nb], ty[:nb], Bo_full)[0] if tx is not None and ty is not None else None

        out[r] = {"A": round(A[r][0][0], 4), "A_montage": f"{'+'.join(A[r][0][3])}/{'+'.join(A[r][0][4])}",
                  "B": round(B[r][0][0], 4), "B_montage": f"{'+'.join(B[r][0][3])}/{'+'.join(B[r][0][4])}",
                  "B_inferior": sorted([e for e in set(B[r][0][3]) | set(B[r][0][4]) if e in INFERIOR]),
                  "C": round(cS, 4), "C_T": round(cT, 5), "C_O": round(cO, 5),
                  "C4": round(c4, 4) if c4 else None,
                  "seed_check": round(chk, 4), "n_cand": len(cand)}
        print(f"    {r:>12}: A={out[r]['A']:.3f} B={out[r]['B']:.3f} "
              f"C={out[r]['C']:.3f} C4={out[r]['C4']} ({time.time()-t2:.0f}s)", flush=True)
    return out


def read_cohort(path):
    """(case ids, {case id: age}) from a cohort CSV. v1 and v2 both use CaseID / Age_Scan.

    Read by column NAME, not position: v2 inserted a `Carried_over` column, and the previous
    positional parse (split on '","', take field 3) would break silently on any such edit.
    """
    ids, ages = [], {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            cid = (r.get("CaseID") or "").strip()
            if not cid:
                continue
            ids.append(cid)
            try:
                ages[cid] = float(r["Age_Scan"])
            except (KeyError, TypeError, ValueError):
                pass
    return ids, ages


def main(outroot, dx=4.0, outdir=None, cohort_csv_path=None, shard=0, nshards=1):
    outdir = outdir or os.path.join(ROOT, "results", "montage_optimization", "cohort_deep")
    os.makedirs(outdir, exist_ok=True)
    # Ages MUST come from the same CSV that supplies the subject list. This used to derive the
    # path from outroot (L:\BrainProbe\SPICEhead\charm -> the v1 100-case CSV) while the subjects came
    # from the v2 300-case CSV, so 236/300 subjects were written with age=null and every
    # downstream age analysis silently ran on the 64 that happened to overlap.
    age_csv = cohort_csv_path or os.path.join(os.path.dirname(outroot), "SPICEhead_cohort.csv")
    ids, ages = read_cohort(age_csv)
    SUBJECTS = ids if cohort_csv_path else SUBSET
    if nshards > 1:
        SUBJECTS = [c for i, c in enumerate(SUBJECTS) if i % nshards == shard]
        print(f"shard {shard}/{nshards}: {len(SUBJECTS)} subjects", flush=True)
    missing = [c for c in SUBJECTS if c not in ages]
    if missing:
        print(f"WARNING: no age for {len(missing)}/{len(SUBJECTS)} subjects in "
              f"{os.path.basename(age_csv)} (e.g. {', '.join(missing[:3])})", flush=True)
    t0 = time.time()
    for i, cid in enumerate(SUBJECTS, 1):
        dest = os.path.join(outdir, f"{cid}.json")
        prev = {}
        if os.path.exists(dest):
            prev = json.load(open(dest)).get("rois", {})
        todo = [r for r in ROI_ORDER if r not in prev]
        if not todo:
            print(f"[{i}/{len(SUBJECTS)}] {cid} - all {len(ROI_ORDER)} ROIs done, skip", flush=True)
            continue
        m2m = resolve(outroot, cid)
        if not os.path.exists(os.path.join(m2m, "electrodes_1020ext.json")):
            print(f"[{i}/{len(SUBJECTS)}] {cid} - no electrodes_1020ext.json, skip", flush=True)
            continue
        print(f"[{i}/{len(SUBJECTS)}] {cid} (age {ages.get(cid, '?')})  computing: {', '.join(todo)}", flush=True)
        try:
            res = run_subject(m2m, dx, np.random.default_rng(0), todo)
            json.dump({"case": cid, "age": ages.get(cid), "dx": dx, "rois": {**prev, **res}},
                      open(dest, "w"), indent=2)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
        print(f"    elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- aggregate ----
    rows = [json.load(open(os.path.join(outdir, f))) for f in sorted(os.listdir(outdir))
            if f.endswith(".json") and f != "SUMMARY.json"]
    if not rows:
        return
    summ = {}
    print(f"\n  ===== COHORT (n={len(rows)}) median focality =====", flush=True)
    print(f"  {'ROI':>12} {'A':>7} {'B':>7} {'C':>7} {'C4':>7}   {'B/A':>7} {'C/A':>7} {'C4/A':>7}", flush=True)
    for r in ROI_ORDER:
        have = [x for x in rows if isinstance(x["rois"].get(r), dict)
                and all(x["rois"][r].get(k) is not None for k in ("A", "B", "C", "C4"))]
        if not have:
            continue
        v = {k: np.array([x["rois"][r][k] for x in have], float) for k in ("A", "B", "C", "C4")}
        gB = 100 * np.median(v["B"] / v["A"] - 1)
        gC = 100 * np.median(v["C"] / v["A"] - 1)
        g4 = 100 * np.median(v["C4"] / v["A"] - 1)
        nInf = sum(1 for x in have if x["rois"][r]["B_inferior"])
        summ[r] = {"n": len(have), "A": round(float(np.median(v["A"])), 4),
                   "B": round(float(np.median(v["B"])), 4), "C": round(float(np.median(v["C"])), 4),
                   "C4": round(float(np.median(v["C4"])), 4),
                   "gain_B_pct": round(float(gB), 2), "gain_C_pct": round(float(gC), 2),
                   "gain_C4_pct": round(float(g4), 2),
                   "gain_C_iqr": [round(float(q), 2) for q in
                                  np.percentile(100 * (v["C"] / v["A"] - 1), [25, 75])],
                   "n_B_uses_inferior": nInf}
        print(f"  {r:>12} {np.median(v['A']):7.3f} {np.median(v['B']):7.3f} {np.median(v['C']):7.3f} "
              f"{np.median(v['C4']):7.3f}   {gB:+6.1f}% {gC:+6.1f}% {g4:+6.1f}%   "
              f"(B uses inferior in {nInf}/{len(have)})", flush=True)
    json.dump(summ, open(os.path.join(outdir, "SUMMARY.json"), "w"), indent=2)
    print(f"  -> {outdir}\\SUMMARY.json   total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1],
         float(sys.argv[2]) if len(sys.argv) > 2 else 4.0,
         sys.argv[3] if len(sys.argv) > 3 else None,
         sys.argv[4] if len(sys.argv) > 4 else None,
         int(sys.argv[5]) if len(sys.argv) > 5 else 0,
         int(sys.argv[6]) if len(sys.argv) > 6 else 1)
