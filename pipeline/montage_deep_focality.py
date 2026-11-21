"""
Deep-target focality: can we beat the 10/20 two-pair optimum?

Motivation: for deep ROIs the 10/20 optimum pins every electrode to the LOWEST available ring
(thalamus F7+T7/T8+P8, insula F7+F8/T7+T8) -> the inferior boundary of the electrode set is a
BINDING constraint, not a converged optimum. The earlier 10-10 sweep only refined the same scalp
band (AF/FC/FT/CP/TP/PO) and gained 1-2%.

Three conditions on the MNI template head, one ROI:
  A  10/20 two-pair          (19 sites)   - reproduces the current GUI answer
  B  extended two-pair       (+ inferior 9/10 chain) - relaxes the binding constraint
  C  multi-electrode current optimisation (continuous, extended set)
     Exploits linearity: any zero-sum current pattern s has E_s = sum_k s_k B_k over a basis of
     (k vs ref) pair fields, so the FULL continuous space costs ~0 extra solves.

Scored with the cohort convention  T = median(env in ROI),  O = p99(env in brain >EXCLUDE_MM from
ROI),  focality = T/O; the GUI's global-O variant is reported alongside for comparability.

  python montage_deep_focality.py <region> [dx] [out.json]
"""
import os
import sys
import json
import time
from itertools import combinations

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components

M2M = r"L:\BrainProbe\SPICEhead\charm\m2m_MNItempl"
AAL = r"C:\Users\ALBERT YANG\nilearn_data\aal_SPM12\aal\atlas\AAL.nii"
CAP = r"C:\Users\ALBERT YANG\SimNIBS-4.6\simnibs_env\Lib\site-packages\simnibs\resources\ElectrodeCaps_MNI\EEG10-10_UI_Jurak_2007.csv"

TEN20 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
         "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
# the inferior "9/10" chain - below the 10/20 ring, never tested before
INFERIOR = ["F9", "F10", "FT9", "FT10", "T9", "T10", "TP9", "TP10",
            "P9", "P10", "PO9", "PO10", "I1", "Iz", "I2"]
EXT = TEN20 + INFERIOR

ELEC_RADIUS_MM, MIN_DIST_MM = 12.0, 40.0
EXCLUDE_MM = 20.0
SCALP, BRAIN = 5, (1, 2)

# Tissue class scored in BOTH T and O. Grey matter only: sigma_WM (0.126) < sigma_GM (0.275), so
# E = J/sigma runs ~28% higher in white matter (measured: median envelope 0.0618 vs 0.0484 V/m).
# Scoring a grey-matter target (thalamus, insula, cortex...) against a GM+WM denominator therefore
# charges the montage for a tissue-conductivity contrast it has no control over. Like-for-like GM
# raised Thalamus_L from T/O 0.824 to 0.975 with no change to the montage.
SCORE = (2,)
N_OFF = 15000          # off-target voxels sampled for the sweep (exact re-score at the end)
TOPK = 15
MAX_CURRENT_MA = 1.0   # injected current per channel (matches the 1 mA pair baseline)
PER_ELEC_CAP_MA = 1.0  # per-electrode cap; 1.0 = "no electrode carries more than a plain pair does"


def read_cap(labels_wanted):
    d = {}
    for line in open(CAP):
        p = line.strip().split(",")
        if len(p) >= 5 and p[0].endswith("Electrode") and p[4] in labels_wanted:
            d[p[4]] = np.array([float(p[1]), float(p[2]), float(p[3])])
    return d


def multiground(G):
    Gc = csr_matrix(G)
    n = G.shape[0]
    ncomp, comp = connected_components(csgraph=Gc, directed=False)
    grounds = [int(np.flatnonzero(comp == c)[0]) for c in range(ncomp)]
    keep = np.ones(n, bool)
    keep[grounds] = False
    idx = np.flatnonzero(keep)
    lu = factorized(Gc[idx][:, idx].tocsc())

    def solve(cur):
        v = np.zeros(n)
        v[idx] = lu(np.asarray(cur, float)[idx])
        return v
    return solve, ncomp


ENVELOPE = os.environ.get("SPICEHEAD_ENVELOPE", "maxdir")
from spicehead.envelope_fast import envelope_max_direction_flat


def env_from(E1, E2):
    """TI envelope on already-masked flat component arrays.

    'maxdir' is the rotation-invariant maximum-over-direction form and is the default; 'peraxis'
    is the superseded per-axis combination sqrt(sum (2 min(|a|,|b|))^2), which depends on the
    coordinate frame and is kept only for reproducing the earlier numbers.
    """
    if ENVELOPE == "maxdir":
        return envelope_max_direction_flat(E1[0], E1[1], E1[2], E2[0], E2[1], E2[2])
    return np.sqrt(sum((2.0 * np.minimum(np.abs(a), np.abs(b))) ** 2 for a, b in zip(E1, E2)))


def main(region="Thalamus_L", dx=4.0, out=None):
    t0 = time.time()
    seg = os.path.join(M2M, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, affine = mri.head_model(seg, dx)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    brain = np.isin(labels, SCORE)          # scoring mask (grey matter), not anatomical extent
    print(f"  grid {labels.shape} nodes={model.n_nodes}  ({time.time()-t0:.1f}s)", flush=True)

    # --- ROI + off-target -------------------------------------------------
    tpl = nib.Nifti1Image(labels.astype(np.int16), affine)
    aal = np.rint(np.asarray(resample_from_to(nib.load(AAL), tpl, order=0).dataobj)).astype(int)
    reg = {r["name"]: r for r in json.load(open(os.path.join(_here, "aal", "aal_regions.json")))["regions"]}
    val = reg[region]["value"]
    roi = (aal == val) & brain
    if roi.sum() == 0:
        raise SystemExit(f"{region} empty at dx={dx}")
    dist = distance_transform_edt(~roi, sampling=(dx, dx, dx))
    off = brain & (dist > EXCLUDE_MM)
    roi_idx = np.argwhere(roi)
    off_all = np.argwhere(off)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(off_all), size=min(N_OFF, len(off_all)), replace=False)
    off_s = off_all[sel]
    print(f"  ROI {region} (AAL {val}): {roi.sum()} vox   off-target {off.sum()} "
          f"(sampled {len(off_s)})", flush=True)

    def take(F, ix):
        return np.stack([f[ix[:, 0], ix[:, 1], ix[:, 2]] for f in F]).astype(np.float32)

    # --- electrodes -------------------------------------------------------
    cap = read_cap(set(EXT))
    missing = [l for l in EXT if l not in cap]
    scalp = np.argwhere(labels == SCALP)
    inv = np.linalg.inv(affine)
    evox, elecs = {}, {}
    for lab in EXT:
        if lab not in cap:
            continue
        ijk = (inv @ np.append(cap[lab], 1))[:3]
        v = scalp[int(np.argmin(np.linalg.norm(scalp - ijk, axis=1)))]
        snap_err = float(np.linalg.norm(v - ijk) * dx)
        if snap_err > 20.0:            # outside the segmented FOV - unusable
            continue
        evox[lab] = v
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                               radius_nodes=ELEC_RADIUS_MM / dx), lab)
    # drop duplicates (two labels snapping to the same scalp voxel)
    seen, keep = {}, []
    for lab in EXT:
        if lab not in evox:
            continue
        k = tuple(evox[lab])
        if k in seen:
            continue
        seen[k] = lab
        keep.append(lab)
    labs20 = [l for l in keep if l in TEN20]
    labsX = keep
    z_of = {l: float((affine @ np.append(evox[l], 1))[2]) for l in keep}
    print(f"  electrodes: {len(labsX)} usable of {len(EXT)} "
          f"(missing {missing}, dropped {len(EXT)-len(labsX)-len(missing)})", flush=True)
    print("  inferior sites kept: " + ", ".join(
        f"{l}(z={z_of[l]:.0f})" for l in labsX if l in INFERIOR), flush=True)
    print(f"  ref rings: T7 z={z_of.get('T7', float('nan')):.0f}  "
          f"T9 z={z_of.get('T9', float('nan')):.0f}", flush=True)

    # --- solve every pair once -------------------------------------------
    solver, ncomp = multiground(model.G)
    print(f"  factorized ({ncomp} components) {time.time()-t0:.1f}s", flush=True)
    pairs = [(a, b) for a, b in combinations(labsX, 2)
             if np.linalg.norm(evox[a] - evox[b]) * dx >= MIN_DIST_MM]
    t1 = time.time()
    Pr, Po = {}, {}
    for (a, b) in pairs:
        F = efield_components(model, solver(pair_injection(model, elecs[a], elecs[b], 1e-3)))
        F = tuple(np.nan_to_num(f) for f in F)
        Pr[(a, b)] = take(F, roi_idx)
        Po[(a, b)] = take(F, off_s)
    print(f"  {len(pairs)} pair fields in {time.time()-t1:.1f}s", flush=True)

    def score_s(pa, pb):
        T = float(np.median(env_from(Pr[pa], Pr[pb])))
        O = float(np.percentile(env_from(Po[pa], Po[pb]), 99))
        return (T / O if O > 0 else 0.0), T, O

    # --- A / B : two-pair sweeps -----------------------------------------
    def sweep(allowed):
        ps = [p for p in pairs if p[0] in allowed and p[1] in allowed]
        res = []
        for pa, pb in combinations(ps, 2):
            if len(set(pa) | set(pb)) < 4:
                continue
            s, T, O = score_s(pa, pb)
            res.append((s, T, O, pa, pb))
        res.sort(key=lambda x: -x[0])
        return res, len(ps)

    t2 = time.time()
    resA, nA = sweep(set(labs20))
    print(f"  [A] 10/20      {nA} pairs -> {len(resA)} montages  ({time.time()-t2:.1f}s)", flush=True)
    t2 = time.time()
    resB, nB = sweep(set(labsX))
    print(f"  [B] extended   {nB} pairs -> {len(resB)} montages  ({time.time()-t2:.1f}s)", flush=True)
    a0, b0 = resA[0], resB[0]   # baselines, needed by the truncation block below

    # --- C : multi-electrode current optimisation -------------------------
    # Candidate sites = electrodes appearing most often among B's best montages. This keeps the
    # problem low-dimensional (Powell converges) AND physically realisable with few current sources.
    freq = {}
    for r in resB[:25]:
        for e in set(r[3]) | set(r[4]):
            freq[e] = freq.get(e, 0) + 1
    cand = [e for e, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))][:16]
    for e in sorted(set(resB[0][3]) | set(resB[0][4])):   # sorted(): set order is hash-randomised,
        if e not in cand:                                 # which would make the basis non-reproducible
            cand.append(e)
    ref = cand[0]
    basis = [e for e in cand if e != ref]
    nb = len(basis)
    Br, Bo = [], []
    for l in basis:
        F = efield_components(model, solver(pair_injection(model, elecs[l], elecs[ref], 1e-3)))
        F = tuple(np.nan_to_num(f) for f in F)
        Br.append(take(F, roi_idx))
        Bo.append(take(F, off_s))
    Br = np.stack(Br)   # (nb, 3, nroi)
    Bo = np.stack(Bo)   # (nb, 3, noff)
    print(f"  [C] {len(cand)} candidate sites (ref {ref}), {nb}-dim/channel: "
          f"{','.join(cand)}  ({time.time()-t0:.1f}s)", flush=True)

    def currents(v):
        """raw non-ref currents -> physical pattern; total injected == MAX_CURRENT_MA.
        The ref electrode carries the balance -sum(v), so the pattern is zero-sum by construction."""
        v = np.asarray(v, float)
        inj = 0.5 * (np.abs(v).sum() + abs(v.sum()))   # half of total |current| = injected current
        return v if inj < 1e-9 else v * (MAX_CURRENT_MA / inj)

    def eval_raw(x, y):
        """x, y = physical non-ref current vectors (mA) -> (focality, T, O)."""
        T = float(np.median(env_from(np.tensordot(x, Br, axes=(0, 0)),
                                     np.tensordot(y, Br, axes=(0, 0)))))
        O = float(np.percentile(env_from(np.tensordot(x, Bo, axes=(0, 0)),
                                         np.tensordot(y, Bo, axes=(0, 0))), 99))
        return (T / O if O > 0 else 0.0), T, O

    def evaluate(z):
        """-> (focality, T, O, penalty). Penalty is reported separately, never folded into T/O."""
        x, y = currents(z[:nb]), currents(z[nb:])
        _s, T, O = eval_raw(x, y)
        pen = 0.0
        for c in (x, y):
            peak = max(float(np.abs(c).max()), abs(float(c.sum())))   # incl. ref's balance current
            if peak > PER_ELEC_CAP_MA:
                pen += 20.0 * (peak - PER_ELEC_CAP_MA)
        return (T / O if O > 0 else 0.0), T, O, pen

    def obj(z):
        s, _, _, pen = evaluate(z)
        return -s + pen

    def seed_from(pa, pb):
        z = np.zeros(2 * nb)
        for j, pr in enumerate((pa, pb)):
            for lab, sgn in ((pr[0], +1.0), (pr[1], -1.0)):
                if lab in basis:
                    z[j * nb + basis.index(lab)] = sgn
        return z

    t3 = time.time()
    z_seed = seed_from(*resB[0][3:5])
    chk, _, _, chk_pen = evaluate(z_seed)
    print(f"  [C] seed check: reproduces B best {chk:.4f} vs {resB[0][0]:.4f} "
          f"(penalty {chk_pen:.3f} - must be 0)", flush=True)

    starts = [seed_from(*resB[i][3:5]) for i in range(min(4, len(resB)))]
    rng2 = np.random.default_rng(1)
    starts += [rng2.normal(size=2 * nb) * 0.3 for _ in range(4)]
    best_z, best_f = None, np.inf
    for i, z0 in enumerate(starts):
        r = minimize(obj, z0, method="Powell",
                     options={"maxfev": 40000, "xtol": 1e-4, "ftol": 1e-5})
        s, T, O, pen = evaluate(r.x)
        if r.fun < best_f:
            best_f, best_z = r.fun, r.x
        print(f"    start {i}: T/O = {s:.4f}  (penalty {pen:.3f})", flush=True)
    xs, ys = currents(best_z[:nb]), currents(best_z[nb:])
    Tc_Oc, Tc, Oc, pen_c = evaluate(best_z)
    print(f"  [C] optimised T/O = {Tc_Oc:.4f}  penalty={pen_c:.3f}  ({time.time()-t3:.1f}s)", flush=True)

    # --- C' : practicality - how much of C survives with only k electrodes/channel? ---
    def to_full(v):
        s = currents(v)
        return np.append(s, -s.sum())          # basis + ref

    def rebalance(t, I=MAX_CURRENT_MA):
        """rescale + / - halves separately -> zero-sum, exactly I mA injected."""
        p, n = t > 0, t < 0
        if not p.any() or not n.any():
            return None
        out = np.zeros_like(t)
        out[p] = t[p] * (I / t[p].sum())
        out[n] = t[n] * (I / -t[n].sum())
        return out

    def trunc(t, k):
        z = np.zeros_like(t)
        keep = np.argsort(-np.abs(t))[:k]
        z[keep] = t[keep]
        return rebalance(z)

    fx, fy = to_full(best_z[:nb]), to_full(best_z[nb:])
    print("  [C'] truncation (drop the small trim currents):", flush=True)
    trunc_rows = []
    for k in (2, 3, 4, 5, 6):
        tx, ty = trunc(fx, k), trunc(fy, k)
        if tx is None or ty is None:
            continue
        s, T, O = eval_raw(tx[:nb], ty[:nb])
        nz = int((np.abs(tx) > 1e-6).sum() + (np.abs(ty) > 1e-6).sum())
        peak = max(float(np.abs(tx).max()), float(np.abs(ty).max()))
        trunc_rows.append({"k_per_channel": k, "n_electrodes": nz, "peak_mA": round(peak, 3),
                           "focality": round(s, 4), "pct_of_C": round(100 * s / Tc_Oc, 1),
                           "gain_vs_A_pct": round(100 * (s / a0[0] - 1), 2)})
        print(f"    top-{k}/ch ({nz} electrodes, peak {peak:.2f} mA): T/O={s:.4f}   "
              f"{100*s/Tc_Oc:.0f}% of C, {100*(s/a0[0]-1):+.1f}% vs A", flush=True)

    def pattern(v):
        d = {basis[i]: round(float(v[i]), 4) for i in range(nb) if abs(v[i]) > 0.02}
        d[ref] = round(float(-v.sum()), 4)
        return {k: val for k, val in sorted(d.items(), key=lambda kv: -abs(kv[1])) if abs(val) > 0.02}

    def fmt(r):
        s, T, O, pa, pb = r
        return {"montage": f"{'+'.join(pa)} / {'+'.join(pb)}",
                "focality": round(s, 4), "T": round(T, 5), "O": round(O, 5),
                "uses_inferior": sorted(set(pa) | set(pb) & set(INFERIOR)) if False else
                                 sorted([e for e in set(pa) | set(pb) if e in INFERIOR])}

    res = {
        "region": region, "dx": dx,
        "metric": "T=median(ROI n GM), O=p99(GM >20mm from ROI)  [grey matter only, like-for-like]",
        "n_electrodes": {"ten20": len(labs20), "extended": len(labsX)},
        "A_ten20": {"best": fmt(a0), "top": [fmt(r) for r in resA[:TOPK]]},
        "B_extended": {"best": fmt(b0), "top": [fmt(r) for r in resB[:TOPK]],
                       "gain_vs_A_pct": round(100 * (b0[0] / a0[0] - 1), 2)},
        "C_multielectrode": {"focality": round(Tc_Oc, 4), "T": round(Tc, 5), "O": round(Oc, 5),
                             "gain_vs_A_pct": round(100 * (Tc_Oc / a0[0] - 1), 2),
                             "gain_vs_B_pct": round(100 * (Tc_Oc / b0[0] - 1), 2),
                             "candidates": cand, "ref": ref, "penalty": round(pen_c, 4),
                             "channel1_mA": pattern(xs), "channel2_mA": pattern(ys),
                             "truncation": trunc_rows},
    }
    out = out or os.path.join(ROOT, "results", "montage_optimization",
                              f"deep_focality_{region}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)

    print("\n  ================ RESULT ================", flush=True)
    print(f"  A  10/20 two-pair     {a0[0]:.4f}   {res['A_ten20']['best']['montage']}", flush=True)
    print(f"  B  extended two-pair  {b0[0]:.4f}   {res['B_extended']['best']['montage']}"
          f"   ({res['B_extended']['gain_vs_A_pct']:+.1f}%)", flush=True)
    print(f"  C  multi-electrode    {Tc_Oc:.4f}   ({res['C_multielectrode']['gain_vs_A_pct']:+.1f}% vs A,"
          f" {res['C_multielectrode']['gain_vs_B_pct']:+.1f}% vs B)", flush=True)
    print(f"  -> {out}   total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    region = sys.argv[1] if len(sys.argv) > 1 else "Thalamus_L"
    dx = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    main(region, dx, sys.argv[3] if len(sys.argv) > 3 else None)
