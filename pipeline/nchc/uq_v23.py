# -*- coding: utf-8 -*-
"""v23 uncertainty analysis, one head (24 Oct 2026; PI: "run the uncertainty analysis on nano4").

Pre-specified in results_v23/v23_uq/ANALYSIS_PLAN.md. Runs, for one head, the nominal v23 model and the perturbed
models of the plan:
  cond_00..39  tissue conductivities (Latin hypercube, log-normal factors truncated at +-2 SD, McCann 2019 spreads)
  jit_00..19   independent tangential electrode displacement, SD 5 mm per axis, snapped to the scalp
  rot_00..09   whole-cap rotation about the head centroid, 6.37 degrees about a random axis, snapped to the scalp
  seg_*        one 0.5 mm label layer at one interface (skull +/- into scalp, CSF +/- into grey matter)
Each run: 18 basis solves (GPU), all 11,628 montages scored for the six targets (GPU scorer), and for fixed
montages (cohort winner and the head's nominal winner per target) the peak and 99.9th-percentile grey-matter
envelope and the peak single-channel |E|, all at 1 mA per channel.
Output: $SPICEHEAD_UQ_OUT/<code>/<run>.npz (F, T, O: targets x montages float32; peaks), montages.json, DONE.

    SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu python pipeline/nchc/uq_v23.py --head h001 [--m2m DIR] [--only nominal,cond_00]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "pipeline")); sys.path.insert(0, HERE)
assert os.environ.get("SPICEHEAD_MODEL") == "v23", "run with SPICEHEAD_MODEL=v23"
OUT = os.environ.get("SPICEHEAD_UQ_OUT", os.path.join(ROOT, "results_v23", "v23_uq", "runs"))
FIXED = os.environ.get("SPICEHEAD_UQ_FIXED", os.path.join(ROOT, "results_v23", "v23_uq", "uq_fixed_montages.json"))
DX = 1.0
BASE_SEED = 20261024
N_COND, N_JIT, N_ROT = 40, 20, 10
JIT_SD_MM = 5.0
ROT_DEG = float(np.degrees(10.0 / 90.0))            # 6.37 degrees = 10 mm of arc at 90 mm
TRUNC = 2.0
# tissue -> (conductivity keys scaled together, log-SD from McCann 2019 Table 3 CVs)
COND_TISSUES = [("scalp", ("scalp",), 0.412), ("compact_bone", ("compact_bone", "bone"), 0.340),
                ("spongy_bone", ("spongy_bone",), 1.077), ("csf", ("csf",), 0.173),
                ("gray_matter", ("gray_matter",), 0.481), ("white_matter", ("white_matter",), 0.695)]
SEG = ["seg_skull_plus", "seg_skull_minus", "seg_csf_plus", "seg_csf_minus"]
BONE, SCALP, CSF, GM, COMPACT = (4, 7, 8), 5, 3, 2, 7


def head_number(code):
    return int(code.lstrip("h"))


def cond_samples(code):
    """N_COND x 6 log-factors: Latin hypercube in the truncated standard normal, then times the log-SDs."""
    from scipy.stats import norm
    rng = np.random.default_rng([BASE_SEED, head_number(code), 1])
    k = len(COND_TISSUES)
    lo, hi = norm.cdf(-TRUNC), norm.cdf(TRUNC)
    U = (np.stack([rng.permutation(N_COND) for _ in range(k)], 1) + rng.random((N_COND, k))) / N_COND
    Z = norm.ppf(lo + U * (hi - lo))
    return Z * np.array([s for _, _, s in COND_TISSUES])


def cond_dict(logf):
    from spicehead.conductivity import CONDUCTIVITY
    c = dict(CONDUCTIVITY)
    for (name, keys, _), lf in zip(COND_TISSUES, logf):
        for key in keys:
            c[key] = CONDUCTIVITY[key] * float(np.exp(lf))
    return c


def seg_labels(L, which):
    from scipy.ndimage import binary_dilation, generate_binary_structure
    st = generate_binary_structure(3, 1)
    L = L.copy()
    bone, scalp, csf, gm = np.isin(L, BONE), L == SCALP, L == CSF, L == GM
    if which == "seg_skull_plus":
        L[scalp & binary_dilation(bone, st)] = COMPACT
    elif which == "seg_skull_minus":
        L[bone & binary_dilation(scalp, st)] = SCALP
    elif which == "seg_csf_plus":
        L[gm & binary_dilation(csf, st)] = CSF
    elif which == "seg_csf_minus":
        L[csf & binary_dilation(gm, st)] = GM
    return L


def tangent_basis(n):
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def jitter_positions(code, k, pos, c):
    rng = np.random.default_rng([BASE_SEED, head_number(code), 2, k])
    out = {}
    for lab in sorted(pos):
        p = np.asarray(pos[lab], float); n = (p - c) / np.linalg.norm(p - c)
        t1, t2 = tangent_basis(n)
        d = rng.normal(0.0, JIT_SD_MM, 2)
        out[lab] = p + d[0] * t1 + d[1] * t2
    return out


def rotate_positions(code, k, pos, c):
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng([BASE_SEED, head_number(code), 3, k])
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = np.radians(ROT_DEG) * (1 if rng.random() < 0.5 else -1)
    R = Rotation.from_rotvec(axis * ang).as_matrix()
    return {lab: c + R @ (np.asarray(p, float) - c) for lab, p in pos.items()}, dict(axis=axis.round(6).tolist(), deg=float(np.degrees(ang)))


def main():
    a = sys.argv[1:]
    code = a[a.index("--head") + 1]
    only = a[a.index("--only") + 1].split(",") if "--only" in a else None
    import v22_fem_fullsearch as fs
    from spicehead import config, mri
    from spicehead.homogenize import homogenized_model
    from spicehead.electrodes import Electrode, contact_nodes, pair_injection
    from spicehead.field import efield_components
    from spicehead.score_gpu import score_montages_gpu, _envelope
    from montage_search import snap
    from v21_head_convergence import make_solver
    import torch
    assert config.NETWORK == "homogenized"
    m2m = a[a.index("--m2m") + 1] if "--m2m" in a else os.path.join(os.environ["SPICEHEAD_CHARM_ROOT"], f"m2m_{code}")
    od = os.path.join(OUT, code); os.makedirs(od, exist_ok=True)
    t0 = time.time()
    geo = fs.grid_geometry(code, m2m)
    # grid_geometry returns float64 coordinates on a cache miss and float32 from its cache; the validated 1 mm runs
    # (300-head comparison and homogenized run) always sampled at the cached float32 values, so use those here too
    gm_world = np.asarray(geo["gm_world"], np.float32)
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    fine, aff_f, _ = mri.load_label_volume(seg)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs, pairs, montages = fs.candidate_set(ej)
    others = [x for x in labs if x != fs.REF]
    keys = [fs.label_of(pa, pb) for pa, pb in montages]
    by_key = dict(zip(keys, montages))
    json.dump(keys, open(os.path.join(od, "montages.json"), "w"))
    cohort = json.load(open(FIXED))["montages"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{code}: geometry and labels ready ({time.time()-t0:.0f} s), {len(geo['gm_idx']):,} GM voxels, device {dev}", flush=True)

    def build(labels, cond):
        model, aff_c = homogenized_model(labels, aff_f, DX, min_fill=config.HOMOG_MIN_FILL, cond=cond)
        solver, ncomp = make_solver(model.G)
        return model, aff_c, solver

    def run(name, model, aff_c, solver, pos, params, fixed_extra):
        p = os.path.join(od, f"{name}.npz")
        if os.path.exists(p):
            return "cached"
        t1 = time.time()
        scalp_vox = np.argwhere(model.labels == fs.SCALP_LABEL)
        elecs, snapd = {}, {}
        for lab in labs:
            v = snap(scalp_vox, mri.world_to_voxel(aff_c, pos[lab]))
            elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v), radius_nodes=fs.ELEC_RADIUS_MM / DX), lab)
            snapd[lab] = float(np.linalg.norm((aff_c[:3, :3] @ np.asarray(v, float) + aff_c[:3, 3]) - np.asarray(pos[lab], float)))
        basis = np.zeros((len(others), 3, len(geo["gm_idx"])), np.float32)
        for i, lab in enumerate(others):
            V = solver(pair_injection(model, elecs[lab], elecs[fs.REF], fs.CURRENT))
            basis[i] = np.stack([fs.sample_on_gm(c, aff_c, gm_world) for c in efield_components(model, V)])
        t_solve = time.time() - t1
        res = score_montages_gpu(basis, others, fs.REF, pairs, montages, geo["roi_sel"], geo["out_sel"], fs.ROI_ORDER, fs.label_of, verbose=False)
        F = np.array([[res[r][k][0] for k in keys] for r in fs.ROI_ORDER], np.float32)
        T = np.array([[res[r][k][1] for k in keys] for r in fs.ROI_ORDER], np.float32)
        O = np.array([[res[r][k][2] for k in keys] for r in fs.ROI_ORDER], np.float32)
        # fixed montages: cohort winner per target, then the head's nominal winner per target
        # (fixed_extra None = this run's own winners: used by the nominal run, whose winners are the nominal ones)
        if fixed_extra is None:
            fixed_extra = [keys[int(F[t].argmax())] for t in range(len(fs.ROI_ORDER))]
        fixed = [cohort[r] for r in fs.ROI_ORDER] + list(fixed_extra)
        idx = {lab: i for i, lab in enumerate(others)}
        Bz = torch.from_numpy(basis).to(dev); zero = torch.zeros_like(Bz[0])
        pf = lambda pr: (Bz[idx[pr[0]]] if pr[0] != fs.REF else zero) - (Bz[idx[pr[1]]] if pr[1] != fs.REF else zero)
        peaks = []
        for m in fixed:
            pa, pb = by_key[m]; A, B = pf(pa), pf(pb)
            env = _envelope(A[None], B[None])[0]
            peaks.append([float(env.max()), float(torch.quantile(env, 0.999)), float(torch.linalg.norm(A, dim=0).max()), float(torch.linalg.norm(B, dim=0).max())])
        del Bz
        params = dict(params, code=code, run=name, n_nodes=int(model.n_nodes), dropped_nodes=int(model.dropped_nodes),
                      snap_mm_median=float(np.median(list(snapd.values()))), snap_mm_max=float(max(snapd.values())),
                      t_solve_s=round(t_solve, 1), t_run_s=round(time.time() - t1, 1))
        np.savez_compressed(p + ".part.npz", F=F, T=T, O=O, peaks=np.array(peaks, np.float32), fixed=np.array(fixed),
                            peak_cols=np.array(["env_max", "env_p999", "ch1_absE_max", "ch2_absE_max"]), params=json.dumps(params))
        os.replace(p + ".part.npz", p)
        return f"ok ({params['t_run_s']:.0f} s)"

    want = lambda n: only is None or n in only or any(n.startswith(o.rstrip("*")) for o in only if o.endswith("*"))
    # nominal
    model, aff_c, solver = build(fine, None)
    pos0 = {lab: np.asarray(ej[lab], float) for lab in labs}
    ii, jj, kk = model.grid_coords
    centroid = aff_c[:3, :3] @ np.array([ii.mean(), jj.mean(), kk.mean()]) + aff_c[:3, 3]
    nom_p = os.path.join(od, "nominal.npz")
    if want("nominal") or not os.path.exists(nom_p):
        print(f"{code} nominal: {run('nominal', model, aff_c, solver, pos0, dict(scenario='nominal'), None)}", flush=True)
    with np.load(nom_p) as z:
        Fn = z["F"]
    winners = [keys[int(Fn[t].argmax())] for t in range(len(fs.ROI_ORDER))]
    json.dump(dict(zip(fs.ROI_ORDER, winners)), open(os.path.join(od, "nominal_winners.json"), "w"), indent=1)
    # placement: same network, new contacts
    for k in range(N_JIT):
        n = f"jit_{k:02d}"
        if want(n):
            print(f"{code} {n}: {run(n, model, aff_c, solver, jitter_positions(code, k, pos0, centroid), dict(scenario='jitter', sd_mm=JIT_SD_MM), winners)}", flush=True)
    for k in range(N_ROT):
        n = f"rot_{k:02d}"
        if want(n):
            pos, rp = rotate_positions(code, k, pos0, centroid)
            print(f"{code} {n}: {run(n, model, aff_c, solver, pos, dict(scenario='rotation', **rp), winners)}", flush=True)
    del model, solver
    # conductivity: new network per sample
    Z = cond_samples(code)
    for k in range(N_COND):
        n = f"cond_{k:02d}"
        if want(n) and not os.path.exists(os.path.join(od, f"{n}.npz")):
            m, ac, s = build(fine, cond_dict(Z[k]))
            print(f"{code} {n}: {run(n, m, ac, s, pos0, dict(scenario='conductivity', log_factors=dict(zip([t for t, _, _ in COND_TISSUES], Z[k].round(6).tolist()))), winners)}", flush=True)
            del m, s
    # segmentation boundaries: new labels per scenario
    for n in SEG:
        if want(n) and not os.path.exists(os.path.join(od, f"{n}.npz")):
            Lm = seg_labels(fine, n)
            m, ac, s = build(Lm, None)
            print(f"{code} {n}: {run(n, m, ac, s, pos0, dict(scenario='segmentation', changed_voxels=int((Lm != fine).sum())), winners)}", flush=True)
            del m, s, Lm
    if only is None:
        open(os.path.join(od, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{code}: finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
