# -*- coding: utf-8 -*-
"""A2 of Rev0.5/10_ANALYSIS_REQUEST_v23_2026-11-02.md: perturbed SPICEhead scores at the SPICEhead-selected cohort montages whose
scores the uncertainty archive did not keep (3 Nov 2026). Prepared for a pilot; a campaign needs the PI's approval of its measured cost.

The perturbed basis fields of the uncertainty run were never written, so these montages cannot be rescored from the archive. This
script repeats only the basis solves, with the seeds, perturbation functions, solver and scoring of uq_v23_atlas.py, and scores
two montages per listed target instead of all 11,628: the SPICEhead cohort montage (new) and the FEM cohort montage (control, must
reproduce the archive). The denominators Tmax and F_constrained stay those of the archived run of the same perturbed model.
Output: $SPICEHEAD_UQ_SP_OUT/<code>/<run>.npz (F, T, O per target for each protocol).
    SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu python pipeline/nchc/uq_v23_sp_protocol.py --head h002 --targets <code>__uqtargets2.npz
        --also-thal <code>__exttargets.npz --protocols uq_sp_protocols.json [--m2m DIR] [--only nominal,cond_00]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "pipeline")); sys.path.insert(0, HERE)
import uq_v23 as U                                                     # noqa: E402  (perturbations and constants)

OUT = os.environ.get("SPICEHEAD_UQ_SP_OUT", os.path.join(ROOT, "results_v23", "v23_uq", "runs_sp_protocol"))


def mask_variant(tg, which, shape, idx_all):
    """One-voxel erosion or dilation of each target within grey matter, as in uq_v23_atlas.py; exclusion masks unchanged."""
    from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
    st = generate_binary_structure(3, 1)
    out_t, changed = [], 0
    for name, tr, roi, out in tg:
        ii = idx_all[roi]
        lo = np.maximum(ii.min(0) - 2, 0); hi = np.minimum(ii.max(0) + 3, np.asarray(shape))
        sub = np.zeros(hi - lo, bool); sub[tuple((ii - lo).T)] = True
        sub = binary_erosion(sub, st) if which == "mask_erode" else binary_dilation(sub, st)
        inside = np.all((idx_all >= lo) & (idx_all < hi), 1)
        new = np.zeros_like(roi)
        new[inside] = sub[tuple((idx_all[inside] - lo).T)]
        changed += int((new != roi).sum())
        out_t.append((name, tr, new, out))
    return out_t, changed


def main():
    a = sys.argv[1:]
    code = a[a.index("--head") + 1]
    only = a[a.index("--only") + 1].split(",") if "--only" in a else None
    proto = json.load(open(a[a.index("--protocols") + 1], encoding="utf-8"))["targets"]
    import v22_fem_fullsearch as fs
    import score_targets_v2 as ST
    from atlas_thal_score import build_targets_thal
    from spicehead import config, mri
    from spicehead.homogenize import homogenized_model
    from spicehead.electrodes import Electrode, contact_nodes, pair_injection
    from spicehead.field import efield_components
    from montage_search import snap
    from v21_head_convergence import make_solver
    import torch
    assert config.NETWORK == "homogenized"
    m2m = a[a.index("--m2m") + 1] if "--m2m" in a else os.path.join(os.environ["SPICEHEAD_CHARM_ROOT"], f"m2m_{code}")
    od = os.path.join(OUT, code); os.makedirs(od, exist_ok=True)
    t0 = time.time()
    geo = fs.grid_geometry(code, m2m)
    gm_world = np.asarray(geo["gm_world"], np.float32)
    zt, zx = np.load(a[a.index("--targets") + 1]), np.load(a[a.index("--also-thal") + 1])
    for z in (zt, zx):
        if not np.array_equal(z["gm_idx"].astype(np.int64), np.asarray(geo["gm_idx"]).astype(np.int64)):
            raise SystemExit(f"{code}: the target grid differs from the scoring grid")
    cereb = json.loads(str(zt["cereb_spec"]))
    tg = ST.build_targets(zt, json.loads(str(zt["centres"])), json.loads(str(zt["aal_names"])), None,
                          dict(labels=zt["t2_labels"], spec={**json.loads(str(zt["t2_spec"])), **cereb}))
    th, _ = build_targets_thal(zx, fs.EXCLUDE_MM)
    have = {t[0]: t for t in tg}
    have.update({t[0]: t for t in th if t[0] not in have})
    tnames = sorted(proto)
    targets = [have[n] for n in tnames]
    assert all(t[2].sum() >= ST.MIN_REGION_VOX for t in targets), f"{code}: a listed target is empty"
    idx_all = np.asarray(geo["gm_idx"]).astype(np.int64)
    grid_shape = tuple(int(x) for x in zt["shape"])
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    fine, aff_f, _ = mri.load_label_volume(seg)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs, pairs, montages = fs.candidate_set(ej)
    others = [x for x in labs if x != fs.REF]
    kidx = {fs.label_of(pa, pb): i for i, (pa, pb) in enumerate(montages)}
    sp_lab = [proto[n]["spicehead"] for n in tnames]; fem_lab = [proto[n]["fem"] for n in tnames]
    uniq = sorted(set(sp_lab) | set(fem_lab))
    sub = [montages[kidx[m]] for m in uniq]
    pos_of = {m: i for i, m in enumerate(uniq)}
    isp, ifem = np.array([pos_of[m] for m in sp_lab]), np.array([pos_of[m] for m in fem_lab])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{code}: {len(targets)} targets, {len(uniq)} montages, {len(idx_all):,} GM voxels ({time.time()-t0:.0f} s)", flush=True)

    def build(labels, cond):
        model, aff_c = homogenized_model(labels, aff_f, U.DX, min_fill=config.HOMOG_MIN_FILL, cond=cond)
        solver, _ = make_solver(model.G)
        return model, aff_c, solver

    def run(name, model, aff_c, solver, pos, params, masks=()):
        names = [name] + list(masks)
        if all(os.path.exists(os.path.join(od, f"{n}.npz")) for n in names):
            return "cached"
        t1 = time.time()
        scalp_vox = np.argwhere(model.labels == fs.SCALP_LABEL)
        elecs = {}
        for lab in labs:
            v = snap(scalp_vox, mri.world_to_voxel(aff_c, pos[lab]))
            elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v), radius_nodes=fs.ELEC_RADIUS_MM / U.DX), lab)
        basis = np.zeros((len(others), 3, len(gm_world)), np.float32)
        for i, lab in enumerate(others):
            V = solver(pair_injection(model, elecs[lab], elecs[fs.REF], fs.CURRENT))
            basis[i] = np.stack([fs.sample_on_gm(c, aff_c, gm_world) for c in efield_components(model, V)])
        t_solve = time.time() - t1
        for n in names:
            tg_, changed = targets, 0
            if n != name:
                tg_, changed = mask_variant(targets, n, grid_shape, idx_all)
            F, T, O = (x.astype(np.float64) for x in ST.score_model(basis, others, pairs, sub, tg_, dev))
            ar = np.arange(len(tnames))
            pick = lambda w: np.stack([F[ar, w], T[ar, w], O[ar, w]], 1).astype(np.float32)
            prm = dict(params, code=code, run=n, n_nodes=int(model.n_nodes), t_solve_s=round(t_solve, 1), t_run_s=round(time.time() - t1, 1), **({"which": n, "mask_changed_points": changed, "n_roi_masked": [int(t[2].sum()) for t in tg_]} if n != name else {}))
            np.savez(os.path.join(od, f"{n}.part.npz"), sp=pick(isp), fem=pick(ifem), targets=np.array(tnames), sp_labels=np.array(sp_lab), fem_labels=np.array(fem_lab), cols=np.array(["F", "T", "O"]), params=json.dumps(prm))
            os.replace(os.path.join(od, f"{n}.part.npz"), os.path.join(od, f"{n}.npz"))
        return f"ok (solve {t_solve:.0f} s, run {time.time()-t1:.0f} s)"

    want = lambda n: only is None or n in only
    t1 = time.time(); model, aff_c, solver = build(fine, None); t_build = time.time() - t1
    print(f"{code} model built in {t_build:.0f} s", flush=True)
    pos0 = {lab: np.asarray(ej[lab], float) for lab in labs}
    ii, jj, kk = model.grid_coords
    centroid = aff_c[:3, :3] @ np.array([ii.mean(), jj.mean(), kk.mean()]) + aff_c[:3, 3]
    print(f"{code} nominal: {run('nominal', model, aff_c, solver, pos0, dict(scenario='nominal'), masks=('mask_erode', 'mask_dilate'))}", flush=True)
    for k in range(U.N_JIT):
        n = f"jit_{k:02d}"
        if want(n):
            print(f"{code} {n}: {run(n, model, aff_c, solver, U.jitter_positions(code, k, pos0, centroid), dict(scenario='jitter', sd_mm=U.JIT_SD_MM))}", flush=True)
    for k in range(U.N_ROT):
        n = f"rot_{k:02d}"
        if want(n):
            pos, rp = U.rotate_positions(code, k, pos0, centroid)
            print(f"{code} {n}: {run(n, model, aff_c, solver, pos, dict(scenario='rotation', **rp))}", flush=True)
    del model, solver
    Z = U.cond_samples(code)
    for k in range(U.N_COND):
        n = f"cond_{k:02d}"
        if want(n) and not os.path.exists(os.path.join(od, f"{n}.npz")):
            t1 = time.time(); m, ac, s = build(fine, U.cond_dict(Z[k])); b = time.time() - t1
            print(f"{code} {n}: build {b:.0f} s, {run(n, m, ac, s, pos0, dict(scenario='conductivity', log_factors=dict(zip([t for t, _, _ in U.COND_TISSUES], Z[k].round(6).tolist()))))}", flush=True)
            del m, s
    for n in U.SEG:
        if want(n) and not os.path.exists(os.path.join(od, f"{n}.npz")):
            Lm = U.seg_labels(fine, n)
            t1 = time.time(); m, ac, s = build(Lm, None); b = time.time() - t1
            print(f"{code} {n}: build {b:.0f} s, {run(n, m, ac, s, pos0, dict(scenario='segmentation', changed_voxels=int((Lm != fine).sum())))}", flush=True)
            del m, s, Lm
    if only is None:
        open(os.path.join(od, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{code}: finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
