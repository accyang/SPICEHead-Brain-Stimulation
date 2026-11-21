# -*- coding: utf-8 -*-
"""SPICEhead bases and scoring for the cerebellum and brainstem extension of the 300-head atlas (24 Oct 2026; PI: "build
the FEM pipeline for all 300 heads"). GPU interpreter, SPICEHEAD_MODEL=v22 (v23 networks are built directly with
spicehead.homogenize, exactly as in the 300-head homogenized run).

Per head, six models on the same points (the grey-matter scoring grid of the 300-head run, then the brainstem
nucleus points of prep_brainstem.py):
  fem      atlas_ext_fem.py (grey matter from the grey-matter leadfield, i.e. the published FEM basis)
  sp2, sp1, sp5          SPICEhead v22 at 2, 1, 5 mm (published construction: mri.head_model under v22)
  sp2v23, sp1v23         SPICEhead v23 at 2, 1 mm (homogenized network)
Coordinates follow the published runs so the regression targets reproduce them: the 2 mm bases of the 300-head runs
were sampled at float64 grid coordinates (cache miss), the 1 and 5 mm bases at the cached float32 values.
Targets (44): the six Paper A spheres (ref6, regression check against the published scores), the 26 AAL3 cerebellar
regions (grey matter), the 12 AAL3 brainstem nuclei (T = median envelope over the nucleus points, O = 99th percentile
over grey matter beyond 20 mm). Region and sphere definitions are those of score_targets_v2.build_targets.
Output: <out>/<code>__ext.npz with F, T (models x targets x montages, float32; O = T / F is not stored, to fit the
/work quota), models, targets, tiers, n_roi, n_out, montages, meta.
    SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu python pipeline/nchc/atlas_ext_score.py --head h001 --m2m DIR --targets <code>__exttargets.npz --fem <code>__fem_ext.npz --out DIR
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline"), HERE]
MIN_REGION_VOX = 5
SP_MODELS = [("sp2", "v22", 2.0), ("sp1", "v22", 1.0), ("sp5", "v22", 5.0), ("sp2v23", "v23", 2.0), ("sp1v23", "v23", 1.0)]


def build_targets_ext(zt, exclude_mm):
    """ref6 spheres + cerebellar regions on grey matter (as score_targets_v2.build_targets) + brainstem nuclei on the extra points."""
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    gm_idx = zt["gm_idx"].astype(np.int64)
    aff = np.asarray(zt["affine"], float); shape = tuple(int(x) for x in zt["shape"])
    gm_world = (aff[:3, :3] @ gm_idx.T).T + aff[:3, 3]
    vox = np.sqrt((aff[:3, :3] ** 2).sum(0)); pad_vox = int(np.ceil(exclude_mm / vox.min())) + 2
    bs_idx = zt["bs_idx"].astype(np.int64); bs_lab = zt["bs_labels"]
    bs_world = (aff[:3, :3] @ bs_idx.T).T + aff[:3, 3]
    n_gm, n_bs = len(gm_idx), len(bs_idx)
    z_bs = np.zeros(n_bs, bool)
    T = []
    for name, d in json.loads(str(zt["centres"])).items():                       # ref6 spheres
        roi = np.zeros(n_gm, bool); near = np.zeros(n_gm, bool)
        for c in d["centres"]:
            dist = np.linalg.norm(gm_world - np.asarray(c, float), axis=1)
            roi |= dist <= float(d["radius_mm"]); near |= dist <= exclude_mm
        T.append((name, d.get("tier", "ref6"), np.r_[roi, z_bs], np.r_[~near, z_bs]))
    for name, vals in json.loads(str(zt["cereb_spec"])).items():                 # cerebellum: grey-matter regions
        sel = np.isin(zt["t2_labels"], vals)
        if sel.sum() < MIN_REGION_VOX:
            T.append((name, "cerebellum", np.zeros(n_gm + n_bs, bool), np.zeros(n_gm + n_bs, bool))); continue
        ri = gm_idx[sel]; lo = np.maximum(ri.min(0) - pad_vox, 0); hi = np.minimum(ri.max(0) + pad_vox + 1, shape)
        box = np.ones(tuple(hi - lo), bool); box[tuple((ri - lo).T)] = False
        dist = distance_transform_edt(box, sampling=vox)
        inb = np.all((gm_idx >= lo) & (gm_idx < hi), axis=1); near = np.zeros(n_gm, bool)
        g = gm_idx[inb] - lo; near[inb] = dist[g[:, 0], g[:, 1], g[:, 2]] <= exclude_mm
        T.append((name, "cerebellum", np.r_[sel, z_bs], np.r_[~near, z_bs]))
    for name, vals in json.loads(str(zt["brainstem_spec"])).items():             # brainstem nuclei on the extra points
        sel = np.isin(bs_lab, vals)
        if sel.sum() < MIN_REGION_VOX:
            T.append((f"brainstem:{name}", "brainstem", np.zeros(n_gm + n_bs, bool), np.zeros(n_gm + n_bs, bool))); continue
        dist, _ = cKDTree(bs_world[sel]).query(gm_world, k=1)
        T.append((f"brainstem:{name}", "brainstem", np.r_[np.zeros(n_gm, bool), sel], np.r_[dist > exclude_mm, z_bs]))   # prefix as in uq_v23_atlas.py
    return T, gm_world, bs_world


def main():
    a = sys.argv[1:]
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    code, m2m, tfile, ffile, out = arg("--head"), arg("--m2m"), arg("--targets"), arg("--fem"), arg("--out")
    assert os.environ.get("SPICEHEAD_MODEL", "v22") == "v22", "run with SPICEHEAD_MODEL=v22 (v23 networks are built directly)"
    p_out = os.path.join(out, f"{code}__ext.npz")
    if os.path.exists(p_out):
        print(f"{code}: cached"); return
    import torch
    import v22_fem_fullsearch as fs
    import score_targets_v2 as ST
    from spicehead import mri
    from spicehead.homogenize import homogenized_model
    from spicehead.electrodes import Electrode, contact_nodes, pair_injection
    from spicehead.field import efield_components
    from montage_search import snap
    from v21_head_convergence import make_solver
    t0 = time.time()
    zt = np.load(tfile)
    targets, gm_world, bs_world = build_targets_ext(zt, fs.EXCLUDE_MM)
    pts64 = np.concatenate([gm_world, bs_world])
    pts32 = np.concatenate([gm_world.astype(np.float32), bs_world.astype(np.float32)])
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs, pairs, montages = fs.candidate_set(ej)
    others = [x for x in labs if x != fs.REF]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    zf = np.load(ffile)
    assert list(zf["labs"]) == others, "FEM electrode order differs"
    bases = {"fem": np.concatenate([zf["basis_gm"], zf["basis_bs"]], axis=2)}
    # brainstem points where the FEM field is zero for every electrode lie outside the WM+GM mesh volume (3 of 2,104 in
    # the h001 test): leave them out of the nucleus masks, for every model alike
    n_gm = len(gm_world)
    dead = np.flatnonzero(np.abs(zf["basis_bs"]).max(axis=(0, 1)) == 0)
    if dead.size:
        for _, tier, roi, _ in targets:
            if tier == "brainstem":
                roi[n_gm + dead] = False
    timing = {}
    fine = None
    for name, ver, dx in SP_MODELS:
        t1 = time.time()
        if ver == "v22":
            model, aff_c = mri.head_model(seg, dx)                                 # published v22 construction
        else:
            if fine is None:
                fine, aff_f, _ = mri.load_label_volume(seg)
            model, aff_c = homogenized_model(fine, aff_f, dx, min_fill=0.5)
        scalp_vox = np.argwhere(model.labels == fs.SCALP_LABEL)
        elecs = {lab: Electrode(contact_nodes(model, tuple(int(x) for x in snap(scalp_vox, mri.world_to_voxel(aff_c, ej[lab]))),
                                              radius_nodes=fs.ELEC_RADIUS_MM / dx), lab) for lab in labs}
        solver, _ = make_solver(model.G)
        pts = pts64 if dx == 2.0 else pts32
        B = np.zeros((len(others), 3, len(pts)), np.float32)
        for i, lab in enumerate(others):
            V = solver(pair_injection(model, elecs[lab], elecs[fs.REF], fs.CURRENT))
            B[i] = np.stack([fs.sample_on_gm(c, aff_c, pts) for c in efield_components(model, V)])
        bases[name] = B
        timing[name] = round(time.time() - t1, 1)
        del model, solver
        print(f"  {code} {name}: basis in {timing[name]:.0f} s", flush=True)
    models = ["fem"] + [m for m, _, _ in SP_MODELS]
    t1 = time.time()
    res = {m: ST.score_model(bases[m], others, pairs, montages, targets, dev) for m in models}
    F = np.stack([res[m][0] for m in models]); T = np.stack([res[m][1] for m in models])
    meta = dict(code=code, n_gm=int(len(gm_world)), n_bs=int(len(bs_world)), n_bs_zero_fem_excluded=int(dead.size), t_bases_s=timing, t_score_s=round(time.time() - t1, 1),
                t_total_s=round(time.time() - t0, 1), fem=json.loads(str(zf["meta"])))
    np.savez(p_out + ".part.npz", F=F, T=T, models=np.array(models), targets=np.array([t[0] for t in targets]),
             tiers=np.array([t[1] for t in targets]), n_roi=np.array([int(t[2].sum()) for t in targets]),
             n_out=np.array([int(t[3].sum()) for t in targets]), montages=np.array([fs.label_of(pa, pb) for pa, pb in montages]),
             meta=np.array(json.dumps(meta)))
    os.replace(p_out + ".part.npz", p_out)
    print(f"{code}: extension scored ({meta['t_total_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
