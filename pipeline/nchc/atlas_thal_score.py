# -*- coding: utf-8 -*-
"""AAL3 thalamic nuclei and supracallosal ACC for the 300-head atlas (25 Oct 2026; PI: "add the thalamic nuclei and
ACC_sup on nano4"). GPU interpreter, SPICEHEAD_MODEL=v22, as atlas_ext_score.py.

Targets (22): the six Paper A spheres (ref6, regression check against the published scores) and 16 AAL3 regions on
grey matter: Thal_AV, Thal_LP, Thal_VA, Thal_VL, Thal_VPL, Thal_IL, Thal_LGN (values 121-132, 139-140) and ACC_sup
(155-156), left and right. Regions are built by atlas_ext_score.build_targets_ext on its cerebellar code path (that of
score_targets_v2.build_targets) with no brainstem points. Left out: Thal_Re and Thal_MGN (median 4 and 6 grey-matter
voxels per head; 5 or more in only 99-106 and 195-202 of the 300 heads). MD and pulvinar are tier-2 targets already.
Models: fem (the published FEM grey-matter basis, heads/<code>/<code>__fem_basis.npz), sp2, sp1, sp5 (v22) and
sp2v23, sp1v23 (homogenized network), built and sampled exactly as in atlas_ext_score.py.
Output: <out>/<code>__thal.npz with F, T (models x targets x montages, float32), models, targets, tiers, n_roi, n_out,
montages, meta.
    SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu python pipeline/nchc/atlas_thal_score.py --head h001 --m2m DIR --targets <code>__exttargets.npz --fem <code>__fem_basis.npz --out DIR
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline"), HERE]
from atlas_ext_score import SP_MODELS, build_targets_ext                     # noqa: E402

NEW = {"Thal_AV_L": 121, "Thal_AV_R": 122, "Thal_LP_L": 123, "Thal_LP_R": 124, "Thal_VA_L": 125, "Thal_VA_R": 126,
       "Thal_VL_L": 127, "Thal_VL_R": 128, "Thal_VPL_L": 129, "Thal_VPL_R": 130, "Thal_IL_L": 131, "Thal_IL_R": 132,
       "Thal_LGN_L": 139, "Thal_LGN_R": 140, "ACC_sup_L": 155, "ACC_sup_R": 156}


def build_targets_thal(zt, exclude_mm):
    """ref6 spheres + the 16 grey-matter regions (build_targets_ext with the new regions in place of the cerebellar ones)."""
    z = {k: zt[k] for k in ("gm_idx", "affine", "shape", "t2_labels", "centres")}
    z.update(bs_idx=np.zeros((0, 3), np.int64), bs_labels=np.zeros(0, np.int64),
             cereb_spec=json.dumps({n: [v] for n, v in NEW.items()}), brainstem_spec="{}")
    T, gm_world, _ = build_targets_ext(z, exclude_mm)
    tier = lambda n, t: ("thalamus" if n.startswith("Thal_") else "cingulate") if n in NEW else t
    return [(n, tier(n, t), roi, out) for n, t, roi, out in T], gm_world


def main():
    a = sys.argv[1:]
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    code, m2m, tfile, ffile, out = arg("--head"), arg("--m2m"), arg("--targets"), arg("--fem"), arg("--out")
    assert os.environ.get("SPICEHEAD_MODEL", "v22") == "v22", "run with SPICEHEAD_MODEL=v22 (v23 networks are built directly)"
    p_out = os.path.join(out, f"{code}__thal.npz")
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
    targets, gm_world = build_targets_thal(zt, fs.EXCLUDE_MM)
    pts64, pts32 = gm_world, gm_world.astype(np.float32)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs, pairs, montages = fs.candidate_set(ej)
    others = [x for x in labs if x != fs.REF]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    zf = np.load(ffile)
    assert list(zf["labs"]) == others, "FEM electrode order differs"
    assert zf["basis"].shape == (len(others), 3, len(gm_world)), "FEM basis is not on this grey-matter grid"
    bases = {"fem": zf["basis"]}
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
    meta = dict(code=code, n_gm=int(len(gm_world)), t_bases_s=timing, t_score_s=round(time.time() - t1, 1),
                t_total_s=round(time.time() - t0, 1), fem=str(zf["meta"]))
    np.savez(p_out + ".part.npz", F=F, T=T, models=np.array(models), targets=np.array([t[0] for t in targets]),
             tiers=np.array([t[1] for t in targets]), n_roi=np.array([int(t[2].sum()) for t in targets]),
             n_out=np.array([int(t[3].sum()) for t in targets]), montages=np.array([fs.label_of(pa, pb) for pa, pb in montages]),
             meta=np.array(json.dumps(meta)))
    os.replace(p_out + ".part.npz", p_out)
    print(f"{code}: thalamic nuclei + ACC_sup scored ({meta['t_total_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
