# -*- coding: utf-8 -*-
"""SPICEhead v23 on the 111 targets of the 300-head atlas of 22 Oct (26 Oct 2026; PI: "score v23 on the 109 targets on
nano4; i want this SPCIEHead paper as the most accuarate and comprehensive TI simulations"). GPU interpreter,
SPICEHEAD_MODEL=v22 (v23 networks are built directly with spicehead.homogenize, as in atlas_ext_score.py).

Targets: those of the atlas (score_targets_v2.build_targets on the head's grid, AAL90 labels, tier-2 labels and spheres,
ref6 spheres), from prep_uq_targets.py --atlas300 (<code>__atlastargets.npz); VTA_L/R stay empty (no grey matter).
Models: sp2v23 and sp1v23, the homogenized network at 2 and 1 mm, built and sampled exactly as in atlas_ext_score.py
(2 mm at float64 grid coordinates, 1 mm at float32, as the published 300-head homogenized run). FEM and v22 scores of
the same targets are the atlas's own (targets_out/<code>__targets.npz); the collector joins them by name.
Output: <out>/<code>__v23atlas.npz with F, T (models x targets x montages, float32), models, targets, tiers, n_roi,
n_out, montages, meta.
    SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu python pipeline/nchc/atlas_v23_score.py --head h001 --m2m DIR --targets <code>__atlastargets.npz --out DIR
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline"), HERE]
from atlas_ext_score import SP_MODELS                                         # noqa: E402

V23_MODELS = [m for m in SP_MODELS if m[1] == "v23"]                         # sp2v23, sp1v23


def main():
    a = sys.argv[1:]
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    code, m2m, tfile, out = arg("--head"), arg("--m2m"), arg("--targets"), arg("--out")
    assert os.environ.get("SPICEHEAD_MODEL", "v22") == "v22", "run with SPICEHEAD_MODEL=v22 (v23 networks are built directly)"
    p_out = os.path.join(out, f"{code}__v23atlas.npz")
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
    targets = ST.build_targets(zt, json.loads(str(zt["centres"])), json.loads(str(zt["aal_names"])), None,
                               dict(labels=zt["t2_labels"], spec=json.loads(str(zt["t2_spec"]))))
    aff = np.asarray(zt["affine"], float); gm_idx = zt["gm_idx"].astype(np.int64)
    gm_world = (aff[:3, :3] @ gm_idx.T).T + aff[:3, 3]
    pts64, pts32 = gm_world, gm_world.astype(np.float32)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs, pairs, montages = fs.candidate_set(ej)
    others = [x for x in labs if x != fs.REF]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    fine, aff_f, _ = mri.load_label_volume(seg)
    bases, timing = {}, {}
    for name, _, dx in V23_MODELS:
        t1 = time.time()
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
    models = [m for m, _, _ in V23_MODELS]
    t1 = time.time()
    res = {m: ST.score_model(bases[m], others, pairs, montages, targets, dev) for m in models}
    F = np.stack([res[m][0] for m in models]); T = np.stack([res[m][1] for m in models])
    meta = dict(code=code, n_gm=int(len(gm_world)), t_bases_s=timing, t_score_s=round(time.time() - t1, 1), t_total_s=round(time.time() - t0, 1))
    np.savez(p_out + ".part.npz", F=F, T=T, models=np.array(models), targets=np.array([t[0] for t in targets]),
             tiers=np.array([t[1] for t in targets]), n_roi=np.array([int(t[2].sum()) for t in targets]),
             n_out=np.array([int(t[3].sum()) for t in targets]), montages=np.array([fs.label_of(pa, pb) for pa, pb in montages]),
             meta=np.array(json.dumps(meta)))
    os.replace(p_out + ".part.npz", p_out)
    print(f"{code}: v23 atlas scored ({meta['t_total_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
