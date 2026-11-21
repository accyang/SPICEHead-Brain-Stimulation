# -*- coding: utf-8 -*-
"""Experiment 3 (23 Oct 2026): does the field EXTRACTION inflate SPICEhead's grey-matter field?

SPICEhead's E = -grad V by central differences over +-1 voxel, then trilinear sampling onto the 1 mm grid.
At a grey-matter voxel next to white matter (lower sigma, higher E) or CSF (higher sigma, lower E) the
central difference mixes the neighbour's gradient into the GM value, and trilinear sampling mixes
neighbouring voxels' E as well. FEM reports the field inside each GM element. This test solves the
standard v22 network for one head and compares four extraction variants with the stored FEM basis:

  central / trilinear      the production definition
  central / nearest        no interpolation across voxels
  tissue  / trilinear      differences use only neighbours of the SAME tissue label (one-sided where
                           the opposite neighbour differs; central where both agree)
  tissue  / nearest

reported over all GM voxels and separately for GM voxels whose six face neighbours are all GM (interior)
and the rest (boundary), per model grid (2 and 1 mm).

    set SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu
    python pipeline/nchc/field_definition_test.py --heads h001 [--dx 2,1]
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "pipeline"))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"
CHARM = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")
OUT = os.path.join(ROOT, "results_v22", "v22_field_definition_test")
os.environ.setdefault("SPICEHEAD_MODEL", "v22")
os.environ.setdefault("SPICEHEAD_FULLSEARCH_CACHE", r"C:\spicehead_v22_logs\pad_experiment\cache")
os.environ["SPICEHEAD_CHARM_ROOT"] = CHARM


def tissue_gradient(Vg, labels, dx, mask):
    """-dV/dx per axis using only same-tissue neighbours: central if both neighbours share the centre's label,
    one-sided towards the one that does, central (production) otherwise."""
    comps = []
    for axis in range(3):
        E = np.full(Vg.shape, np.nan)
        c = [slice(1, -1)] * 3; f = [slice(1, -1)] * 3; b = [slice(1, -1)] * 3
        f[axis] = slice(2, None); b[axis] = slice(0, -2)
        c, f, b = tuple(c), tuple(f), tuple(b)
        Vc, Vf, Vb = Vg[c], Vg[f], Vg[b]
        same_f = (labels[f] == labels[c]) & mask[f]; same_b = (labels[b] == labels[c]) & mask[b]
        with np.errstate(invalid="ignore"):
            central = -(Vf - Vb) / (2 * dx)
            fwd = -(Vf - Vc) / dx
            bwd = -(Vc - Vb) / dx
        Ec = np.where(same_f & same_b, central, np.where(same_f, fwd, np.where(same_b, bwd, central)))
        E[c] = Ec
        comps.append(E)
    return comps


def sample(comp, aff, gm_world, order):
    from scipy.ndimage import map_coordinates
    inv = np.linalg.inv(aff)
    vox = (inv[:3, :3] @ gm_world.T) + inv[:3, 3:4]
    return map_coordinates(np.nan_to_num(np.asarray(comp, np.float64)), vox, order=order, mode="nearest")


def run(code, cid, dx):
    import v22_fem_fullsearch as fs
    from spicehead import mri
    from spicehead.electrodes import Electrode, contact_nodes, pair_injection
    from spicehead.field import efield_components
    from montage_search import snap
    from v21_head_convergence import make_solver
    t0 = time.time()
    m2m = os.path.join(CHARM, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)
    geo = fs.grid_geometry(cid, m2m)
    labels, aff, _ = mri.load_label_volume(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"), dx_mm=dx)
    model = mri.model_from_labels(labels, None, dx_mm=dx)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs, pairs, montages = fs.candidate_set(ej)
    scalp = np.argwhere(labels == fs.SCALP_LABEL)
    elecs = {lab: Electrode(contact_nodes(model, tuple(int(x) for x in snap(scalp, mri.world_to_voxel(aff, ej[lab]))), radius_nodes=fs.ELEC_RADIUS_MM / dx), lab) for lab in labs}
    solver, _ = make_solver(model.G)
    others = [x for x in labs if x != fs.REF]
    with np.load(os.path.join(L, "bundles", f"{code}.npz")) as z:
        Bf, labs_f = z["basis_fem"], list(z["labs_fem"]); gw = z["gm_world"].astype(float); gi = z["gm_idx"].astype(np.int64)
    assert labs_f == others
    # interior GM voxels of the 1 mm grid: all six face neighbours GM on the (1 mm) label grid
    lab1 = mri.load_label_volume(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"), dx_mm=1.0)[0] if dx != 1.0 else labels
    gm1 = np.pad(lab1 == fs.GM_LABEL, 1)
    inter = np.ones(len(gi), bool)
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        inter &= gm1[gi[:, 0] + 1 + d[0], gi[:, 1] + 1 + d[1], gi[:, 2] + 1 + d[2]]
    variants = {("central", "trilinear"): 1, ("central", "nearest"): 0, ("tissue", "trilinear"): 1, ("tissue", "nearest"): 0}
    acc = {k: dict(rel=[], ratio=[], rel_int=[], ratio_int=[], rel_bnd=[], ratio_bnd=[], bias=[]) for k in variants}
    for i, lab in enumerate(others):
        V = solver(pair_injection(model, elecs[lab], elecs[fs.REF], fs.CURRENT))
        Vg = model.nodes_to_grid(V, fill=np.nan)
        comps = {"central": efield_components(model, V), "tissue": tissue_gradient(Vg, labels, model.dx, model.mask)}
        ef = Bf[i].astype(np.float64); nf = np.linalg.norm(ef, axis=0); ok = nf > 1e-9 * nf.max()
        for (grad, interp), order in variants.items():
            es = np.stack([sample(c, aff, gw, order) for c in comps[grad]])
            ns = np.linalg.norm(es, axis=0)
            rel = np.linalg.norm(es - ef, axis=0) / np.maximum(nf, 1e-30); ratio = ns / np.maximum(nf, 1e-30)
            a = acc[(grad, interp)]
            a["rel"].append(np.median(rel[ok])); a["ratio"].append(np.median(ratio[ok])); a["bias"].append(100 * (ns[ok].mean() / nf[ok].mean() - 1))
            a["rel_int"].append(np.median(rel[ok & inter])); a["ratio_int"].append(np.median(ratio[ok & inter]))
            a["rel_bnd"].append(np.median(rel[ok & ~inter])); a["ratio_bnd"].append(np.median(ratio[ok & ~inter]))
    res = {f"{g}/{s}": {k: float(np.median(v)) for k, v in a.items()} for (g, s), a in acc.items()}
    res["_meta"] = dict(code=code, dx_mm=dx, n_gm=int(len(gi)), interior_share=float(inter.mean()), t_s=round(time.time() - t0, 1))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{code}__sp{dx:g}_fielddef.json"), "w"), indent=1)
    print(f"{code} {dx:g} mm: GM voxels {len(gi):,}, interior {100*inter.mean():.0f}% ({time.time()-t0:.0f} s)")
    print(f"  {'variant':20s} {'rel all':>8s} {'ratio all':>10s} {'bias %':>7s} | {'rel int':>8s} {'ratio int':>10s} | {'rel bnd':>8s} {'ratio bnd':>10s}")
    for k, v in res.items():
        if k != "_meta":
            print(f"  {k:20s} {v['rel']:8.3f} {v['ratio']:10.3f} {v['bias']:7.1f} | {v['rel_int']:8.3f} {v['ratio_int']:10.3f} | {v['rel_bnd']:8.3f} {v['ratio_bnd']:10.3f}")


def main():
    a = sys.argv[1:]
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    codes = a[a.index("--heads") + 1].split(",")
    dxs = [float(x) for x in a[a.index("--dx") + 1].split(",")] if "--dx" in a else [2.0]
    for c in codes:
        for dx in dxs:
            run(c, cmap[c], dx)


if __name__ == "__main__":
    main()
