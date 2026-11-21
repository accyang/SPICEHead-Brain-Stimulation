# -*- coding: utf-8 -*-
"""Experiment 3b (23 Oct 2026): the standard v22 network at 0.5 mm (the CHARM label resolution; ~30 M nodes),
to measure the convergence trend 2 -> 1 -> 0.5 mm against FEM. GPU solver required (H200: Nano 4).

Same construction as v22_fem_fullsearch.stage_sp_basis (nearest-label grid, v22 skin-surface contacts,
harmonic rule), sampled on the shared 1 mm grid, scored for the six Paper A targets; the basis is saved
as <code>__sp0.5_basis.npz for the field comparison with the FEM bases on the desktop (pad_experiment
--summary style: fine_grid --summary).

    SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu python fine_grid_experiment.py --heads h001 [--dx 0.5]
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
OUT = os.environ.get("SPICEHEAD_EXP_OUT", os.path.join(ROOT, "results_v22", "v22_fine_grid_experiment"))
FEMJ = os.environ.get("SPICEHEAD_FULLSEARCH_OUT", "")
os.environ.setdefault("SPICEHEAD_MODEL", "v22")
os.environ.setdefault("SPICEHEAD_FULLSEARCH_CACHE", r"C:\spicehead_v22_logs\pad_experiment\cache")
os.environ["SPICEHEAD_CHARM_ROOT"] = CHARM


def run_head(code, cmap, dx):
    import v22_fem_fullsearch as fs
    from spicehead import mri
    from spicehead.electrodes import Electrode, contact_nodes, pair_injection
    from spicehead.field import efield_components
    from spicehead.score_gpu import score_montages_gpu
    from montage_search import snap
    from v21_head_convergence import make_solver
    tag = f"sp{dx:g}"
    outp = os.path.join(OUT, f"{code}__{tag}.json")
    if os.path.exists(outp):
        return "cached"
    t0 = time.time()
    cid = cmap[code] if cmap else code
    m2m = os.path.join(CHARM, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)
    geo = fs.grid_geometry(cid, m2m)
    labels, aff, _ = mri.load_label_volume(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"), dx_mm=dx)
    model = mri.model_from_labels(labels, None, dx_mm=dx)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs, pairs, montages = fs.candidate_set(ej)
    scalp = np.argwhere(labels == fs.SCALP_LABEL)
    elecs = {lab: Electrode(contact_nodes(model, tuple(int(x) for x in snap(scalp, mri.world_to_voxel(aff, ej[lab]))), radius_nodes=fs.ELEC_RADIUS_MM / dx), lab) for lab in labs}
    t1 = time.time()
    solver, ncomp = make_solver(model.G)
    others = [x for x in labs if x != fs.REF]
    basis = np.zeros((len(others), 3, len(geo["gm_idx"])), np.float32)
    Rdir = aff[:3, :3] / np.sqrt((aff[:3, :3] ** 2).sum(0))        # resample_to_output grids are RAS, but be safe
    for i, lab in enumerate(others):
        V = solver(pair_injection(model, elecs[lab], elecs[fs.REF], fs.CURRENT))
        comps = efield_components(model, V)
        world = [sum(Rdir[w, v] * comps[v] for v in range(3) if abs(Rdir[w, v]) > 1e-12) for w in range(3)]
        basis[i] = np.stack([fs.sample_on_gm(c, aff, geo["gm_world"]) for c in world])
        print(f"      basis {i+1}/18 ({(time.time()-t1)/(i+1):.0f} s/solve)", flush=True)
    t_basis = time.time() - t1
    res = score_montages_gpu(basis, others, fs.REF, pairs, montages, geo["roi_sel"], geo["out_sel"], fs.ROI_ORDER, fs.label_of, verbose=False)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{code}__{tag}_basis.part.npz"), basis=basis, labs=np.array(others), meta=json.dumps(dict(model=tag, dx_mm=dx)))
    os.replace(os.path.join(OUT, f"{code}__{tag}_basis.part.npz"), os.path.join(OUT, f"{code}__{tag}_basis.npz"))
    out = dict(code=code, model=tag, dx_mm=dx, n_nodes=int(model.n_nodes), n_components=int(ncomp), t_build_s=round(t1 - t0, 1),
               t_basis_s=round(t_basis, 1), t_total_s=round(time.time() - t0, 1), rois=res)
    json.dump(out, open(outp + ".part", "w"), indent=0); os.replace(outp + ".part", outp)
    return f"ok ({model.n_nodes:,} nodes, build {t1-t0:.0f} s, basis {t_basis:.0f} s)"


def main():
    a = sys.argv[1:]
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))} if os.path.exists(CODEMAP) else None
    codes = a[a.index("--heads") + 1].split(",")
    dxs = [float(x) for x in a[a.index("--dx") + 1].split(",")] if "--dx" in a else [0.5]
    os.makedirs(os.environ["SPICEHEAD_FULLSEARCH_CACHE"], exist_ok=True)
    for c in codes:
        for dx in dxs:
            try:
                msg = run_head(c, cmap, dx)
            except Exception as e:
                import traceback; traceback.print_exc(); msg = f"FAILED {type(e).__name__}: {e}"
            print(f"{c} {dx:g} mm: {msg}", flush=True)


if __name__ == "__main__":
    main()
