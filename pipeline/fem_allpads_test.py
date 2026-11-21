# -*- coding: utf-8 -*-
"""Is FEM superposition exact when every electrode pad is present in every run?

The first FEM basis (v22_fem_fullsearch.py --fem-basis) meshed only the two electrodes of each
run, so E(a,Cz) - E(b,Cz) mixed geometries and missed the direct E(a,b) by 4-8% at the worst
voxel: a pad with zero net current still shunts current. This test runs, on one head, three
SimNIBS simulations with all 19 pads meshed (zero current on the others): T8-Cz, P8-Cz and the
direct T8-P8, then compares the reconstruction with the direct run (should be at solver precision)
and both with the existing two-pad direct run (the passive-pad effect itself).

  "<SimNIBS>\\simnibs_env\\python.exe" pipeline/fem_allpads_test.py
"""
import json
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
M2M = r"L:\BrainProbe\SPICEhead\charm\m2m_SUBJ01"
OUT = r"C:\spicehead_v22_logs\fem_tests\allpads"
OLD = os.path.join(ROOT, "results", "v21_simnibs_validation", "SUBJ01", "ch1")   # T8-P8, two pads
RUNS = (("T8", "Cz", "T8_Cz"), ("P8", "Cz", "P8_Cz"), ("T8", "P8", "T8_P8"))


def run(ej, labs, pos, neg, name):
    from simnibs import sim_struct, run_simnibs
    fem = os.path.join(OUT, name)
    vol = os.path.join(fem, "subject_volumes")
    if os.path.isdir(vol) and any(f.endswith("_E.nii.gz") for f in os.listdir(vol)):
        print(name, "(cached)", flush=True)
        return
    os.makedirs(fem, exist_ok=True)
    S = sim_struct.SESSION()
    S.subpath, S.pathfem = M2M, fem
    S.fields, S.map_to_vol, S.open_in_gmsh = "E", True, False
    t = S.add_tdcslist()
    cur = []
    for i, lab in enumerate(labs):
        e = t.add_electrode()
        e.channelnr = i + 1
        e.centre = ej[lab]
        e.shape = "ellipse"
        e.dimensions = [24.0, 24.0]
        e.thickness = [2.0]
        cur.append(1e-3 if lab == pos else (-1e-3 if lab == neg else 0.0))
    t.currents = cur
    t0 = time.time()
    run_simnibs(S)
    print(f"{name} OK ({(time.time()-t0)/60:.1f} min, {len(labs)} pads)", flush=True)
    for f in os.listdir(fem):
        if f.endswith((".msh", ".geo")):
            os.remove(os.path.join(fem, f))


def load_E(fem):
    import nibabel as nib
    vol = os.path.join(fem, "subject_volumes")
    p = [os.path.join(vol, f) for f in sorted(os.listdir(vol)) if f.endswith("_E.nii.gz")][0]
    img = nib.load(p)
    return np.nan_to_num(np.asanyarray(img.dataobj).astype(np.float64)), img.affine


def gm_mask(aff, shape):
    import nibabel as nib
    lab = nib.load(os.path.join(M2M, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    d = np.asanyarray(lab.dataobj)
    inv = np.linalg.inv(lab.affine) @ aff
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    v = np.rint(inv[:3, :3] @ np.stack([ii.ravel(), jj.ravel(), kk.ravel()]) + inv[:3, 3:4]).astype(int)
    ok = np.all((v >= 0) & (v < np.array(d.shape)[:, None]), axis=0)
    gm = np.zeros(v.shape[1], bool)
    gm[ok] = d[v[0, ok], v[1, ok], v[2, ok]] == 2
    return gm.reshape(shape)


def compare(rec, ref, gm, what):
    r, d = rec[gm], ref[gm]
    mag = np.linalg.norm(d, axis=1)
    rel_max = float(np.abs(r - d).max() / np.abs(d).max())
    rel_med = float(np.median(np.linalg.norm(r - d, axis=1) / np.maximum(mag, 1e-12)))
    print(f"  {what}: max|diff|/max|ref| = {rel_max:.2e}; median voxel-wise relative = {rel_med:.2e}; "
          f"ROI-free magnitude ratio median = {np.median(np.linalg.norm(r, axis=1) / np.maximum(mag, 1e-12)):.4f}",
          flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    ej = json.load(open(os.path.join(M2M, "electrodes_1020.json")))
    labs = sorted(ej)
    for pos, neg, name in RUNS:
        run(ej, labs, pos, neg, name)
    A, aff = load_E(os.path.join(OUT, "T8_Cz"))
    B, _ = load_E(os.path.join(OUT, "P8_Cz"))
    D, _ = load_E(os.path.join(OUT, "T8_P8"))
    O, _ = load_E(OLD)
    gm = gm_mask(aff, A.shape[:3])
    print(f"GM voxels: {int(gm.sum())}", flush=True)
    compare(A - B, D, gm, "all-pads basis difference vs all-pads direct T8-P8 (superposition)")
    compare(D, O, gm, "all-pads direct vs two-pad direct T8-P8 (passive-pad effect)")
    compare(A - B, O, gm, "all-pads basis difference vs two-pad direct T8-P8")


if __name__ == "__main__":
    main()
