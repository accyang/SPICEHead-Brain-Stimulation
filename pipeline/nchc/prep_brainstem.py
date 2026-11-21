# -*- coding: utf-8 -*-
"""Brainstem target points per head (24 Oct 2026; PI: "approve AAL3 and add cerebellum/brainstem"). SimNIBS interpreter.

The cerebellum is grey matter in CHARM, and its 26 AAL3 regions are already labelled on every grey-matter voxel
of the scoring grid (t2/<code>__t2labels.npz). The brainstem nuclei of AAL3v1 are not: CHARM labels the
brainstem white matter, so none of their voxels are in the grey-matter scoring set. The 12 nuclei are:
  VTA (159/160), substantia nigra pars compacta (161/162) and pars reticulata (163/164),
  red nucleus (165/166), locus coeruleus (167/168), dorsal and median raphe (169, 170).
This script finds, per head, the T1-grid voxels of these nuclei that lie in brain tissue (grey or white matter).
  - AAL3v1 is warped into the head exactly as prep_tier2.py does (simnibs warp_volume, mni2subject, non-linear,
    nearest neighbour; h088 through the validated warp_fallback).
  - Brain tissue = CHARM WM or GM resampled to the T1 grid with the grey-matter rule of
    v22_fem_fullsearch.grid_geometry (trilinear > 0.5).
Output: L:/BrainProbe/SPICEhead/nchc_fullsearch/t2/<code>__bslabels.npz (bs_idx: T1-grid voxel indices, bs_labels:
AAL3 values, affine, voxel counts per nucleus); h-codes only. The code map is read only.
    "<SimNIBS>/simnibs_env/python.exe" pipeline/nchc/prep_brainstem.py [--workers 3] [--heads h001,...]
"""
import csv
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
OUTD = os.path.join(L, "t2")
CHARM = r"L:\BrainProbe\SPICEhead\charm"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only
AAL3 = r"C:\Users\ALBERT YANG\atlases\AAL3\AAL3\AAL3v1_1mm.nii"
BRAINSTEM = {159: "VTA_L", 160: "VTA_R", 161: "SN_pc_L", 162: "SN_pc_R", 163: "SN_pr_L", 164: "SN_pr_R",
             165: "Red_N_L", 166: "Red_N_R", 167: "LC_L", 168: "LC_R", 169: "Raphe_D", 170: "Raphe_M"}
WM, GM = 1, 2


def m2m_of(cid):
    return os.path.join(CHARM, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)


def one(code, cid):
    import nibabel as nib
    sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline"), HERE]
    from v21_simnibs_compare import resample_to
    out = os.path.join(OUTD, f"{code}__bslabels.npz")
    if os.path.exists(out):
        return code, "cached", 0.0
    t0 = time.time()
    m2m = m2m_of(cid)
    t1 = nib.load(os.path.join(m2m, "T1.nii.gz")); aff, shape = t1.affine, t1.shape[:3]
    lab_img = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    lab = np.asanyarray(lab_img.dataobj)
    brain = resample_to(np.isin(lab, (WM, GM)).astype(np.float32), lab_img.affine, aff, shape) > 0.5
    complete = os.path.exists(os.path.join(m2m, "toMNI", "MNI2Conform_nonl.nii.gz"))
    if complete:
        from simnibs.utils.transformations import warp_volume
        warp = "simnibs warp_volume (mni2subject, nonl, nearest)"
        with tempfile.TemporaryDirectory(dir=r"C:\spicehead_v22_logs") as td:
            wf = os.path.join(td, "aal3_subj.nii.gz")
            warp_volume(AAL3, m2m, wf, transformation_direction="mni2subject", transformation_type="nonl",
                        reference=os.path.join(m2m, "T1.nii.gz"), order=0, method="nearest")
            a = nib.load(wf)
            if a.shape[:3] != shape or not np.allclose(a.affine, aff):
                return code, "warped AAL3 not on the T1 grid", time.time() - t0
            A = np.asarray(a.dataobj).astype(np.int16)
        sel = np.isin(A, list(BRAINSTEM)) & brain
        idx = np.argwhere(sel); vals = A[sel]
    else:                                   # h088: MNI2Conform_nonl missing (warp_fallback.py, validated in prep_tier2)
        from warp_fallback import pull_labels
        warp = "warp_fallback (Conform2MNI lookup)"
        cand = np.argwhere(brain)
        v = pull_labels(m2m, AAL3, cand).astype(np.int16)
        keep = np.isin(v, list(BRAINSTEM)); idx = cand[keep]; vals = v[keep]
    counts = {BRAINSTEM[k]: int((vals == k).sum()) for k in BRAINSTEM}
    np.savez(out + ".part.npz", bs_idx=idx.astype(np.int16), bs_labels=vals.astype(np.int16), affine=aff,
             names=np.array(json.dumps(BRAINSTEM)), counts=np.array(json.dumps(counts)), warp=np.array(warp))
    os.replace(out + ".part.npz", out)
    return code, "ok " + " ".join(f"{k}={n}" for k, n in counts.items()), time.time() - t0


def main():
    a = sys.argv[1:]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 3
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    codes = sorted(cmap) if "--heads" not in a else a[a.index("--heads") + 1].split(",")
    os.makedirs(OUTD, exist_ok=True)
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(one, c, cmap[c]): c for c in codes}
        for f in as_completed(futs):
            try:
                code, msg, dt = f.result()
            except Exception as e:
                code, msg, dt = futs[f], f"FAILED {type(e).__name__}: {e}", 0.0
            done += 1
            print(f"[{done}/{len(codes)}] {code} {msg} ({dt:.0f} s) | elapsed {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
