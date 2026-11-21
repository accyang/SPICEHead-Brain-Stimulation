# -*- coding: utf-8 -*-
"""Desktop preparation for the region-target scoring on Nano 4 (tier 1 = AAL90; tier 2 added later).

Written 18 Oct 2026 after the PI chose "tier 1 and 2", with all three models scored. Runs under the
SimNIBS interpreter, which provides warp_volume. For every head of the 300-head set:
  1. Warp the SPM12 AAL atlas (the GUI's file) into the head's T1 grid (mni2subject, non-linear,
     nearest neighbour) with the desktop CHARM folder, which holds the warps not staged to the
     cluster. Read the label at each grey-matter voxel of the scoring grid, in the grid npz order.
     Labels are 1..90 in AAL.xml order (the 90 cerebral regions BrainProbe scores); cerebellum,
     vermis and unlabelled voxels are 0.
  2. Write one bundle per head:
     - the three 18-field bases (sp2, sp1, fem) as float32. The GPU scorer casts to float32 on load,
       so results are bit-identical to scoring the float64 files;
     - gm_idx, gm_world, grid affine/shape, electrodes_1020 and the AAL90 labels.
Output: L:/BrainProbe/SPICEhead/nchc_fullsearch/bundles/<code>.npz (~215 MB each), keyed by h-code
only. The PI uploads the bundles (subject-derived data). The code map is read here and never written.

    "<SimNIBS>/simnibs_env/python.exe" pipeline/nchc/prep_target_bundles.py [--workers 3] [--heads h001,h002]
"""
import csv
import json
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
HEADS = os.path.join(L, "heads")
OUTD = os.path.join(L, "bundles")
CHARM = r"L:\BrainProbe\SPICEhead\charm"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only
AAL_NII = r"C:\Users\ALBERT YANG\nilearn_data\aal_SPM12\aal\atlas\AAL.nii"
AAL_XML = r"C:\Users\ALBERT YANG\nilearn_data\aal_SPM12\aal\atlas\AAL.xml"


def aal90():
    labs = [(int(l.find("index").text), l.find("name").text) for l in ET.parse(AAL_XML).getroot().iter("label")]
    cer = [x for x in labs if not (x[1].startswith("Cerebelum") or x[1].startswith("Vermis"))]
    assert len(labs) == 116 and len(cer) == 90, (len(labs), len(cer))
    return cer                                     # [(atlas value, name)] in AAL.xml order


def m2m_of(cid):
    return os.path.join(CHARM, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)


def one(code, cid):
    import nibabel as nib
    from simnibs.utils.transformations import warp_volume
    out = os.path.join(OUTD, f"{code}.npz")
    if os.path.exists(out):
        return code, "cached", 0.0
    t0 = time.time()
    m2m = m2m_of(cid)
    hd = os.path.join(HEADS, code)
    if not os.path.exists(os.path.join(hd, ".verified")):
        return code, "no verified outputs on L:", 0.0
    g = np.load(os.path.join(hd, f"{code}__grid.npz"), allow_pickle=True)
    gi = g["gm_idx"].astype(np.int64)
    t1 = nib.load(os.path.join(m2m, "T1.nii.gz"))
    if os.path.exists(os.path.join(m2m, "toMNI", "MNI2Conform_nonl.nii.gz")):
        warp = "simnibs warp_volume"
        with tempfile.TemporaryDirectory(dir=r"C:\spicehead_v22_logs") as td:
            wf = os.path.join(td, "aal_subj.nii.gz")
            warp_volume(AAL_NII, m2m, wf, transformation_direction="mni2subject", transformation_type="nonl",
                        reference=os.path.join(m2m, "T1.nii.gz"), order=0, method="nearest")
            a = nib.load(wf)
            if a.shape[:3] != t1.shape[:3] or not np.allclose(a.affine, t1.affine):
                return code, "warped atlas not on the T1 grid", time.time() - t0
            atlas = np.asarray(a.dataobj).astype(np.int32)
        v = atlas[gi[:, 0], gi[:, 1], gi[:, 2]]
    else:                                   # h088: MNI2Conform_nonl missing; identical result (warp_fallback.py)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from warp_fallback import pull_labels
        warp = "warp_fallback.pull_labels (Conform2MNI lookup)"
        v = pull_labels(m2m, AAL_NII, gi)
    lut = {val: i + 1 for i, (val, _) in enumerate(aal90())}
    lab = np.zeros(len(gi), np.int16)
    for val, i in lut.items():
        lab[v == val] = i
    bases, labs = {}, {}
    for m in ("sp2", "sp1", "fem"):
        z = np.load(os.path.join(hd, f"{code}__{m}_basis.npz"), allow_pickle=True)
        bases[m] = np.ascontiguousarray(z["basis"], dtype=np.float32)
        labs[m] = np.array(list(z["labs"]))
        if bases[m].shape[2] != len(gi):
            return code, f"{m} basis has {bases[m].shape[2]} voxels, grid {len(gi)}", time.time() - t0
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    tmp = out + ".part.npz"
    np.savez(tmp, basis_sp2=bases["sp2"], basis_sp1=bases["sp1"], basis_fem=bases["fem"],
             labs_sp2=labs["sp2"], labs_sp1=labs["sp1"], labs_fem=labs["fem"],
             gm_idx=g["gm_idx"], gm_world=g["gm_world"].astype(np.float32),
             affine=t1.affine, shape=np.array(t1.shape[:3]), aal90=lab, warp=np.array(warp),
             electrodes=np.array(json.dumps(ej)))
    os.replace(tmp, out)
    return code, f"ok ({int((lab > 0).sum())} labelled GM voxels, {len(np.unique(lab[lab > 0]))} regions)", time.time() - t0


def main():
    a = sys.argv[1:]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 3
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    codes = sorted(cmap)
    if "--heads" in a:
        codes = a[a.index("--heads") + 1].split(",")
    os.makedirs(OUTD, exist_ok=True)
    json.dump([{"index": i + 1, "aal_value": v, "name": n} for i, (v, n) in enumerate(aal90())],
              open(os.path.join(OUTD, "aal90_targets.json"), "w"), indent=1)
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
