# -*- coding: utf-8 -*-
"""Desktop preparation of the tier-2 (psychiatric) and ref6 targets per head (18 Oct 2026; the PI
approved the tier-2 list and the AAL3 download). Runs under the SimNIBS interpreter.

Per head, with the desktop CHARM folder (which holds the MNI warps):
  <code>__t2labels.npz  AAL3v1 (1 mm; Rolls et al. 2020; AAL3v1_for_SPM12.zip from oxcns.org, sha256 45a89d28...)
                        warped into the head (mni2subject, non-linear, nearest neighbour). Holds the AAL3
                        value at each grey-matter voxel of the scoring grid, plus the tier-2 spec
                        {target: [AAL3 values]}.
  <code>__centres.json  sphere targets in subject world coordinates. ref6 (the six Paper A targets,
                        from the head's roi_coords.json, radius 10 mm) is a regression check. Tier 2 adds
                        the three Fox et al. 2012 left-DLPFC coordinates, warped with mni2subject_coords
                        (radius 10 mm).
Output: L:/BrainProbe/SPICEhead/nchc_fullsearch/t2/, keyed by h-code. The code map is read only.

    "<SimNIBS>/simnibs_env/python.exe" pipeline/nchc/prep_tier2.py [--workers 2] [--heads h001,...]
"""
import csv
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
OUTD = os.path.join(L, "t2")
CHARM = r"L:\BrainProbe\SPICEhead\charm"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only
AAL3 = r"C:\Users\ALBERT YANG\atlases\AAL3\AAL3\AAL3v1_1mm.nii"
SPEC = {"sgACC_L": [151], "sgACC_R": [152], "pgACC_L": [153], "pgACC_R": [154],
        "NAcc_L": [157], "NAcc_R": [158], "MDthal_L": [135, 137], "MDthal_R": [136, 138],
        "Pulvinar_L": [143, 145, 147, 149], "Pulvinar_R": [144, 146, 148, 150],
        "VTA_L": [159], "VTA_R": [160]}
FOX = {"DLPFC_L_Fox2012_optimized": [-38.0, 44.0, 26.0],          # Fox et al. 2012, doi:10.1016/j.biopsych.2012.04.028
       "DLPFC_L_Fox2012_sgACC_anticorr": [-44.0, 38.0, 34.0],
       "DLPFC_L_5cm_site_Fox2012": [-41.0, 16.0, 54.0]}
REF6 = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]


def m2m_of(cid):
    return os.path.join(CHARM, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)


def one(code, cid):
    import nibabel as nib
    from simnibs.utils.transformations import warp_volume, mni2subject_coords
    fl, fc = os.path.join(OUTD, f"{code}__t2labels.npz"), os.path.join(OUTD, f"{code}__centres.json")
    if os.path.exists(fl) and os.path.exists(fc):
        return code, "cached", 0.0
    t0 = time.time()
    m2m = m2m_of(cid)
    g = np.load(os.path.join(L, "heads", code, f"{code}__grid.npz"), allow_pickle=True)
    gi = g["gm_idx"].astype(np.int64)
    complete = os.path.exists(os.path.join(m2m, "toMNI", "MNI2Conform_nonl.nii.gz"))
    if complete:
        warp = "simnibs warp_volume / mni2subject_coords"
        with tempfile.TemporaryDirectory(dir=r"C:\spicehead_v22_logs") as td:
            wf = os.path.join(td, "aal3_subj.nii.gz")
            warp_volume(AAL3, m2m, wf, transformation_direction="mni2subject", transformation_type="nonl",
                        reference=os.path.join(m2m, "T1.nii.gz"), order=0, method="nearest")
            a = nib.load(wf)
            t1 = nib.load(os.path.join(m2m, "T1.nii.gz"))
            if a.shape[:3] != t1.shape[:3] or not np.allclose(a.affine, t1.affine):
                return code, "warped AAL3 not on the T1 grid", time.time() - t0
            lab = np.asarray(a.dataobj).astype(np.int16)[gi[:, 0], gi[:, 1], gi[:, 2]]
    else:                                   # h088: MNI2Conform_nonl missing (warp_fallback.py, validated)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from warp_fallback import pull_labels, points_to_subject
        warp = "warp_fallback (Conform2MNI lookup; points within 0.4 mm of mni2subject_coords)"
        lab = pull_labels(m2m, AAL3, gi).astype(np.int16)
    np.savez(fl + ".part.npz", labels=lab, spec=np.array(json.dumps(SPEC)), warp=np.array(warp))
    os.replace(fl + ".part.npz", fl)
    rc = json.load(open(os.path.join(m2m, "roi_coords.json")))
    cen = {f"ref6:{r}": {"centres": rc[r], "radius_mm": 10.0, "tier": "ref6"} for r in REF6}
    sub = (mni2subject_coords(np.array(list(FOX.values())), m2m) if complete
           else points_to_subject(m2m, np.array(list(FOX.values()))))
    for (name, _), xyz in zip(FOX.items(), sub):
        cen[name] = {"centres": [[float(v) for v in xyz]], "radius_mm": 10.0, "tier": "tier2"}
    json.dump(cen, open(fc + ".part", "w"), indent=1)
    os.replace(fc + ".part", fc)
    nvox = {k: int(np.isin(lab, v).sum()) for k, v in SPEC.items()}
    return code, "ok " + " ".join(f"{k}={n}" for k, n in nvox.items()), time.time() - t0


def main():
    a = sys.argv[1:]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 2
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
