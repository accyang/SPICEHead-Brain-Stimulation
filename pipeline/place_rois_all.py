"""
Warp target-ROI centers from MNI into every cohort subject (SimNIBS mni2subject_coords via the
toMNI warp) and save each subject's ROI centers to m2m/roi_coords.json.

RUN WITH THE SimNIBS ENV PYTHON (loops all subjects in one process = one simnibs import):
  "<SimNIBS>\simnibs_env\python.exe" place_rois_all.py <cohort.csv> <charm_outroot>

ROI centers are standard MNI coordinates (mm), bilateral where the structure is bilateral.
Adjust here if you want different definitions; radius is applied later in analyze_rois.py.
"""
import os
import sys
import csv
import json

import numpy as np
import simnibs

ROIS_MNI = {
    "M1_handknob": [[-37, -21, 58], [37, -21, 58]],  # precentral hand-knob (motor montage targets L)
    "thalamus":    [[-10, -19, 8],  [10, -19, 8]],
    "insula":      [[-38, 6, 2],    [38, 6, 2]],
    "ACC":         [[0, 32, 18]],                     # anterior cingulate (midline)
    # Hippocampus kept as SEPARATE left/right ROIs rather than a bilateral union like the four
    # above: a union forces one compromise montage across both sides, which hides any lateralised
    # optimum. Centres are the AAL Hippocampus_L/R centroids, so they match the template analysis.
    "hippocampus_L": [[-25, -22, -11]],
    "hippocampus_R": [[29, -21, -12]],
}


def resolve_m2m(outroot, caseid):
    # SUBJ01 lives under the ASCII alias (non-ASCII CaseID path breaks SimNIBS/ITK)
    if caseid.startswith("SUBJ01"):
        return os.path.join(outroot, "m2m_SUBJ01asc")
    return os.path.join(outroot, "m2m_" + caseid)


def main(cohort_csv, outroot):
    names, pts = [], []
    for name, centers in ROIS_MNI.items():
        for c in centers:
            names.append(name)
            pts.append(c)
    pts = np.asarray(pts, float)

    with open(cohort_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    done = skipped = 0
    for r in rows:
        cid = r["CaseID"]
        m2m = resolve_m2m(outroot, cid)
        warp = os.path.join(m2m, "toMNI", "MNI2Conform_nonl.nii.gz")
        if not os.path.exists(warp):
            print("  skip (no warp):", cid)
            skipped += 1
            continue
        sub = simnibs.mni2subject_coords(pts, m2m)
        out = {}
        for name in ROIS_MNI:
            out[name] = [[float(v) for v in sub[i]] for i, n in enumerate(names) if n == name]
        json.dump(out, open(os.path.join(m2m, "roi_coords.json"), "w"), indent=2)
        done += 1
        print(f"  {cid}: roi_coords.json saved")
    print(f"done: {done} subjects, {skipped} skipped")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
