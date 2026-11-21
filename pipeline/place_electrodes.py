"""
Compute subject-space EEG-10-10 electrode positions for a CHARM m2m folder, using
SimNIBS's MNI -> subject nonlinear warp (the toMNI/ data that survived segmentation).

RUN WITH THE SimNIBS ENV PYTHON (bypass the buggy simnibs_python.cmd on spaced paths):
  "<SimNIBS>\simnibs_env\python.exe" place_electrodes.py <m2m_dir> <out.json>

Writes {label: [x, y, z]} in subject (conform) world mm — the same world space as
m2m/label_prep/tissue_labeling_upsampled.nii.gz, so SPICEhead can map them to voxels.
"""
import os
import sys
import json

import numpy as np
import simnibs

# Motor montage (FC3-C1 / C3-CP3) + thalamic montage (F7-PO7 / F8-PO8).
LABELS = ["FC3", "C1", "C3", "CP3", "F7", "PO7", "F8", "PO8"]
CAP = os.path.join(os.path.dirname(simnibs.__file__), "resources",
                   "ElectrodeCaps_MNI", "EEG10-10_UI_Jurak_2007.csv")


def read_cap_mni(labels):
    coords = {}
    with open(CAP) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 5 and p[0] == "Electrode" and p[4] in labels:
                coords[p[4]] = [float(p[1]), float(p[2]), float(p[3])]
    missing = set(labels) - set(coords)
    if missing:
        raise SystemExit(f"electrodes missing from cap {CAP}: {sorted(missing)}")
    return coords


def main(m2m, out_json):
    cap = read_cap_mni(LABELS)
    mni = np.array([cap[l] for l in LABELS])
    sub = simnibs.mni2subject_coords(mni, m2m)          # -> subject conform mm
    out = {l: [float(v) for v in c] for l, c in zip(LABELS, sub)}
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", out_json)
    for l in LABELS:
        print(f"  {l:>4}: {np.round(out[l], 1)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
