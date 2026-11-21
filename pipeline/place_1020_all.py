"""
Warp the 19 standard EEG 10/20 scalp positions from MNI into every cohort subject
(SimNIBS mni2subject_coords via the toMNI warp); save each to m2m/electrodes_1020.json.

RUN WITH THE SimNIBS ENV PYTHON (one process, loops all subjects):
  "<SimNIBS>\simnibs_env\python.exe" place_1020_all.py <cohort.csv> <charm_outroot>
"""
import os
import sys
import csv
import json

import numpy as np
import simnibs

# 19 standard 10/20 positions (10-10 nomenclature as used in the SimNIBS cap).
LABELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
          "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
CAP = os.path.join(os.path.dirname(simnibs.__file__), "resources",
                   "ElectrodeCaps_MNI", "EEG10-10_UI_Jurak_2007.csv")


def read_cap(labels):
    coords = {}
    with open(CAP) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 5 and p[0].endswith("Electrode") and p[4] in labels:
                coords[p[4]] = [float(p[1]), float(p[2]), float(p[3])]
    missing = set(labels) - set(coords)
    if missing:
        raise SystemExit(f"cap missing: {sorted(missing)}")
    return coords


def resolve_m2m(outroot, caseid):
    if caseid.startswith("SUBJ01"):
        return os.path.join(outroot, "m2m_SUBJ01asc")
    return os.path.join(outroot, "m2m_" + caseid)


def main(cohort_csv, outroot):
    cap = read_cap(LABELS)
    mni = np.array([cap[l] for l in LABELS])
    rows = list(csv.DictReader(open(cohort_csv, encoding="utf-8-sig")))
    done = skipped = 0
    for r in rows:
        cid = r["CaseID"]
        m2m = resolve_m2m(outroot, cid)
        if not os.path.exists(os.path.join(m2m, "toMNI", "MNI2Conform_nonl.nii.gz")):
            print("  skip (no warp):", cid); skipped += 1; continue
        sub = simnibs.mni2subject_coords(mni, m2m)
        out = {l: [float(v) for v in c] for l, c in zip(LABELS, sub)}
        json.dump(out, open(os.path.join(m2m, "electrodes_1020.json"), "w"), indent=2)
        done += 1
        print(f"  {cid}: electrodes_1020.json saved")
    print(f"done: {done} subjects, {skipped} skipped")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
