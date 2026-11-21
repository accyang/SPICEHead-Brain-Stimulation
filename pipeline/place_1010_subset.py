"""
Warp an expanded 10-10 electrode set (10/20 + key intermediate positions, incl. PO7/PO8) into a
decade-spread 14-subject subset; save m2m/electrodes_1010.json. Run with SimNIBS env python.
  "<SimNIBS>\simnibs_env\python.exe" place_1010_subset.py <cohort.csv> <charm_outroot>
"""
import os
import sys
import csv
import json

import numpy as np
import simnibs

LABELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
          "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
          "AF7", "AF8", "FC3", "FC4", "FT7", "FT8", "CP3", "CP4", "TP7", "TP8", "PO7", "PO8", "POz"]

# 2 per decade, ASCII names, avoiding the 3 unsegmented HC_Chicago
SUBSET = ["SUBJ01", "SUBJ02", "SUBJ03", "SUBJ04", "SUBJ05", "SUBJ06",
          "SUBJ07", "SUBJ08", "SUBJ09", "SUBJ10", "SUBJ11", "SUBJ12",
          "SUBJ13", "SUBJ14"]

CAP = os.path.join(os.path.dirname(simnibs.__file__), "resources", "ElectrodeCaps_MNI", "EEG10-10_UI_Jurak_2007.csv")


def read_cap(labels):
    d = {}
    for line in open(CAP):
        p = line.strip().split(",")
        if len(p) >= 5 and p[0].endswith("Electrode") and p[4] in labels:
            d[p[4]] = [float(p[1]), float(p[2]), float(p[3])]
    miss = set(labels) - set(d)
    if miss:
        raise SystemExit("cap missing: " + str(sorted(miss)))
    return d


def resolve(outroot, cid):
    return os.path.join(outroot, "m2m_SUBJ15asc") if cid.startswith("SUBJ15") else os.path.join(outroot, "m2m_" + cid)


def main(cohort_csv, outroot):
    cap = read_cap(LABELS)
    mni = np.array([cap[l] for l in LABELS])
    done = 0
    rows = list(csv.DictReader(open(cohort_csv, encoding='utf-8-sig')))
    for cid in [r['CaseID'] for r in rows]:
        m2m = resolve(outroot, cid)
        if not os.path.exists(os.path.join(m2m, "toMNI", "MNI2Conform_nonl.nii.gz")):
            print("  skip (no warp):", cid); continue
        sub = simnibs.mni2subject_coords(mni, m2m)
        json.dump({l: [float(v) for v in c] for l, c in zip(LABELS, sub)},
                  open(os.path.join(m2m, "electrodes_1010.json"), "w"))
        done += 1
        print("  ", cid, "ok")
    print(f"done: {done}/{len(rows)} subjects, {len(LABELS)} electrodes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
