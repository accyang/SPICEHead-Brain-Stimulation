"""
Warp the EXTENDED electrode set (19 x 10/20 + the inferior 9/10 chain) into the decade-spread
subset; save m2m/electrodes_1020ext.json. Run with the SimNIBS env python.

The inferior chain (F9/10, FT9/10, T9/10, TP9/10, P9/10, PO9/10, I1/Iz/I2) sits ~40 mm below the
10/20 ring (T7 z=-8 vs T9 z=-48 on the MNI template) and was never placed before -
place_1010_subset.py only added AF/FC/FT/CP/TP/PO, i.e. the same scalp band.

  "<SimNIBS>\\simnibs_env\\python.exe" place_1020ext_all.py <cohort.csv> <charm_outroot>
"""
import os
import sys
import csv
import json

import numpy as np
import simnibs

TEN20 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
         "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
INFERIOR = ["F9", "F10", "FT9", "FT10", "T9", "T10", "TP9", "TP10",
            "P9", "P10", "PO9", "PO10", "I1", "Iz", "I2"]
LABELS = TEN20 + INFERIOR

# same 14 used by place_1010_subset.py: 2/decade, ASCII names, skips the 3 unsegmented HC_Chicago.
# Reused deliberately so the inferior-chain result is directly comparable to the 10-10 result.
SUBSET = ["SUBJ01", "SUBJ02", "SUBJ03", "SUBJ04", "SUBJ05", "SUBJ06",
          "SUBJ07", "SUBJ08", "SUBJ09", "SUBJ10", "SUBJ11", "SUBJ12",
          "SUBJ13", "SUBJ14"]

CAP = os.path.join(os.path.dirname(simnibs.__file__), "resources",
                   "ElectrodeCaps_MNI", "EEG10-10_UI_Jurak_2007.csv")


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
    return os.path.join(outroot, "m2m_SUBJ15asc") if cid.startswith("SUBJ15") \
        else os.path.join(outroot, "m2m_" + cid)


def main(cohort_csv, outroot):
    cap = read_cap(LABELS)
    mni = np.array([cap[l] for l in LABELS])
    done = 0
    rows = list(csv.DictReader(open(cohort_csv, encoding='utf-8-sig')))
    for cid in [r['CaseID'] for r in rows]:
        m2m = resolve(outroot, cid)
        if not os.path.exists(os.path.join(m2m, "toMNI", "MNI2Conform_nonl.nii.gz")):
            print("  skip (no warp):", cid, flush=True)
            continue
        sub = simnibs.mni2subject_coords(mni, m2m)
        json.dump({l: [float(v) for v in c] for l, c in zip(LABELS, sub)},
                  open(os.path.join(m2m, "electrodes_1020ext.json"), "w"))
        done += 1
        print(f"   {cid} ok", flush=True)
    print(f"done: {done}/{len(rows)} subjects, {len(LABELS)} electrodes", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
