# -*- coding: utf-8 -*-
"""Head lists for the Paper A v23 recompute on Forerunner 1 (26 Oct 2026; PI: "Use forerunner").

Writes to C:/spicehead_v22_logs/f1_stage/ (lab-local), h-codes only, no case IDs and no demographics:
  heads300.csv    CaseID = h-code for the 300 heads, in cohort order; Site = NA (the cluster jobs read only CaseID;
                  the 2 mm check reads Site only when its sample is not pinned, and Paper A pins it)
  t7_pinned.csv   the 20 pinned 2 mm heads of Paper A (verification_8subjects/cohort_table7.csv) as h-codes, same order
  code_sha.txt    SHA-256 of the staged head tarballs (copy of nchc_stage/code_sha.txt)
    python pipeline/nchc/prep_f1_stage.py
"""
import csv
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
STAGE = r"C:\spicehead_v22_logs\nchc_stage"
OUT = r"C:\spicehead_v22_logs\f1_stage"
COHORT = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"                        # private; read only
T7 = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "verification_8subjects", "cohort_table7.csv")


def main():
    os.makedirs(OUT, exist_ok=True)
    code = {r["CaseID"]: r["code"] for r in csv.DictReader(open(os.path.join(STAGE, "head_codes.csv"), encoding="utf-8-sig"))}
    cohort = [r["CaseID"] for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))]
    t7 = [r["CaseID"] for r in csv.DictReader(open(T7, encoding="utf-8-sig"))]
    missing = [c for c in cohort + t7 if c not in code]
    assert not missing, f"{len(missing)} case IDs without an h-code"
    with open(os.path.join(OUT, "heads300.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["CaseID", "Site"]); w.writerows([code[c], "NA"] for c in cohort)
    with open(os.path.join(OUT, "t7_pinned.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["CaseID"]); w.writerows([code[c]] for c in t7)
    shutil.copyfile(os.path.join(STAGE, "code_sha.txt"), os.path.join(OUT, "code_sha.txt"))
    print(f"{len(cohort)} heads, {len(t7)} pinned 2 mm heads, {sum(1 for _ in open(os.path.join(OUT, 'code_sha.txt')))} tarball checksums -> {OUT}")


if __name__ == "__main__":
    main()
