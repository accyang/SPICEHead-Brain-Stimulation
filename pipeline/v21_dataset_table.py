# -*- coding: utf-8 -*-
"""Dataset table for reviewer comment G.

The review asks for sample size, source/site, scanner and T1 sequence, age and sex distribution,
inclusion criteria, purpose, ethics approval and discovery/validation status.

This script supplies everything derivable from the data itself: n, site, age, sex, voxel geometry
and any TR/TE the NIfTI headers preserved. Scanner make/model, sequence name, inclusion criteria
and ethics approvals are NOT in the imaging data and are marked as needing the author.

Note the cohort is n = 300 only -- the earlier 97-case cohort is superseded and deliberately absent.

Usage:  python pipeline/v21_dataset_table.py
Output: results/v21_dataset/{dataset_table.csv, dataset_table.md}
"""
import collections
import csv
import os
import re
import statistics as st
import sys

import nibabel as nib

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
OUT = os.path.join(ROOT, "results", "v21_dataset")
CSVP = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"
CH = r"L:\BrainProbe\SPICEhead\charm"


def header_facts(cid):
    """TR/TE and geometry, where the conversion preserved them."""
    m2m = os.path.join(CH, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)
    p = os.path.join(m2m, "T1.nii.gz")
    if not os.path.exists(p):
        return None
    h = nib.load(p).header
    txt = " ".join(bytes(h[k]).decode("ascii", "ignore").strip("\x00")
                   for k in ("db_name", "descrip", "aux_file"))
    tr = re.search(r"TR[:=]\s*([\d.]+)", txt)
    te = re.search(r"TE[:=]\s*([\d.]+)", txt)
    z = h.get_zooms()[:3]
    return {"tr": float(tr.group(1)) if tr else None,
            "te": float(te.group(1)) if te else None,
            "vox": "x".join(f"{v:g}" for v in z),
            "dim": "x".join(str(d) for d in h.get_data_shape()[:3])}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(CSVP, encoding="utf-8-sig")))
    print(f"cohort: {len(rows)} subjects")

    bysite = collections.defaultdict(list)
    for r in rows:
        bysite[r["Site"]].append(r)

    out = []
    for site in sorted(bysite):
        rs = bysite[site]
        ages = [float(r["Age_Scan"]) for r in rs]
        sx = collections.Counter(r["Sex_1M0F"] for r in rs)
        # probe headers for a handful per site -- they are identical within a site in practice
        facts, probed = [], 0
        for r in rs:
            f = header_facts(r["CaseID"])
            if f:
                facts.append(f)
                probed += 1
            if probed >= 8:
                break
        trs = sorted({f["tr"] for f in facts if f["tr"]})
        tes = sorted({f["te"] for f in facts if f["te"]})
        vox = sorted({f["vox"] for f in facts})
        dim = sorted({f["dim"] for f in facts})
        out.append({
            "site": site, "n": len(rs),
            "age_mean": round(st.mean(ages), 1), "age_sd": round(st.pstdev(ages), 1),
            "age_min": int(min(ages)), "age_max": int(max(ages)),
            "male": sx.get("1", 0), "female": sx.get("0", 0),
            "TR_ms": "/".join(f"{t:g}" for t in trs) if trs else "not in header",
            "TE_ms": "/".join(f"{t:g}" for t in tes) if tes else "not in header",
            "voxel_mm": "/".join(vox) if vox else "?",
            "matrix": "/".join(dim) if dim else "?",
            "n_probed": probed,
        })
    ages = [float(r["Age_Scan"]) for r in rows]
    sx = collections.Counter(r["Sex_1M0F"] for r in rows)
    total = {"site": "ALL", "n": len(rows), "age_mean": round(st.mean(ages), 1),
             "age_sd": round(st.pstdev(ages), 1), "age_min": int(min(ages)),
             "age_max": int(max(ages)), "male": sx.get("1", 0), "female": sx.get("0", 0),
             "TR_ms": "", "TE_ms": "", "voxel_mm": "", "matrix": "", "n_probed": ""}

    with open(os.path.join(OUT, "dataset_table.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out + [total])

    L = ["# Dataset table (reviewer comment G)\n",
         f"Single cohort, n = {len(rows)}. The earlier 97-case cohort is superseded and is not "
         "reported anywhere in the manuscript.\n",
         "| Site | n | Age mean ± SD | Age range | M / F | TR (ms) | TE (ms) | Voxel (mm) | Matrix |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        L.append(f"| {r['site']} | {r['n']} | {r['age_mean']} ± {r['age_sd']} | "
                 f"{r['age_min']}–{r['age_max']} | {r['male']} / {r['female']} | "
                 f"{r['TR_ms']} | {r['TE_ms']} | {r['voxel_mm']} | {r['matrix']} |")
    L.append(f"| **All** | **{total['n']}** | **{total['age_mean']} ± {total['age_sd']}** | "
             f"**{total['age_min']}–{total['age_max']}** | **{total['male']} / {total['female']}** "
             "| | | | |")
    L += ["", "## Still required from the author (not recoverable from the imaging data)", "",
          "- Scanner manufacturer, model and field strength, per site",
          "- Sequence name (MPRAGE / MP2RAGE / SPGR ...) and remaining parameters "
          "(flip angle, TI, bandwidth)",
          "- Inclusion and exclusion criteria",
          "- Ethics approval reference per site, and consent procedure",
          "- Whether any site served a distinct role (all four currently pooled for every analysis)",
          "",
          "TR/TE above were recovered from the NIfTI `db_name` field, which the DICOM→NIfTI "
          "conversion truncates to 18 characters; TE is therefore often cut off mid-value and any "
          "value shown should be confirmed against the acquisition protocol before publication."]
    open(os.path.join(OUT, "dataset_table.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    print(f"\n{'site':<12}{'n':>5}{'age':>16}{'M/F':>9}{'TR':>14}{'TE':>14}{'voxel':>10}")
    for r in out:
        age = "%s+-%s" % (r["age_mean"], r["age_sd"])
        mf = "%s/%s" % (r["male"], r["female"])
        print(f"{r['site']:<12}{r['n']:>5}{age:>16}{mf:>9}"
              f"{r['TR_ms']:>14}{r['TE_ms']:>14}{r['voxel_mm']:>10}")
    print(f"\nwrote {OUT}/dataset_table.md")


if __name__ == "__main__":
    main()
