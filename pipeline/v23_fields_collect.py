# -*- coding: utf-8 -*-
"""Per-head field agreement with FEM under v23, in the v23 results folder (2 Nov 2026).

The 300-head comparison of 23 Oct 2026 wrote results_v22/v22_fem_fullsearch_300_homog/fields_perhead.csv with four
models: sp2 and sp1 (v22 nearest-label network) and sp2hom and sp1hom (the homogenized network, identical to the packaged
v23; results_v23/v23_package_check). The manuscript uses only sp2hom and sp1hom. This script copies those rows, unchanged,
to results_v23/v23_fem_fullsearch_300/fields_perhead.csv, so the v23 number chain reads nothing from a v22 folder. The
source stays in place: values_v23.py reads all four models for the v22-to-v23 comparison.

  python pipeline/v23_fields_collect.py
"""
import csv
import hashlib
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results_v22", "v22_fem_fullsearch_300_homog", "fields_perhead.csv")
OUT = os.path.join(ROOT, "results_v23", "v23_fem_fullsearch_300", "fields_perhead.csv")
MODELS = ("sp2hom", "sp1hom")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    keep = [r for r in rows if r["model"] in MODELS]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(keep)
    back = list(csv.DictReader(open(OUT, encoding="utf-8")))
    assert back == keep, "rows changed on writing"
    heads = {m: len({r["subject"] for r in back if r["model"] == m}) for m in MODELS}
    assert all(n == 300 for n in heads.values()), heads
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M"), script="pipeline/v23_fields_collect.py",
                   source=os.path.relpath(SRC, ROOT).replace(os.sep, "/"), source_sha256=sha(SRC),
                   models=MODELS, rows=len(back), heads_per_model=heads, output_sha256=sha(OUT),
                   note="Rows copied unchanged; sp2hom and sp1hom are v23 at 2 and 1 mm."),
              open(OUT.replace(".csv", ".PROVENANCE.json"), "w", encoding="utf-8"), indent=1)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(back)} rows, {heads}")


if __name__ == "__main__":
    main()
