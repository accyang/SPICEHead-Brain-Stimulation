# -*- coding: utf-8 -*-
"""Per-head target files for the cerebellum and brainstem extension of the 300-head atlas (24 Oct 2026).

For every head: the scoring grid of the 300-head run (heads/<code>/<code>__grid.npz: gm_idx; T1 affine and shape),
the AAL3 label on every grey-matter voxel (t2/<code>__t2labels.npz), the brainstem nucleus points
(t2/<code>__bslabels.npz, prep_brainstem.py), the ref6 spheres (t2/<code>__centres.json, regression check against
the published scores) and the AAL3 specs: 26 cerebellar regions (values 95-120) and 12 brainstem nuclei (159-170).
Output: L:/BrainProbe/SPICEhead/nchc_fullsearch/atlas_ext/targets/<code>__exttargets.npz and SHA256SUMS.txt
(subject-derived, h-codes only: the PI uploads them with upload_atlas_ext.sh).
    python pipeline/nchc/prep_ext_targets.py [--heads h001,...]
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
OUTD = os.path.join(L, "atlas_ext", "targets")
AAL3_TXT = r"C:\Users\ALBERT YANG\atlases\AAL3\AAL3\AAL3v1_1mm.nii.txt"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only (head list)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    a = sys.argv[1:]
    codes = a[a.index("--heads") + 1].split(",") if "--heads" in a else sorted(r["code"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig")))
    os.makedirs(OUTD, exist_ok=True)
    lut = {int(p[0]): p[1] for p in (l.split() for l in open(AAL3_TXT, encoding="utf-8", errors="replace")) if len(p) >= 2 and p[0].isdigit()}
    cereb = {n: [v] for v, n in sorted(lut.items()) if 95 <= v <= 120}
    bstem = {n: [v] for v, n in sorted(lut.items()) if 159 <= v <= 170}
    assert len(cereb) == 26 and len(bstem) == 12
    lines, problems = [], []
    for code in codes:
        out = os.path.join(OUTD, f"{code}__exttargets.npz")
        try:
            with np.load(os.path.join(L, "heads", code, f"{code}__grid.npz"), allow_pickle=True) as g:
                gm_idx = g["gm_idx"]
            with np.load(os.path.join(L, "bundles", f"{code}.npz")) as z:                     # affine and shape of the grid
                aff, shape, gm_b = z["affine"], z["shape"], z["gm_idx"]
            assert np.array_equal(gm_idx.astype(np.int64), gm_b.astype(np.int64)), "grid cache and bundle differ"
            with np.load(os.path.join(L, "t2", f"{code}__t2labels.npz")) as e:
                t2 = e["labels"]
            with np.load(os.path.join(L, "t2", f"{code}__bslabels.npz")) as b:
                assert np.allclose(b["affine"], aff), "brainstem points on another grid"
                bs_idx, bs_lab = b["bs_idx"], b["bs_labels"]
            assert len(t2) == len(gm_idx)
            cen = json.load(open(os.path.join(L, "t2", f"{code}__centres.json"), encoding="utf-8"))
            ref6 = {k: v for k, v in cen.items() if k.startswith("ref6:")}
            np.savez_compressed(out, gm_idx=gm_idx.astype(np.int16), affine=aff, shape=shape, t2_labels=t2, bs_idx=bs_idx, bs_labels=bs_lab,
                                centres=np.array(json.dumps(ref6)), cereb_spec=np.array(json.dumps(cereb)), brainstem_spec=np.array(json.dumps(bstem)))
            lines.append(f"{sha(out)}  {os.path.basename(out)}")
        except Exception as ex:
            problems.append(f"{code}: {type(ex).__name__}: {ex}")
    open(os.path.join(OUTD, "SHA256SUMS.txt"), "w", newline="\n").write("\n".join(lines) + "\n")
    print(f"{len(lines)} target files in {OUTD}; problems: {problems or 'none'}")


if __name__ == "__main__":
    main()
