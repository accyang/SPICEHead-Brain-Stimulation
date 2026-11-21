# -*- coding: utf-8 -*-
"""Per-head target definitions for the atlas extension of the v23 uncertainty analysis (24 Oct 2026; PI: "the
uncertainty run should also generalize to all brain region targets").

For each head of results_v23/v23_uq/uq_heads.txt, collects from the verified atlas inputs on L: exactly what
score_targets_v2.build_targets needs, and nothing else:
  bundles/<code>.npz:     gm_idx, affine, shape (the scoring grid) and aal90 (AAL label per GM voxel)
  t2/<code>__t2labels.npz and t2/<code>__centres.json: tier-2 labels and spheres, ref6 spheres
  bundles/aal90_targets.json: AAL region names
The bases in the bundles are not copied. Output: L:/BrainProbe/SPICEhead/nchc_fullsearch/uq_v23/targets/<code>__uqtargets.npz
Version 2 (--v2, 24 Oct 2026, PI: "approve AAL3 and add cerebellum/brainstem"): <code>__uqtargets2.npz adds the
brainstem nucleus points (t2/<code>__bslabels.npz from prep_brainstem.py) and the AAL3 cerebellar and brainstem
target specs; SHA256SUMS2.txt.
(subject-derived: the PI uploads them with upload_uq_targets.sh), plus SHA256SUMS.txt.
Also writes results_v23/v23_uq/uq_atlas_fixed_montages.json: the FEM atlas cohort choices per target (aggregate).
    python pipeline/nchc/prep_uq_targets.py
"""
import csv
import hashlib
import json
import os

import numpy as np

AAL3_TXT = r"C:\Users\ALBERT YANG\atlases\AAL3\AAL3\AAL3v1_1mm.nii.txt"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
OUTD = os.path.join(L, "uq_v23", "targets")
UQ = os.path.join(ROOT, "results_v23", "v23_uq")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import sys
    v2 = "--v2" in sys.argv
    # --atlas300 (26 Oct 2026; PI: "score v23 on the 109 targets on nano4"): the same version-1 content for all 300 heads,
    # as L:/.../atlas_v23/targets/<code>__atlastargets.npz + SHA256SUMS.txt, for atlas_v23_score.py; nothing else is written
    atlas300 = "--atlas300" in sys.argv
    # --ext (31 Oct 2026; PI: the 35-head UQ is "not acceptable", extend to all 300): version-2 files for the 265 heads
    # not in uq_heads.txt, listed in results_v23/v23_uq/uq_heads_ext.txt, checksums in SHA256SUMS2_ext.txt; the 35
    # existing files and the fixed-montage JSON are left untouched
    ext = "--ext" in sys.argv
    if ext:
        v2 = True
    outd = os.path.join(L, "atlas_v23", "targets") if atlas300 else OUTD
    os.makedirs(outd, exist_ok=True)
    uq35 = [l.strip() for l in open(os.path.join(UQ, "uq_heads.txt")) if l.strip()]
    all300 = sorted(f.split("__")[0] for f in os.listdir(os.path.join(L, "atlas_ext", "targets")) if f.endswith("__exttargets.npz"))
    heads = all300 if atlas300 else [c for c in all300 if c not in uq35] if ext else uq35
    if ext:
        assert len(all300) == 300 and len(heads) == 265, (len(all300), len(heads))
        open(os.path.join(UQ, "uq_heads_ext.txt"), "w", newline="\n").write("\n".join(heads) + "\n")
    lut = {int(p[0]): p[1] for p in (l.split() for l in open(AAL3_TXT, encoding="utf-8", errors="replace")) if len(p) >= 2 and p[0].isdigit()}
    cereb = {n: [v] for v, n in sorted(lut.items()) if 95 <= v <= 120}          # 26 cerebellar AAL3 regions (incl. vermis)
    bstem = {n: [v] for v, n in sorted(lut.items()) if 159 <= v <= 170}         # 12 brainstem nuclei
    assert len(cereb) == 26 and len(bstem) == 12, (len(cereb), len(bstem))
    names = [d["name"] for d in json.load(open(os.path.join(L, "bundles", "aal90_targets.json")))]
    lines = []
    for code in heads:
        out = os.path.join(outd, f"{code}__atlastargets.npz" if atlas300 else f"{code}__uqtargets2.npz" if v2 else f"{code}__uqtargets.npz")
        with np.load(os.path.join(L, "bundles", f"{code}.npz")) as z:
            gm_idx, aff, shape, aal = z["gm_idx"], z["affine"], z["shape"], z["aal90"]
        with np.load(os.path.join(L, "t2", f"{code}__t2labels.npz")) as e:
            t2_labels, t2_spec = e["labels"], str(e["spec"])
        centres = open(os.path.join(L, "t2", f"{code}__centres.json"), encoding="utf-8").read()
        assert len(aal) == len(gm_idx) == len(t2_labels), code
        extra = {}
        if v2:
            with np.load(os.path.join(L, "t2", f"{code}__bslabels.npz")) as b2:
                assert np.allclose(b2["affine"], aff), code
                extra = dict(bs_idx=b2["bs_idx"], bs_labels=b2["bs_labels"], cereb_spec=np.array(json.dumps(cereb)),
                             brainstem_spec=np.array(json.dumps(bstem)))
        np.savez_compressed(out, gm_idx=gm_idx, affine=aff, shape=shape, aal90=aal, t2_labels=t2_labels,
                            t2_spec=np.array(t2_spec), centres=np.array(centres), aal_names=np.array(json.dumps(names)),
                            gm_idx_sha256=np.array(hashlib.sha256(np.ascontiguousarray(gm_idx.astype(np.int16)).tobytes()).hexdigest()), **extra)
        lines.append(f"{sha(out)}  {os.path.basename(out)}")
    open(os.path.join(outd, "SHA256SUMS2_ext.txt" if ext else "SHA256SUMS2.txt" if v2 else "SHA256SUMS.txt"), "w", newline="\n").write("\n".join(lines) + "\n")
    if ext:
        print(f"{len(lines)} extension target files in {outd}; heads listed in results_v23/v23_uq/uq_heads_ext.txt")
        return
    if atlas300:
        print(f"{len(lines)} target files in {outd} ({sum(os.path.getsize(os.path.join(outd, f)) for f in os.listdir(outd)) / 1e6:.0f} MB)")
        return
    # FEM atlas cohort choices (constrained rule + band + accessibility, and the unconstrained winner)
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "results_v22", "v22_target_atlas_300", "target_summary.csv"), encoding="utf-8"))
            if r["model"] == "fem"]
    fixed = {r["target"]: dict(cohort_choice=r["cohort_choice"], cohort_unconstrained=r["cohort_unconstrained_winner"], tier=r["tier"])
             for r in rows if r.get("cohort_choice")}
    json.dump(dict(source="results_v22/v22_target_atlas_300/target_summary.csv, model fem (300 heads)", targets=fixed),
              open(os.path.join(UQ, "uq_atlas_fixed_montages.json"), "w"), indent=1)
    print(f"{len(lines)} target files in {OUTD} ({sum(os.path.getsize(os.path.join(OUTD, f)) for f in os.listdir(OUTD)) / 1e6:.0f} MB); "
          f"FEM cohort choices for {len(fixed)} targets")


if __name__ == "__main__":
    main()
