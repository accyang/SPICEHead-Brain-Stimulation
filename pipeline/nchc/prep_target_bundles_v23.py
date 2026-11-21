# -*- coding: utf-8 -*-
"""v23 bundles for the region-target atlas (24 Oct 2026, after the PI approved v23 and asked to port the pipelines).

A v23 bundle is the head's v22 bundle (L:/.../nchc_fullsearch/bundles/<code>.npz: FEM basis, scoring grid,
electrodes, AAL90 labels) with the two SPICEhead bases replaced by the v23 bases of the same head, under the same
keys (basis_sp2 / basis_sp1, labs_sp2 / labs_sp1), so score_targets_v2.py scores it unchanged and the
collector reads it with SPICEHEAD_MODEL=v23. The v23 bases are those of the 300-head homogenized run
(L:/.../nchc_fullsearch/homog300/<code>__sp{2,1}hom_basis.npz, verified by SHA-256 on download), whose
construction equals the packaged v23 model (results_v23/v23_package_check: network bit-identical, scores identical).
The bundle records model="v23" and the source of each basis.

Output: L:/BrainProbe/SPICEhead/nchc_fullsearch/bundles_v23/<code>.npz (h-codes only; subject-derived data:
the PI uploads them with upload_bundles_v23.sh).
    python pipeline/nchc/prep_target_bundles_v23.py [--heads h001,h002] [--workers 3]
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
SRC, HOM, OUTD = os.path.join(L, "bundles"), os.path.join(L, "homog300"), os.path.join(L, "bundles_v23")


def one(code):
    t0 = time.time()
    out = os.path.join(OUTD, f"{code}.npz")
    if os.path.exists(out):
        return code, "exists", 0.0
    with np.load(os.path.join(SRC, f"{code}.npz")) as z:
        d = {k: z[k] for k in z.files}
    for m in ("sp2", "sp1"):
        with np.load(os.path.join(HOM, f"{code}__{m}hom_basis.npz")) as h:
            b, labs = np.ascontiguousarray(h["basis"], dtype=np.float32), list(h["labs"])
        if labs != list(d[f"labs_{m}"]) or b.shape != d[f"basis_{m}"].shape:
            return code, f"{m}: electrode order or shape differs from the v22 bundle", time.time() - t0
        d[f"basis_{m}"], d[f"labs_{m}"] = b, np.array(labs)
    d.pop("basis_sp5", None); d.pop("labs_sp5", None)
    d["model"] = np.array("v23")
    d["sources"] = np.array(json.dumps({"fem": "v22 bundle (model-independent)", "grid": "v22 bundle",
                                        "sp2": f"homog300/{code}__sp2hom_basis.npz", "sp1": f"homog300/{code}__sp1hom_basis.npz"}))
    tmp = out + ".part.npz"
    np.savez(tmp, **d)
    os.replace(tmp, out)
    return code, "ok", time.time() - t0


def main():
    a = sys.argv[1:]
    os.makedirs(OUTD, exist_ok=True)
    codes = a[a.index("--heads") + 1].split(",") if "--heads" in a else sorted(f[:-4] for f in os.listdir(SRC) if f.startswith("h") and f.endswith(".npz"))
    nw = int(a[a.index("--workers") + 1]) if "--workers" in a else 3
    for f in ("aal90_targets.json",):
        if os.path.exists(os.path.join(SRC, f)) and not os.path.exists(os.path.join(OUTD, f)):
            open(os.path.join(OUTD, f), "wb").write(open(os.path.join(SRC, f), "rb").read())
    bad = 0
    with ProcessPoolExecutor(nw) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, c) for c in codes]), 1):
            c, msg, dt = fut.result()
            bad += msg not in ("ok", "exists")
            if msg not in ("ok", "exists") or i % 25 == 0 or i == len(codes):
                print(f"[{i}/{len(codes)}] {c}: {msg} ({dt:.0f} s)", flush=True)
    print(f"done: {len(codes) - bad} bundles in {OUTD}, {bad} problems")


if __name__ == "__main__":
    main()
