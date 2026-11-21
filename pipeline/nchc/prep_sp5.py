# -*- coding: utf-8 -*-
"""Desktop preparation of the 5 mm SPICEhead model for the region-target atlas (18 Oct 2026; the PI
asked to add 5 mm after the tier-2 preparation).

Per head:
  - v22_fem_fullsearch.stage_sp_basis(dx=5.0) builds the 5 mm network from the desktop CHARM folder,
    solves the 18 electrode-vs-Cz fields (CPU direct solver, about 40 s per head) and samples them
    trilinearly on the same 1 mm T1-grid GM voxels as FEM, SP2 and SP1. It uses the head's grid file
    from L: (identical GM ordering).
  - One small bundle is written: basis_sp5 (float32) plus the grid, AAL90 labels and electrodes, taken
    from the tier-1 bundle.
Output: L:/BrainProbe/SPICEhead/nchc_fullsearch/bundles_sp5/<code>.npz (~135 MB), keyed by h-code.
Intermediate files named by study ID stay in C:/spicehead_v22_logs/sp5_cache and are deleted per head.

    python pipeline/nchc/prep_sp5.py [--workers 3] [--heads h001,...]
"""
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
OUTD = os.path.join(L, "bundles_sp5")
CACHE = r"C:\spicehead_v22_logs\sp5_cache"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only


def init():
    os.environ.update(SPICEHEAD_MODEL="v22", SPICEHEAD_FULLSEARCH_CACHE=CACHE,
                      SPICEHEAD_CHARM_ROOT=r"L:\BrainProbe\SPICEhead\charm",
                      OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for p in (ROOT, os.path.join(ROOT, "pipeline")):
        if p not in sys.path:
            sys.path.insert(0, p)


def one(code, cid):
    out = os.path.join(OUTD, f"{code}.npz")
    if os.path.exists(out):
        return code, "cached", 0.0
    t0 = time.time()
    import v22_fem_fullsearch as fs
    grid_src = os.path.join(L, "heads", code, f"{code}__grid.npz")
    grid_c = os.path.join(CACHE, f"{cid}__grid.npz")
    if not os.path.exists(grid_c):
        shutil.copy2(grid_src, grid_c)
    hl = os.path.join(CACHE, f"{cid}.csv")
    open(hl, "w", encoding="utf-8").write(f"CaseID\n{cid}\n")
    os.environ["SPICEHEAD_FS_HEADS"] = hl
    fs.stage_sp_basis(["--heads", cid], 5.0)
    bp = os.path.join(CACHE, f"{cid}__sp5_basis.npz")
    with np.load(bp, allow_pickle=True) as z:                     # closed before the file is deleted
        basis = np.ascontiguousarray(z["basis"], dtype=np.float32)
        labs = np.array(list(z["labs"]))
        meta = json.loads(str(z["meta"]))
    with np.load(os.path.join(L, "bundles", f"{code}.npz")) as b:
        keep = {k: b[k] for k in ("gm_idx", "gm_world", "affine", "shape", "aal90", "electrodes")}
    if basis.shape[2] != len(keep["gm_idx"]):
        return code, f"sp5 basis has {basis.shape[2]} voxels, grid {len(keep['gm_idx'])}", time.time() - t0
    meta.pop("subject", None)                                     # keep study IDs out of the bundle
    np.savez(out + ".part.npz", basis_sp5=basis, labs_sp5=labs, meta_sp5=np.array(json.dumps(meta)), **keep)
    os.replace(out + ".part.npz", out)
    for f in (bp, grid_c, hl):
        if os.path.exists(f):
            os.remove(f)                                          # fail loudly: study-ID files must not linger
    return code, f"ok (check {max(meta['basis_check'].values()):.1e}, {meta['n_nodes']} nodes)", time.time() - t0


def main():
    a = sys.argv[1:]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 3
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    codes = sorted(cmap) if "--heads" not in a else a[a.index("--heads") + 1].split(",")
    os.makedirs(OUTD, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(workers, initializer=init) as ex:
        futs = {ex.submit(one, c, cmap[c]): c for c in codes}
        for f in as_completed(futs):
            try:
                code, msg, dt = f.result()
            except Exception as e:
                code, msg, dt = futs[f], f"FAILED {type(e).__name__}: {e}", 0.0
            done += 1
            print(f"[{done}/{len(codes)}] {code} {msg} ({dt:.0f} s) | elapsed {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
