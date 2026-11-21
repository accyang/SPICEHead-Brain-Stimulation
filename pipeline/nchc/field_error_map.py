# -*- coding: utf-8 -*-
"""Analysis 1 (23 Oct 2026): where do the SPICEhead fields depart from FEM? Field-level error anatomy of the
18 single-electrode basis fields in all 300 heads, from the stored bundles (no new simulation).

Per head and electrode basis (E(a, Cz), 1 mA), on the grey-matter voxels of the shared 1 mm grid:
  rel   = |E_sp - E_fem| / |E_fem|                 (vector relative error, per voxel)
  ratio = |E_sp| / |E_fem|                          (magnitude ratio)
  ang   = angle between E_sp and E_fem (deg)
for SPICEhead 1 mm and 2 mm. Each voxel carries anatomical features from the head's CHARM label volume:
  depth    distance from the voxel to the nearest scalp-surface voxel (mm; EDT of the conductor mask)
  d_elec   distance to the nearest electrode centre (mm) and to THIS basis electrode's centre
  csf5     CSF fraction within 5 mm; bone10: bone fraction within 10 mm (skull-thickness proxy)
  aal      AAL90 region (0 outside)
Summaries written per head (aggregates only): medians of rel/ratio/ang by electrode; by depth bin
(0-10, 10-20, 20-30, 30-40, >40 mm); by distance-to-electrode bin; by csf5 and bone10 quartile; by AAL
region; plus whole-GM r, CCC and bias per basis. Collected into results_v22/v22_field_error_map/.

    python pipeline/nchc/field_error_map.py [--workers 3] [--heads h001,...]
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
CHARM = r"L:\BrainProbe\SPICEhead\charm"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only
OUT = os.path.join(ROOT, "results_v22", "v22_field_error_map")
PERHEAD = os.path.join(L, "field_error_perhead")
DEPTH_BINS = [0, 10, 20, 30, 40, 1e9]
DIST_BINS = [0, 20, 40, 60, 80, 1e9]
GM, CSF, BONE = 2, 3, (4, 7, 8)


def features(m2m, gi, aff, shape):
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt, uniform_filter
    from v21_simnibs_compare import resample_to
    img = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    lab = np.asanyarray(img.dataobj)
    # labels on the 1 mm scoring grid (nearest): resample each needed mask
    cond = resample_to((lab > 0).astype(np.float32), img.affine, aff, shape) > 0.5
    csf = resample_to((lab == CSF).astype(np.float32), img.affine, aff, shape)
    bone = resample_to(np.isin(lab, BONE).astype(np.float32), img.affine, aff, shape)
    vox = np.sqrt((aff[:3, :3] ** 2).sum(0))
    depth = distance_transform_edt(cond, sampling=vox)                 # distance to the nearest non-conductor
    csf5 = uniform_filter(csf, size=np.maximum(1, np.round(10 / vox)).astype(int))
    bone10 = uniform_filter(bone, size=np.maximum(1, np.round(20 / vox)).astype(int))
    g = (gi[:, 0], gi[:, 1], gi[:, 2])
    return depth[g], csf5[g], bone10[g]


def one(code, cid):
    out = os.path.join(PERHEAD, f"{code}.json")
    if os.path.exists(out):
        return code, "cached", 0.0
    t0 = time.time()
    sys.path.insert(0, os.path.join(ROOT, "pipeline")); sys.path.insert(0, ROOT)
    m2m = os.path.join(CHARM, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)
    with np.load(os.path.join(L, "bundles", f"{code}.npz")) as z:
        gi = z["gm_idx"].astype(np.int64); aff = np.asarray(z["affine"], float); shape = tuple(int(x) for x in z["shape"])
        gw = z["gm_world"].astype(float); aal = z["aal90"]
        ej = json.loads(str(z["electrodes"])); labs = list(z["labs_fem"])
        assert list(z["labs_sp1"]) == labs and list(z["labs_sp2"]) == labs
        Bf, B1, B2 = z["basis_fem"], z["basis_sp1"], z["basis_sp2"]
    depth, csf5, bone10 = features(m2m, gi, aff, shape)
    cen = {k: np.asarray(v, float) for k, v in ej.items()}
    d_any = np.min(np.stack([np.linalg.norm(gw - c, axis=1) for c in cen.values()]), axis=0)
    res = dict(code=code, n_gm=int(len(gi)), by_electrode={}, by_depth={}, by_dist_this={}, by_dist_any={}, by_csf5={}, by_bone10={}, by_aal={})
    acc = {k: {} for k in ("by_depth", "by_dist_this", "by_dist_any", "by_csf5", "by_bone10", "by_aal")}
    qc = np.quantile(csf5, [0.25, 0.5, 0.75]); qb = np.quantile(bone10, [0.25, 0.5, 0.75])
    cls = dict(by_depth=np.digitize(depth, DEPTH_BINS[1:-1]), by_dist_any=np.digitize(d_any, DIST_BINS[1:-1]),
               by_csf5=np.digitize(csf5, qc), by_bone10=np.digitize(bone10, qb), by_aal=aal.astype(int))
    def stats(ef, es):
        nf = np.linalg.norm(ef, axis=0); ns = np.linalg.norm(es, axis=0)
        ok = nf > 1e-9 * nf.max()
        rel = np.linalg.norm(es - ef, axis=0)[ok] / nf[ok]
        ratio = ns[ok] / nf[ok]
        cosang = np.clip((ef * es).sum(0)[ok] / (nf[ok] * ns[ok] + 1e-30), -1, 1)
        return rel, ratio, np.degrees(np.arccos(cosang)), ok
    for i, lab in enumerate(labs):
        ef = Bf[i].astype(np.float64)
        row = {}
        for m, B in (("sp1", B1), ("sp2", B2)):
            es = B[i].astype(np.float64)
            rel, ratio, ang, ok = stats(ef, es)
            nf = np.linalg.norm(ef, axis=0)[ok]; ns = np.linalg.norm(es, axis=0)[ok]
            r = float(np.corrcoef(nf, ns)[0, 1])
            ccc = float(2 * np.cov(nf, ns, bias=True)[0, 1] / (nf.var() + ns.var() + (nf.mean() - ns.mean()) ** 2))
            row[m] = dict(rel_median=float(np.median(rel)), rel_p90=float(np.percentile(rel, 90)), ratio_median=float(np.median(ratio)),
                          ang_median=float(np.median(ang)), r=r, ccc=ccc, bias_pct=float(100 * (ns.mean() / nf.mean() - 1)))
            d_this = np.linalg.norm(gw - cen[lab], axis=1)[ok]
            cls_this = np.digitize(d_this, DIST_BINS[1:-1])
            for key, c in list(cls.items()) + [("by_dist_this", cls_this)]:
                cc = c[ok] if key != "by_dist_this" else c
                for b in np.unique(cc):
                    sel = cc == b
                    a = acc[key].setdefault((m, int(b)), [[], [], [], 0])
                    a[0].append(float(np.median(rel[sel]))); a[1].append(float(np.median(ratio[sel]))); a[2].append(float(np.median(ang[sel]))); a[3] += int(sel.sum())
        res["by_electrode"][lab] = row
    for key, d in acc.items():
        res[key] = {f"{m}:{b}": dict(rel=float(np.median(v[0])), ratio=float(np.median(v[1])), ang=float(np.median(v[2])), n=v[3]) for (m, b), v in d.items()}
    res["quartiles"] = dict(csf5=qc.tolist(), bone10=qb.tolist())
    os.makedirs(PERHEAD, exist_ok=True)
    json.dump(res, open(out, "w"), indent=0)
    return code, "ok", time.time() - t0


def main():
    a = sys.argv[1:]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 3
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    codes = sorted(cmap) if "--heads" not in a else a[a.index("--heads") + 1].split(",")
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(workers) as ex:
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
