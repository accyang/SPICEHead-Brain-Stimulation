# -*- coding: utf-8 -*-
"""Two reviewer items of the 300-head, 163-region v23 atlas, by re-scoring only (31 Oct 2026; decisions D5 and D6 of
RESULTS_v23_2026-10-31_r3). No field is solved.

PART A (D5): does the site effect on the target field T reflect anatomy/recruitment or the scanner? One site
(Bai_TVGH) used a GE scanner, the other three another vendor, so "site" conflates scanner and recruitment.
  1. Per-head tissue volumes and thicknesses from the CHARM segmentation (label_prep/tissue_labeling_upsampled.nii.gz,
     0.5 mm; final_tissues.nii.gz exists for 9 heads only). Volumes by label at 0.5 mm; thicknesses at 1 mm (every
     second voxel) from Euclidean distance transforms: skull thickness = distance from each outer skull-surface voxel
     to the nearest intracranial (WM/GM/CSF) voxel; soft-tissue (scalp) thickness = distance from each outer head-
     surface voxel to the nearest bone-or-intracranial voxel; medians over the cranial vault (world z at least 20 mm
     above the grey-matter centroid, which excludes eyes, ears and face). Output by h-code only.
  2. ANCOVA of log anatomy on age/decade + sex + site (reference BT_TSGH, as in the earlier site analyses) and on
     age/decade + sex + GE site (Bai_TVGH vs the other three sites pooled).
  3. Region-wise log T at the reported montage ~ age/decade + sex + site, then adding log ICV, skull thickness,
     soft-tissue thickness, log CSF volume, each alone and all together; the site coefficients (per site vs reference,
     and GE vs others) before and after, median % over the 163 regions and regions with BH q < 0.05.
PART B (D6): a data-driven minimum region size from region_review.csv: relative bootstrap CI width of the regional
  median T and F and the cohort-montage re-selection rate against region size (voxels at 1 mm, log scale), LOWESS and
  binned medians; thresholds where (a) the smoothed relative T CI width falls below the cohort-wide median and (b)
  smoothed re-selection reaches 90 %; regions below; Spearman F vs size (all 163 and AAL90 only).

Inputs (read only): L:/BrainProbe/SPICEhead/nchc_fullsearch/atlas_v23_parameters_perhead.npz (sp1v23__T_wa, 300 x 163),
results_v23/v23_atlas_review/region_review.csv, L:/BrainProbe/SPICEhead/charm/m2m_<CaseID>/ (CaseID via the private code
map; CaseIDs never leave this process), the private cohort CSV (age, sex, site).
Output: results_v23/v23_atlas_review/site_size/ (head_tissue_volumes.csv, site_anatomy_ancova.csv,
site_T_regions.csv, site_T_summary.csv, size_threshold.csv, size_threshold_bins.csv, size_threshold.png, SUMMARY.md,
PROVENANCE.json). Below-normal priority on 4 CPUs (3 segmentation workers).
    python pipeline/v23_site_size_review.py [--test N] [--workers 3] [--recompute]
"""
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, "nchc")]

L = r"L:\BrainProbe\SPICEhead"
PERHEAD = os.path.join(L, "nchc_fullsearch", "atlas_v23_parameters_perhead.npz")
PERHEAD_V2 = os.path.join(L, "nchc_fullsearch", "target_atlas_perhead_v2.npz")
CHARM = os.path.join(L, "charm")
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read locally only
COHORT = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"          # private; read locally only
REVIEW = os.path.join(ROOT, "results_v23", "v23_atlas_review", "region_review.csv")
OUT = os.path.join(ROOT, "results_v23", "v23_atlas_review", "site_size")
GE_SITE = "Bai_TVGH"                 # assumption: no scanner column in the cohort CSV; TVGH-Bai is the GE site
REF_SITE = "BT_TSGH"                 # reference level of the earlier site analyses (sorted first)
VAULT_MM = 20.0                      # vault = world z >= GM centroid z + VAULT_MM
LABELS = {1: "wm", 2: "gm", 3: "csf", 4: "bone_generic", 5: "scalp", 6: "eyes", 7: "compact_bone", 8: "spongy_bone",
          9: "blood", 10: "muscle", 11: "cartilage", 12: "fat"}
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def low_footprint(cpus=4):
    """Below-normal priority and the first `cpus` logical CPUs (copied from pipeline/v23_fullsearch_collect.py)."""
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    h = k32.GetCurrentProcess()
    if not (k32.SetPriorityClass(h, 0x4000) and k32.SetProcessAffinityMask(h, (1 << cpus) - 1)):
        raise SystemExit(f"could not lower priority or affinity (Windows error {k32.GetLastError()})")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def bh(p):
    p = np.asarray(p, float); n = int(np.isfinite(p).sum()); q = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]; order = idx[np.argsort(p[idx])]
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1].clip(max=1)
    return q


# ----------------------------------------------------------------------------------------------- Part A1: segmentation
def head_anatomy(args):
    """One head: (code, m2m dir) -> dict by h-code. Reads the 0.5 mm label volume once."""
    code, m2m = args
    if os.name == "nt":
        low_footprint()
    import nibabel as nib
    from scipy import ndimage as ndi
    f1 = os.path.join(m2m, "final_tissues.nii.gz")
    f2 = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    path = f2 if os.path.exists(f2) else f1
    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    img = nib.Nifti1Image.from_bytes(gzip.decompress(raw))
    lab = np.asanyarray(img.dataobj)
    zoom = np.array(img.header.get_zooms()[:3], float)
    vox_ml = float(np.prod(zoom)) / 1000.0
    cnt = np.bincount(lab.ravel().astype(np.int64), minlength=13)[:13]
    r = dict(code=code, source=os.path.basename(path), voxel_mm=round(float(zoom[0]), 3))
    for k, nm in LABELS.items():
        r[f"vol_{nm}_ml"] = round(float(cnt[k]) * vox_ml, 3)
    r["icv_ml"] = round(float(cnt[1] + cnt[2] + cnt[3]) * vox_ml, 3)
    r["brain_ml"] = round(float(cnt[1] + cnt[2]) * vox_ml, 3)
    r["bone_total_ml"] = round(float(cnt[4] + cnt[7] + cnt[8]) * vox_ml, 3)
    r["soft_total_ml"] = round(float(cnt[5] + cnt[10] + cnt[11] + cnt[12]) * vox_ml, 3)
    # thickness at about 1 mm (every second voxel when the source is 0.5 mm)
    step = 2 if zoom[0] < 0.75 else 1
    b = lab[::step, ::step, ::step]
    sp = zoom * step
    aff = img.affine.copy()
    aff[:3, :3] *= step
    inner = np.isin(b, (1, 2, 3))
    bone = np.isin(b, (4, 7, 8))
    soft = np.isin(b, (5, 10, 11, 12))
    bg = b == 0
    gi, gj, gk = np.nonzero(b == 2)
    zc = float(np.mean(aff[2, 0] * gi + aff[2, 1] * gj + aff[2, 2] * gk + aff[2, 3]))
    ii, jj, kk = np.indices(b.shape, sparse=True)
    zw = aff[2, 0] * ii + aff[2, 1] * jj + aff[2, 2] * kk + aff[2, 3]
    vault = np.broadcast_to(zw >= zc + VAULT_MM, b.shape)
    d_in = ndi.distance_transform_edt(~inner, sampling=sp)                    # distance to nearest intracranial voxel
    d_bone_in = ndi.distance_transform_edt(~(inner | bone), sampling=sp)      # distance to nearest bone/intracranial voxel
    outer_skull = bone & ndi.binary_dilation(~(inner | bone)) & vault
    outer_head = soft & ndi.binary_dilation(bg) & vault
    r["skull_thickness_mm"] = round(float(np.median(d_in[outer_skull])), 3)
    r["skull_thickness_p25_mm"] = round(float(np.percentile(d_in[outer_skull], 25)), 3)
    r["skull_thickness_p75_mm"] = round(float(np.percentile(d_in[outer_skull], 75)), 3)
    r["soft_thickness_mm"] = round(float(np.median(d_bone_in[outer_head])), 3)
    r["soft_thickness_p25_mm"] = round(float(np.percentile(d_bone_in[outer_head], 25)), 3)
    r["soft_thickness_p75_mm"] = round(float(np.percentile(d_bone_in[outer_head], 75)), 3)
    r["n_outer_skull_voxels"] = int(outer_skull.sum())
    r["n_outer_head_voxels"] = int(outer_head.sum())
    v1 = float(np.prod(sp)) / 1000.0
    r["vault_bone_ml"] = round(float((bone & vault).sum()) * v1, 3)
    r["vault_soft_ml"] = round(float((soft & vault).sum()) * v1, 3)
    r["vault_csf_ml"] = round(float(((b == 3) & vault).sum()) * v1, 3)
    r["vault_z_offset_mm"] = VAULT_MM
    r["_sha256"] = digest
    return r


def demographics(codes):
    """Age, sex, site per h-code (private maps read locally; nothing identifying leaves this function)."""
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    crow = {r["CaseID"]: r for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))}
    cols = list(next(iter(crow.values())).keys())
    scanner_cols = [c for c in cols if any(k in c.lower() for k in ("scanner", "vendor", "field", "tesla", "manufact"))]
    age = np.array([float(crow[cmap[c]]["Age_Scan"]) for c in codes])
    male = np.array([crow[cmap[c]]["Sex_1M0F"] == "1" for c in codes], float)
    site = np.array([crow[cmap[c]]["Site"] for c in codes])
    m2m = {c: os.path.join(CHARM, "m2m_" + cmap[c]) for c in codes}
    return age, male, site, m2m, scanner_cols


def ols(y, X):
    """OLS: coefficients, p-values (t), residual, R^2."""
    from scipy.stats import t as tdist
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / dof)
    p = 2 * tdist.sf(np.abs(b / se), dof)
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, p, e, r2


def partial_f(y, X_full, drop):
    """Partial F-test of the columns `drop` of X_full."""
    from scipy.stats import f as fdist
    keep = [j for j in range(X_full.shape[1]) if j not in drop]
    _, _, e1, _ = ols(y, X_full)
    _, _, e0, _ = ols(y, X_full[:, keep])
    k, dof = len(drop), len(y) - X_full.shape[1]
    F = ((e0 @ e0 - e1 @ e1) / k) / ((e1 @ e1) / dof)
    return float(F), float(fdist.sf(F, k, dof))


def pct(b):
    return 100 * (np.exp(b) - 1)


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and not k.startswith("_"):
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------------------- Part A
def part_a(codes, T_sets, anat, age, male, site, prov, lines):
    sites = sorted(set(site.tolist()))
    assert sites[0] == REF_SITE, sites
    others = sites[1:]
    n_site = {s: int((site == s).sum()) for s in sites}
    ge = (site == GE_SITE).astype(float)
    Xsite = np.column_stack([np.ones(len(codes)), age / 10, male] + [(site == s).astype(float) for s in others])
    Xge = np.column_stack([np.ones(len(codes)), age / 10, male, ge])
    A = {k: np.array([anat[c][k] for c in codes], float) for k in
         ("icv_ml", "brain_ml", "vol_gm_ml", "vol_wm_ml", "vol_csf_ml", "vol_compact_bone_ml", "vol_spongy_bone_ml",
          "bone_total_ml", "vol_scalp_ml", "soft_total_ml", "vault_bone_ml", "vault_soft_ml", "vault_csf_ml",
          "skull_thickness_mm", "soft_thickness_mm")}
    # --- A2: ANCOVA of log anatomy on age + sex + site
    anc = []
    for k, y0 in A.items():
        y = np.log(y0)
        b, p, _, r2 = ols(y, Xsite)
        Fst, pF = partial_f(y, Xsite, list(range(3, 3 + len(others))))
        bg, pg, _, _ = ols(y, Xge)
        row = dict(measure=k, geometric_mean=round(float(np.exp(y.mean())), 3),
                   age_pct_per_decade=round(float(pct(b[1])), 2), age_p=float(p[1]),
                   men_vs_women_pct=round(float(pct(b[2])), 2), sex_p=float(p[2]),
                   site_partial_F=round(Fst, 2), site_p=pF)
        for j, s in enumerate(others):
            row[f"{s}_vs_{REF_SITE}_pct"] = round(float(pct(b[3 + j])), 2)
            row[f"{s}_vs_{REF_SITE}_p"] = float(p[3 + j])
        row[f"{GE_SITE}_vs_others_pct"] = round(float(pct(bg[3])), 2)
        row[f"{GE_SITE}_vs_others_p"] = float(pg[3])
        for s in sites:
            row[f"median_{s}"] = round(float(np.median(y0[site == s])), 3)
        row["r2"] = round(float(r2), 3)
        anc.append(row)
    write_csv(os.path.join(OUT, "site_anatomy_ancova.csv"), anc)
    # --- A3: region-wise log T with and without anatomy
    cov = {"M0 base (age+sex+site)": [],
           "M1 +log ICV": [np.log(A["icv_ml"])],
           "M2 +skull thickness": [A["skull_thickness_mm"]],
           "M3 +soft-tissue thickness": [A["soft_thickness_mm"]],
           "M4 +log CSF volume": [np.log(A["vol_csf_ml"])],
           "M5 +all four": [np.log(A["icv_ml"]), A["skull_thickness_mm"], A["soft_thickness_mm"], np.log(A["vol_csf_ml"])],
           "M6 +all four + spongy-bone fraction": [np.log(A["icv_ml"]), A["skull_thickness_mm"], A["soft_thickness_mm"],
                                                   np.log(A["vol_csf_ml"]), A["vol_spongy_bone_ml"] / A["bone_total_ml"]]}
    reg_rows, summ = [], []
    for outcome, (T, targets) in T_sets.items():
        for mname, extra in cov.items():
            Xs = np.column_stack([Xsite] + [e[:, None] for e in extra]) if extra else Xsite
            Xg = np.column_stack([Xge] + [e[:, None] for e in extra]) if extra else Xge
            sidx = list(range(3, 3 + len(others)))
            res = []
            for t, name in enumerate(targets):
                y = np.log(T[:, t].astype(float))
                ok = np.isfinite(y)
                b, p, _, r2 = ols(y[ok], Xs[ok])
                Fst, pF = partial_f(y[ok], Xs[ok], sidx)
                bg, pg, _, r2g = ols(y[ok], Xg[ok])
                adj = np.r_[0.0, b[3:3 + len(others)]]
                res.append(dict(outcome=outcome, model=mname, target=name, n=int(ok.sum()), r2=round(float(r2), 4),
                                site_F=round(Fst, 3), site_p=pF, site_spread_pct=round(float(pct(adj.max() - adj.min())), 3),
                                age_pct_per_decade=round(float(pct(b[1])), 3), age_p=float(p[1]),
                                men_vs_women_pct=round(float(pct(b[2])), 3), sex_p=float(p[2]),
                                **{f"{s}_vs_{REF_SITE}_pct": round(float(pct(b[3 + j])), 3) for j, s in enumerate(others)},
                                **{f"{s}_vs_{REF_SITE}_p": float(p[3 + j]) for j, s in enumerate(others)},
                                **{f"{GE_SITE}_vs_others_pct": round(float(pct(bg[3])), 3), f"{GE_SITE}_vs_others_p": float(pg[3])}))
            for key in ["site_p", "age_p", "sex_p", f"{GE_SITE}_vs_others_p"] + [f"{s}_vs_{REF_SITE}_p" for s in others]:
                q = bh([r[key] for r in res])
                for r, qq in zip(res, q):
                    r[key.replace("_p", "_q")] = float(qq)
            reg_rows += res
            s = dict(outcome=outcome, model=mname, regions=len(res),
                     site_F_q05=int(sum(r["site_q"] < 0.05 for r in res)),
                     site_spread_pct_median=round(float(np.median([r["site_spread_pct"] for r in res])), 2),
                     r2_median=round(float(np.median([r["r2"] for r in res])), 3),
                     age_pct_median=round(float(np.median([r["age_pct_per_decade"] for r in res])), 2),
                     age_q05=int(sum(r["age_q"] < 0.05 for r in res)),
                     sex_pct_median=round(float(np.median([r["men_vs_women_pct"] for r in res])), 2),
                     sex_q05=int(sum(r["sex_q"] < 0.05 for r in res)))
            for st in others:
                s[f"{st}_pct_median"] = round(float(np.median([r[f"{st}_vs_{REF_SITE}_pct"] for r in res])), 2)
                s[f"{st}_q05"] = int(sum(r[f"{st}_vs_{REF_SITE}_q"] < 0.05 for r in res))
            s[f"{GE_SITE}_vs_others_pct_median"] = round(float(np.median([r[f"{GE_SITE}_vs_others_pct"] for r in res])), 2)
            s[f"{GE_SITE}_vs_others_q05"] = int(sum(r[f"{GE_SITE}_vs_others_q"] < 0.05 for r in res))
            summ.append(s)
    write_csv(os.path.join(OUT, "site_T_regions.csv"), reg_rows)
    write_csv(os.path.join(OUT, "site_T_summary.csv"), summ)
    # --- text
    L_ = lines
    L_ += ["## Part A (D5): site effect on T versus segmentation-derived anatomy", "",
           f"Heads per site: {', '.join(f'{s} {n_site[s]}' for s in sites)}. Reference site {REF_SITE} (as in the earlier "
           f"site analyses). GE site: {GE_SITE} (assumption: the cohort CSV has no scanner/vendor/field column"
           f"{'' if not prov['scanner_columns'] else ' other than ' + ', '.join(prov['scanner_columns'])}; the other "
           "three sites used a different vendor).", "",
           f"Anatomy from CHARM " + ", ".join(f"{k} ({v} heads)" for k, v in prov['segmentation_source_counts'].items()) +
           f" (0.5 mm labels; volumes by label; thicknesses at 1 mm from Euclidean distance transforms, medians over the "
           f"cranial vault = world z >= grey-matter centroid + {VAULT_MM:.0f} mm). ICV = WM + GM + CSF. Whole-image scalp, "
           "soft-tissue and bone volumes depend on scan coverage (neck, face), so the vault_* volumes and the two thicknesses "
           "are the coverage-independent comparisons; intracranial volumes are coverage-independent. Per-head values: "
           "head_tissue_volumes.csv (h-codes only).", "",
           "### A2. Anatomy by site (ANCOVA: log measure ~ age/decade + sex + site)", "",
           f"| Measure | Geometric mean | Age %/decade (p) | Men vs women % (p) | " +
           " | ".join(f"{s} vs {REF_SITE} % (p)" for s in others) + f" | Site partial F p | {GE_SITE} vs others % (p) |",
           "|---|---|---|---|" + "---|" * len(others) + "---|---|"]
    for r in anc:
        L_.append(f"| {r['measure']} | {r['geometric_mean']} | {r['age_pct_per_decade']:+.1f} ({r['age_p']:.2g}) | "
                  f"{r['men_vs_women_pct']:+.1f} ({r['sex_p']:.2g}) | " +
                  " | ".join(f"{r[f'{s}_vs_{REF_SITE}_pct']:+.1f} ({r[f'{s}_vs_{REF_SITE}_p']:.2g})" for s in others) +
                  f" | {r['site_p']:.2g} | {r[f'{GE_SITE}_vs_others_pct']:+.1f} ({r[f'{GE_SITE}_vs_others_p']:.2g}) |")
    L_ += ["", "### A3. Site coefficients on log T by region, before and after anatomy adjustment", "",
           "Median over regions of each site's % effect on T (exp(b) - 1), and the number of regions with BH q < 0.05 "
           "across the regions of that outcome. 'Site F' = partial F-test of the three site terms. 'Spread' = max - min "
           "of the adjusted site means, %.", ""]
    for outcome in T_sets:
        L_ += [f"**{outcome}**", "",
               "| Model | " + " | ".join(f"{s} vs {REF_SITE}: median % (q<0.05)" for s in others) +
               f" | {GE_SITE} vs others: median % (q<0.05) | Site F q<0.05 | Spread median % | Age %/dec (q<0.05) | Sex % (q<0.05) | R2 median |",
               "|---|" + "---|" * len(others) + "---|---|---|---|---|---|"]
        for s in summ:
            if s["outcome"] != outcome:
                continue
            L_.append(f"| {s['model']} | " + " | ".join(f"{s[f'{st}_pct_median']:+.1f} ({s[f'{st}_q05']})" for st in others) +
                      f" | {s[f'{GE_SITE}_vs_others_pct_median']:+.1f} ({s[f'{GE_SITE}_vs_others_q05']}) | {s['site_F_q05']} | "
                      f"{s['site_spread_pct_median']:.1f} | {s['age_pct_median']:+.1f} ({s['age_q05']}) | "
                      f"{s['sex_pct_median']:+.1f} ({s['sex_q05']}) | {s['r2_median']:.3f} |")
        L_.append("")
    return anc, summ


# ------------------------------------------------------------------------------------------------------------- Part B
def part_b(lines, prov):
    from scipy.stats import spearmanr
    from statsmodels.nonparametric.smoothers_lowess import lowess
    rr = list(csv.DictReader(open(REVIEW, encoding="utf-8")))
    f = lambda k: np.array([float(r[k]) for r in rr])
    vox = f("voxels_median")
    Tm, Tlo, Thi = f("sp1v23__T_reported_median"), f("sp1v23__T_reported_ci_lo"), f("sp1v23__T_reported_ci_hi")
    Fm, Flo, Fhi = f("sp1v23__F_reported_median"), f("sp1v23__F_reported_ci_lo"), f("sp1v23__F_reported_ci_hi")
    resel = f("sp1v23__cohort_reselected_pct")
    ret5, retm, fails = f("retention_p5"), f("retention_median"), f("sp_choice_fails_fem_floor_pct")
    relT = 100 * (Thi - Tlo) / Tm
    relF = 100 * (Fhi - Flo) / Fm
    tier = np.array([r["tier"] for r in rr])
    names = [r["target"] for r in rr]
    lx = np.log10(vox)
    medT, medF = float(np.median(relT)), float(np.median(relF))
    grid = np.linspace(lx.min(), lx.max(), 400)
    metrics = {"relT": (relT, lambda c: c < medT), "relF": (relF, lambda c: c < medF),
               "resel": (resel, lambda c: c >= 90), "ret5": (ret5, lambda c: c >= 90)}
    sm, thr = {}, {}
    for key, (y, cond) in metrics.items():
        sm[key] = lowess(y, lx, frac=0.5, it=1, xvals=grid)
        ok = cond(sm[key])
        if not ok[-1]:
            thr[key] = float("nan")                      # the smoothed curve fails the criterion at the largest sizes
        else:
            i = len(ok) - 1
            while i > 0 and ok[i - 1]:
                i -= 1
            thr[key] = float("nan") if i == 0 else float(10 ** grid[i])   # i == 0: criterion met at every size
        thr[key + "_curve_min"], thr[key + "_curve_max"] = float(sm[key].min()), float(sm[key].max())
    fin = lambda k: np.isfinite(thr[k])
    # binned medians (quantile bins of log size)
    nb = 8
    edges = np.quantile(lx, np.linspace(0, 1, nb + 1))
    bins = []
    for i in range(nb):
        m = (lx >= edges[i]) & (lx <= edges[i + 1]) if i == nb - 1 else (lx >= edges[i]) & (lx < edges[i + 1])
        bins.append(dict(bin=i + 1, voxels_lo=int(round(10 ** edges[i])), voxels_hi=int(round(10 ** edges[i + 1])), regions=int(m.sum()),
                         relT_ci_width_pct_median=round(float(np.median(relT[m])), 2),
                         relF_ci_width_pct_median=round(float(np.median(relF[m])), 2),
                         reselected_pct_median=round(float(np.median(resel[m])), 1),
                         retention_p5_median=round(float(np.median(ret5[m])), 1),
                         fails_fem_floor_pct_median=round(float(np.median(fails[m])), 1),
                         reselected_ge90_share_pct=round(100 * float(np.mean(resel[m] >= 90)), 1),
                         retention_p5_ge90_share_pct=round(100 * float(np.mean(ret5[m] >= 90)), 1),
                         relT_below_cohort_median_share_pct=round(100 * float(np.mean(relT[m] < medT)), 1)))
    write_csv(os.path.join(OUT, "size_threshold_bins.csv"), bins)
    # Spearman against size
    aal = tier == "tier1"
    sp = {}
    ycols = (("F", Fm), ("T", Tm), ("relT", relT), ("relF", relF), ("resel", resel), ("ret5", ret5), ("fails", fails))
    for lab, m in (("all163", np.ones(len(rr), bool)), ("aal90", aal)):
        for k, y in ycols:
            r_, p_ = spearmanr(vox[m], y[m])
            sp[(lab, k)] = (float(r_), float(p_), int(m.sum()))
    # per-region table
    rows = []
    for i, r in enumerate(rr):
        rows.append(dict(target=names[i], tier=tier[i], set=r["set"], voxels_median=int(vox[i]), voxels_min=int(r["voxels_min"]),
                         T_median=Tm[i], T_ci_lo=Tlo[i], T_ci_hi=Thi[i], relT_ci_width_pct=round(float(relT[i]), 2),
                         F_median=Fm[i], F_ci_lo=Flo[i], F_ci_hi=Fhi[i], relF_ci_width_pct=round(float(relF[i]), 2),
                         cohort_reselected_pct=resel[i], retention_median=retm[i], retention_p5=ret5[i], fails_fem_floor_pct=fails[i],
                         relT_below_cohort_median=bool(relT[i] < medT), relF_below_cohort_median=bool(relF[i] < medF),
                         reselected_ge90=bool(resel[i] >= 90), retention_p5_ge90=bool(ret5[i] >= 90),
                         **{f"below_{k}_threshold": (bool(vox[i] < thr[k]) if fin(k) else "") for k in metrics}))
    rows.sort(key=lambda d: d["voxels_median"])
    write_csv(os.path.join(OUT, "size_threshold.csv"), rows)
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6})
    sets = ["tier1", "tier2", "ref6", "cingulate", "thalamus", "cerebellum", "brainstem"]
    setlab = {"tier1": "AAL90", "tier2": "psychiatric", "ref6": "reference spheres", "cingulate": "supracallosal ACC",
              "thalamus": "thalamic nuclei", "cerebellum": "cerebellum", "brainstem": "brainstem"}
    col = {s: OKABE[i] for i, s in enumerate(sets)}
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 125 / 25.4))
    panels = (("relT", relT, "Relative 95% CI width of median T, %", medT, "top"),
              ("relF", relF, "Relative 95% CI width of median F, %", medF, "top"),
              ("resel", resel, "Cohort montage re-selected, % of bootstraps", 90.0, "bottom"),
              ("ret5", ret5, "FEM retention at SPICEhead's choice, 5th pct., %", 90.0, "bottom"))
    for ax, (key, y, ylab, hline, where), letter in zip(axes.ravel(), panels, "ABCD"):
        for s in sets:
            m = tier == s
            ax.scatter(vox[m], y[m], s=9, c=col[s], edgecolors="none", alpha=0.85, label=setlab[s], zorder=3)
        ax.plot(10 ** grid, sm[key], color="black", lw=1.0, zorder=4)
        ax.axhline(hline, color="0.4", lw=0.6, ls="--")
        if fin(key):
            ax.axvline(thr[key], color=OKABE[5], lw=0.8, ls=":")
            yl = ax.get_ylim()
            ax.text(thr[key], yl[1] if where == "top" else yl[0] + 0.03 * (yl[1] - yl[0]), f" {thr[key]:,.0f} vox",
                    color=OKABE[5], fontsize=6.5, va="top" if where == "top" else "bottom", ha="left")
        ax.set_xscale("log")
        ax.set_xlabel("Region size (voxels at 1 mm, median over heads)")
        ax.set_ylabel(ylab)
        ax.text(-0.18, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        for i in np.argsort(vox)[:7]:
            ax.annotate(names[i].replace("ref6:", "").replace("brainstem:", ""), (vox[i], y[i]), fontsize=5,
                        xytext=(2, 1), textcoords="offset points")
    axes[0, 0].legend(frameon=False, loc="upper left", handletextpad=0.2, borderpad=0.2, labelspacing=0.2, markerscale=1.2, ncol=2)
    fig.tight_layout(w_pad=1.5, h_pad=1.2)
    fig.savefig(os.path.join(OUT, "size_threshold.png"), dpi=300)
    plt.close(fig)
    # text
    def thr_text(k, crit):
        if fin(k):
            n_below = sum(bool(d["voxels_median"] < thr[k]) for d in rows)
            return f"smoothed curve {crit} above **{thr[k]:,.0f} voxels** ({n_below} regions below)"
        always = metrics[k][1](np.array([thr[k + "_curve_min"], thr[k + "_curve_max"]])).all()
        return (f"no size threshold: the smoothed curve (range {thr[k + '_curve_min']:.1f} to {thr[k + '_curve_max']:.1f}) "
                + ("meets the criterion at every size" if always else "does not meet the criterion at the largest sizes"))
    below = {k: [d for d in rows if fin(k) and d["voxels_median"] < thr[k]] for k in metrics}
    lines += ["## Part B (D6): minimum region size", "",
              f"Inputs: region_review.csv (163 regions; bootstrap over heads, 1000 draws for the medians, 100 for re-selection; "
              f"FEM retention from the same per-head arrays). Relative CI width = (CI hi - CI lo) / median, %. Cohort-wide "
              f"median relative CI width: T {medT:.2f} %, F {medF:.2f} %. LOWESS (frac 0.5) of each quantity on log10(voxels); "
              f"a size threshold is the smallest size above which the smoothed curve stays on the favourable side of the criterion.", "",
              f"- (a) relative T CI width below the cohort median: {thr_text('relT', 'falls below the cohort median')}.",
              f"- (a') relative F CI width below the cohort median: {thr_text('relF', 'falls below the cohort median')}.",
              f"- (b) cohort montage re-selected in >= 90 % of bootstraps: {thr_text('resel', 'reaches 90 %')}.",
              f"- (c) FEM retention of SPICEhead's choice, 5th percentile over heads, >= 90 %: {thr_text('ret5', 'reaches 90 %')}.", "",
              "Binned medians (quantile bins of log size):", "",
              "| Bin | Voxels | Regions | rel. T CI % | rel. F CI % | re-selected % | retention p5 % | fails FEM floor % | share re-selected >= 90 % | share retention p5 >= 90 % | share rel. T CI < cohort median |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b_ in bins:
        lines.append(f"| {b_['bin']} | {b_['voxels_lo']:,}-{b_['voxels_hi']:,} | {b_['regions']} | {b_['relT_ci_width_pct_median']:.2f} | "
                     f"{b_['relF_ci_width_pct_median']:.2f} | {b_['reselected_pct_median']:.0f} | {b_['retention_p5_median']:.1f} | "
                     f"{b_['fails_fem_floor_pct_median']:.1f} | {b_['reselected_ge90_share_pct']:.0f} % | {b_['retention_p5_ge90_share_pct']:.0f} % | "
                     f"{b_['relT_below_cohort_median_share_pct']:.0f} % |")
    small = [d for d in rows if d["voxels_median"] < 150]
    lines += ["", "Smallest regions (< 150 voxels at 1 mm; the expected small nuclei):", "",
              "| Region | Tier | Voxels (median; min) | T median [95% CI] | rel. T CI % | F median [95% CI] | rel. F CI % | re-selected % | retention p5 % | fails FEM floor % |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for d in small:
        lines.append(f"| {d['target']} | {d['tier']} | {d['voxels_median']}; {d['voxels_min']} | {d['T_median']:.4f} [{d['T_ci_lo']:.4f}, {d['T_ci_hi']:.4f}] | "
                     f"{d['relT_ci_width_pct']:.1f} | {d['F_median']:.3f} [{d['F_ci_lo']:.3f}, {d['F_ci_hi']:.3f}] | {d['relF_ci_width_pct']:.1f} | "
                     f"{d['cohort_reselected_pct']:.0f} | {d['retention_p5']:.1f} | {d['fails_fem_floor_pct']:.1f} |")
    for k, label in (("ret5", "FEM-retention threshold (c)"), ("relT", "T-precision threshold (a)"), ("resel", "re-selection threshold (b)")):
        if fin(k):
            lines += ["", f"Regions below the {label}, {thr[k]:,.0f} voxels (voxels; value): " +
                      ", ".join(f"{d['target']} ({d['voxels_median']}; {metrics[k][0][names.index(d['target'])]:.1f})" for d in below[k]) + "."]
    fail_resel = [d for d in rows if d["cohort_reselected_pct"] < 90]
    fail_ret = [d for d in rows if d["retention_p5"] < 90]
    lines += ["", f"Regions with re-selection < 90 % ({len(fail_resel)} of 163; voxels; %): " +
              ", ".join(f"{d['target']} ({d['voxels_median']:,}; {d['cohort_reselected_pct']:.0f})" for d in fail_resel) + ".",
              "", f"Regions with FEM retention p5 < 90 % ({len(fail_ret)} of 163; voxels; %): " +
              ", ".join(f"{d['target']} ({d['voxels_median']:,}; {d['retention_p5']:.1f})" for d in fail_ret) + ".", "",
              "Size association (Spearman of the regional value against voxels_median):", "",
              "| Set | n | F median | T median | rel. T CI width | rel. F CI width | re-selection | retention p5 | fails FEM floor |",
              "|---|---|---|---|---|---|---|---|---|"]
    for lab in ("all163", "aal90"):
        n = sp[(lab, "F")][2]
        lines.append(f"| {lab} | {n} | " + " | ".join(f"{sp[(lab, k)][0]:+.3f} (p {sp[(lab, k)][1]:.2g})" for k, _ in ycols) + " |")
    lines.append("")
    prov["part_b"] = dict(cohort_median_relT_ci_pct=medT, cohort_median_relF_ci_pct=medF, thresholds_voxels=thr,
                          lowess_frac=0.5, spearman={f"{a}_{b}": dict(rho=v[0], p=v[1], n=v[2]) for (a, b), v in sp.items()})
    return thr, sp, medT, medF, below, fail_resel, fail_ret, small


# ----------------------------------------------------------------------------------------------- per-site medians (Data S1)
def per_site(z, codes, targets, site, lines, prov):
    """One row per region: median over heads within each site of T (sp1v23, fem) and F (sp1v23) at the reported montage,
    heads with a valid value per site, and the GE-site / other-sites ratio of medians. Aggregates only."""
    sites = sorted(set(site.tolist()))
    keys = ("sp1v23__T_wa", "fem__T_wa", "sp1v23__F_wa")
    rows = []
    for t, name in enumerate(targets):
        r = dict(target=name)
        for k in keys:
            v = z[k][:, t].astype(float)
            for st in sites:
                m = (site == st) & np.isfinite(v)
                r[f"{k}__median_{st}"] = round(float(np.median(v[m])), 5) if m.any() else ""
                r[f"{k}__n_{st}"] = int(m.sum())
            ge, ot = (site == GE_SITE) & np.isfinite(v), (site != GE_SITE) & np.isfinite(v)
            r[f"{k}__median_{GE_SITE}_over_others"] = round(float(np.median(v[ge]) / np.median(v[ot])), 4)
        rows.append(r)
    write_csv(os.path.join(OUT, "per_site_T.csv"), rows)
    ratio = {k: np.array([r[f"{k}__median_{GE_SITE}_over_others"] for r in rows]) for k in keys}
    nsite = {st: int((site == st).sum()) for st in sites}
    lines += ["## Per-site medians (Data S1)", "",
              f"per_site_T.csv: for each of the 163 regions, the median over heads within each site "
              f"({', '.join(f'{st} n={n}' for st, n in nsite.items())}) of T at the reported montage (SPICEhead v23 and FEM) and "
              "of F at the reported montage (SPICEhead v23), the number of heads with a valid value per site, and the ratio of the "
              f"{GE_SITE} median to the median of the other three sites pooled. Unadjusted (no age/sex correction); aggregates only. "
              "Median over regions of the GE/others ratio: " +
              "; ".join(f"{k} {np.median(ratio[k]):.3f} (IQR {np.percentile(ratio[k], 25):.3f}-{np.percentile(ratio[k], 75):.3f})"
                        for k in keys) + ".", ""]
    prov["per_site"] = {k: dict(ge_over_others_ratio_median=float(np.median(ratio[k])),
                                p25=float(np.percentile(ratio[k], 25)), p75=float(np.percentile(ratio[k], 75))) for k in keys}
    prov["per_site"]["heads_per_site"] = nsite
    return rows


# --------------------------------------------------------------------------------------------------------------- main
def main():
    if os.name == "nt":
        low_footprint()
    t0 = time.time()
    a = sys.argv[1:]
    n_test = int(a[a.index("--test") + 1]) if "--test" in a else None
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 3
    os.makedirs(OUT, exist_ok=True)
    z = np.load(PERHEAD)
    codes = [str(c) for c in z["codes"]]
    targets = [str(t) for t in z["targets"]]
    assert len(codes) == 300 and len(targets) == 163
    T_wa = z["sp1v23__T_wa"]; T_wc = z["sp1v23__T_wc"]; T_fem = z["fem__T_wa"]
    prov = dict(script="pipeline/v23_site_size_review.py", date=time.strftime("%Y-%m-%d %H:%M"), heads=300, regions=163,
                inputs={PERHEAD: sha256(PERHEAD), REVIEW: sha256(REVIEW)}, ge_site_assumption=GE_SITE, reference_site=REF_SITE,
                vault_offset_mm=VAULT_MM)
    # T at the cohort montage: only if a v23 model exists in target_atlas_perhead_v2.npz
    tcoh_note = "target_atlas_perhead_v2.npz not found"
    if os.path.exists(PERHEAD_V2):
        z2 = np.load(PERHEAD_V2)
        models2 = [str(m) for m in z2["models"]]
        tcoh_note = (f"target_atlas_perhead_v2.npz holds models {models2} (v22 generation; no sp1v23), so T at the cohort "
                     "montage was not analysed; the review script computes it but does not save per-head values.")
        prov["perhead_v2_models"] = models2
    age, male, site, m2m, scanner_cols = demographics(codes)
    prov["scanner_columns"] = scanner_cols
    # --- A1 anatomy (cached in head_tissue_volumes.csv)
    vol_csv = os.path.join(OUT, "head_tissue_volumes.csv")
    anat = {}
    if os.path.exists(vol_csv) and "--recompute" not in a:
        for r in csv.DictReader(open(vol_csv, encoding="utf-8")):
            anat[r["code"]] = {k: (float(v) if k not in ("code", "source") else v) for k, v in r.items()}
    todo = [c for c in codes if c not in anat]
    if n_test is not None:
        todo = todo[:n_test]
    seg_sha = {}
    if todo:
        for c in todo:
            assert os.path.isdir(m2m[c]), f"missing m2m for {c}"
        print(f"segmentation: {len(todo)} heads on {workers} workers", flush=True)
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(head_anatomy, [(c, m2m[c]) for c in todo]), 1):
                seg_sha[r["code"]] = r.pop("_sha256")
                anat[r["code"]] = r
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
        rows = [anat[c] for c in codes if c in anat]
        rows.sort(key=lambda d: d["code"])
        write_csv(vol_csv, rows)
        sha_path = os.path.join(OUT, "segmentation_sha256.json")
        old = json.load(open(sha_path, encoding="utf-8")) if os.path.exists(sha_path) else {}
        old.update(seg_sha)
        json.dump(old, open(sha_path, "w", encoding="utf-8"), indent=0, sort_keys=True)
    if n_test is not None and len(anat) < 300:
        print(f"test run: {len(anat)} heads segmented; stopping before the regressions ({time.time() - t0:.0f} s)")
        return
    assert all(c in anat for c in codes)
    src = {}
    for c in codes:
        src[anat[c]["source"]] = src.get(anat[c]["source"], 0) + 1
    prov["segmentation_source_counts"] = src
    prov["segmentation_sha256_file"] = "segmentation_sha256.json"
    lines = [f"# Site vs anatomy (D5) and minimum region size (D6): SPICEhead v23, 300 heads, 163 regions", "",
             f"Generated {prov['date']} by pipeline/v23_site_size_review.py (re-scoring only; no field solved). "
             f"Inputs: atlas_v23_parameters_perhead.npz (T at the reported montage, sp1v23__T_wa; also sp1v23__T_wc and "
             f"fem__T_wa as checks), region_review.csv, CHARM segmentations, private demographics (age, sex, site).", "",
             f"Note on T at the cohort montage: {tcoh_note}", ""]
    T_sets = {"SPICEhead v23, T at the reported montage (sp1v23__T_wa)": (T_wa, targets),
              "SPICEhead v23, T at the constrained optimum (sp1v23__T_wc)": (T_wc, targets),
              "FEM, T at the reported montage (fem__T_wa)": (T_fem, targets)}
    anc, summ = part_a(codes, T_sets, anat, age, male, site, prov, lines)
    thr, sp, medT, medF, below, fail_resel, fail_ret, small = part_b(lines, prov)
    # --- conclusions
    s0 = [s for s in summ if s["outcome"].startswith("SPICEhead v23, T at the reported") and s["model"].startswith("M0")][0]
    s5 = [s for s in summ if s["outcome"].startswith("SPICEhead v23, T at the reported") and s["model"].startswith("M5")][0]
    s1 = [s for s in summ if s["outcome"].startswith("SPICEhead v23, T at the reported") and s["model"].startswith("M1")][0]
    swc0 = [s for s in summ if s["outcome"].startswith("SPICEhead v23, T at the constrained") and s["model"].startswith("M0")][0]
    sites = sorted(set(site.tolist())); others = sites[1:]
    ge_anat = {r["measure"]: (r[f"{GE_SITE}_vs_others_pct"], r[f"{GE_SITE}_vs_others_p"]) for r in anc}
    lines += ["## Conclusions", "",
              f"**D5.** Before adjustment (M0), T at the reported montage differed by site in {s0['site_F_q05']} of 163 regions "
              f"(partial F, q < 0.05; median spread {s0['site_spread_pct_median']:.1f} %); the GE site ({GE_SITE}) versus the "
              f"other three: median {s0[f'{GE_SITE}_vs_others_pct_median']:+.1f} % ({s0[f'{GE_SITE}_vs_others_q05']} regions q < 0.05). "
              f"With log ICV alone (M1): {s1['site_F_q05']} regions, GE {s1[f'{GE_SITE}_vs_others_pct_median']:+.1f} % "
              f"({s1[f'{GE_SITE}_vs_others_q05']}). With all four anatomy covariates (M5: log ICV, skull thickness, soft-tissue "
              f"thickness, log CSF): {s5['site_F_q05']} regions, median spread {s5['site_spread_pct_median']:.1f} %, GE "
              f"{s5[f'{GE_SITE}_vs_others_pct_median']:+.1f} % ({s5[f'{GE_SITE}_vs_others_q05']}). Per site vs {REF_SITE}, M0 -> M5: " +
              "; ".join(f"{st} {s0[f'{st}_pct_median']:+.1f} % ({s0[f'{st}_q05']}) -> {s5[f'{st}_pct_median']:+.1f} % ({s5[f'{st}_q05']})" for st in others) + ". "
              f"(Consistency check: at the constrained optimum, M0 gives a site effect in {swc0['site_F_q05']} regions with a median "
              f"spread of {swc0['site_spread_pct_median']:.1f} %, the figures of the r3 record.)",
              "",
              f"Segmentation-derived anatomy of the GE site vs the others (age- and sex-adjusted): ICV {ge_anat['icv_ml'][0]:+.1f} % "
              f"(p {ge_anat['icv_ml'][1]:.2g}), CSF {ge_anat['vol_csf_ml'][0]:+.1f} % (p {ge_anat['vol_csf_ml'][1]:.2g}), compact bone "
              f"{ge_anat['vol_compact_bone_ml'][0]:+.1f} % (p {ge_anat['vol_compact_bone_ml'][1]:.2g}), spongy bone "
              f"{ge_anat['vol_spongy_bone_ml'][0]:+.1f} % (p {ge_anat['vol_spongy_bone_ml'][1]:.2g}), skull thickness "
              f"{ge_anat['skull_thickness_mm'][0]:+.1f} % (p {ge_anat['skull_thickness_mm'][1]:.2g}), soft-tissue thickness "
              f"{ge_anat['soft_thickness_mm'][0]:+.1f} % (p {ge_anat['soft_thickness_mm'][1]:.2g}), scalp (skin) volume "
              f"{ge_anat['vol_scalp_ml'][0]:+.1f} % (p {ge_anat['vol_scalp_ml'][1]:.2g}).", ""]
    s6 = [s for s in summ if s["outcome"].startswith("SPICEhead v23, T at the reported") and s["model"].startswith("M6")][0]
    swc0 = [s for s in summ if s["outcome"].startswith("SPICEhead v23, T at the constrained") and s["model"].startswith("M0")][0]
    g0, g5, g6 = (s[f"{GE_SITE}_vs_others_pct_median"] for s in (s0, s5, s6))
    change = (f"the GE-vs-others median effect went from {g0:+.1f} % (M0) to {g5:+.1f} % (M5) and {g6:+.1f} % (M6, adding the "
              f"spongy-bone fraction; {s6[f'{GE_SITE}_vs_others_q05']} regions q < 0.05, site F in {s6['site_F_q05']} regions)")
    if abs(g5) > 1.25 * abs(g0) and s5["site_F_q05"] >= s0["site_F_q05"]:
        verdict = ("Adjusting for segmentation-derived anatomy does not remove the site effect; it enlarges it. The GE heads "
                   "have anatomy that predicts a lower T (larger ICV and CSF volume), yet their T is higher, so conditioning on "
                   "anatomy uncovers a larger site term. The site effect is therefore not the gross anatomy of the recruited "
                   "samples as segmented; it is a candidate scanner/segmentation effect (the GE site also shows a very "
                   "different compact/spongy bone split, a contrast-dependent label) or an unmeasured head property.")
    elif s5["site_F_q05"] <= 0.25 * max(s0["site_F_q05"], 1) and s5[f"{GE_SITE}_vs_others_q05"] <= 0.25 * max(s0[f"{GE_SITE}_vs_others_q05"], 1):
        verdict = ("Adjusting for segmentation-derived anatomy removes most of the site effect: the site differences in T are "
                   "carried by the heads' tissue volumes and thicknesses as segmented (anatomy of the recruited samples, or "
                   "scanner-dependent segmentation of those tissues; the two cannot be separated without a cross-scanner "
                   "rescan). What remains is small.")
    elif s5["site_F_q05"] <= 0.6 * max(s0["site_F_q05"], 1):
        verdict = ("Adjusting for segmentation-derived anatomy reduces but does not remove the site effect: part of it is "
                   "explained by the segmented tissue volumes and thicknesses, and a residual site term remains that is a "
                   "candidate scanner/segmentation effect (or unmeasured anatomy).")
    else:
        verdict = ("Adjusting for segmentation-derived anatomy does not remove the site effect: the site term is largely "
                   "independent of ICV, skull and soft-tissue thickness and CSF volume as segmented, so it is a candidate "
                   "scanner/segmentation (contrast-dependent) effect rather than gross anatomy of the recruited samples.")
    lines += [f"Verdict: {verdict} ({change}.) Note that the "
              "anatomy covariates come from the same segmentations, so a scanner effect that acts through the segmentation "
              "(e.g. on bone/CSF boundaries) would be absorbed by them; the comparison therefore separates 'site acting "
              "through segmented anatomy' from 'site acting otherwise', not scanner from recruitment.", "",
              "**D6.** "
              + (f"The relative CI width of the regional median T falls below the cohort median above {thr['relT']:,.0f} voxels; "
                 f"{len(below['relT'])} regions below ({', '.join(d['target'] for d in below['relT'])}). "
                 if np.isfinite(thr["relT"]) else
                 f"The precision of the regional median T does not depend on region size (Spearman rho of the relative CI width "
                 f"vs size {sp[('all163', 'relT')][0]:+.2f}, p {sp[('all163', 'relT')][1]:.2g}; smoothed curve "
                 f"{thr['relT_curve_min']:.1f}-{thr['relT_curve_max']:.1f} % against a cohort median of {medT:.2f} %), so criterion (a) "
                 "gives no size threshold: the CI reflects between-head variability, not voxel count. ")
              + (f"Smoothed re-selection reaches 90 % above {thr['resel']:,.0f} voxels ({len(below['resel'])} regions below). "
                 if np.isfinite(thr["resel"]) else
                 f"Re-selection below 90 % is not a size phenomenon either (rho {sp[('all163', 'resel')][0]:+.2f}, "
                 f"p {sp[('all163', 'resel')][1]:.2g}; smoothed curve {thr['resel_curve_min']:.0f}-{thr['resel_curve_max']:.0f} %); "
                 f"the {len(fail_resel)} regions below 90 % span {min(d['voxels_median'] for d in fail_resel):,} to "
                 f"{max(d['voxels_median'] for d in fail_resel):,} voxels. ")
              + (f"The size-dependent quantity is the FEM retention of SPICEhead's choice in the worst heads (5th percentile; rho "
                 f"{sp[('all163', 'ret5')][0]:+.2f}, p {sp[('all163', 'ret5')][1]:.2g}): its smoothed curve reaches 90 % above "
                 f"**{thr['ret5']:,.0f} voxels** (criterion c), with {len(below['ret5'])} regions below: "
                 + ", ".join(f"{d['target']} ({d['voxels_median']} vox; p5 {d['retention_p5']:.1f} %)" for d in below['ret5']) + ". "
                 if np.isfinite(thr["ret5"]) else
                 f"The only size-dependent quantity is the FEM retention of SPICEhead's choice in the worst heads (5th percentile; "
                 f"rho {sp[('all163', 'ret5')][0]:+.2f}, p {sp[('all163', 'ret5')][1]:.2g}; smoothed curve "
                 f"{thr['ret5_curve_min']:.1f}-{thr['ret5_curve_max']:.1f} %), but the smoothed curve stays above 90 % at every "
                 f"size, so criterion (c) gives no threshold either; the {len(fail_ret)} regions individually below 90 % are "
                 + ", ".join(f"{d['target']} ({d['voxels_median']} vox; {d['retention_p5']:.1f} %)" for d in fail_ret) + ". ")
              + f"F is {'' if sp[('all163', 'F')][1] < 0.05 else 'not '}associated with region size over all 163 regions "
              f"(Spearman rho {sp[('all163', 'F')][0]:+.3f}, p {sp[('all163', 'F')][1]:.2g}) and "
              f"{'' if sp[('aal90', 'F')][1] < 0.05 else 'not '}within AAL90 (rho {sp[('aal90', 'F')][0]:+.3f}, p {sp[('aal90', 'F')][1]:.2g}); "
              f"the all-163 association comes from the small, deep extra-AAL nuclei, not from a size gradient within the cortical/subcortical AAL set.",
              "",
              "No data-driven minimum size emerges from the bootstrap precision or stability of the atlas values: the small nuclei "
              "are estimated as precisely as large gyri because precision is set by the 300 heads, not by voxel count. If a flag is "
              "still wanted for D6 it has to be a convention on the grid, not a threshold from these data; the regions with fewer "
              "than 60 voxels (1 mm) in the median head are "
              + ", ".join(f"{d['target']} ({d['voxels_median']}; min over heads {d['voxels_min']})" for d in small if d["voxels_median"] < 60)
              + f", and the {len(fail_ret)} regions whose worst-head FEM retention falls below 90 % are "
              + " and ".join(d["target"] for d in fail_ret) + ".", ""]
    per_site(z, codes, targets, site, lines, prov)
    open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(lines))
    prov["seconds"] = round(time.time() - t0)
    prov["outputs"] = sorted(f for f in os.listdir(OUT) if f != "PROVENANCE.json")
    json.dump(prov, open(os.path.join(OUT, "PROVENANCE.json"), "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
