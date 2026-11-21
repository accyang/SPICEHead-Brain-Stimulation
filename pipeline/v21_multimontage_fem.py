# -*- coding: utf-8 -*-
"""Multi-montage FEM comparison — second-round review, comments 3 and 5.

The first FEM comparison used one montage (the thalamic optimum), yet the manuscript argues that
between-montage comparisons are trustworthy because the discrepancy behaves as a common scale
factor. One montage cannot establish that. The reviewer asks specifically whether

  * the regression slope is stable across geometrically distinct montages,
  * montage rank ordering is preserved between SPICEhead and FEM,
  * target-intensity differences are preserved.

Rank preservation matters more to the optimization claim than voxel-wise correlation does, because
the paper never uses absolute magnitude to choose a montage — it uses the ordering.

Four montages spanning different current paths:
  M1        F3+F4 / Pz+O1     superficial, fronto-parietal
  THAL      T8+P8 / F7+T7     bilateral temporal, the montage already tested
  HIPP      Fp2+F7 / P8+O2    fronto-occipital, medial-temporal target
  INFERIOR  F9+T9 / F10+T10   the inferior 9/10 chain, well below the 10-20 ring

Stages (each resumable, --sim under the SimNIBS interpreter, the rest under the project one):
  --sim       FEM runs for the montages not yet simulated
  --compare   agreement per montage plus rank preservation by target intensity
  --focality  rank preservation by focality F = T/O, the quantity Table 3 actually ranks by

The --compare rank test orders montages by median target intensity, which is what the reviewer
asked for. But the optimization ranks by focality, and the off-target term O never enters that
test, so --focality repeats it on F. It re-uses the same cached envelopes and adds no simulations.

  python pipeline/v21_multimontage_fem.py --sim|--compare|--focality [n_subjects] [dx_mm]
"""
import csv
import json
import os
import pickle
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)
from v21_simnibs_validation import subjects, OUT, CURRENT, ELEC_RADIUS_MM

MONTAGES = {
    "M1":       ((("F3", "F4"), ("Pz", "O1")), "electrodes_1020.json"),
    "THAL":     ((("T8", "P8"), ("F7", "T7")), "electrodes_1020.json"),
    "HIPP":     ((("Fp2", "F7"), ("P8", "O2")), "electrodes_1020.json"),
    "INFERIOR": ((("F9", "T9"), ("F10", "T10")), "electrodes_1020ext.json"),
}
# THAL was simulated by v21_simnibs_validation.py into <subject>/ch1, ch2 — reuse it.
LEGACY = {"THAL": ("ch1", "ch2")}
CACHE = os.path.join(os.environ.get("TEMP", "."), config.tagged("spicehead_multimont_cache"))


def fem_dir(cid, tag, ch):
    if tag in LEGACY:
        return os.path.join(OUT, cid, LEGACY[tag][ch - 1])
    return os.path.join(OUT, cid, f"{tag}_ch{ch}")


def write_merged(path, rows, key):
    """Write `rows`, preserving rows from other resolutions.

    Both stages are run at more than one dx and every row carries dx_mm, so a plain overwrite
    would silently discard the other resolution's results. Rows matching an incoming row on
    `key` are replaced; everything else is kept.
    """
    if not rows:
        return
    keep = []
    if os.path.exists(path):
        incoming = {tuple(str(r[k]) for k in key) for r in rows}
        for old in csv.DictReader(open(path, encoding="utf-8")):
            if all(k in old for k in key) and \
                    tuple(str(old[k]) for k in key) not in incoming:
                keep.append(old)
    fields = list(rows[0])
    for r in keep:                      # tolerate older files with fewer columns
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in keep + rows:
            w.writerow({k: r.get(k, "") for k in fields})


# ---------------------------------------------------------------- stage: simulate
def stage_sim(n):
    from simnibs import sim_struct, run_simnibs
    for i, (cid, m2m, age, site) in enumerate(subjects(n), 1):
        for tag, (pairs, ejfile) in MONTAGES.items():
            ej = json.load(open(os.path.join(m2m, ejfile)))
            for ch, (a, b) in enumerate(pairs, 1):
                d = fem_dir(cid, tag, ch)
                sv = os.path.join(d, "subject_volumes")
                if os.path.isdir(sv) and os.listdir(sv):
                    print(f"  [{i}] {cid} {tag} ch{ch} (cached)", flush=True)
                    continue
                if a not in ej or b not in ej:
                    print(f"  [{i}] {cid} {tag} ch{ch} - missing electrode, skip", flush=True)
                    continue
                os.makedirs(d, exist_ok=True)
                S = sim_struct.SESSION()
                S.subpath, S.pathfem = m2m, d
                S.fields, S.map_to_vol, S.open_in_gmsh = "eE", True, False
                t = S.add_tdcslist()
                t.currents = [CURRENT, -CURRENT]
                for lab in (a, b):
                    e = t.add_electrode()
                    e.channelnr = 1 if lab == a else 2
                    e.centre = ej[lab]
                    e.shape = "ellipse"
                    e.dimensions = [2 * ELEC_RADIUS_MM, 2 * ELEC_RADIUS_MM]
                    e.thickness = [2.0]
                t0 = time.time()
                try:
                    run_simnibs(S)
                    print(f"  [{i}] {cid} {tag} ch{ch} OK ({(time.time()-t0)/60:.1f} min)",
                          flush=True)
                except Exception as ex:
                    print(f"  [{i}] {cid} {tag} ch{ch} FAILED {type(ex).__name__}: {ex}",
                          flush=True)


# ---------------------------------------------------------------- stage: compare
def stage_compare(n, dx=2.0):
    import nibabel as nib
    from scipy.stats import spearmanr
    from spicehead import mri
    from spicehead.electrodes import Electrode, contact_nodes, pair_injection
    from spicehead.field import efield_components
    from montage_search import snap, roi_mask, ROI_RADIUS_MM, SCALP, SCORE, ROI_ORDER
    from v21_head_convergence import make_solver
    from v21_simnibs_compare import load_simnibs_E, envelope, resample_to
    from v21_agreement_raw import stats

    os.makedirs(CACHE, exist_ok=True)
    rows, ranks = [], []
    for si, (cid, m2m, age, site) in enumerate(subjects(n), 1):
        # --- SPICEhead: one model, one factorization, all montages
        cp = os.path.join(CACHE, f"{cid}__dx{dx:g}.pkl")
        if os.path.exists(cp):
            with open(cp, "rb") as fh:
                sp_env, saff = pickle.load(fh)
        else:
            model, saff = mri.head_model(   # v21/v22: nearest-label network; v23: homogenized
                os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"), dx)
            labels = model.labels
            scalp = np.argwhere(labels == SCALP)
            solver, _ = make_solver(model.G)
            ejs = {f: json.load(open(os.path.join(m2m, f)))
                   for f in {v[1] for v in MONTAGES.values()}}
            sp_env = {}
            for tag, (pairs, ejfile) in MONTAGES.items():
                ej = ejs[ejfile]
                F = []
                for (a, b) in pairs:
                    el = []
                    for lab in (a, b):
                        v = snap(scalp, mri.world_to_voxel(saff, ej[lab]))
                        el.append(Electrode(contact_nodes(
                            model, tuple(int(x) for x in v),
                            radius_nodes=ELEC_RADIUS_MM / dx), lab))
                    V = solver(pair_injection(model, el[0], el[1], CURRENT))
                    F.append(tuple(np.nan_to_num(f).astype(np.float32)
                                   for f in efield_components(model, V)))
                sp_env[tag] = np.nan_to_num(envelope(F[0], F[1])).astype(np.float32)
            with open(cp + ".part", "wb") as fh:
                pickle.dump((sp_env, saff), fh, protocol=4)
            os.replace(cp + ".part", cp)

        # --- FEM per montage, on the FEM grid
        per_tag = {}
        for tag in MONTAGES:
            try:
                E1, aff, sh = load_simnibs_E(fem_dir(cid, tag, 1))
                E2, _, _ = load_simnibs_E(fem_dir(cid, tag, 2))
            except Exception:
                continue
            per_tag[tag] = (np.nan_to_num(envelope(E1, E2)), aff, sh)
        if len(per_tag) < 2:
            print(f"[{si}] {cid}: <2 montages with FEM output - skip", flush=True)
            continue

        aff = next(iter(per_tag.values()))[1]
        sh = next(iter(per_tag.values()))[2]
        li = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
        gm = resample_to(np.isin(np.asanyarray(li.dataobj), SCORE).astype(np.float32),
                         li.affine, aff, sh) > 0.5
        rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
        gidx = np.argwhere(gm)
        gw = (aff[:3, :3] @ gidx.T).T + aff[:3, 3]
        rmask = {r: roi_mask(sh, gidx, gw, rois[r], ROI_RADIUS_MM) for r in ROI_ORDER if r in rois}

        tgt = {}
        for tag, (fem, _, _) in per_tag.items():
            spg = resample_to(sp_env[tag], saff, aff, sh)
            x, y = spg[gm].astype(float), fem[gm].astype(float)
            k = np.isfinite(x) & np.isfinite(y) & (y > 0)
            st = stats(x[k], y[k])
            st.update(subject=cid, site=site, montage=tag, dx_mm=dx)
            rows.append(st)
            tgt[tag] = {r: (float(np.median(spg[m_])), float(np.median(fem[m_])))
                        for r, m_ in rmask.items()}
            print(f"[{si}] {cid} {tag:<9} slope={st['slope']:.3f} r={st['r']:.3f} "
                  f"CCC={st['ccc']:.3f} bias={st['raw_bias']:+.5f}", flush=True)

        # --- rank preservation: do both solvers order the montages the same way per target?
        tags = sorted(tgt)
        for r in rmask:
            sp = [tgt[t][r][0] for t in tags]
            fe = [tgt[t][r][1] for t in tags]
            rho = spearmanr(sp, fe).statistic if len(tags) > 2 else float("nan")
            ranks.append(dict(subject=cid, roi=r, dx_mm=dx, n_montages=len(tags),
                              spearman=round(float(rho), 4),
                              best_spicehead=tags[int(np.argmax(sp))],
                              best_fem=tags[int(np.argmax(fe))],
                              agree=tags[int(np.argmax(sp))] == tags[int(np.argmax(fe))]))

    d = os.path.join(config.RESULTS, "v21_multimontage")
    os.makedirs(d, exist_ok=True)
    write_merged(os.path.join(d, "agreement_by_montage.csv"), rows,
                 ("subject", "montage", "dx_mm"))
    write_merged(os.path.join(d, "rank_preservation.csv"), ranks, ("subject", "roi", "dx_mm"))

    print("\n=== slope stability across montages ===")
    print(f"{'montage':<10}{'n':>4}{'slope':>9}{'r':>8}{'CCC':>8}{'raw bias':>11}")
    for tag in MONTAGES:
        s = [r for r in rows if r["montage"] == tag]
        if not s:
            continue
        f = lambda k: float(np.mean([r[k] for r in s]))
        print(f"{tag:<10}{len(s):>4}{f('slope'):>9.3f}{f('r'):>8.3f}{f('ccc'):>8.3f}"
              f"{f('raw_bias'):>+11.5f}")
    sl = [float(np.mean([r["slope"] for r in rows if r["montage"] == t]))
          for t in MONTAGES if any(r["montage"] == t for r in rows)]
    if sl:
        print(f"\nslope range across montages: {min(sl):.3f} to {max(sl):.3f} "
              f"(spread {max(sl)-min(sl):.3f})")
    if ranks:
        ag = sum(1 for r in ranks if r["agree"])
        rho = np.nanmean([r["spearman"] for r in ranks])
        print(f"best-montage agreement: {ag}/{len(ranks)} subject-target pairs")
        print(f"mean Spearman of montage ordering: {rho:.3f}")
    print("wrote", d)


# ---------------------------------------------------------------- stage: focality
def stage_focality(n, dx=2.0):
    """Rank preservation on F = T/O rather than on target intensity.

    Both solvers are scored identically on the FEM grid: T = median envelope in the 10 mm ROI,
    O = 99th percentile of grey-matter envelope farther than 20 mm from every centre of that ROI.
    F is a ratio, so the montage-dependent magnitude bias found by --compare largely divides out —
    whether it divides out *enough* to preserve the ordering is the question.
    """
    import nibabel as nib
    from scipy.stats import spearmanr
    from montage_search import roi_mask, ROI_RADIUS_MM, EXCLUDE_MM, SCORE, ROI_ORDER
    from v21_simnibs_compare import load_simnibs_E, envelope, resample_to

    rows, ranks = [], []
    for si, (cid, m2m, age, site) in enumerate(subjects(n), 1):
        cp = os.path.join(CACHE, f"{cid}__dx{dx:g}.pkl")
        if not os.path.exists(cp):
            print(f"[{si}] {cid}: no cached SPICEhead envelope - run --compare first", flush=True)
            continue
        with open(cp, "rb") as fh:
            sp_env, saff = pickle.load(fh)

        per_tag = {}
        for tag in MONTAGES:
            try:
                E1, aff, sh = load_simnibs_E(fem_dir(cid, tag, 1))
                E2, _, _ = load_simnibs_E(fem_dir(cid, tag, 2))
            except Exception:
                continue
            per_tag[tag] = (np.nan_to_num(envelope(E1, E2)), aff, sh)
        if len(per_tag) < 2:
            continue
        aff, sh = per_tag[next(iter(per_tag))][1], per_tag[next(iter(per_tag))][2]

        li = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
        gm = resample_to(np.isin(np.asanyarray(li.dataobj), SCORE).astype(np.float32),
                         li.affine, aff, sh) > 0.5
        rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
        gidx = np.argwhere(gm)
        gw = (aff[:3, :3] @ gidx.T).T + aff[:3, 3]
        rmask = {r: roi_mask(sh, gidx, gw, rois[r], ROI_RADIUS_MM) for r in ROI_ORDER if r in rois}
        omask = {r: gm & ~roi_mask(sh, gidx, gw, rois[r], EXCLUDE_MM) for r in rmask}

        def foc(env, r):
            T = float(np.median(env[rmask[r]]))
            O = float(np.percentile(env[omask[r]], 99))
            return (T / O if O > 0 else float("nan")), T, O

        F = {}
        for tag, (fem, _, _) in per_tag.items():
            spg = resample_to(sp_env[tag], saff, aff, sh)
            F[tag] = {}
            for r in rmask:
                fs, Ts, Os = foc(spg, r)
                ff, Tf, Of = foc(fem, r)
                F[tag][r] = (fs, ff)
                # 7 significant digits: rounding to 4 decimals can manufacture ties between
                # montages when choices are recomputed from this file
                rows.append(dict(subject=cid, site=site, montage=tag, roi=r, dx_mm=dx,
                                 F_spicehead=float(f"{fs:.7g}"), F_fem=float(f"{ff:.7g}"),
                                 T_spicehead=float(f"{Ts:.7g}"), T_fem=float(f"{Tf:.7g}"),
                                 O_spicehead=float(f"{Os:.7g}"), O_fem=float(f"{Of:.7g}"),
                                 F_ratio=float(f"{fs / ff:.7g}") if ff else None))
            print(f"[{si}] {cid} {tag:<9} " +
                  " ".join(f"{r.split('_')[0][:4]}:{F[tag][r][0]:.2f}/{F[tag][r][1]:.2f}"
                           for r in rmask), flush=True)

        tags = sorted(F)
        for r in rmask:
            sp = [F[t][r][0] for t in tags]
            fe = [F[t][r][1] for t in tags]
            rho = spearmanr(sp, fe).statistic if len(tags) > 2 else float("nan")
            # sensitivity (review R4): the same choice with the inferior chain removed from both
            # methods' candidates, because 13 of its 16 reference runs carry the SimNIBS warning
            t3 = [t for t in tags if t != "INFERIOR"]
            sp3 = [F[t][r][0] for t in t3]
            fe3 = [F[t][r][1] for t in t3]
            ranks.append(dict(subject=cid, roi=r, dx_mm=dx, n_montages=len(tags),
                              spearman=round(float(rho), 4),
                              best_spicehead=tags[int(np.argmax(sp))],
                              best_fem=tags[int(np.argmax(fe))],
                              agree=tags[int(np.argmax(sp))] == tags[int(np.argmax(fe))],
                              best3_spicehead=t3[int(np.argmax(sp3))],
                              best3_fem=t3[int(np.argmax(fe3))],
                              agree3=t3[int(np.argmax(sp3))] == t3[int(np.argmax(fe3))]))

    d = os.path.join(config.RESULTS, "v21_multimontage")
    os.makedirs(d, exist_ok=True)
    write_merged(os.path.join(d, "focality_by_montage.csv"), rows,
                 ("subject", "montage", "roi", "dx_mm"))
    write_merged(os.path.join(d, "rank_preservation_focality.csv"), ranks,
                 ("subject", "roi", "dx_mm"))

    print("\n=== focality by montage (mean over subjects and targets) ===")
    print(f"{'montage':<10}{'n':>4}{'F SPICEhead':>13}{'F FEM':>9}{'F ratio':>9}")
    for tag in MONTAGES:
        s = [r for r in rows if r["montage"] == tag]
        if not s:
            continue
        f = lambda k: float(np.nanmean([r[k] for r in s if r[k] is not None]))
        print(f"{tag:<10}{len(s):>4}{f('F_spicehead'):>13.3f}{f('F_fem'):>9.3f}{f('F_ratio'):>9.3f}")
    fr = [r["F_ratio"] for r in rows if r["F_ratio"] is not None]
    print(f"\nF_SPICEhead / F_FEM over all {len(fr)} subject-montage-target cells: "
          f"mean {np.mean(fr):.3f}, sd {np.std(fr, ddof=1):.3f}, "
          f"range {min(fr):.3f} to {max(fr):.3f}")
    if ranks:
        ag = sum(1 for r in ranks if r["agree"])
        print(f"best-montage-by-focality agreement: {ag}/{len(ranks)} subject-target pairs")
        print(f"mean Spearman of focality ordering: {np.nanmean([r['spearman'] for r in ranks]):.3f}")
        print(f"\n{'ROI':<15}{'agree':>8}{'mean rho':>10}")
        for r_ in ROI_ORDER:
            g = [x for x in ranks if x["roi"] == r_]
            if g:
                print(f"{r_:<15}{sum(x['agree'] for x in g):>5}/{len(g):<2}"
                      f"{np.nanmean([x['spearman'] for x in g]):>10.3f}")
    print("wrote", d)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dx = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    if sys.argv[1] == "--sim":
        stage_sim(n)
    elif sys.argv[1] == "--compare":
        stage_compare(n, dx)
    elif sys.argv[1] == "--focality":
        stage_focality(n, dx)
    else:
        raise SystemExit(__doc__)
