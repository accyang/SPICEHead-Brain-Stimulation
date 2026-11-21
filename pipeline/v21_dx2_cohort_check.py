# -*- coding: utf-8 -*-
"""Partial 2-mm validation of the 5-mm cohort optimization — fourth-round review, comment 2.

The 1-mm multi-montage experiment showed ranking quality improves strongly with resolution, which
sharpened rather than settled the reviewer's concern: the 300-subject optimization itself runs at
5 mm, a resolution the paper's own convergence study shows is far from converged (r = 0.61 against
1 mm, up to 19% focality forfeited by individual selection). Leave-one-site-out establishes the
reproducibility of the 5-mm objective, not that the 5-mm winner would win at finer resolution.

The reviewer's prescription, followed exactly: take the top candidates per target from the 5-mm
cohort ranking and re-score them at 2 mm in a representative subject subset — no need to recompute
all 11,600 x 300.

  candidates   top K = 25 montages per target, ranked by median focality across all 300 subjects
               at 5 mm (the same statistic Table 3 uses, from the per-subject score caches)
  subjects     5 per acquisition site x 4 sites = 20, deterministic seed
  re-scoring   identical metric at dx = 2 mm: F = median(env in ROI, GM) / p99(env in GM > 20 mm),
               rotation-invariant maximum-over-direction envelope

Stages (resumable; per-subject JSON cache):
  --score [n_per_site]     solve and score the sampled subjects
  --collect                aggregate: 2-mm vs 5-mm ranking of the same candidates

Reported per target: Spearman of the candidate ordering (2 mm cohort-median vs 5 mm, both over the
same 20 subjects and vs the full-300 ranking), whether the 5-mm winner stays the 2-mm winner, its
2-mm rank, and the focality forfeited at 2 mm by deploying the 5-mm winner.

  python pipeline/v21_dx2_cohort_check.py --score 5
  python pipeline/v21_dx2_cohort_check.py --collect

Full sweep (no shortlist). The shortlist cannot see a 2-mm optimum that ranked below 25th at
5 mm. These stages re-score EVERY two-pair montage at 2 mm instead, and cost less than the
shortlist: each pair field is the difference of two single-electrode fields against a common
reference, E(a,b) = E(a,Cz) - E(b,Cz), exactly (pair_injection spreads each electrode's current
over its own nodes whatever its partner), so 18 solves give all pair fields where the
shortlist needed 98. Each subject also solves two pairs directly and fails if the difference
exceeds BASIS_TOL.

  --score-full [n_per_site] [--limit K]   solve and score every montage (K: first K subjects)
  --collect-full                          2-mm vs 5-mm over all montages, the 5-mm rank of the
                                          2-mm winner, the loss the top-25 shortlist incurred,
                                          and an exact cross-check against the shortlist run

  python pipeline/v21_dx2_cohort_check.py --score-full 5
  python pipeline/v21_dx2_cohort_check.py --collect-full

Data roots can be moved without editing this file: SPICEHEAD_CHARM_ROOT, SPICEHEAD_COHORT_CSV,
SPICEHEAD_DX5_CACHE (the 300 per-subject 5-mm caches), SPICEHEAD_T7_SUBJECTS (pinned subject
list) and SPICEHEAD_T7_SHORTLIST_DIR (the shortlist run's per-subject JSONs, for the
cross-check; defaults to this script's own output directory).
"""
import csv
import json
import os
import pickle
import sys
import time
from itertools import combinations

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)

from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components
from spicehead.envelope_fast import envelope_max_direction_flat
from montage_search import (snap, roi_mask, ELEC_RADIUS_MM, ROI_RADIUS_MM, EXCLUDE_MM,
                            MIN_DIST_MM, SCALP, SCORE, ROI_ORDER)
from run_montage_cohort import montage_str, resolve_m2m
from v21_head_convergence import make_solver

CACHE5 = os.environ.get("SPICEHEAD_DX5_CACHE", os.path.join(
    config.RESULTS, "montage_optimization", "cohort_gm300_dx5_maxdir", "_subject_scores"))
OUT = os.path.join(config.RESULTS, "v21_dx2_check")
CSVP = os.environ.get("SPICEHEAD_COHORT_CSV", r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv")
CH = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")
DX = 2.0
K = 25            # candidates per target (reviewer: 20-50)
MIN_FRAC = 0.8
SEED = 0


def canon(ms):
    """Order-independent montage label.

    Montage keys are frozensets of two electrode pairs, so `montage_str` emits the two pairs in
    whatever order set iteration happens to give — and because string hashing is randomised per
    process, that order differs between the scoring run and this aggregation run. Sorting within
    and across pairs makes the label reproducible, which is what lets the two stages match.
    """
    pairs = [tuple(sorted(p.split("+"))) for p in ms.split(" / ")]
    return " / ".join("+".join(p) for p in sorted(pairs))


def load5():
    """{cid: {roi: {frozenset_key: F}}} for all cached 5-mm subjects."""
    data = {}
    for f in sorted(os.listdir(CACHE5)):
        if f.endswith("_maxdir.pkl"):
            cid = f.split("__dx")[0]
            with open(os.path.join(CACHE5, f), "rb") as fh:
                data[cid] = pickle.load(fh)
    return data


def candidates(data):
    """Top-K montage keys per ROI by median 5-mm focality over all subjects."""
    ids = sorted(data)
    top = {}
    for roi in ROI_ORDER:
        acc = {}
        for c in ids:
            for k, v in data[c].get(roi, {}).items():
                acc.setdefault(k, []).append(v)
        ranked = [(float(np.median(v)), k) for k, v in acc.items()
                  if len(v) >= MIN_FRAC * len(ids)]
        ranked.sort(key=lambda x: -x[0])
        top[roi] = [k for _, k in ranked[:K]]
    return top


def sample_subjects(data, n_per_site):
    """Five subjects per acquisition site, deterministic under SEED.

    The pool is filtered by which subjects have CHARM inputs on disk, so the same seed draws a
    different sample from a partial copy of the cohort. SPICEHEAD_T7_SUBJECTS therefore lets a
    re-run pin the published sample: point it at a CSV with a CaseID column (or a JSON list) and
    the listed subjects are used in file order. Unset, the original sampling is unchanged.
    """
    pin = os.environ.get("SPICEHEAD_T7_SUBJECTS")
    if pin:
        if pin.lower().endswith(".json"):
            picked = json.load(open(pin, encoding="utf-8"))
        else:
            picked = [r["CaseID"] for r in csv.DictReader(open(pin, encoding="utf-8-sig"))]
        print(f"  subjects pinned from {os.path.basename(pin)}: {len(picked)}", flush=True)
        return picked
    site = {r["CaseID"]: r["Site"] for r in csv.DictReader(open(CSVP, encoding="utf-8-sig"))}
    by = {}
    for cid in sorted(data):
        m2m = resolve_m2m(CH, cid)
        if (os.path.exists(os.path.join(m2m, "electrodes_1020.json"))
                and os.path.exists(os.path.join(m2m, "roi_coords.json"))):
            by.setdefault(site.get(cid, "?"), []).append(cid)
    rng = np.random.default_rng(SEED)
    picked = []
    for s in sorted(by):
        pool = by[s]
        take = min(n_per_site, len(pool))
        picked += list(rng.choice(pool, size=take, replace=False))
    return picked


def score_subject_at_2mm(cid, cands):
    """F at dx=2 for every candidate montage, per ROI. One factorization, union-of-pairs solves."""
    m2m = resolve_m2m(CH, cid)
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, affine = mri.head_model(seg, DX)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    gm = np.isin(labels, SCORE)
    gm_idx = np.argwhere(gm)
    gm_world = (affine[:3, :3] @ gm_idx.T).T + affine[:3, 3]
    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    present = [r for r in ROI_ORDER if r in rois]
    roi_sel = {r: roi_mask(labels.shape, gm_idx, gm_world, rois[r], ROI_RADIUS_MM)[gm]
               for r in present}
    out_sel = {r: ~roi_mask(labels.shape, gm_idx, gm_world, rois[r], EXCLUDE_MM)[gm]
               for r in present}

    scalp_vox = np.argwhere(labels == SCALP)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    pairs = sorted({p for roi in present for key in cands[roi] for p in tuple(key)})
    elecs = {}
    for p in pairs:
        for lab in p:
            if lab not in elecs:
                v = snap(scalp_vox, mri.world_to_voxel(affine, ej[lab]))
                elecs[lab] = Electrode(contact_nodes(
                    model, tuple(int(x) for x in v), radius_nodes=ELEC_RADIUS_MM / DX), lab)

    solver, ncomp = make_solver(model.G)
    print(f"    {cid}: {model.n_nodes} nodes ({ncomp} components), {len(pairs)} distinct pairs, "
          f"{int(gm.sum())} GM voxels", flush=True)
    flatE = {}
    t0 = time.time()
    for i, (a, b) in enumerate(pairs, 1):
        V = solver(pair_injection(model, elecs[a], elecs[b], 1e-3))
        Ex, Ey, Ez = efield_components(model, V)
        flatE[(a, b)] = tuple(np.nan_to_num(c[gm]).astype(np.float32) for c in (Ex, Ey, Ez))
        if i % 20 == 0 or i == len(pairs):
            print(f"      pair {i}/{len(pairs)} ({(time.time()-t0)/i:.0f} s/pair)", flush=True)

    res = {}
    for roi in present:
        res[roi] = {}
        for key in cands[roi]:
            pa, pb = tuple(key)
            a, b = flatE[pa], flatE[pb]
            env = envelope_max_direction_flat(a[0], a[1], a[2], b[0], b[1], b[2])
            rv = env[roi_sel[roi]]
            if rv.size == 0:
                continue
            O = float(np.percentile(env[out_sel[roi]], 99))
            res[roi][montage_str(key)] = dict(
                F=(float(np.median(rv)) / O) if O > 0 else 0.0,
                T=float(np.median(rv)), O=O)
    return res


def stage_score(n_per_site):
    os.makedirs(OUT, exist_ok=True)
    data = load5()
    cands = candidates(data)
    with open(os.path.join(OUT, "candidates.json"), "w", encoding="utf-8") as fh:
        json.dump({r: [montage_str(k) for k in cands[r]] for r in cands}, fh, indent=1)
    subs = sample_subjects(data, n_per_site)
    print(f"{len(subs)} subjects, K={K} candidates/target, dx={DX:g} mm", flush=True)
    for i, cid in enumerate(subs, 1):
        p = os.path.join(OUT, f"{cid}__dx{DX:g}.json")
        if os.path.exists(p):
            print(f"  [{i}/{len(subs)}] {cid} (cached)", flush=True)
            continue
        t = time.time()
        try:
            res = score_subject_at_2mm(cid, cands)
        except Exception as ex:
            print(f"  [{i}/{len(subs)}] {cid} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        with open(p + ".part", "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        os.replace(p + ".part", p)
        print(f"  [{i}/{len(subs)}] {cid} OK ({(time.time()-t)/60:.1f} min)", flush=True)


def stage_collect():
    from scipy.stats import spearmanr
    data = load5()
    cands = candidates(data)
    subs, per2 = [], {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith(f"__dx{DX:g}.json"):
            cid = f.split("__dx")[0]
            subs.append(cid)
            per2[cid] = json.load(open(os.path.join(OUT, f), encoding="utf-8"))
    print(f"collected {len(subs)} subjects\n")

    rows = []
    print(f"{'ROI':<15}{'n':>3}{'rho vs 300':>12}{'rho same-20':>13}{'winner kept':>13}"
          f"{'5mm winner rank@2mm':>21}{'forfeit@2mm':>13}")
    for roi in ROI_ORDER:
        names = [canon(montage_str(k)) for k in cands[roi]]   # 5-mm rank order, best first
        # per-subject 2-mm scores, re-keyed canonically (they were written by another process)
        sc2 = {c: {canon(m): v for m, v in per2[c].get(roi, {}).items()} for c in subs}
        med2, med5s = {}, {}
        for m in names:
            v2 = [sc2[c][m]["F"] for c in subs if m in sc2[c]]
            if len(v2) >= MIN_FRAC * len(subs):
                med2[m] = float(np.median(v2))
        for key in cands[roi]:
            v5 = [data[c][roi][key] for c in subs if key in data[c].get(roi, {})]
            if len(v5) >= MIN_FRAC * len(subs):
                med5s[canon(montage_str(key))] = float(np.median(v5))
        common = [m for m in names if m in med2 and m in med5s]
        if len(common) < 3:
            continue
        r300 = [names.index(m) + 1 for m in common]           # full-300 5-mm rank
        rho_300 = float(spearmanr([-r for r in r300], [med2[m] for m in common]).statistic)
        rho_s20 = float(spearmanr([med5s[m] for m in common],
                                  [med2[m] for m in common]).statistic)
        best2 = max(common, key=lambda m: med2[m])
        win5 = names[0]
        if win5 not in common:
            # the 5-mm winner was scored in too few of the sampled subjects to compare; say so
            # rather than reporting a rank for a montage we did not evaluate
            print(f"{roi:<15}{len(common):>3}  5-mm winner {win5} not evaluable in this subset")
            continue
        kept = (win5 == best2)
        order = sorted(common, key=lambda m: -med2[m])
        rank2 = order.index(win5) + 1
        forfeit = 100 * (med2[best2] - med2[win5]) / med2[best2]
        # top-5 overlap between resolutions (fifth-round review, comment 9)
        top5_5 = set(names[:5])
        top5_2 = set(order[:5])
        jac = len(top5_5 & top5_2) / len(top5_5 | top5_2)
        rows.append(dict(roi=roi, n_candidates=len(common), n_subjects=len(subs),
                         spearman_vs_300rank=round(rho_300, 4),
                         spearman_same_subjects=round(rho_s20, 4),
                         winner_kept=bool(kept), rank_of_5mm_winner_at_2mm=rank2,
                         forfeit_pct_at_2mm=round(float(forfeit), 3),
                         top5_jaccard=round(float(jac), 4),
                         winner_5mm=win5, winner_2mm=best2))
        print(f"{roi:<15}{len(common):>3}{rho_300:>12.3f}{rho_s20:>13.3f}"
              f"{('yes' if kept else 'NO'):>13}{rank2:>21}{forfeit:>12.2f}%{jac:>8.2f}")

    with open(os.path.join(OUT, "dx2_vs_dx5.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    kept = sum(r["winner_kept"] for r in rows)
    ff = [r["forfeit_pct_at_2mm"] for r in rows]
    print(f"\n5-mm winner kept at 2 mm in {kept}/{len(rows)} targets; "
          f"median forfeit {np.median(ff):.2f}%, worst {max(ff):.2f}%")
    print(f"mean Spearman vs full-300 ranking {np.mean([r['spearman_vs_300rank'] for r in rows]):.3f}")
    print("wrote", os.path.join(OUT, "dx2_vs_dx5.csv"))


# ---------------------------------------------------------------- full sweep (no shortlist)
REF = "Cz"
OUT_FULL = os.path.join(config.RESULTS, "v21_dx2_check_full")
BASIS_TOL = 1e-6        # max|basis - direct| / max|direct| allowed on the per-subject check
CHECK_PAIRS = (("P8", "T8"), ("F7", "T7"))   # the Table 3 thalamic channels


def g7(x):
    return float(f"{x:.7g}")


def run_subjects(n_per_site):
    """The Table 7 subjects without loading any cache contents.

    sample_subjects() only iterates the cache's subject ids, so the filenames reproduce the
    original draw; a pinned list (SPICEHEAD_T7_SUBJECTS) needs no cache at all.
    """
    if os.environ.get("SPICEHEAD_T7_SUBJECTS"):
        return [str(c) for c in sample_subjects({}, n_per_site)]
    ids = {f.split("__dx")[0]: None for f in sorted(os.listdir(CACHE5))
           if f.endswith("_maxdir.pkl")}
    return [str(c) for c in sample_subjects(ids, n_per_site)]


def score_subject_full(cid):
    """F, T and O at dx=2 for every two-pair montage and ROI, from single-electrode fields."""
    m2m = resolve_m2m(CH, cid)
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, affine = mri.head_model(seg, DX)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    gm = np.isin(labels, SCORE)
    gm_idx = np.argwhere(gm)
    gm_world = (affine[:3, :3] @ gm_idx.T).T + affine[:3, 3]
    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    present = [r for r in ROI_ORDER if r in rois]
    roi_sel = {r: roi_mask(labels.shape, gm_idx, gm_world, rois[r], ROI_RADIUS_MM)[gm]
               for r in present}
    out_sel = {r: ~roi_mask(labels.shape, gm_idx, gm_world, rois[r], EXCLUDE_MM)[gm]
               for r in present}

    scalp_vox = np.argwhere(labels == SCALP)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs = sorted(ej)
    if REF not in labs:
        raise RuntimeError(f"reference electrode {REF} missing from electrodes_1020.json")
    epos, elecs = {}, {}
    for lab in labs:
        v = snap(scalp_vox, mri.world_to_voxel(affine, ej[lab]))
        epos[lab] = v.astype(float)
        elecs[lab] = Electrode(contact_nodes(
            model, tuple(int(x) for x in v), radius_nodes=ELEC_RADIUS_MM / DX), lab)
    pairs = [(a, b) for a, b in combinations(labs, 2)
             if np.linalg.norm(epos[a] - epos[b]) * DX >= MIN_DIST_MM]

    solver, ncomp = make_solver(model.G)
    print(f"    {cid}: {model.n_nodes} nodes ({ncomp} components), {len(labs)} electrodes, "
          f"{len(pairs)} valid pairs, {int(gm.sum())} GM voxels", flush=True)

    def field(inj):
        return np.stack([np.nan_to_num(c[gm]) for c in efield_components(model, solver(inj))])

    t0 = time.time()
    basis = {}
    others = [lab for lab in labs if lab != REF]
    for i, lab in enumerate(others, 1):
        basis[lab] = field(pair_injection(model, elecs[lab], elecs[REF], 1e-3))
        if i % 6 == 0 or i == len(others):
            print(f"      basis {i}/{len(others)} ({(time.time()-t0)/i:.0f} s/solve)", flush=True)
    solve_s = time.time() - t0

    def pair_field(a, b):
        return (basis[a] if a != REF else 0.0) - (basis[b] if b != REF else 0.0)

    check = {}
    probes = [p for p in CHECK_PAIRS if p in pairs] or [p for p in pairs if REF not in p][:2]
    for a, b in probes:
        direct = field(pair_injection(model, elecs[a], elecs[b], 1e-3))
        rel = float(np.abs(pair_field(a, b) - direct).max() / np.abs(direct).max())
        check[f"{a}-{b}"] = rel
        if rel > BASIS_TOL:
            raise RuntimeError(f"superposition check failed for {a}-{b}: {rel:.2e} > {BASIS_TOL}")
    print(f"      superposition check: " +
          ", ".join(f"{k} {v:.1e}" for k, v in check.items()), flush=True)

    flatE = {p: tuple(c.astype(np.float32) for c in pair_field(*p)) for p in pairs}
    del basis

    montages = [(pa, pb) for pa, pb in combinations(pairs, 2) if len(set(pa) | set(pb)) == 4]
    res = {r: {} for r in present}
    t1 = time.time()
    for i, (pa, pb) in enumerate(montages, 1):
        a, b = flatE[pa], flatE[pb]
        env = envelope_max_direction_flat(a[0], a[1], a[2], b[0], b[1], b[2])
        label = canon(montage_str(frozenset((pa, pb))))
        for r in present:
            rv = env[roi_sel[r]]
            if rv.size == 0:
                continue
            T = float(np.median(rv))
            O = float(np.percentile(env[out_sel[r]], 99))
            res[r][label] = [g7(T / O if O > 0 else 0.0), g7(T), g7(O)]
        if i % 2000 == 0 or i == len(montages):
            print(f"      scored {i}/{len(montages)} montages ({time.time()-t1:.0f} s)",
                  flush=True)
    return dict(subject=cid, dx=DX, ref=REF, n_nodes=int(model.n_nodes),
                n_gm_voxels=int(gm.sum()), n_pairs=len(pairs), n_montages=len(montages),
                basis_check=check, solve_s=round(solve_s, 1),
                score_s=round(time.time() - t1, 1), rois=res)


def stage_score_full(n_per_site, limit=0, shard=(0, 1)):
    os.makedirs(OUT_FULL, exist_ok=True)
    subs = run_subjects(n_per_site)
    if limit:
        subs = subs[:limit]
    k, n = shard
    if n > 1:
        subs = [c for i, c in enumerate(subs) if i % n == k]
        print(f"shard {k}/{n}: {len(subs)} subjects", flush=True)
    print(f"{len(subs)} subjects, ALL montages, dx={DX:g} mm, reference {REF}", flush=True)
    for i, cid in enumerate(subs, 1):
        p = os.path.join(OUT_FULL, f"{cid}__dx{DX:g}_full.json")
        if os.path.exists(p):
            print(f"  [{i}/{len(subs)}] {cid} (cached)", flush=True)
            continue
        t = time.time()
        try:
            res = score_subject_full(cid)
        except Exception as ex:
            print(f"  [{i}/{len(subs)}] {cid} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        with open(p + ".part", "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        os.replace(p + ".part", p)
        print(f"  [{i}/{len(subs)}] {cid} OK ({(time.time()-t)/60:.1f} min)", flush=True)


def load5_matrix():
    """5-mm F for every cached subject and montage: (ids, labels, {roi: [n_subj, n_mont]}).

    Streams one cache at a time into float32 arrays (NaN = not scored). load5() keeps all 300
    caches as Python dicts, which is fine for a 25-montage shortlist but several GB when every
    montage is needed.
    """
    files = sorted(f for f in os.listdir(CACHE5) if f.endswith("_maxdir.pkl"))
    ids = [f.split("__dx")[0] for f in files]
    keylab, labcol, labels = {}, {}, []
    parts = {r: [] for r in ROI_ORDER}
    for f in files:
        with open(os.path.join(CACHE5, f), "rb") as fh:
            d = pickle.load(fh)

        def col(k):
            lab = keylab.get(k)
            if lab is None:
                lab = keylab[k] = canon(montage_str(k))
            j = labcol.get(lab)
            if j is None:
                j = labcol[lab] = len(labels)
                labels.append(lab)
            return j

        for r in ROI_ORDER:
            dr = d.get(r, {})
            parts[r].append((np.fromiter((col(k) for k in dr), np.int32, len(dr)),
                             np.fromiter(dr.values(), np.float32, len(dr))))
        del d
    F = {}
    for r in ROI_ORDER:
        A = np.full((len(ids), len(labels)), np.nan, np.float32)
        for s, (j, v) in enumerate(parts[r]):
            A[s, j] = v
        F[r] = A
    return ids, labels, F


def cohort_median(A, frac=MIN_FRAC):
    """Column medians over subjects, NaN where fewer than frac of them scored the montage."""
    n = np.isfinite(A).sum(0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(A, axis=0)
    med[n < frac * A.shape[0]] = np.nan
    return med


def _pct(a, b):
    return 100.0 * (a - b) / a if a > 0 else float("nan")


def _summ(v):
    v = [x for x in v if np.isfinite(x)]
    if not v:
        return dict(n=0)
    q = np.percentile(v, [25, 50, 75])
    return dict(n=len(v), median=round(float(q[1]), 3), q1=round(float(q[0]), 3),
                q3=round(float(q[2]), 3), max=round(float(max(v)), 3),
                n_over_1pct=int(sum(x > 1 for x in v)), n_over_5pct=int(sum(x > 5 for x in v)))


def stage_collect_full(n_per_site=5, limit=0):
    """The 2-mm full sweep against the 5-mm ranking, at cohort, same-subject and subject level.

    Candidate set. For each target a montage is admissible when it has a finite 300-subject 5-mm
    median (scored in >= MIN_FRAC of the cached subjects) AND a finite 2-mm median over the run's
    subjects (scored in >= MIN_FRAC of them). Every optimum and regret below is taken over that
    set; the montages it excludes are counted and reported. The shortlist is the published one:
    the top K of the 300-subject 5-mm ranking.

    Cohort level: the 5-mm 300-subject winner against the optimum of the 2-mm cohort-median
    focality. Same-subject level: the optimum of the SAME subjects' 5-mm medians, so resolution is
    not confounded with sampling (300 versus 20 heads). Subject level: each subject's own 2-mm
    optimum and the regret, relative to it, of the 5-mm cohort winner, the 2-mm cohort winner,
    the best shortlisted montage and the subject's own 5-mm optimum.

    A run is complete only when every expected subject has a result; otherwise every output is
    labelled PARTIAL and written under a _partial file name.
    """
    from scipy.stats import spearmanr
    expected = run_subjects(n_per_site)
    if limit:
        expected = expected[:limit]
    per2 = {}
    for cid in expected:
        p = os.path.join(OUT_FULL, f"{cid}__dx{DX:g}_full.json")
        if os.path.exists(p):
            per2[cid] = json.load(open(p, encoding="utf-8"))
    subs = [c for c in expected if c in per2]
    missing = [c for c in expected if c not in per2]
    if not subs:
        raise SystemExit(f"no full-sweep results for the expected subjects in {OUT_FULL}")
    status = ("COMPLETE" if not missing else
              f"PARTIAL ({len(subs)}/{len(expected)} subjects; missing {', '.join(missing)})")
    suffix = "" if not missing else "_partial"
    print(f"full sweep {status}", flush=True)
    print("loading the 5-mm caches...", flush=True)
    ids5, labs5, F5 = load5_matrix()
    col5 = {m: j for j, m in enumerate(labs5)}
    rows5 = [ids5.index(c) for c in subs if c in ids5]
    checks = [v for c in subs for v in per2[c].get("basis_check", {}).values()]
    print(f"{len(ids5)} subjects at 5 mm, {len(labs5)} distinct montages; superposition "
          f"checks: {len(checks)}" + (f", worst {max(checks):.1e}" if checks else "") + "\n")

    # exact cross-check: shortlist entries must match the full sweep for the same subject
    sl_dir = os.environ.get("SPICEHEAD_T7_SHORTLIST_DIR", OUT)
    rel = []
    for c in subs:
        p = os.path.join(sl_dir, f"{c}__dx{DX:g}.json")
        if not os.path.exists(p):
            continue
        for roi, d in json.load(open(p, encoding="utf-8")).items():
            for m, v in d.items():
                g = per2[c]["rois"].get(roi, {}).get(canon(m))
                if g is not None and v["F"]:
                    rel.append(abs(g[0] - v["F"]) / abs(v["F"]))
    if rel:
        flag = "OK" if max(rel) < 1e-4 else "MISMATCH, investigate before using these results"
        print(f"cross-check vs the published shortlist run: {len(rel)} montage-ROI values, "
              f"max relative difference {max(rel):.1e}  [{flag}]\n")
    else:
        print(f"cross-check skipped: no shortlist JSONs for these subjects in {sl_dir}\n")

    cohort_rows, subj_rows, summary = [], [], {}
    for roi in ROI_ORDER:
        med300 = cohort_median(F5[roi])
        med5s = cohort_median(F5[roi][rows5]) if rows5 else np.full(len(labs5), np.nan)
        ranked5 = [labs5[j] for j in np.argsort(-np.nan_to_num(med300, nan=-np.inf))
                   if np.isfinite(med300[j])]
        rank5 = {m: i + 1 for i, m in enumerate(ranked5)}
        shortlist = set(ranked5[:K])

        labs2 = sorted({m for c in subs for m in per2[c]["rois"].get(roi, {})})
        A2 = np.full((len(subs), len(labs2)), np.nan)
        T2, O2 = A2.copy(), A2.copy()
        for s, c in enumerate(subs):
            d = per2[c]["rois"].get(roi, {})
            for j, m in enumerate(labs2):
                if m in d:
                    A2[s, j], T2[s, j], O2[s, j] = d[m]
        med2 = cohort_median(A2)
        j2 = {m: j for j, m in enumerate(labs2)}
        universe = set(labs2) | set(labs5)
        adm = [m for m in labs2 if np.isfinite(med2[j2[m]]) and m in rank5]
        excl = dict(n_universe=len(universe), n_admissible=len(adm),
                    n_excluded_5mm_coverage=len([m for m in universe if m not in rank5]),
                    n_excluded_2mm_coverage=len([m for m in universe if m in rank5 and
                                                 (m not in j2 or not np.isfinite(med2[j2[m]]))]))
        if len(adm) < 3:
            print(f"{roi:<15} fewer than 3 admissible montages; skipped")
            continue
        v2 = np.array([med2[j2[m]] for m in adm])
        order2 = [adm[i] for i in np.argsort(-v2)]
        win2, win5 = order2[0], ranked5[0]
        F2 = {m: med2[j2[m]] for m in adm}
        if win5 not in F2:
            print(f"{roi:<15} the 5-mm winner {win5} is not admissible at 2 mm; skipped")
            continue
        sl_adm = [m for m in ranked5[:K] if m in F2]
        best_sl = max(sl_adm, key=lambda m: F2[m]) if sl_adm else None
        rho300 = float(spearmanr([med300[col5[m]] for m in adm], v2).statistic)
        ok5s = [i for i, m in enumerate(adm) if np.isfinite(med5s[col5[m]])]
        rho_s = float(spearmanr([med5s[col5[adm[i]]] for i in ok5s], v2[ok5s]).statistic) \
            if len(ok5s) >= 3 else float("nan")
        win5s = max((adm[i] for i in ok5s), key=lambda m: med5s[col5[m]]) if ok5s else None
        medT = lambda m: float(np.nanmedian(T2[:, j2[m]]))
        medO = lambda m: float(np.nanmedian(O2[:, j2[m]]))
        t5a, t5b = set(ranked5[:5]), set(order2[:5])
        cohort_rows.append(dict(
            roi=roi, status=status, n_subjects=len(subs), **excl,
            spearman_vs_300rank=round(rho300, 4), spearman_same_subjects=round(rho_s, 4),
            winner_5mm_300=win5, winner_2mm=win2, winner_kept=bool(win5 == win2),
            rank_of_5mm_winner_at_2mm=order2.index(win5) + 1,
            cohort_regret_5mm_winner_pct=round(_pct(F2[win2], F2[win5]), 3),
            rank_at_5mm_of_2mm_winner=rank5[win2],
            shortlist_contains_2mm_winner=bool(win2 in shortlist),
            shortlist_cohort_loss_pct=round(_pct(F2[win2], F2[best_sl]), 3) if best_sl else None,
            winner_5mm_same_subjects=win5s,
            same_subject_winner_kept=bool(win5s == win2) if win5s else None,
            same_subject_regret_pct=round(_pct(F2[win2], F2[win5s]), 3) if win5s else None,
            top25_overlap=len(set(ranked5[:K]) & set(order2[:K])),
            top5_jaccard=round(len(t5a & t5b) / len(t5a | t5b), 4),
            F_2mm_winner=round(F2[win2], 4), T_2mm_winner=round(medT(win2), 5),
            O_2mm_winner=round(medO(win2), 5), F_5mm_winner_at_2mm=round(F2[win5], 4),
            T_5mm_winner_at_2mm=round(medT(win5), 5), O_5mm_winner_at_2mm=round(medO(win5), 5)))

        # subject level: each subject's own 2-mm optimum over the admissible set
        reg = {k: [] for k in ("cohort5", "cohort2", "shortlist", "own5")}
        contained = 0
        for s, c in enumerate(subs):
            Fs = {m: A2[s, j2[m]] for m in adm if np.isfinite(A2[s, j2[m]])}
            if not Fs:
                continue
            own2 = max(Fs, key=Fs.get)
            Fstar = Fs[own2]
            r5 = _pct(Fstar, Fs[win5]) if win5 in Fs else float("nan")
            r2 = _pct(Fstar, Fs[win2]) if win2 in Fs else float("nan")
            sl_s = [m for m in shortlist if m in Fs]
            rsl = _pct(Fstar, max(Fs[m] for m in sl_s)) if sl_s else float("nan")
            own5, ro5 = None, float("nan")
            if c in ids5:
                row5 = F5[roi][ids5.index(c)]
                f5 = {m: row5[col5[m]] for m in Fs if np.isfinite(row5[col5[m]])}
                if f5:
                    own5 = max(f5, key=f5.get)
                    ro5 = _pct(Fstar, Fs[own5])
            contained += own2 in shortlist
            for k, v in (("cohort5", r5), ("cohort2", r2), ("shortlist", rsl), ("own5", ro5)):
                reg[k].append(v)
            subj_rows.append(dict(
                subject=c, roi=roi, status=status, n_admissible=len(Fs),
                optimum_2mm=own2, F_optimum_2mm=round(float(Fstar), 4),
                cohort_5mm_rank_of_optimum=rank5.get(own2),
                shortlist_contains_optimum=bool(own2 in shortlist),
                regret_5mm_cohort_winner_pct=round(r5, 3),
                regret_2mm_cohort_winner_pct=round(r2, 3),
                regret_best_shortlisted_pct=round(rsl, 3),
                optimum_5mm_same_subject=own5,
                regret_own_5mm_optimum_pct=round(ro5, 3)))
        summary[roi] = dict(admissible=excl, n_subjects=len(subs),
                            shortlist_contains_subject_optimum=f"{contained}/{len(subs)}",
                            **{f"regret_{k}": _summ(v) for k, v in reg.items()})

    os.makedirs(OUT_FULL, exist_ok=True)
    out1 = os.path.join(OUT_FULL, f"dx2_full_vs_dx5{suffix}.csv")
    out2 = os.path.join(OUT_FULL, f"dx2_full_subject_regret{suffix}.csv")
    out3 = os.path.join(OUT_FULL, f"dx2_full_summary{suffix}.json")
    for path, rows in ((out1, cohort_rows), (out2, subj_rows)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(out3, "w", encoding="utf-8") as fh:
        json.dump(dict(status=status, expected=expected, missing=missing,
                       cross_check_max_rel=max(rel) if rel else None,
                       superposition_worst=max(checks) if checks else None,
                       per_target=summary), fh, indent=1)

    print(f"{'ROI':<15}{'adm':>6}{'rho300':>8}{'rho20':>7}{'kept':>6}{'5mm@2mm':>8}"
          f"{'regret':>8}{'2mm@5mm':>8}{'SL loss':>8}{'same-20 kept':>13}")
    for r in cohort_rows:
        print(f"{r['roi']:<15}{r['n_admissible']:>6}{r['spearman_vs_300rank']:>8.3f}"
              f"{r['spearman_same_subjects']:>7.3f}{('yes' if r['winner_kept'] else 'NO'):>6}"
              f"{r['rank_of_5mm_winner_at_2mm']:>8}{r['cohort_regret_5mm_winner_pct']:>7.2f}%"
              f"{r['rank_at_5mm_of_2mm_winner']:>8}{(format(r['shortlist_cohort_loss_pct'], '>7.2f') + '%') if r['shortlist_cohort_loss_pct'] is not None else '    n/a '}"
              f"{str(r['same_subject_winner_kept']):>13}")
    print("\nsubject-level regret, % of each subject's own 2-mm optimum (median [IQR], max):")
    for roi, sm in summary.items():
        def f(k):
            s = sm[f"regret_{k}"]
            return (f"{s['median']:.2f} [{s['q1']:.2f}-{s['q3']:.2f}] max {s['max']:.2f}"
                    if s.get("n") else "n/a")
        print(f"  {roi:<14} 5mm-cohort {f('cohort5')} | 2mm-cohort {f('cohort2')} | "
              f"best-shortlisted {f('shortlist')} | own-5mm {f('own5')} | "
              f"optimum in shortlist {sm['shortlist_contains_subject_optimum']}")
    print(f"\n{status}")
    print("wrote", out1, "|", os.path.basename(out2), "|", os.path.basename(out3))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rest = sys.argv[2:]
    lim, shard = 0, (0, 1)
    if "--limit" in rest:
        i = rest.index("--limit")
        lim = int(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]
    if "--shard" in rest:
        i = rest.index("--shard")
        a, b = rest[i + 1].split("/")
        shard = (int(a), int(b))
        rest = rest[:i] + rest[i + 2:]
    nps = int(rest[0]) if rest else 5
    if sys.argv[1] == "--score":
        stage_score(nps)
    elif sys.argv[1] == "--collect":
        stage_collect()
    elif sys.argv[1] == "--score-full":
        stage_score_full(nps, lim, shard)
    elif sys.argv[1] == "--collect-full":
        stage_collect_full(nps, lim)
    else:
        raise SystemExit(__doc__)
