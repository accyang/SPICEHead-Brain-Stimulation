"""
Cohort montage optimizer: run the per-subject 10/20 two-pair TI montage search over the whole
cohort and rank montages by MEDIAN focality (T/O) across subjects, per ROI. Reports the robust
population-optimal montages with cross-subject spread (IQR) and placement convenience.

  python run_montage_cohort.py <cohort.csv> <charm_outroot> <out_dir> [dx] [subset_n]

subset_n>0 scores only the first N subjects (for a quick validation run).
"""
import os
import sys
import csv
import json
import pickle
import time
from itertools import combinations

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))  # project root (spicehead)
sys.path.insert(0, _here)                    # pipeline dir (montage_search)

from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components, ti_envelope_magnitude
from spicehead.envelope_fast import envelope_max_direction_flat

# TI envelope definition used for scoring. 'maxdir' is the rotation-invariant
# maximum-over-direction form (Grossman et al.); 'peraxis' is the superseded coordinate-dependent
# form and exists only to reproduce pre-2026-08 numbers.
ENVELOPE = os.environ.get("SPICEHEAD_ENVELOPE", "maxdir")

from montage_search import (make_multiground_solver, snap, roi_mask,
                            ELEC_RADIUS_MM, MIN_DIST_MM, ROI_RADIUS_MM, EXCLUDE_MM,
                            SCALP, BRAIN, SCORE, ROI_ORDER, EASY)


def score_subject(m2m, dx, tissue='gm'):
    """Return {roi: {frozenset(pairA,pairB): T/O}} for every valid 2-pair montage."""
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, affine = mri.head_model(seg, dx)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    brain = np.isin(labels, SCORE if tissue == 'gm' else BRAIN)  # scored tissue, T and O
    brain_idx = np.argwhere(brain)
    brain_world = (affine[:3, :3] @ brain_idx.T).T + affine[:3, 3]
    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    present = [r for r in ROI_ORDER if r in rois]
    roi_masks = {r: roi_mask(labels.shape, brain_idx, brain_world, rois[r], ROI_RADIUS_MM) for r in present}
    near = {r: roi_mask(labels.shape, brain_idx, brain_world, rois[r], EXCLUDE_MM) for r in present}
    outside = {r: (brain & ~near[r]) for r in present}

    scalp_vox = np.argwhere(labels == SCALP)
    epos, elecs = {}, {}
    for lab, world in json.load(open(os.path.join(m2m, "electrodes_1020.json"))).items():
        v = snap(scalp_vox, mri.world_to_voxel(affine, world))
        epos[lab] = v.astype(float)
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v), radius_nodes=ELEC_RADIUS_MM / dx), lab)
    labs = list(elecs)
    pairs = [(a, b) for a, b in combinations(labs, 2) if np.linalg.norm(epos[a] - epos[b]) * dx >= MIN_DIST_MM]

    solver, _ = make_multiground_solver(model.G)
    pairE = {}
    for (a, b) in pairs:
        V = solver(pair_injection(model, elecs[a], elecs[b], 1e-3))
        Ex, Ey, Ez = efield_components(model, V)
        pairE[(a, b)] = (Ex.astype(np.float32), Ey.astype(np.float32), Ez.astype(np.float32))

    # Score on the flat scored-tissue voxels only. The envelope is evaluated ~11,600 times per
    # subject, so restricting it to the ~18% of voxels that are actually scored, and using the
    # componentwise rotation-invariant form, is what makes ENVELOPE='maxdir' affordable at all
    # (40.3 ms -> 3.3 ms per call; the readable reference form would cost 37 h for the cohort).
    flatE = {k: tuple(c[brain] for c in v) for k, v in pairE.items()}
    roi_sel = {r: roi_masks[r][brain] for r in present}
    out_sel = {r: outside[r][brain] for r in present}

    scores = {r: {} for r in present}
    for pa, pb in combinations(pairs, 2):
        if len(set(pa) | set(pb)) < 4:
            continue
        if ENVELOPE == "maxdir":
            a, b = flatE[pa], flatE[pb]
            env = envelope_max_direction_flat(a[0], a[1], a[2], b[0], b[1], b[2])
        else:                                   # 'peraxis' reproduces the superseded definition
            env = np.nan_to_num(ti_envelope_magnitude(pairE[pa], pairE[pb]))[brain]
        key = frozenset((pa, pb))
        for r in present:
            rv = env[roi_sel[r]]
            if rv.size == 0:
                continue
            O = float(np.percentile(env[out_sel[r]], 99))
            scores[r][key] = (float(np.median(rv)) / O) if O > 0 else 0.0
    return scores


def montage_str(key):
    """Canonical, process-independent montage label.

    A montage key is a frozenset of two electrode pairs, so `tuple(key)` yields whichever order
    set iteration happens to produce — and since string hashing is randomised per process, the same
    montage could be labelled "A+B / C+D" in one run and "C+D / A+B" in the next. That silently
    broke a cross-run comparison once (Brain Stimulation review, comment 8). Sorting within and
    across pairs makes the label reproducible everywhere it is written.
    """
    pairs = sorted(tuple(sorted(p)) for p in key)
    return " / ".join("+".join(p) for p in pairs)


def convenience(key):
    pa, pb = tuple(key)
    return sum(1 for e in (set(pa) | set(pb)) if e in EASY)


def resolve_m2m(outroot, cid):
    return os.path.join(outroot, "m2m_SUBJ01asc") if cid.startswith("SUBJ01") else os.path.join(outroot, "m2m_" + cid)


def main(cohort_csv, outroot, out_dir, dx=5.0, subset_n=0, tissue='gm', shard=0, nshards=1):
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(cohort_csv, encoding="utf-8-sig")))
    subs = []
    for r in rows:
        m2m = resolve_m2m(outroot, r["CaseID"])
        if os.path.exists(os.path.join(m2m, "electrodes_1020.json")) and os.path.exists(os.path.join(m2m, "roi_coords.json")):
            subs.append((r["CaseID"], m2m))
    if subset_n > 0:
        subs = subs[:subset_n]
    warm = nshards > 1          # cache-warming worker: compute a slice, skip aggregation
    if warm:
        subs = [x for i, x in enumerate(subs) if i % nshards == shard]
        print(f"shard {shard}/{nshards}: {len(subs)} subjects")
    print(f"scoring {len(subs)} subjects at dx={dx} mm, tissue={tissue}")

    # per-subject checkpoint: this run is ~2 h over 97 subjects, so cache each subject's scores
    # and skip them on restart rather than losing everything to an interruption
    cache = os.path.join(out_dir, "_subject_scores")
    os.makedirs(cache, exist_ok=True)

    accum = {r: {} for r in ROI_ORDER}
    for i, (cid, m2m) in enumerate(subs):
        t = time.time()
        cpath = os.path.join(cache, f"{cid}__dx{dx:g}_{tissue}" + ("" if ENVELOPE == "peraxis" else "_" + ENVELOPE) + ".pkl")
        if os.path.exists(cpath):
            with open(cpath, "rb") as fh:
                sc = pickle.load(fh)
            note = "cached"
        else:
            sc = score_subject(m2m, dx, tissue)
            with open(cpath, "wb") as fh:
                pickle.dump(sc, fh, protocol=4)
            note = f"{time.time()-t:.0f}s"
        if not warm:
            for r in sc:
                for key, val in sc[r].items():
                    accum[r].setdefault(key, []).append(val)
        else:
            del sc
        print(f"  [{i+1}/{len(subs)}] {cid} ({note})", flush=True)
    if warm:
        print(f"shard {shard}/{nshards} done", flush=True)
        return

    nsub = len(subs)
    out = {"n_subjects": nsub, "dx": dx, "tissue": tissue, "envelope": ENVELOPE,
           "metric": ("median(env in ROI n %s) / p99(env in %s >20mm from ROI)"
                      % (("GM","GM") if tissue=="gm" else ("GM+WM","GM+WM"))),
           "min_present_frac": 0.8, "top": {}}
    for r in ROI_ORDER:
        ranked = []
        for key, vals in accum[r].items():
            if len(vals) >= 0.8 * nsub:
                ranked.append((float(np.median(vals)), float(np.percentile(vals, 25)),
                               float(np.percentile(vals, 75)), len(vals), convenience(key), montage_str(key)))
        ranked.sort(key=lambda x: -x[0])
        out["top"][r] = [{"rank": i + 1, "median_score": round(m, 3), "iqr": [round(q1, 3), round(q3, 3)],
                          "n": n, "easy_electrodes": conv, "montage": s}
                         for i, (m, q1, q3, n, conv, s) in enumerate(ranked[:15])]
    json.dump(out, open(os.path.join(out_dir, "cohort_montage_optima.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(out_dir, 'cohort_montage_optima.json')}")
    for r in ROI_ORDER:
        print(f"\n== {r}: top montages by median focality (T/O), n={nsub} ==")
        for e in out["top"][r][:5]:
            print(f"  {e['median_score']:.3f}  IQR[{e['iqr'][0]:.2f}-{e['iqr'][1]:.2f}]  easy={e['easy_electrodes']}/4  {e['montage']}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    dx = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    subset_n = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    tissue = sys.argv[6] if len(sys.argv) > 6 else 'gm'
    shard = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    nshards = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    main(sys.argv[1], sys.argv[2], sys.argv[3], dx, subset_n, tissue, shard, nshards)
