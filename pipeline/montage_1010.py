"""
10-10 (expanded ~32-electrode) montage search on a decade-spread subset. Since the set contains
the 10/20 positions, this is a superset search: the 10-10 best is >= the 10/20 best on the same
subjects, and the question is whether the winner USES any finer (non-10/20) position.

  python montage_1010.py <cohort.csv> <charm_outroot> <out.json> [dx]
"""
import os
import sys
import gzip
import json
import pickle
import time
from itertools import combinations

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here)); sys.path.insert(0, _here)
from spicehead import config
from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components, ti_envelope_magnitude
# TI envelope definition. 'maxdir' = rotation-invariant maximum-over-direction (Grossman et al.);
# 'peraxis' = superseded coordinate-dependent form, retained only to reproduce pre-2026-08 numbers.
ENVELOPE = os.environ.get("SPICEHEAD_ENVELOPE", "maxdir")
from spicehead.envelope_fast import envelope_max_direction_flat

from montage_search import (make_multiground_solver, snap, roi_mask, ELEC_RADIUS_MM,
                            MIN_DIST_MM, ROI_RADIUS_MM, EXCLUDE_MM, SCALP, SCORE, ROI_ORDER)

# kept in sync with place_1010_subset.py (which can't be imported here — it needs simnibs)
LABELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
          "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
          "AF7", "AF8", "FC3", "FC4", "FT7", "FT8", "CP3", "CP4", "TP7", "TP8", "PO7", "PO8", "POz"]
SUBSET = ["SUBJ01", "SUBJ02", "SUBJ03", "SUBJ04", "SUBJ05", "SUBJ06",
          "SUBJ07", "SUBJ08", "SUBJ09", "SUBJ10", "SUBJ11", "SUBJ12",
          "SUBJ13", "SUBJ14"]

TEN20 = {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"}
EASY = {"Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "FT7", "FT8", "T7", "T8", "TP7", "TP8", "P7", "P8", "PO7", "PO8", "O1", "O2"}


def score(m2m, dx):
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, affine = mri.head_model(seg, dx)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    # grey matter only in BOTH T and O -- keep in sync with montage_search.SCORE
    brain = np.isin(labels, SCORE); bidx = np.argwhere(brain)
    bw = (affine[:3, :3] @ bidx.T).T + affine[:3, 3]
    rj = json.load(open(os.path.join(m2m, "roi_coords.json")))
    rmask = {r: roi_mask(labels.shape, bidx, bw, rj[r], ROI_RADIUS_MM) for r in ROI_ORDER}
    outside = {r: (brain & ~roi_mask(labels.shape, bidx, bw, rj[r], EXCLUDE_MM)) for r in ROI_ORDER}
    scalp = np.argwhere(labels == SCALP); ej = json.load(open(os.path.join(m2m, "electrodes_1010.json")))
    epos = {}; elecs = {}
    for lab, world in ej.items():
        v = snap(scalp, mri.world_to_voxel(affine, world)); epos[lab] = v.astype(float)
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v), radius_nodes=ELEC_RADIUS_MM / dx), lab)
    labs = list(elecs)
    pairs = [(a, b) for a, b in combinations(labs, 2) if np.linalg.norm(epos[a] - epos[b]) * dx >= MIN_DIST_MM]
    solver, _ = make_multiground_solver(model.G); pE = {}
    for a, b in pairs:
        Ex, Ey, Ez = efield_components(model, solver(pair_injection(model, elecs[a], elecs[b], 1e-3)))
        pE[(a, b)] = (Ex.astype(np.float32), Ey.astype(np.float32), Ez.astype(np.float32))
    # flat scored-tissue arrays: the envelope is evaluated once per montage, so restricting it to
    # the ~18% of voxels actually scored is what makes the rotation-invariant form affordable
    _flat = {k: tuple(c[brain] for c in v) for k, v in pE.items()}
    _rsel = {r: rmask[r][brain] for r in rmask}
    _osel = {r: outside[r][brain] for r in outside}
    out = {r: {} for r in ROI_ORDER}
    for pa, pb in combinations(pairs, 2):
        if len(set(pa) | set(pb)) < 4:
            continue
        if ENVELOPE == "maxdir":
            a, b = _flat[pa], _flat[pb]
            env = envelope_max_direction_flat(a[0], a[1], a[2], b[0], b[1], b[2])
        else:
            env = np.nan_to_num(ti_envelope_magnitude(pE[pa], pE[pb]))[brain]
        key = "%s+%s / %s+%s" % (pa[0], pa[1], pb[0], pb[1])
        for r in ROI_ORDER:
            rv = env[_rsel[r]]
            if rv.size:
                O = float(np.percentile(env[_osel[r]], 99))
                out[r][key] = (float(np.median(rv)) / O) if O > 0 else 0.0
    return out


def uses_finer(key):
    es = set(key.replace(" / ", "+").split("+"))
    return sorted(es - TEN20)


def convenience(key):
    es = set(key.replace(" / ", "+").split("+"))
    return sum(1 for e in es if e in EASY)


def main(cohort_csv, outroot, outjson, dx=5.0, shard=0, nshards=1):
    """nshards>1 = CACHE-WARMING worker: computes only its slice of subjects and exits without
    aggregating. Run N of them in parallel, then one plain (nshards=1) pass to aggregate all 300
    from cache. The aggregation path is untouched by sharding -- which process computed a subject
    cannot change that subject's scores, so the result is identical to a serial run."""
    subs = []
    import csv as _csv
    _rows = list(_csv.DictReader(open(cohort_csv, encoding='utf-8-sig')))
    for cid in [r['CaseID'] for r in _rows]:
        m2m = os.path.join(outroot, "m2m_SUBJ15asc") if cid.startswith("SUBJ15") else os.path.join(outroot, "m2m_" + cid)
        if os.path.exists(os.path.join(m2m, "electrodes_1010.json")):
            subs.append((cid, m2m))
    print(f"10-10 search on {len(subs)} subjects, {len(LABELS)} electrodes", flush=True)
    # Per-subject cache. This sweep is ~7 min/subject (496 usable pairs -> ~123k montages), so a
    # 300-subject run is >30 h and used to lose everything if the console was closed: it only wrote
    # at the very end. Cache outside the Dropbox tree -- these are ~10 MB each and must not sync.
    cache = os.environ.get("MONTAGE1010_CACHE") or os.path.join(
        os.environ.get("TEMP", "."), config.tagged("spicehead_1010_cache"))
    os.makedirs(cache, exist_ok=True)
    print(f"cache: {cache}", flush=True)
    warm = nshards > 1
    if warm:
        subs = [s for i, s in enumerate(subs) if i % nshards == shard]
        print(f"shard {shard}/{nshards}: warming cache for {len(subs)} subjects", flush=True)
    accum = {r: {} for r in ROI_ORDER}
    for i, (cid, m2m) in enumerate(subs):
        t = time.time()
        cf = os.path.join(cache, f"{cid}__dx{dx:g}_gm.pkl.gz")
        if os.path.exists(cf):
            if warm:            # nothing to accumulate here; don't pay to unpickle ~10 MB
                print(f"  [{i+1}/{len(subs)}] {cid} (cached, skip)", flush=True)
                continue
            with gzip.open(cf, "rb") as fh:
                sc = pickle.load(fh)
            note = "cached"
        else:
            sc = score(m2m, dx)
            # write-then-rename: this job has been killed mid-run twice, and a truncated .pkl.gz
            # would be silently treated as a valid cached subject on the next pass.
            tmp = cf + ".part"
            with gzip.open(tmp, "wb", compresslevel=1) as fh:
                pickle.dump(sc, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cf)
            note = f"{time.time()-t:.0f}s"
        if not warm:
            for r in ROI_ORDER:
                for k, v in sc[r].items():
                    accum[r].setdefault(k, []).append(v)
        print(f"  [{i+1}/{len(subs)}] {cid} ({note})", flush=True)
    if warm:
        print(f"shard {shard}/{nshards} done", flush=True)
        return
    ns = len(subs); out = {"n_subjects": ns, "n_electrodes": len(LABELS), "per_roi": {}}
    for r in ROI_ORDER:
        ranked = sorted([(float(np.median(v)), convenience(k), k) for k, v in accum[r].items() if len(v) >= 0.8 * ns],
                        key=lambda x: -x[0])
        # best montage that uses ONLY 10/20 positions, for a same-subset baseline
        best_1020_only = next(((s, c, k) for s, c, k in ranked if not uses_finer(k)), None)
        out["per_roi"][r] = {
            "best": [{"score": round(s, 3), "easy": c, "finer_used": uses_finer(k), "montage": k} for s, c, k in ranked[:5]],
            "best_1020_only_same_subset": ({"score": round(best_1020_only[0], 3), "montage": best_1020_only[2]} if best_1020_only else None),
        }
    json.dump(out, open(outjson, "w"), indent=2)
    for r in ROI_ORDER:
        top = out["per_roi"][r]["best"][0]; base = out["per_roi"][r]["best_1020_only_same_subset"]
        fu = ("+".join(top["finer_used"]) if top["finer_used"] else "none")
        print(f"  {r}: 10-10 best={top['montage']} ({top['score']}, finer={fu}) | 10/20-only same subset={base['score'] if base else 'NA'}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    dx = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    shard = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    nshards = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    main(sys.argv[1], sys.argv[2], sys.argv[3], dx, shard, nshards)
