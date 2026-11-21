"""
Montage optimizer for one subject: rank 10/20 two-pair TI montages by focality (T/O) per ROI.

Exploits SPICEhead's factorize-once property: factorize G one time, precompute the E-field for
every valid single 10/20 pair, then evaluate every 2-pair montage by combining two precomputed
fields (no new solve). Focality objective per ROI:
    T = median envelope in ROI,  O = 99th-pctile envelope in brain OUTSIDE ROI,  score = T / O.
Convenience is reported (not optimized): count of "easy-access" electrodes in the montage.

  python montage_search.py <m2m_dir> <out.json> [dx_mm] [topk]
"""
import os
import sys
import json
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized

from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components, ti_envelope_magnitude


def make_multiground_solver(G):
    """Prefactorized direct solver that grounds one node per connected component, so
    isolated conductor islands don't make the SuperLU factorization singular."""
    N = G.shape[0]
    Gc = csr_matrix(G)
    ncomp, comp = connected_components(csgraph=Gc, directed=False)
    grounds = [int(np.flatnonzero(comp == c)[0]) for c in range(ncomp)]
    keep = np.ones(N, bool)
    keep[grounds] = False
    idx = np.flatnonzero(keep)
    lu = factorized(Gc[idx][:, idx].tocsc())

    def solve(current):
        V = np.zeros(N)
        V[idx] = lu(np.asarray(current, float)[idx])
        return V
    return solve, ncomp

ELEC_RADIUS_MM = 12.0    # electrode radius (mm), converted to voxels per grid
MIN_DIST_MM = 40.0       # minimum separation between the two electrodes of a pair
ROI_RADIUS_MM = 10.0     # target ROI radius for T
EXCLUDE_MM = 20.0        # off-target O excludes brain within this distance of any ROI center
SCALP, BRAIN = 5, (1, 2)
# Tissue scored in BOTH T and O. Grey matter only: sigma_WM (0.126) < sigma_GM (0.275), so the
# envelope runs ~28% higher in white matter and a GM+WM denominator penalises a grey-matter target
# for a conductivity contrast the montage cannot influence. BRAIN is retained for anatomical use.
SCORE = (2,)
ROI_ORDER = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]
EASY = {"Fp1", "Fp2", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2"}  # skin/edge sites


def snap(scalp_vox, ijk):
    return scalp_vox[int(np.argmin(np.linalg.norm(scalp_vox - np.asarray(ijk, float), axis=1)))]


def roi_mask(shape, brain_idx, brain_world, centers, radius):
    m = np.zeros(brain_idx.shape[0], bool)
    for c in centers:
        m |= np.linalg.norm(brain_world - np.asarray(c, float), axis=1) <= radius
    full = np.zeros(shape, bool)
    sel = brain_idx[m]
    full[sel[:, 0], sel[:, 1], sel[:, 2]] = True
    return full


def main(m2m, outjson, dx=3.0, topk=20):
    t0 = time.time()
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, affine = mri.head_model(seg, dx)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    brain = np.isin(labels, SCORE)      # scoring mask = grey matter, not anatomical extent
    brain_idx = np.argwhere(brain)
    brain_world = (affine[:3, :3] @ brain_idx.T).T + affine[:3, 3]

    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    roi_masks = {r: roi_mask(labels.shape, brain_idx, brain_world, rois[r], ROI_RADIUS_MM) for r in ROI_ORDER}
    # off-target for O = brain farther than EXCLUDE_MM from every ROI center (skip the peri-target halo)
    near = {r: roi_mask(labels.shape, brain_idx, brain_world, rois[r], EXCLUDE_MM) for r in ROI_ORDER}
    outside = {r: (brain & ~near[r]) for r in ROI_ORDER}

    scalp_vox = np.argwhere(labels == SCALP)
    epos, elecs = {}, {}
    for lab, world in json.load(open(os.path.join(m2m, "electrodes_1020.json"))).items():
        v = snap(scalp_vox, mri.world_to_voxel(affine, world))
        epos[lab] = v.astype(float)
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v), radius_nodes=ELEC_RADIUS_MM / dx), lab)
    labs = list(elecs)
    pairs = [(a, b) for a, b in combinations(labs, 2)
             if np.linalg.norm(epos[a] - epos[b]) * dx >= MIN_DIST_MM]
    print(f"  model {labels.shape} nodes={model.n_nodes}  pairs={len(pairs)}  setup={time.time()-t0:.1f}s")

    t1 = time.time()
    solver, ncomp = make_multiground_solver(model.G)
    print(f"  factorized in {time.time()-t1:.1f}s  ({ncomp} conductor components)")
    t2 = time.time()
    pairE = {}
    for (a, b) in pairs:
        V = solver(pair_injection(model, elecs[a], elecs[b], 1e-3))
        Ex, Ey, Ez = efield_components(model, V)
        pairE[(a, b)] = (Ex.astype(np.float32), Ey.astype(np.float32), Ez.astype(np.float32))
    print(f"  {len(pairs)} pair fields in {time.time()-t2:.1f}s")

    t3 = time.time()
    results = {r: [] for r in ROI_ORDER}
    n = 0
    for pa, pb in combinations(pairs, 2):
        if len(set(pa) | set(pb)) < 4:
            continue
        n += 1
        env = np.nan_to_num(ti_envelope_magnitude(pairE[pa], pairE[pb]))
        conv = sum(1 for e in (set(pa) | set(pb)) if e in EASY)
        for r in ROI_ORDER:
            rv = env[roi_masks[r]]
            if rv.size == 0:
                continue
            T = float(np.median(rv))
            O = float(np.percentile(env[outside[r]], 99))
            results[r].append((T / O if O > 0 else 0.0, T, O, pa, pb, conv))
    print(f"  swept {n} montages in {time.time()-t3:.1f}s")

    out = {"m2m": m2m, "dx": dx, "n_pairs": len(pairs), "n_montages": n, "montages_per_roi": {}}
    for r in ROI_ORDER:
        results[r].sort(key=lambda x: -x[0])
        out["montages_per_roi"][r] = [
            {"rank": i + 1, "score": round(s, 3), "T": round(T, 4), "O": round(O, 4),
             "pair1": list(pa), "pair2": list(pb), "easy_electrodes": conv}
            for i, (s, T, O, pa, pb, conv) in enumerate(results[r][:topk])]
    json.dump(out, open(outjson, "w"), indent=2)
    for r in ROI_ORDER:
        b = out["montages_per_roi"][r][0]
        print(f"  BEST {r:>12}: {'+'.join(b['pair1'])} / {'+'.join(b['pair2'])}  "
              f"score={b['score']} (T={b['T']} O={b['O']}, easy={b['easy_electrodes']}/4)")
    print(f"  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    dx = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    topk = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    main(sys.argv[1], sys.argv[2], dx, topk)
