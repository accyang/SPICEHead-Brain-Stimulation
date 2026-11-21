# -*- coding: utf-8 -*-
"""Network size and electrode contact supports for every head at one grid spacing.

Review of 1 October, Results section 1 ("what is calculated, at what spacing, for how many
heads?"). For each cohort head: nodes, edges, connected conductor components, scored gray-matter
voxels, and the number of contact nodes of each of the 19 electrodes under the active model
(spicehead.config; v22 = skin-surface contacts). A head whose electrode gets no contact node is
reported, not skipped.

  python pipeline/v22_scale_census.py [dx[,dx...]] [--fem8]   (default 5; --fem8: the eight
                                                               FEM-comparison heads only)

Output: <results>/v22_scale/census_dx<dx>.csv (census_fem8_dx<dx>.csv with --fem8)
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.sparse.csgraph import connected_components

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)

from spicehead import mri
from spicehead.electrodes import contact_nodes
from montage_search import snap, SCALP, SCORE, ELEC_RADIUS_MM
from run_montage_cohort import resolve_m2m

CSVP = os.environ.get("SPICEHEAD_COHORT_CSV", r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv")
CH = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")
OUT = os.path.join(config.RESULTS, "v22_scale")


def main(dx=5.0, fem8=False):
    os.makedirs(OUT, exist_ok=True)
    print(config.describe(), f"dx={dx:g} mm", flush=True)
    rows = []
    if fem8:                                    # the eight FEM-comparison / co-simulation heads
        from v21_simnibs_validation import subjects
        ids = [s[0] for s in subjects(8)]
    else:
        ids = [r["CaseID"] for r in csv.DictReader(open(CSVP, encoding="utf-8-sig"))]
    for i, cid in enumerate(ids, 1):
        t = time.time()
        m2m = resolve_m2m(CH, cid)
        model, affine = mri.head_model(   # v21/v22: nearest-label network; v23: homogenized
            os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"), dx)
        labels = model.labels
        ncomp = connected_components(model.G, directed=False)[0]
        scalp_vox = np.argwhere(labels == SCALP)
        n = []
        for lab, world in json.load(open(os.path.join(m2m, "electrodes_1020.json"))).items():
            v = tuple(int(x) for x in snap(scalp_vox, mri.world_to_voxel(affine, world)))
            try:
                n.append(contact_nodes(model, v, ELEC_RADIUS_MM / dx).size)
            except ValueError:
                n.append(0)
        rows.append(dict(subject=cid, dx_mm=dx, n_nodes=int(model.n_nodes),
                         n_edges=int((model.G.nnz - model.n_nodes) // 2), n_components=int(ncomp),
                         n_gm_voxels=int(np.isin(labels, SCORE).sum()), n_electrodes=len(n),
                         contact_min=min(n), contact_median=float(np.median(n)),
                         contact_max=max(n), n_empty_contacts=sum(1 for x in n if x == 0)))
        print(f"  [{i}/{len(ids)}] {cid}: {model.n_nodes} nodes, contacts {min(n)}-{max(n)} "
              f"({time.time()-t:.0f}s)", flush=True)
    p = os.path.join(OUT, f"census_{'fem8_' if fem8 else ''}dx{dx:g}.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    nn = np.array([r["n_nodes"] for r in rows])
    print(f"\n{len(rows)} heads: nodes median {int(np.median(nn))} (range {nn.min()}-{nn.max()}); "
          f"empty contacts {sum(r['n_empty_contacts'] for r in rows)}")
    print("wrote", p)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--fem8"]
    for d in (a[0].split(",") if a else ["5"]):
        main(float(d), fem8="--fem8" in sys.argv)
