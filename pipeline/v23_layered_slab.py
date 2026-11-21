# -*- coding: utf-8 -*-
"""Layered slab against its exact series resistance: nearest-label (v22) and homogenized (v23) networks.

v23 has no interface rule, so the v21/v22 slab test (pipeline/v21_interface_rule.py: arithmetic vs harmonic,
layers built directly at each grid) has no v23 counterpart. This test asks the question that decides between
the two v22/v23 constructions instead: the slab is defined once on a 0.5 mm label grid, as CHARM's
segmentation is, and each network is built from those labels at 3, 2 and 1 mm the way the package builds a
head:
  v22  labels resampled to dx by nearest neighbour (nibabel resample_to_output, order 0, as
       spicehead.mri.load_label_volume), then the harmonic network (SPICEHeadModel, conductance_mean="harmonic");
  v23  spicehead.homogenize.homogenized_model on the 0.5 mm labels.

Current (1 mA) enters on one face of node centres and leaves on the opposite face, normal to the layers, so the
exact resistance between the two face planes is a series sum over the layers between them; it is computed for
each network's own face positions. Two stacks, scalp / compact bone / CSF / grey matter with the cohort's
conductivities:
  aligned   6 / 6 / 3 / 15 mm (the v21 stack; every layer a whole number of 1 mm cells)
  thin      5.5 / 4.5 / 2.5 / 15 mm (no layer a whole number of 1, 2 or 3 mm cells, as in a real head)

Synthetic; no subject data. Seconds on one core.
    python pipeline/v23_layered_slab.py
Output: results_v23/v23_layered_slab/{layered_slab.csv, layered_slab.md}
"""
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
import spicehead as sh                                           # noqa: E402
from spicehead.homogenize import homogenized_model              # noqa: E402
from spicehead.model import SPICEHeadModel                       # noqa: E402

OUT = os.path.join(ROOT, "results_v23", "v23_layered_slab")
FINE = 0.5                                                       # mm, CHARM's voxel size
CURRENT = 1e-3                                                   # A
CROSS = 12.0                                                     # mm, square cross-section (whole cells at 3, 2, 1 mm)
COND = {"scalp": 0.465, "compact_bone": 0.008, "csf": 1.654, "grey_matter": 0.275}
NAMES = list(COND)
LMAP = {i + 1: n for i, n in enumerate(NAMES)}
STACKS = {"aligned": [6.0, 6.0, 3.0, 15.0], "thin": [5.5, 4.5, 2.5, 15.0]}
DXS = (3.0, 2.0, 1.0)


def fine_slab(thick):
    """0.5 mm labels: layers along axis 0, label 1..4; voxel j covers x in [0.5 j, 0.5 (j + 1)) mm."""
    col = np.concatenate([np.full(int(round(t / FINE)), i + 1, np.int16) for i, t in enumerate(thick)])
    n = int(round(CROSS / FINE))
    return np.repeat(np.repeat(col[:, None, None], n, 1), n, 2)


def sigma_at(x, thick):
    edges = np.cumsum([0.0] + list(thick))
    i = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(thick) - 1)
    return np.array([COND[NAMES[k]] for k in np.atleast_1d(i)])


def exact_R(x0, x1, thick, area_m2):
    """Series resistance between the planes x0 < x1 (mm), by exact integration over the layers."""
    edges = np.cumsum([0.0] + list(thick))
    R = 0.0
    for k in range(len(thick)):
        a, b = max(x0, edges[k]), min(x1, edges[k + 1])
        if b > a:
            R += (b - a) * 1e-3 / (COND[NAMES[k]] * area_m2)
    return R


def measure(m, x_nodes, dx):
    """Resistance between the first and last node planes along axis 0, the plane positions (mm) and the
    conducting cross-section (one dx x dx column per node of a face plane)."""
    ii = m.grid_coords[0]
    lo, hi = np.flatnonzero(ii == ii.min()), np.flatnonzero(ii == ii.max())
    assert lo.size == hi.size and m.n_nodes % lo.size == 0, "every plane must carry the same nodes"
    inj = np.zeros(m.n_nodes)
    inj[lo] += CURRENT / lo.size
    inj[hi] -= CURRENT / hi.size
    V = sh.solve_potentials(m.G, inj, ground_node=int(hi[0]), method="direct")
    return (float((V[lo].mean() - V[hi].mean()) / CURRENT), x_nodes[ii.min()], x_nodes[ii.max()],
            lo.size * (dx * 1e-3) ** 2)


def v22_network(fine, dx):
    import nibabel as nib
    from nibabel.processing import resample_to_output
    # fine voxel j has its centre at 0.5 j + 0.25 mm
    aff = np.diag([FINE, FINE, FINE, 1.0]); aff[:3, 3] = FINE / 2
    img = resample_to_output(nib.Nifti1Image(fine, aff), voxel_sizes=(dx,) * 3, order=0)
    lab = np.rint(np.asarray(img.dataobj)).astype(np.int64)
    sig = np.vectorize(lambda v: COND[LMAP[v]] if v in LMAP else 0.0)(lab)
    m = SPICEHeadModel(sig, dx=dx * 1e-3, mask=sig > 0, conductance_mean="harmonic")
    x = img.affine[0, 3] + dx * np.arange(lab.shape[0])
    return m, x


def v23_network(fine, dx):
    aff = np.diag([FINE, FINE, FINE, 1.0]); aff[:3, 3] = FINE / 2
    m, caff = homogenized_model(fine, aff, dx, label_map=LMAP, cond=COND)
    x = caff[0, 3] + dx * np.arange(m.labels.shape[0])
    return m, x


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sname, thick in STACKS.items():
        fine = fine_slab(thick)
        for dx in DXS:
            for model, build in (("v22", v22_network), ("v23", v23_network)):
                m, x = build(fine, dx)
                Rm, x0, x1, area = measure(m, x, dx)
                Re = exact_R(x0, x1, thick, area)
                rows.append(dict(stack=sname, dx_mm=dx, model=model, n_nodes=m.n_nodes, x_first_mm=round(x0, 3),
                                 x_last_mm=round(x1, 3), R_exact_ohm=round(Re, 4), R_model_ohm=round(Rm, 4),
                                 R_ratio=round(Rm / Re, 6), err_pct=round(100 * (Rm / Re - 1), 4)))
                print(f"{sname:8s} {dx:g} mm {model}: R/R_exact {Rm / Re:.6f} ({100 * (Rm / Re - 1):+.3f}%)", flush=True)
    with open(os.path.join(OUT, "layered_slab.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = ["# Layered slab: nearest-label (v22) and homogenized (v23) networks against the exact series resistance", "",
         "Slab defined on a 0.5 mm label grid; networks built from it at each grid as the package builds a head. "
         "Current normal to the layers; the exact resistance is integrated between each network's own face planes. "
         "Conductivities (S/m): " + ", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in COND.items()) + ".", "",
         "| Stack (scalp / bone / CSF / GM, mm) | Grid | v22 R / R_exact | v23 R / R_exact |", "|---|---|---|---|"]
    for sname, thick in STACKS.items():
        for dx in DXS:
            r = {x["model"]: x for x in rows if x["stack"] == sname and x["dx_mm"] == dx}
            L.append(f"| {sname} ({' / '.join(f'{t:g}' for t in thick)}) | {dx:g} mm | {r['v22']['R_ratio']:.4f} | {r['v23']['R_ratio']:.6f} |")
    open(os.path.join(OUT, "layered_slab.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
