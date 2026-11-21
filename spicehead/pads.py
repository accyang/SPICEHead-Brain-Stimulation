"""Gel-pad electrodes for the resistor network (experiment of 23 Oct 2026).

SimNIBS's TDCS leadfield meshes every electrode as a 2 mm layer of gel (1.0 S/m) on the scalp and
injects the current uniformly over the gel's outer surface (Neumann), so the current spreads
laterally in the gel before it enters the skin. SPICEhead's v22 contact injects uniformly over the
skin-surface nodes under the pad footprint. This module adds the gel layer to the label volume so
that the network can carry the same electrode model:

  labels, pads = add_gel_pads(labels, centres_ijk, radius_vox, thickness_vox, label=GEL_LABEL)

  gel voxels   = non-conductor voxels within `radius_vox` of a pad centre and within `thickness_vox`
                 of the conductor surface (Euclidean distance transform), i.e. a shell of gel over
                 the footprint
  pads[name]   = (i, j, k) index arrays of that pad's gel voxels and of its OUTER layer (gel voxels
                 with a non-conductor face neighbour), where the current is injected

All pads are added at once (passive pads conduct, as in the leadfield). Use GEL_LABEL with
`label_map`/`cond` overrides in mri.model_from_labels; the default model is unchanged.
"""
from __future__ import annotations

import numpy as np

GEL_LABEL = 11
GEL_SIGMA = 1.0          # S/m, SimNIBS default for saline/gel


def add_gel_pads(labels, centres_ijk, radius_vox, thickness_vox, label=GEL_LABEL):
    from scipy.ndimage import distance_transform_edt
    lab = np.array(labels, copy=True)
    cond = lab > 0
    # distance of every non-conductor voxel to the nearest conductor voxel
    dist = distance_transform_edt(~cond)
    shell = (~cond) & (dist <= thickness_vox + 1e-9)
    ii, jj, kk = np.indices(lab.shape, sparse=True)
    pads = {}
    for name, (ci, cj, ck) in centres_ijk.items():
        inside = ((ii - ci) ** 2 + (jj - cj) ** 2 + (kk - ck) ** 2) <= radius_vox ** 2
        sel = shell & inside
        if not sel.any():
            raise ValueError(f"pad {name}: no gel voxels (centre {ci},{cj},{ck})")
        lab[sel] = label
        pads[name] = sel
    # outer layer of each pad: gel voxels that still touch a non-conductor voxel after all pads are added
    full = lab > 0
    padded = np.pad(full, 1, constant_values=False)
    exposed = np.zeros_like(full)
    n = full.shape
    for axis in range(3):
        for s in (-1, 1):
            sl = [slice(1, 1 + n[0]), slice(1, 1 + n[1]), slice(1, 1 + n[2])]
            sl[axis] = slice(1 + s, 1 + s + n[axis])
            exposed |= ~padded[tuple(sl)]
    out = {}
    for name, sel in pads.items():
        top = sel & exposed
        if not top.any():
            top = sel
        out[name] = dict(gel=np.argwhere(sel), top=np.argwhere(top))
    return lab, out


def nodes_from_ijk(model, ijk):
    """Node indices of the given (n, 3) voxel indices (all must be conductor nodes)."""
    ids = model.node_id[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    if (ids < 0).any():
        raise ValueError("gel voxel outside the conductor mask")
    return ids
