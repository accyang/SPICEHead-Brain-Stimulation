"""
Electrode definition and current injection.

Electrodes are modelled as current sources (Manuscript Sec. 2.3): a set of node
indices carrying a total current, distributed equally across the nodes. Moving an
electrode changes only the injected-current vector, not the resistor network,
which is the manuscript's headline advantage over FEM remeshing (Sec. 4).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Electrode:
    """A contact = a set of node indices with a label."""
    nodes: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.nodes = np.atleast_1d(np.asarray(self.nodes, dtype=np.int64))
        if self.nodes.size == 0:
            raise ValueError(f"Electrode '{self.name}' selected zero nodes.")


def nodes_in_sphere(model, center_ijk, radius_nodes):
    """Node indices within ``radius_nodes`` grid units of a centre (i,j,k).

    This is a VOLUMETRIC contact support: every conducting node inside the sphere is selected,
    whatever its tissue, so a scalp electrode can include skull, muscle or other nodes below the
    skin. It is what every published SPICEhead analysis used. For a contact restricted to the
    skin surface under the pad, see scalp_surface_nodes_in_sphere().
    """
    ii, jj, kk = model.grid_coords
    ci, cj, ck = center_ijk
    d2 = (ii - ci) ** 2 + (jj - cj) ** 2 + (kk - ck) ** 2
    return np.flatnonzero(d2 <= radius_nodes ** 2)


def scalp_surface_nodes_in_sphere(model, labels, center_ijk, radius_nodes, scalp_label=5):
    """Outer-skin contact support: scalp nodes inside the sphere that touch the non-conductor.

    A node qualifies if it carries ``scalp_label`` and at least one of its six face neighbours
    lies outside the conductor (air or outside the grid). Current is then injected only through
    the skin surface under the pad footprint, which is closer to a physical electrode than the
    volumetric support of nodes_in_sphere(). Restricting by scalp label alone would still be a
    volumetric support, because the scalp is several voxels thick.
    """
    sel = nodes_in_sphere(model, center_ijk, radius_nodes)
    ii, jj, kk = (c[sel] for c in model.grid_coords)
    cond = np.zeros(model.shape, dtype=bool)
    cond[model.grid_coords] = True
    padded = np.pad(cond, 1, constant_values=False)
    pi, pj, pk = ii + 1, jj + 1, kk + 1
    exposed = np.zeros(sel.size, dtype=bool)
    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        exposed |= ~padded[pi + di, pj + dj, pk + dk]
    keep = exposed & (np.asarray(labels)[ii, jj, kk] == scalp_label)
    return sel[keep]


def contact_nodes(model, center_ijk, radius_nodes):
    """An electrode's contact nodes under the active model version (spicehead.config).

    v22: the skin under the pad (scalp_surface_nodes_in_sphere); v21: the volumetric sphere
    (nodes_in_sphere). The v22 support needs the model's label volume, which
    mri.model_from_labels attaches; a model without one is refused rather than silently given
    a volumetric contact.
    """
    from . import config
    if config.CONTACT == "volume":
        return nodes_in_sphere(model, center_ijk, radius_nodes)
    labels = getattr(model, "labels", None)
    if labels is None:
        raise ValueError("skin-surface contacts need model.labels; build the model with "
                         "mri.model_from_labels or set SPICEHEAD_MODEL=v21")
    nodes = scalp_surface_nodes_in_sphere(model, labels, center_ijk, radius_nodes)
    if nodes.size == 0:
        raise ValueError(f"no skin-surface nodes within {radius_nodes:.1f} grid units of "
                         f"{tuple(center_ijk)}")
    return nodes


def nearest_node(model, center_ijk):
    """Single node index closest to a target (i,j,k)."""
    ii, jj, kk = model.grid_coords
    ci, cj, ck = center_ijk
    d2 = (ii - ci) ** 2 + (jj - cj) ** 2 + (kk - ck) ** 2
    return int(np.argmin(d2))


def pair_injection(model, anode, cathode, current):
    """Injected-current vector (N,) for one electrode pair.

    +current is spread over anode nodes, -current over cathode nodes, so the
    vector sums to zero (charge conservation).
    """
    inj = np.zeros(model.n_nodes)
    inj[anode.nodes] += current / anode.nodes.size
    inj[cathode.nodes] -= current / cathode.nodes.size
    return inj
