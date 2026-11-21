"""
Synthetic geometries and electrode placement for validation.

The homogeneous cylinder reproduces the manuscript's Section 3.1 phantom
("diameter of 100 resistors and a height of 10 resistors ... 3 mm spacing ...
all resistors 1 Ohm"), used to validate SPICEhead against COMSOL (Figs. 4-6)
and to probe the TI steering laws (Figs. 7-8).
"""

from __future__ import annotations

import numpy as np

from .model import SPICEHeadModel
from .electrodes import Electrode, nodes_in_sphere, nearest_node


def cylinder_model(diameter_nodes=100, height_nodes=10, dx=3e-3, sigma=1.0):
    """Homogeneous cylindrical conductor discretized on a cubic grid.

    Returns a SPICEHeadModel whose in-mask nodes form a disk of the given
    diameter, extruded over ``height_nodes`` layers.
    """
    nx = ny = int(diameter_nodes)
    nz = int(height_nodes)
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    R = diameter_nodes / 2.0

    ii, jj = np.mgrid[0:nx, 0:ny]
    disk = (ii - cx) ** 2 + (jj - cy) ** 2 <= R ** 2
    mask = np.repeat(disk[:, :, None], nz, axis=2)
    sigma_vol = np.where(mask, float(sigma), 0.0)
    return SPICEHeadModel(sigma_vol, dx=dx, mask=mask)


def cylinder_center(model):
    nx, ny, nz = model.shape
    return ((nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0)


def rim_point(model, angle_deg, radial_frac=1.0, z=None):
    """A target (i,j,k) on (or inside) the cylinder rim at a given angle.

    angle_deg is measured from the +x axis; radial_frac=1.0 is the rim,
    0.0 is the centre.
    """
    nx, ny, nz = model.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    R = min(cx, cy) * radial_frac
    a = np.deg2rad(angle_deg)
    i = cx + R * np.cos(a)
    j = cy + R * np.sin(a)
    if z is None:
        z = nz // 2
    return (i, j, z)


def place_rim_electrode(model, angle_deg, radius_nodes=3.0, radial_frac=1.0,
                        z=None, name=""):
    """Build an Electrode from nodes near a rim location.

    Falls back to the single nearest node if the sphere selects none (e.g. the
    target sits just outside the mask).
    """
    center = rim_point(model, angle_deg, radial_frac=radial_frac, z=z)
    nodes = nodes_in_sphere(model, center, radius_nodes)
    if nodes.size == 0:
        nodes = np.array([nearest_node(model, center)])
    return Electrode(nodes=nodes, name=name)
