"""
Electric-field and temporal-interference envelope computation.

From node potentials V we recover the field by central differences
(Manuscript Eqs. 2-4):

    E_x(i,j,k) = k * [V(i+1,j,k) - V(i-1,j,k)] / (2 dx)      (and y, z)

and the TI envelope amplitude along each axis (Manuscript Eq. 5, after
Huang & Parra 2019):

    A = 2 * min(|a|, |b|)

where a and b are the field amplitudes produced by the two electrode pairs
(carriers f1 and f2). Because the medium is linear and purely resistive, the
spatial amplitude pattern of each pair equals its DC solution scaled by the
pair current, so a and b come from two independent .op solves.

Normalization factor k
----------------------
k rescales the raw circuit output to physical V/m. With the *physical*
conductances used in model.py (g = sigma*dx) and currents in amperes, V is
already in volts and E in V/m, so k ~ 1. ``calibrate_k`` recovers k as the
slope of a reference-vs-model regression (the manuscript's procedure), which is
useful when comparing against an external solver in arbitrary units.
"""

from __future__ import annotations

import numpy as np


def efield_components(model, V, k=1.0):
    """Central-difference E = -k*grad(V) on the grid.

    Returns (Ex, Ey, Ez), each a (nx,ny,nz) array with np.nan where a central
    difference is unavailable (node or a required neighbour is outside the mask).

    The components are along world axes. For v21/v22 the grid is world-aligned, so the
    voxel-axis differences are returned unchanged. A v23 grid follows the segmentation's voxel
    axes; its model carries ``axes_to_world`` (unit voxel-axis directions as columns) and the
    voxel-axis components are rotated, E_world[w] = sum_v R[w, v] * E_voxel[v].
    """
    Vg = model.nodes_to_grid(V, fill=np.nan)
    dx = model.dx
    comps = []
    for axis in range(3):
        E = np.full(model.shape, np.nan)
        if model.shape[axis] < 3:
            comps.append(E)  # can't central-difference a thin axis
            continue
        fwd = np.take(Vg, np.arange(2, model.shape[axis]), axis=axis)
        bwd = np.take(Vg, np.arange(0, model.shape[axis] - 2), axis=axis)
        center = tuple(
            slice(1, -1) if d == axis else slice(None) for d in range(3)
        )
        with np.errstate(invalid="ignore"):
            E[center] = -k * (fwd - bwd) / (2.0 * dx)
        comps.append(E)
    R = getattr(model, "axes_to_world", None)
    if R is not None:
        # same expression as the validated prototype (pipeline/nchc/homog_experiment.py)
        comps = [sum(R[w, v] * comps[v] for v in range(3) if abs(R[w, v]) > 1e-12) for w in range(3)]
    return tuple(comps)


def field_magnitude(components):
    """|E| from (Ex,Ey,Ez), propagating nan.

    Note: on a slab only one voxel thick, the out-of-plane component cannot be
    central-differenced and is nan everywhere, which would make the magnitude
    nan everywhere. Use ``field_magnitude_inplane`` for such 2-D models.
    """
    return np.sqrt(sum(np.asarray(c) ** 2 for c in components))


def field_magnitude_inplane(components, axes=(0, 1)):
    """In-plane |E| from selected components (default x,y).

    The correct magnitude for a 2-D (single-slice) model, where the analytic
    reference is itself two-dimensional and the out-of-plane component is
    identically zero rather than undefined.
    """
    comps = [np.asarray(components[a]) for a in axes]
    return np.sqrt(sum(c ** 2 for c in comps))


def ti_envelope_components(E1, E2):
    """Per-axis TI envelope A_axis = 2*min(|a|,|b|)  (Manuscript Eq. 5).

    E1, E2 are the (Ex,Ey,Ez) tuples from the two electrode pairs.
    """
    return tuple(
        2.0 * np.minimum(np.abs(a), np.abs(b)) for a, b in zip(E1, E2)
    )


def ti_envelope_magnitude(E1, E2):
    """Magnitude of the per-axis envelope vector, sqrt(Ax^2+Ay^2+Az^2).

    NOTE: this quantity is **coordinate-frame dependent** — applying Eq. 5 to
    each Cartesian component and then taking the norm gives a different answer
    in a rotated frame. Use ``ti_envelope_max_direction`` for the rotation-
    invariant maximum-modulation envelope.
    """
    env = ti_envelope_components(E1, E2)
    return field_magnitude(env)


def ti_envelope_max_direction(E1, E2):
    """Rotation-invariant TI envelope: max over directions n of 2*min(|E1.n|,|E2.n|).

    This is the maximum-modulation-amplitude envelope of Grossman et al. (2017),
    evaluated with the standard closed form. Writing the two carrier fields as
    vectors E1, E2 (with |E1| >= |E2| and the acute angle alpha between them):

        A_max = 2|E2|                                if |E2| <= |E1| cos(alpha)
        A_max = 2 |E2 x (E1 - E2)| / |E1 - E2|       otherwise

    Unlike the per-axis form (Eq. 5 applied to Ex, Ey, Ez separately), this
    depends only on the geometry of the two field vectors and is therefore
    invariant to the choice of coordinate frame.

    Parameters
    ----------
    E1, E2 : tuples (Ex, Ey, Ez) of grids, one per electrode pair.

    Returns
    -------
    Grid of the maximum envelope amplitude.
    """
    a = np.stack([np.asarray(c, dtype=float) for c in E1], axis=-1)
    b = np.stack([np.asarray(c, dtype=float) for c in E2], axis=-1)

    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)

    # order so that |a| >= |b|
    swap = nb > na
    a_, b_ = np.where(swap[..., None], b, a), np.where(swap[..., None], a, b)
    na_, nb_ = np.maximum(na, nb), np.minimum(na, nb)

    # take the acute angle between the carriers
    flip = np.sum(a_ * b_, axis=-1) < 0
    b_ = np.where(flip[..., None], -b_, b_)

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_alpha = np.sum(a_ * b_, axis=-1) / (na_ * nb_)
        diff = a_ - b_
        ndiff = np.linalg.norm(diff, axis=-1)
        cross = np.cross(b_, diff)
        oblique = 2.0 * np.linalg.norm(cross, axis=-1) / ndiff
        aligned = 2.0 * nb_
        out = np.where(nb_ <= na_ * cos_alpha, aligned, oblique)

    # degenerate points (either carrier ~ zero) contribute no modulation
    out = np.where((na_ == 0) | (nb_ == 0), 0.0, out)
    return out


def calibrate_k(model_raw, reference):
    """Least-squares slope k so that k * model_raw ~ reference.

    Mirrors the manuscript's k (slope of the SPICEhead-vs-FEM regression,
    Sec. 3.1 / Fig. 6). Operates on matched, finite samples only.
    """
    x = np.asarray(model_raw, dtype=float).ravel()
    y = np.asarray(reference, dtype=float).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    denom = float(np.dot(x, x))
    if denom == 0 or x.size < 2:
        return float("nan"), float("nan")
    k = float(np.dot(x, y) / denom)  # slope through origin
    if x.std() == 0 or y.std() == 0:
        return k, float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return k, r
