# -*- coding: utf-8 -*-
"""Fast rotation-invariant TI envelope, for the cohort sweeps.

``field.ti_envelope_max_direction`` is the readable reference implementation and stays the
definition of record. It is ~25x slower than the per-axis form, which makes a 300-subject
11,600-montage sweep infeasible (37 h for one analysis). The cost is incidental, not intrinsic:
it upcasts float32 fields to float64, allocates several (nx,ny,nz,3) temporaries via stack/where,
and calls ``np.cross``.

This version computes the same closed form componentwise, in the input dtype, on FLAT arrays --
so callers can pass only the voxels they actually score (grey matter is ~18% of the volume).

    A = 2|b|                             if |b| <= |a| cos(alpha)      (a is the larger field)
    A = 2 |b x (a-b)| / |a-b|            otherwise

``test_matches_reference`` asserts agreement with the reference implementation; run this module
directly to check.
"""
import numpy as np


def envelope_max_direction_flat(a1, a2, a3, b1, b2, b3, out=None):
    """Rotation-invariant envelope for flat component arrays. Returns a flat array."""
    # order so that (a1,a2,a3) is the larger-magnitude field
    na2 = a1 * a1 + a2 * a2 + a3 * a3
    nb2 = b1 * b1 + b2 * b2 + b3 * b3
    swap = nb2 > na2
    if swap.any():
        a1, b1 = np.where(swap, b1, a1), np.where(swap, a1, b1)
        a2, b2 = np.where(swap, b2, a2), np.where(swap, a2, b2)
        a3, b3 = np.where(swap, b3, a3), np.where(swap, a3, b3)
        na2, nb2 = np.where(swap, nb2, na2), np.where(swap, na2, nb2)

    # acute angle: flip b if the carriers oppose
    dot = a1 * b1 + a2 * b2 + a3 * b3
    flip = dot < 0
    if flip.any():
        b1 = np.where(flip, -b1, b1)
        b2 = np.where(flip, -b2, b2)
        b3 = np.where(flip, -b3, b3)
        dot = np.abs(dot)

    na = np.sqrt(na2)
    nb = np.sqrt(nb2)

    # d = a - b ;  cross = b x d = b x a   (since b x b = 0)
    d1, d2, d3 = a1 - b1, a2 - b2, a3 - b3
    c1 = b2 * d3 - b3 * d2
    c2 = b3 * d1 - b1 * d3
    c3 = b1 * d2 - b2 * d1
    with np.errstate(invalid="ignore", divide="ignore"):
        ndiff = np.sqrt(d1 * d1 + d2 * d2 + d3 * d3)
        oblique = 2.0 * np.sqrt(c1 * c1 + c2 * c2 + c3 * c3) / ndiff
        # |b| <= |a| cos(alpha)  <=>  |b|^2 <= dot   (multiply both sides by |b|)
        res = np.where(nb2 <= dot, 2.0 * nb, oblique)
    res = np.nan_to_num(res, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if out is not None:
        out[...] = res
        return out
    return res


def envelope_max_direction_masked(E1, E2, mask):
    """Envelope on the masked voxels only. E1/E2 are (Ex,Ey,Ez) grids; returns a flat array."""
    a1, a2, a3 = (np.asarray(c)[mask] for c in E1)
    b1, b2, b3 = (np.asarray(c)[mask] for c in E2)
    return envelope_max_direction_flat(a1, a2, a3, b1, b2, b3)


def test_matches_reference(shape=(24, 31, 29), seed=0, tol=1e-5):
    from .field import ti_envelope_max_direction
    rng = np.random.default_rng(seed)
    cases = []
    for scale in (1.0, 1e-6, 1e3):
        cases.append(tuple(rng.standard_normal(shape).astype(np.float32) * scale for _ in range(3)))
    E1, E2 = cases[0], cases[1]
    ref = np.asarray(ti_envelope_max_direction(E1, E2), dtype=float)
    m = np.ones(shape, bool)
    got = envelope_max_direction_masked(E1, E2, m).reshape(shape)
    den = max(float(np.nanmax(np.abs(ref))), 1e-30)
    err = float(np.nanmax(np.abs(got - ref))) / den
    assert err < tol, f"max relative deviation {err:.2e}"

    # degenerate cases the closed form has to survive
    z = np.zeros(8, np.float32)
    o = np.ones(8, np.float32)
    for args in ((z, z, z, z, z, z),          # both fields zero
                 (o, z, z, o, z, z),          # parallel
                 (o, z, z, -o, z, z),         # antiparallel
                 (o, z, z, z, o, z)):         # orthogonal
        v = envelope_max_direction_flat(*[x.copy() for x in args])
        assert np.all(np.isfinite(v)), "non-finite in degenerate case"
    return err


if __name__ == "__main__":
    import time
    e = test_matches_reference()
    print(f"matches reference to {e:.2e} relative; degenerate cases finite")
    from .field import ti_envelope_max_direction, ti_envelope_magnitude
    rng = np.random.default_rng(1)
    sh = (40, 53, 53)
    E1 = tuple(rng.standard_normal(sh).astype(np.float32) for _ in range(3))
    E2 = tuple(rng.standard_normal(sh).astype(np.float32) for _ in range(3))
    gm = rng.random(sh) < 0.178          # realistic grey-matter fraction
    for fn, name in ((lambda: ti_envelope_magnitude(E1, E2), "per-axis (full grid)"),
                     (lambda: ti_envelope_max_direction(E1, E2), "max-dir reference"),
                     (lambda: envelope_max_direction_masked(E1, E2, gm), "max-dir fast (GM only)")):
        fn()
        t = time.perf_counter()
        for _ in range(5):
            fn()
        print(f"  {name:<26}{(time.perf_counter()-t)/5*1000:7.2f} ms")
