"""
Validation tests for the SPICEhead reconstruction.

Runnable directly (``python tests/test_validation.py``) or under pytest.

1. test_ngspice_agreement   -- native solver == ngspice on the same netlist
                               (independent verification of the circuit solve).
2. test_analytic_disk       -- discrete solution -> analytic 2D-disk Green's
                               function (continuum limit; the COMSOL stand-in).
3. test_uniform_field_k     -- physical conductances give normalization k = 1.
4. test_ti_current_ratio    -- TI envelope peak moves toward the LOWER-current
                               electrode, monotonically (Manuscript Sec. 3.2-3.3).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spicehead as sh
from spicehead import netlist
from spicehead.model import SPICEHeadModel


# ----------------------------------------------------------------------
def analytic_disk_potential(model, src_nodes, src_currents, sigma):
    """Analytic 2D-disk potential via method of images (insulating rim).

    Balanced point sources at the given node locations. Returns the potential
    at each node (V), up to an additive constant.
    """
    nx, ny, nz = model.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    R = min(cx, cy) * model.dx
    ii, jj, kk = model.grid_coords
    z = (ii - cx) * model.dx + 1j * (jj - cy) * model.dx

    phi = np.zeros(model.n_nodes)
    for node, q in zip(src_nodes, src_currents):
        zs = (ii[node] - cx) * model.dx + 1j * (jj[node] - cy) * model.dx
        z_img = R ** 2 / np.conj(zs) if abs(zs) > 0 else np.inf
        with np.errstate(divide="ignore"):
            term = np.log(np.abs(z - zs)) + np.log(np.abs(z - z_img))
        phi += -(q / (2.0 * np.pi * sigma)) * term
    return phi


def test_ngspice_agreement():
    if not netlist.ngspice_available():
        print("[ngspice] SKIP (ngspice not on PATH)")
        return None
    m = sh.cylinder_model(diameter_nodes=16, height_nodes=1, dx=3e-3, sigma=1.0)
    a = sh.place_rim_electrode(m, 20, radius_nodes=1.5, name="A+")
    c = sh.place_rim_electrode(m, 200, radius_nodes=1.5, name="A-")
    inj = sh.pair_injection(m, a, c, 2e-3)
    ground = netlist._pick_ground(inj, None)
    Vn = sh.solve_potentials(m.G, inj, ground_node=ground)
    Vs = netlist.run_ngspice(m, inj, ground_node=ground)
    mask = np.ones(m.n_nodes, bool); mask[ground] = False
    rel = np.abs(Vn[mask] - Vs[mask]).max() / np.abs(Vn[mask]).max()
    print(f"[ngspice] relative max difference = {rel:.2e}")
    assert rel < 1e-4, f"native vs ngspice mismatch: {rel}"
    return rel


def test_analytic_disk():
    sigma = 1.0
    m = sh.cylinder_model(diameter_nodes=80, height_nodes=1, dx=3e-3, sigma=sigma)
    a = sh.Electrode(sh.nearest_node(m, sh.rim_point(m, 25, radial_frac=0.9)), "A+")
    c = sh.Electrode(sh.nearest_node(m, sh.rim_point(m, 205, radial_frac=0.9)), "A-")
    inj = sh.pair_injection(m, a, c, 2e-3)
    ground = netlist._pick_ground(inj, None)
    V = sh.solve_potentials(m.G, inj, ground_node=ground)
    Va = analytic_disk_potential(
        m, [int(a.nodes[0]), int(c.nodes[0])], [2e-3, -2e-3], sigma
    )
    ii, jj, kk = m.grid_coords
    keep = np.ones(m.n_nodes, bool)
    for node in (int(a.nodes[0]), int(c.nodes[0])):
        keep &= np.hypot(ii - ii[node], jj - jj[node]) > 4
    r = np.corrcoef(V[keep], Va[keep])[0, 1]
    print(f"[analytic_disk] Pearson r (discrete vs analytic) = {r:.5f}")
    assert r > 0.99, f"analytic-disk correlation too low: {r}"
    return r


def test_uniform_field_k():
    """Uniform current through a slab -> E = I/(sigma*A) exactly, so k = 1.

    Demonstrates that physical conductances (g = sigma*dx) make the output
    physically scaled: the manuscript's normalization factor k ~ 1, not ~98.
    """
    nx, ny, nz = 24, 10, 10
    dx, sigma, I = 3e-3, 0.3, 1e-3
    m = SPICEHeadModel(np.full((nx, ny, nz), sigma), dx=dx)
    ii, jj, kk = m.grid_coords
    face_in = np.flatnonzero(ii == 0)
    face_out = np.flatnonzero(ii == nx - 1)
    inj = np.zeros(m.n_nodes)
    inj[face_in] += I / face_in.size
    inj[face_out] -= I / face_out.size
    V = sh.solve_potentials(m.G, inj, ground_node=int(face_out[0]))
    Ex, Ey, Ez = sh.efield_components(m, V)
    E_model = np.nanmean(np.abs(Ex[3:-3, 2:-2, 2:-2]))
    A = (ny * nz) * dx ** 2           # each node carries dx^2 of cross-section
    E_analytic = I / (sigma * A)
    k = E_analytic / E_model
    print(f"[uniform_field] E_model={E_model:.4f}  E_analytic={E_analytic:.4f}"
          f"  k={k:.4f} V/m")
    assert abs(k - 1.0) < 0.02, f"k should be ~1 with physical conductances: {k}"
    return k


def _montage_45deg(model, radius_nodes=3.0):
    a1 = sh.place_rim_electrode(model, 45, radius_nodes, name="P1+")
    c1 = sh.place_rim_electrode(model, 135, radius_nodes, name="P1-")
    a2 = sh.place_rim_electrode(model, 315, radius_nodes, name="P2+")
    c2 = sh.place_rim_electrode(model, 225, radius_nodes, name="P2-")
    return (a1, c1), (a2, c2)


def test_ti_current_ratio():
    """Envelope peak shifts toward the lower-current pair, monotonically."""
    m = sh.cylinder_model(diameter_nodes=60, height_nodes=1, dx=3e-3, sigma=1.0)
    pair1, pair2 = _montage_45deg(m)  # pair1 = top (+y), pair2 = bottom (-y)
    cy = (m.shape[1] - 1) / 2.0
    ratios = [(1e-3, 3e-3), (2e-3, 2e-3), (3e-3, 1e-3)]
    offsets = []
    for i1, i2 in ratios:
        res = sh.simulate_ti(m, pair1, pair2, current1=i1, current2=i2)
        Ay = np.nan_to_num(res.envelope_components[1][:, :, 0])
        Ay = np.where(m.mask[:, :, 0], Ay, 0.0)
        j = np.unravel_index(np.argmax(Ay), Ay.shape)[1]
        offsets.append(j - cy)
    print(f"[current_ratio] peak y-offsets {ratios} = {offsets}")
    # pair1 (top) low current -> peak toward top (+y); high -> toward bottom (-y)
    assert offsets[0] > offsets[1] > offsets[2], \
        f"peak did not move monotonically toward lower-current side: {offsets}"
    assert offsets[0] > 0 > offsets[2], f"peak did not change sign: {offsets}"
    return offsets


if __name__ == "__main__":
    print("Running SPICEhead validation tests\n" + "-" * 44)
    test_ngspice_agreement()
    test_analytic_disk()
    test_uniform_field_k()
    test_ti_current_ratio()
    print("-" * 44 + "\nAll validation checks passed.")
