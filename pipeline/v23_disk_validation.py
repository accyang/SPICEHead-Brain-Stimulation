# -*- coding: utf-8 -*-
"""v23 versions of the homogeneous-disk benchmarks of Results section 1 (28 Oct 2026; PI: Paper A's final results are v23).

v21_convergence_validation.py and v21_disk_analytic_gradient.py build the disk as a nearest-label network (a node per
voxel centre inside the circle); Paper A's v22 record reused their outputs. Under v23 the disk is voxelized at the
CHARM fine resolution (0.5 mm): 300 mm diameter, one block thick, homogenized to each grid with spicehead.homogenize
(fill >= 0.5), so partially filled rim blocks carry their true tube conductance. The analytic reference is the same
method-of-images solution, for the true radius (150 mm) about the disk centre. Electrodes (rim angles 25 and 205 deg at
0.9 of the radius), current, the 30 mm source exclusion and every statistic are those of the two v21 scripts, and the
outputs repeat their columns:
  results_v23/v21_validation/convergence.csv            (agreement with the discrete analytic field)
  results_v23/v21_validation/disk_analytic_gradient.csv (unadjusted, against the exact gradient; two-way decomposition)
plus the uniform-slab check (k = 1) on a homogenized slab, printed. No subject data.
    SPICEHEAD_MODEL=v23 python pipeline/v23_disk_validation.py
"""
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)

from spicehead import config                                                    # noqa: E402
from spicehead.homogenize import homogenized_model                              # noqa: E402
from v21_convergence_validation import sh, netlist, agreement_metrics, DIAM_MM, SIGMA, CURRENT   # noqa: E402
from v21_disk_analytic_gradient import unadjusted, EXCLUDE_MM, DXS              # noqa: E402

FINE = 0.5                                       # mm, the CHARM fine grid
R_M = DIAM_MM / 2 * 1e-3                         # true disk radius, m
LMAP, COND = {1: "disk"}, {"disk": SIGMA}
OUT = os.path.join(config.RESULTS, "v21_validation")


def fine_affine(nx, ny):
    aff = np.diag([FINE, FINE, FINE, 1.0])
    aff[:3, 3] = [-(nx * FINE) / 2 + FINE / 2, -(ny * FINE) / 2 + FINE / 2, 0.0]   # voxel centres; grid centred at 0
    return aff


def disk_model(dx_mm):
    f, n = int(round(dx_mm / FINE)), int(round(DIAM_MM / dx_mm))
    c = (np.arange(n * f) + 0.5) * FINE - DIAM_MM / 2                           # fine voxel centres, mm
    X, Y = np.meshgrid(c, c, indexing="ij")
    fine = np.repeat(((X ** 2 + Y ** 2) <= (DIAM_MM / 2) ** 2)[:, :, None], f, axis=2).astype(np.int16)   # one block thick
    model, _ = homogenized_model(fine, fine_affine(n * f, n * f), dx_mm, label_map=LMAP, cond=COND)
    return model


def images(m, src, q):
    """Method-of-images potential and exact in-plane |E| for the true radius, at every node (block centres)."""
    nx, ny, _ = m.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0                                     # disk centre in block coordinates
    ii, jj, _ = m.grid_coords
    z = (ii - cx) * m.dx + 1j * (jj - cy) * m.dx
    phi, E = np.zeros(m.n_nodes), np.zeros(m.n_nodes, dtype=complex)
    for node, qq in zip(src, q):
        zs = (ii[node] - cx) * m.dx + 1j * (jj[node] - cy) * m.dx
        zimg = R_M ** 2 / np.conj(zs)
        k = (qq / m.dx) / (2 * np.pi * SIGMA)                                  # line current through the slab thickness
        with np.errstate(divide="ignore", invalid="ignore"):
            phi += -k * (np.log(np.abs(z - zs)) + np.log(np.abs(z - zimg)))
            E += k * (1.0 / np.conj(z - zs) + 1.0 / np.conj(z - zimg))
    return phi, np.abs(E)


def run_dx(dx_mm):
    m = disk_model(dx_mm)
    a = sh.Electrode(sh.nearest_node(m, sh.rim_point(m, 25, radial_frac=0.9)), "A+")
    c = sh.Electrode(sh.nearest_node(m, sh.rim_point(m, 205, radial_frac=0.9)), "A-")
    inj = sh.pair_injection(m, a, c, CURRENT)
    method = "direct" if m.n_nodes <= 60000 else "cg"
    V = sh.solve_potentials(m.G, inj, ground_node=netlist._pick_ground(inj, None), method=method, tol=1e-9)
    src = [int(a.nodes[0]), int(c.nodes[0])]
    Va, exact = images(m, src, [CURRENT, -CURRENT])
    model = m.grid_to_nodes(np.nan_to_num(sh.field_magnitude_inplane(sh.efield_components(m, V))))
    disc = m.grid_to_nodes(np.nan_to_num(sh.field_magnitude_inplane(sh.efield_components(m, Va))))
    ii, jj, _ = m.grid_coords
    near = np.zeros(m.n_nodes, bool)
    for node in src:
        near |= np.hypot((ii - ii[node]) * dx_mm, (jj - jj[node]) * dx_mm) <= EXCLUDE_MM
    # convergence.csv (v21_convergence_validation.run_dx)
    keep = np.isfinite(model) & np.isfinite(disc) & (model > 0) & ~near
    conv = agreement_metrics(model[keep], disc[keep])
    top = lambda v: v >= np.quantile(v, 0.90)
    mx, my = top(model[keep]), top(disc[keep])
    vi = np.flatnonzero(np.isfinite(model) & np.isfinite(disc) & (model > 0))
    pm = vi[np.argmax(model[vi])]
    conv.update(dx_mm=dx_mm, n_nodes=int(m.n_nodes), dice=2 * np.sum(mx & my) / (mx.sum() + my.sum()),
                peak_err_mm=min(np.hypot((ii[pm] - ii[s]) * dx_mm, (jj[pm] - jj[s]) * dx_mm) for s in src))
    # disk_analytic_gradient.csv (v21_disk_analytic_gradient.run_dx)
    keep2 = keep & np.isfinite(exact) & (exact > 0)
    x, yd, ye = model[keep2], disc[keep2], exact[keep2]
    grad = dict(dx_mm=dx_mm, n_nodes=int(m.n_nodes), n_compared=int(keep2.sum()), solver=method)
    for tag, (p, q) in (("model_vs_discrete", (x, yd)), ("discrete_vs_exact", (yd, ye)), ("model_vs_exact", (x, ye))):
        for k, v in unadjusted(p, q).items():
            grad[f"{tag}_{k}"] = float(v)
    return conv, grad


def uniform_slab_k():
    """v21's uniform-current slab (24 x 10 x 10 nodes at 3 mm) built by homogenization: k = E_analytic / E_model."""
    dx, f = 3.0, 6
    nx, ny, nz = 24, 10, 10
    fine = np.ones((nx * f, ny * f, nz * f), np.int16)
    m, _ = homogenized_model(fine, np.diag([FINE, FINE, FINE, 1.0]), dx, label_map=LMAP, cond=COND)
    ii, _, _ = m.grid_coords
    inj = np.zeros(m.n_nodes)
    inj[ii == 0] += CURRENT / np.sum(ii == 0)
    inj[ii == nx - 1] -= CURRENT / np.sum(ii == nx - 1)
    V = sh.solve_potentials(m.G, inj, ground_node=int(np.flatnonzero(ii == nx - 1)[0]))
    Ex, _, _ = sh.efield_components(m, V)
    return (CURRENT / (SIGMA * (ny * nz) * (dx * 1e-3) ** 2)) / np.nanmean(np.abs(Ex[3:-3, 2:-2, 2:-2]))


def main():
    assert config.MODEL == "v23", "run with SPICEHEAD_MODEL=v23"
    os.makedirs(OUT, exist_ok=True)
    print(f"[physical units] homogenized uniform slab k = {uniform_slab_k():.6f}  (target 1)", flush=True)
    conv, grad = [], []
    for dx in DXS:
        c, g = run_dx(dx)
        conv.append(c); grad.append(g)
        print(f"dx={dx:g} mm ({c['n_nodes']:,} nodes, {g['solver']}): r {c['r']:.5f}, nMAE vs discrete {c['nmae']:.4f}, "
              f"Dice {c['dice']:.3f}, peak error {c['peak_err_mm']:.1f} mm | vs exact gradient nMAE {g['model_vs_exact_nmae_pct']:.2f} %", flush=True)
    ck = ["dx_mm", "n_nodes", "r", "r2", "slope", "intercept", "nmae", "ba_bias", "ba_lo", "ba_hi", "dice", "peak_err_mm"]
    with open(os.path.join(OUT, "convergence.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ck); w.writeheader(); w.writerows({k: r[k] for k in ck} for r in conv)
    with open(os.path.join(OUT, "disk_analytic_gradient.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grad[0])); w.writeheader(); w.writerows(grad)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
