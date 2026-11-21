# -*- coding: utf-8 -*-
"""Homogeneous-disk benchmark against the EXACT analytic field, not a discretized one.

v21_convergence_validation.py builds its reference field by applying SPICEhead's own central
difference to samples of the analytic potential. That measures agreement with an
analytic-potential-derived discrete field: any error the difference operator itself makes is
shared by model and reference and cancels. Here the reference is the closed-form gradient of the
same method-of-images solution, evaluated at the same node positions, so the comparison covers
the differencing error as well. Every statistic is unadjusted (no fitted rescaling), and the
two-way decomposition is reported: model versus discrete reference, discrete reference versus
exact field, and model versus exact field.

For the two point currents +q at z_s1 and -q at z_s2 in a disk of radius R with an insulating rim,
  phi(z) = -(q_line / 2 pi sigma) * sum_s sign_s [ln|z - z_s| + ln|z - R^2/conj(z_s)|]
  E(z)   = -grad phi = (q_line / 2 pi sigma) * sum_s sign_s [1/conj(z - z_s) + 1/conj(z - z_img)]
with E written as the complex number Ex + i Ey and q_line = q / dx (the slab is one voxel thick).

Writes results/v21_validation/disk_analytic_gradient.csv; the original convergence.csv is not
touched.

  python pipeline/v21_disk_analytic_gradient.py
"""
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)

from v21_convergence_validation import (sh, netlist, analytic_disk_potential, agreement_metrics,
                                        DIAM_MM, SIGMA, CURRENT)

OUT = os.path.join(ROOT, "results", "v21_validation", "disk_analytic_gradient.csv")
DXS = [5.0, 4.0, 3.0, 2.0, 1.0]
EXCLUDE_MM = 30.0      # same physical source neighbourhood as the original benchmark


def exact_inplane_field(m, src_nodes, src_q):
    """|E| in the disk plane from the closed-form gradient, at every node."""
    nx, ny, _ = m.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    R = min(cx, cy) * m.dx
    ii, jj, _ = m.grid_coords
    z = (ii - cx) * m.dx + 1j * (jj - cy) * m.dx
    E = np.zeros(m.n_nodes, dtype=complex)
    for node, q in zip(src_nodes, src_q):
        zs = (ii[node] - cx) * m.dx + 1j * (jj[node] - cy) * m.dx
        zimg = R ** 2 / np.conj(zs)
        k = (q / m.dx) / (2 * np.pi * SIGMA)
        with np.errstate(divide="ignore", invalid="ignore"):
            E += k * (1.0 / np.conj(z - zs) + 1.0 / np.conj(z - zimg))
    return np.abs(E)


def unadjusted(x, y):
    """Raw agreement of x with reference y: no slope or intercept is applied first."""
    d = x - y
    return dict(nmae_pct=100 * np.mean(np.abs(d)) / np.mean(np.abs(y)),
                mean_signed_pct=100 * np.mean(d) / np.mean(y),
                median_abs_rel_pct=100 * np.median(np.abs(d) / y),
                p95_abs_rel_pct=100 * np.percentile(np.abs(d) / y, 95),
                slope=agreement_metrics(x, y)["slope"], r=float(np.corrcoef(x, y)[0, 1]))


def run_dx(dx_mm):
    n = int(round(DIAM_MM / dx_mm))
    m = sh.cylinder_model(diameter_nodes=n, height_nodes=1, dx=dx_mm * 1e-3, sigma=SIGMA)
    a = sh.Electrode(sh.nearest_node(m, sh.rim_point(m, 25, radial_frac=0.9)), "A+")
    c = sh.Electrode(sh.nearest_node(m, sh.rim_point(m, 205, radial_frac=0.9)), "A-")
    inj = sh.pair_injection(m, a, c, CURRENT)
    ground = netlist._pick_ground(inj, None)
    method = "direct" if m.n_nodes <= 60000 else "cg"
    V = sh.solve_potentials(m.G, inj, ground_node=ground, method=method, tol=1e-9)
    src = [int(a.nodes[0]), int(c.nodes[0])]
    Va = analytic_disk_potential(m, src, [CURRENT, -CURRENT])

    model = m.grid_to_nodes(np.nan_to_num(sh.field_magnitude_inplane(sh.efield_components(m, V))))
    disc = m.grid_to_nodes(np.nan_to_num(sh.field_magnitude_inplane(sh.efield_components(m, Va))))
    exact = exact_inplane_field(m, src, [CURRENT, -CURRENT])

    ii, jj, _ = m.grid_coords
    keep = np.isfinite(model) & np.isfinite(disc) & np.isfinite(exact) & (model > 0) & (exact > 0)
    for node in src:
        keep &= np.hypot((ii - ii[node]) * dx_mm, (jj - jj[node]) * dx_mm) > EXCLUDE_MM
    x, yd, ye = model[keep], disc[keep], exact[keep]
    row = dict(dx_mm=dx_mm, n_nodes=int(m.n_nodes), n_compared=int(keep.sum()), solver=method)
    for tag, (p, q) in (("model_vs_discrete", (x, yd)), ("discrete_vs_exact", (yd, ye)),
                        ("model_vs_exact", (x, ye))):
        for k, v in unadjusted(p, q).items():
            row[f"{tag}_{k}"] = float(v)
    return row


def main():
    rows = []
    for dx in DXS:
        r = run_dx(dx)
        rows.append(r)
        print(f"dx={dx:g} mm ({r['n_nodes']:,} nodes, {r['solver']}): nMAE model-vs-discrete "
              f"{r['model_vs_discrete_nmae_pct']:.3f}% | discrete-vs-exact "
              f"{r['discrete_vs_exact_nmae_pct']:.3f}% | model-vs-exact "
              f"{r['model_vs_exact_nmae_pct']:.3f}% (signed {r['model_vs_exact_mean_signed_pct']:+.3f}%,"
              f" p95 {r['model_vs_exact_p95_abs_rel_pct']:.2f}%, slope "
              f"{r['model_vs_exact_slope']:.4f})", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
