"""
Reproduce the manuscript's cylinder validation (Figs. 4-8).

Produces:
  - envelope map for the two-pair 45-degree montage (cf. Fig. 4)
  - scatter of SPICEhead vs. analytic 2D-disk field (cf. Fig. 6)
  - current-ratio steering: peak of the y-envelope vs. current ratio (cf. Fig. 8)

Run:  python examples/cylinder_validation.py
Saves PNGs next to this file.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spicehead as sh
from spicehead import netlist
from tests.test_validation import analytic_disk_potential

OUT = os.path.dirname(os.path.abspath(__file__))


def midslice(grid):
    return np.nan_to_num(grid[:, :, grid.shape[2] // 2])


def montage_45deg(model, radius_nodes=3.0):
    """Two pairs forming a rectangle at 45 deg: pair1 across the top, pair2
    across the bottom (electrodes at 45/135 and 225/315 deg)."""
    a1 = sh.place_rim_electrode(model, 45, radius_nodes, name="P1+")
    c1 = sh.place_rim_electrode(model, 135, radius_nodes, name="P1-")
    a2 = sh.place_rim_electrode(model, 315, radius_nodes, name="P2+")
    c2 = sh.place_rim_electrode(model, 225, radius_nodes, name="P2-")
    return (a1, c1), (a2, c2)


def fig_envelope_map(model, res):
    env = midslice(res.envelope)
    disk = midslice(model.mask.astype(float))
    env = np.ma.masked_where(disk < 0.5, env)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(env.T, origin="lower", cmap="viridis")
    ax.set_title("SPICEhead envelope amplitude (cylinder, 2 mA:2 mA)")
    ax.set_xlabel("x (nodes)"); ax.set_ylabel("y (nodes)")
    fig.colorbar(im, ax=ax, label="envelope |A| (V/m)")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_cylinder_envelope.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_scatter_vs_analytic(model):
    sigma = 1.0
    a = sh.Electrode(sh.nearest_node(model, sh.rim_point(model, 25, radial_frac=0.9)), "A+")
    c = sh.Electrode(sh.nearest_node(model, sh.rim_point(model, 205, radial_frac=0.9)), "A-")
    inj = sh.pair_injection(model, a, c, 2e-3)
    ground = netlist._pick_ground(inj, None)
    V = sh.solve_potentials(model.G, inj, ground_node=ground)
    Va = analytic_disk_potential(model, [int(a.nodes[0]), int(c.nodes[0])],
                                 [2e-3, -2e-3], sigma)
    Emod = sh.field_magnitude(sh.efield_components(model, V))
    Eana = sh.field_magnitude(sh.efield_components(model, Va))
    x = model.grid_to_nodes(np.nan_to_num(Emod))
    y = model.grid_to_nodes(np.nan_to_num(Eana))
    ii, jj, kk = model.grid_coords
    keep = np.ones(model.n_nodes, bool)
    for node in (int(a.nodes[0]), int(c.nodes[0])):
        keep &= np.hypot(ii - ii[node], jj - jj[node]) > 5
    keep &= np.isfinite(x) & np.isfinite(y) & (y < np.percentile(y[keep], 99))
    k, r = sh.calibrate_k(x[keep], y[keep])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x[keep], y[keep], s=4, alpha=0.3)
    lim = [0, np.percentile(x[keep], 99.5)]
    ax.plot(lim, [k * v for v in lim], "r-", lw=2, label=f"k={k:.3f}")
    ax.set_xlabel("SPICEhead |E| (V/m)"); ax.set_ylabel("analytic |E| (V/m)")
    ax.set_title(f"SPICEhead vs analytic disk (r={r:.4f})")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig_cylinder_scatter.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p, k, r


def fig_current_ratio(model):
    pair1, pair2 = montage_45deg(model)
    ratios = [(1e-3, 3e-3), (1.5e-3, 2.5e-3), (2e-3, 2e-3),
              (2.5e-3, 1.5e-3), (3e-3, 1e-3)]
    cy = (model.shape[1] - 1) / 2.0
    peaks = []
    maps = []
    for i1, i2 in ratios:
        res = sh.simulate_ti(model, pair1, pair2, current1=i1, current2=i2)
        Ay = np.nan_to_num(res.envelope_components[1][:, :, model.shape[2] // 2])
        disk = midslice(model.mask.astype(float)) >= 0.5
        Ay = np.where(disk, Ay, 0.0)
        pj = np.unravel_index(np.argmax(Ay), Ay.shape)[1]
        peaks.append(pj - cy)
        maps.append((i1, i2, Ay))
    fig, axes = plt.subplots(1, len(ratios), figsize=(4 * len(ratios), 4))
    for ax, (i1, i2, Ay) in zip(axes, maps):
        m = np.ma.masked_where(Ay <= 0, Ay)
        ax.imshow(m.T, origin="lower", cmap="viridis")
        ax.axhline(cy, color="w", ls=":", lw=0.8)
        ax.set_title(f"{i1*1e3:.1f}:{i2*1e3:.1f} mA")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("y-envelope vs current ratio (peak should move toward lower-current pair)")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_cylinder_current_ratio.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return p, list(zip([r[0] for r in ratios], [r[1] for r in ratios], peaks))


if __name__ == "__main__":
    print("Building cylinder (100 x 100 x 10 would be full-res; using 100 x 100 x 3)")
    model = sh.cylinder_model(diameter_nodes=100, height_nodes=3, dx=3e-3, sigma=1.0)
    print(model)

    pair1, pair2 = montage_45deg(model)
    res = sh.simulate_ti(model, pair1, pair2, current1=2e-3, current2=2e-3)
    p1 = fig_envelope_map(model, res)
    print("saved", p1)

    p2, k, r = fig_scatter_vs_analytic(model)
    print(f"saved {p2}  (k={k:.4f}, r={r:.4f})")

    p3, peaks = fig_current_ratio(model)
    print("saved", p3)
    print("current-ratio peak y-offsets (i1_mA, i2_mA, peak_offset_nodes):")
    for i1, i2, off in peaks:
        print(f"   {i1*1e3:.1f} : {i2*1e3:.1f}  ->  {off:+.1f}")
