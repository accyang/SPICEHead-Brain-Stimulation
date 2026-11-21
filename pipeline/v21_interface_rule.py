# -*- coding: utf-8 -*-
"""Arithmetic vs harmonic interface conductance, against an EXACT analytic reference.

Novelty-review item 5: "Explicitly test arithmetic versus harmonic/interface conductance in a
multilayer slab or sphere against an analytic or highly resolved FEM reference. If interface
handling explains much of the convergence problem, that becomes an important methodological result."

A layered slab carrying uniform current normal to the layers has an exact solution: the layers are
resistors in series, R_total = sum_i L_i / (sigma_i * A), and the field inside layer i is
E_i = J / sigma_i with J = I / A. There is no approximation to argue about, so this isolates the
discretization of the conductivity discontinuity itself.

The realistic-head test (v21_head_convergence.py) showed harmonic averaging makes resolution
dependence WORSE, which was surprising. This says which rule is actually right, and by how much,
in the one geometry where "right" is unambiguous.

Usage:  python pipeline/v21_interface_rule.py
Output: results/v21_interface/{interface_rule.csv, interface_rule.md}
"""
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
import spicehead as sh
from spicehead.model import SPICEHeadModel

OUT = os.path.join(ROOT, "results", "v21_interface")
CURRENT = 1e-3          # A
AREA_NODES = (8, 8)     # cross-section in nodes; field is 1-D so this only sets total area

# Scalp / compact bone / CSF / grey matter -- the stack TI current actually crosses,
# with the conductivities used in the cohort work.
LAYERS = [("scalp", 0.465, 6.0), ("compact_bone", 0.008, 6.0),
          ("csf", 1.654, 3.0), ("grey_matter", 0.275, 15.0)]   # (name, sigma S/m, thickness mm)


def analytic(dx_mm):
    """Exact per-layer field for unit cross-section, and total resistance."""
    ny, nz = AREA_NODES
    A = (ny * nz) * (dx_mm * 1e-3) ** 2
    J = CURRENT / A
    return {n: J / s for n, s, _ in LAYERS}, sum((t * 1e-3) / (s * A) for _, s, t in LAYERS)


def run(dx_mm, rule):
    ny, nz = AREA_NODES
    # build the layer stack along x at this resolution
    sig_col = []
    for _, s, t in LAYERS:
        n = int(round(t / dx_mm))
        if n < 1:
            n = 1
        sig_col += [s] * n
    nx = len(sig_col)
    sigma = np.empty((nx, ny, nz))
    for i, s in enumerate(sig_col):
        sigma[i] = s

    m = SPICEHeadModel(sigma, dx=dx_mm * 1e-3, conductance_mean=rule)
    ii, jj, kk = m.grid_coords
    inj = np.zeros(m.n_nodes)
    face_lo = np.flatnonzero(ii == 0)
    face_hi = np.flatnonzero(ii == nx - 1)
    inj[face_lo] += CURRENT / face_lo.size
    inj[face_hi] -= CURRENT / face_hi.size
    V = sh.solve_potentials(m.G, inj, ground_node=int(face_hi[0]), method="direct")
    Ex, _, _ = sh.efield_components(m, V)

    ana, R_ana = analytic(dx_mm)
    # measured field per layer, sampled at interior nodes of each layer only (central differences
    # straddle the interface at layer edges, where no 1-D "layer field" is defined)
    out, i0 = {}, 0
    for name, s, t in LAYERS:
        n = max(1, int(round(t / dx_mm)))
        lo, hi = i0, i0 + n
        i0 = hi
        interior = slice(lo + 1, hi - 1) if n >= 3 else slice(lo, hi)
        vals = np.abs(Ex[interior, 1:-1, 1:-1])
        vals = vals[np.isfinite(vals)]
        out[name] = float(np.mean(vals)) if vals.size else float("nan")
    # total resistance seen by the source
    R_meas = float((V[face_lo].mean() - V[face_hi].mean()) / CURRENT)
    return out, ana, R_meas, R_ana, nx


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print("layer stack:", " / ".join(f"{n} {t:g}mm sigma={s:g}" for n, s, t in LAYERS))
    print(f"\n{'dx':>4}{'rule':>12}{'R_meas/R_exact':>16}   per-layer field error vs exact")
    for dx in (3.0, 2.0, 1.0, 0.5):
        for rule in ("arithmetic", "harmonic"):
            meas, ana, Rm, Ra, nx = run(dx, rule)
            errs = {n: 100 * (meas[n] / ana[n] - 1) for n, _, _ in LAYERS if np.isfinite(meas[n])}
            rows.append({"dx_mm": dx, "rule": rule, "nx": nx,
                         "R_ratio": round(Rm / Ra, 5),
                         **{f"err_{n}_pct": round(errs.get(n, float('nan')), 2)
                            for n, _, _ in LAYERS}})
            s = "  ".join(f"{n.split('_')[0][:5]} {errs[n]:+6.1f}%" for n in errs)
            print(f"{dx:>4.1f}{rule:>12}{Rm/Ra:>16.4f}   {s}")
    with open(os.path.join(OUT, "interface_rule.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    L = ["# Interface conductance rule vs an exact analytic reference\n",
         "Layered slab, current normal to the layers, so the exact solution is resistors in "
         "series. Stack: " + ", ".join(f"{n} ({t:g} mm, σ = {s:g} S/m)" for n, s, t in LAYERS)
         + ".\n",
         "`R_ratio` is total resistance seen by the source divided by the exact series "
         "resistance; 1.000 is perfect.\n",
         "| Δx (mm) | rule | R_meas / R_exact |",
         "|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dx_mm']:g} | {r['rule']} | {r['R_ratio']:.4f} |")
    open(os.path.join(OUT, "interface_rule.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {OUT}/interface_rule.csv")


if __name__ == "__main__":
    main()
