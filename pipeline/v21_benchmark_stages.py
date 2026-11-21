"""
v21 staged benchmark: cost of repeated electrode reconfiguration.

Addresses reviewer comment F: the claimed advantage is repeated source
reconfiguration, but the r20 benchmark reports a single solve on mismatched
hardware. Here every pipeline stage is timed separately on ONE machine:

    construction   voxel -> conductance matrix (graph Laplacian assembly)
    factorization  one-time sparse factorization of G
    first solve    first electrode configuration
    per-config     each additional configuration (back-substitution only)

and the total is projected for 1, 10, 100, 1000 and 10000 configurations.

This isolates the actual claim: the volume conductor is built and factorized
once, after which a new montage costs only a right-hand-side solve — no
remeshing, no re-assembly, no re-factorization. Reported alongside the
iterative (CG) path, which needs no factorization but pays per solve.

Run:  python pipeline/v21_benchmark_stages.py
Output: results/v21_benchmark/stages.csv
"""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spicehead as sh
from spicehead.solve import make_solver, solve_potentials
from spicehead.electrodes import pair_injection

from spicehead import config

OUT = "results/v21_benchmark"
V23 = config.NETWORK == "homogenized"
if V23:
    # v23 (28-30 Oct 2026; Paper A switched to v23, benchmarks on a Forerunner node): the cylinder is built the v23 way,
    # by homogenizing a 0.5 mm voxelization, and "construction" times that; results in the model's own folder
    OUT = os.path.join(config.RESULTS, "v21_benchmark")
N_TIMED = 12            # configurations actually timed (rest projected)


def homogenized_cylinder(diameter_nodes, height_nodes, dx, sigma, fine_mm=0.5):
    """The cylinder of sh.cylinder_model (same diameter and height in blocks), voxelized at fine_mm and homogenized."""
    from spicehead.homogenize import homogenized_model
    dx_mm = dx * 1e3
    f = int(round(dx_mm / fine_mm))
    n = diameter_nodes * f
    c = (np.arange(n) + 0.5) * fine_mm - n * fine_mm / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    disk = (X ** 2 + Y ** 2) <= (n * fine_mm / 2) ** 2
    fine = np.repeat(disk[:, :, None], height_nodes * f, axis=2).astype(np.int16)
    model, _ = homogenized_model(fine, np.diag([fine_mm, fine_mm, fine_mm, 1.0]), dx_mm,
                                 label_map={1: "cylinder"}, cond={"cylinder": sigma})
    return model
DIRECT_MAX_NODES = 100_000   # above this, 3-D sparse-LU fill-in is impractical
PROJECT = [1, 10, 100, 1000, 10000]


def build_montages(model, n):
    """n distinct two-pair montages (rotating electrode positions)."""
    out = []
    for i in range(n):
        a = sh.place_rim_electrode(model, (i * 29) % 360, 3.0)
        c = sh.place_rim_electrode(model, (i * 29 + 180) % 360, 3.0)
        out.append(pair_injection(model, a, c, 2e-3))
    return out


def bench(diameter_nodes, height_nodes, dx=3e-3, sigma=0.3):
    t0 = time.perf_counter()
    m = (homogenized_cylinder if V23 else sh.cylinder_model)(diameter_nodes, height_nodes, dx, sigma)
    t_build = time.perf_counter() - t0          # includes Laplacian assembly (v23: and the homogenization)

    injs = build_montages(m, N_TIMED)
    ground = int(np.flatnonzero(np.abs(injs[0]) < 1e-15)[0])

    # --- direct: factorize once, then back-substitute per configuration ------
    # Skipped above DIRECT_MAX_NODES: fill-in of a 3-D graph Laplacian makes
    # sparse LU factorization impractical at whole-head resolution (this is
    # itself the reason the iterative path is the recommended back end).
    if m.n_nodes <= DIRECT_MAX_NODES:
        t0 = time.perf_counter()
        solve = make_solver(m.G, ground_node=ground)
        t_fact = time.perf_counter() - t0

        t0 = time.perf_counter(); solve(injs[0]); t_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        for inj in injs[1:]:
            solve(inj)
        t_per = (time.perf_counter() - t0) / (N_TIMED - 1)
    else:
        t_fact = t_first = t_per = float("nan")

    # --- iterative (CG): no factorization, pays per solve -------------------
    t0 = time.perf_counter()
    solve_potentials(m.G, injs[0], ground_node=ground, method="cg", tol=1e-8)
    t_cg = time.perf_counter() - t0

    return dict(nodes=m.n_nodes, edges=m.n_edges,
                t_build=t_build, t_fact=t_fact, t_first=t_first,
                t_per_config=t_per, t_cg_per_config=t_cg)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"platform: {platform.platform()}")
    print(f"processor: {platform.processor() or platform.machine()}\n")

    rows = []
    for diam, h, label in [(100, 3, "23k nodes"), (100, 10, "79k nodes"), (100, 25, "196k nodes (whole-head scale)")]:
        r = bench(diam, h)
        r["label"] = label
        rows.append(r)
        print(f"[{label}] nodes={r['nodes']:,}")
        print(f"   construction (voxel->matrix) : {r['t_build']:8.3f} s")
        if np.isnan(r["t_fact"]):
            print("   factorization (one-time)     :  impractical (3-D LU fill-in)")
            print("   first solve                  :  n/a")
            print("   per additional configuration :  n/a")
        else:
            print(f"   factorization (one-time)     : {r['t_fact']:8.3f} s")
            print(f"   first solve                  : {r['t_first']:8.3f} s")
            print(f"   per additional configuration : {r['t_per_config']:8.4f} s")
        print(f"   iterative CG, per config     : {r['t_cg_per_config']:8.4f} s")
        for n in PROJECT:
            tot_d = r["t_build"] + r["t_fact"] + r["t_first"] + (n - 1) * r["t_per_config"]
            tot_i = r["t_build"] + n * r["t_cg_per_config"]
            d = "      n/a" if np.isnan(tot_d) else f"{tot_d:9.2f} s"
            print(f"     total for {n:>5} configs: direct {d} | CG {tot_i:9.2f} s")
        print()

    import csv
    keys = ["label", "nodes", "edges", "t_build", "t_fact", "t_first",
            "t_per_config", "t_cg_per_config"]
    with open(os.path.join(OUT, "stages.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print(f"wrote {OUT}/stages.csv")


if __name__ == "__main__":
    main()
