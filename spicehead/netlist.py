"""
SPICE netlist export and (optional) ngspice execution.

This is the manuscript's defining feature: the head model *is* a SPICE netlist
(Sec. 2.3, Sec. 4), so it can be dropped into a stimulator-circuit design and
co-simulated. Each resistor edge becomes an ``R`` line; each injected current a
``I`` line; the grounded reference node becomes SPICE node ``0``.

``run_ngspice`` writes the netlist, invokes ngspice in batch mode, and parses
the node voltages back, giving an independent check of the native solver on
small models.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy import sparse


def _node_name(idx, ground_node):
    return "0" if idx == ground_node else f"N{idx}"


def _pick_ground(injection, ground_node):
    """Ensure the ground node carries no injected current (else pick one)."""
    if ground_node is not None and abs(injection[ground_node]) <= 1e-15:
        return ground_node
    zero_nodes = np.flatnonzero(np.abs(injection) <= 1e-15)
    if zero_nodes.size == 0:
        raise ValueError("No zero-injection node available to use as ground.")
    return int(zero_nodes[0])


def build_netlist(model, injection, ground_node=None, title="SPICEhead"):
    """Return the SPICE netlist for ``model`` driven by ``injection`` (str)."""
    injection = np.asarray(injection, dtype=float)
    ground_node = _pick_ground(injection, ground_node)

    # Edges from the upper triangle of G: off-diagonal value = -g, so R = 1/g.
    U = sparse.triu(model.G, k=1).tocoo()
    gi, gj, gval = U.row, U.col, -U.data
    R = 1.0 / gval

    lines = [f"* {title}", f"* nodes={model.n_nodes} edges={gi.size} "
             f"dx={model.dx} ground=N{ground_node}->0"]
    for e, (i, j, r) in enumerate(zip(gi, gj, R)):
        ni = _node_name(int(i), ground_node)
        nj = _node_name(int(j), ground_node)
        lines.append(f"R{e} {ni} {nj} {r:.9g}")

    src_nodes = np.flatnonzero(np.abs(injection) > 1e-15)
    for s, idx in enumerate(src_nodes):
        # I<name> 0 Nidx DC val  ->  injects +val into node idx.
        lines.append(f"I{s} 0 {_node_name(int(idx), ground_node)} "
                     f"DC {injection[idx]:.9g}")

    lines.append(".op")
    lines.append(".end")
    return "\n".join(lines) + "\n", ground_node


def write_netlist(model, injection, path, ground_node=None, title="SPICEhead"):
    text, ground_node = build_netlist(model, injection, ground_node, title)
    with open(path, "w") as fh:
        fh.write(text)
    return ground_node


def ngspice_available():
    return shutil.which("ngspice") is not None


def run_ngspice(model, injection, ground_node=None, title="SPICEhead"):
    """Solve via ngspice and return node potentials (N,), with V[ground]=0.

    Uses a .control block with ``print`` to dump every node voltage. Intended
    for small models (cross-validation), not whole-head solves.
    """
    if not ngspice_available():
        raise RuntimeError("ngspice not found on PATH.")

    injection = np.asarray(injection, dtype=float)
    ground_node = _pick_ground(injection, ground_node)

    U = sparse.triu(model.G, k=1).tocoo()
    gi, gj, gval = U.row, U.col, -U.data
    R = 1.0 / gval

    node_names = [f"N{i}" for i in range(model.n_nodes)]
    node_names[ground_node] = "0"

    lines = [f"* {title}"]
    for e, (i, j, r) in enumerate(zip(gi, gj, R)):
        lines.append(f"R{e} {node_names[i]} {node_names[j]} {r:.12g}")
    src_nodes = np.flatnonzero(np.abs(injection) > 1e-15)
    for s, idx in enumerate(src_nodes):
        lines.append(f"I{s} 0 {node_names[idx]} DC {injection[idx]:.12g}")

    probe = " ".join(f"v({node_names[i]})" for i in range(model.n_nodes)
                     if i != ground_node)
    lines += [".control", "op", f"print {probe}", ".endc", ".end"]
    netlist = "\n".join(lines) + "\n"

    with tempfile.TemporaryDirectory() as td:
        cir = os.path.join(td, "net.cir")
        with open(cir, "w") as fh:
            fh.write(netlist)
        proc = subprocess.run(
            ["ngspice", "-b", cir],
            capture_output=True, text=True, timeout=600,
        )
    V = _parse_ngspice_print(proc.stdout, model.n_nodes, ground_node)
    return V


def _parse_ngspice_print(stdout, n_nodes, ground_node):
    """Parse 'v(nXXX) = value' pairs from ngspice batch print output."""
    import re

    V = np.zeros(n_nodes)
    pat = re.compile(r"v\(n(\d+)\)\s*=\s*([-+0-9.eE]+)", re.IGNORECASE)
    for m in pat.finditer(stdout):
        V[int(m.group(1))] = float(m.group(2))
    V[ground_node] = 0.0
    return V
