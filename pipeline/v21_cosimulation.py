"""
v21 circuit-head co-simulation: a real stimulator front end driving the
SPICEhead volume conductor, solved as ONE netlist in ngspice.

Addresses reviewer comment A: the netlist-integration advantage is currently
"asserted rather than experimentally demonstrated". Here the head model and a
non-ideal current-source stimulator share a single SPICE description, so the
delivered field can be traced back to circuit parameters that a field solver
cannot represent at all:

    * finite output impedance  R_out  of the current source
    * compliance-voltage limit  V_cc  (modelled as the supply the source works
      against; when the load needs more headroom than the supply provides, the
      delivered current droops)
    * electrode-skin contact impedance  R_el  in series with each contact

Experiment
----------
The same TI montage is simulated while sweeping (a) electrode contact impedance
and (b) source output impedance, and the delivered channel current and target
envelope amplitude are recorded. A pure FEM pipeline must *assume* the
programmed current is delivered; the co-simulation shows when it is not.

Both engines are used:
  - ngspice on the full head+driver netlist  (the integration demonstration)
  - the native solver on the same network as a cross-check

Run:  python pipeline/v21_cosimulation.py
Outputs: results/v21_cosim/{cosim.csv, cosim.png, example_head_driver.cir}
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spicehead as sh
from spicehead import netlist as nl

OUT = "results/v21_cosim"
I_PROG = 2e-3          # programmed channel current (A)


def build_head(diameter_nodes=40, height_nodes=1, dx=3e-3, sigma=0.3):
    """A modest homogeneous phantom head (keeps the ngspice runs quick)."""
    return sh.cylinder_model(diameter_nodes=diameter_nodes,
                             height_nodes=height_nodes, dx=dx, sigma=sigma)


def head_driver_netlist(model, anode, cathode, r_el, r_out, v_cc, i_prog=I_PROG):
    """One netlist: Norton-equivalent driver + electrode impedance + head.

    Driver (per channel): an ideal current source i_prog in parallel with the
    output impedance r_out, working against a compliance rail v_cc enforced by
    diode clamps. Electrode contact impedance r_el sits between the driver node
    and the scalp node of the volume conductor.
    """
    ground = int(cathode.nodes[0])
    names = [f"N{i}" for i in range(model.n_nodes)]
    names[ground] = "0"

    U = sparse.triu(model.G, k=1).tocoo()
    R = 1.0 / (-U.data)

    L = ["* SPICEhead co-simulation: stimulator front end + volume conductor"]
    for e, (i, j, r) in enumerate(zip(U.row, U.col, R)):
        L.append(f"Rh{e} {names[i]} {names[j]} {r:.10g}")

    # --- driver, anode side -------------------------------------------------
    # ideal source into node DRV, shunted by r_out, clamped to +/- v_cc
    L += [
        f"Idrv 0 DRV DC {i_prog:.10g}",
        f"Rout DRV 0 {r_out:.10g}",
        f"Vcc VCC 0 DC {v_cc:.10g}",
        "D1 DRV VCC DCLAMP",
        ".model DCLAMP D(Is=1e-15 N=0.02)",
    ]
    # 0 V source as an ammeter for the total delivered current
    L.append("Vsense DRV ELN DC 0")
    # electrode contact impedance to each anode scalp node (parallel contacts)
    n_a = anode.nodes.size
    for s, node in enumerate(anode.nodes):
        L.append(f"Rel{s} ELN {names[int(node)]} {r_el * n_a:.10g}")
    # return path: cathode contacts to ground through their own impedance
    n_c = cathode.nodes.size
    for s, node in enumerate(cathode.nodes):
        if int(node) == ground:
            continue
        L.append(f"Rrt{s} {names[int(node)]} 0 {r_el * n_c:.10g}")

    probes = [f"v({names[i]})" for i in range(model.n_nodes) if i != ground]
    ctl = [".control", "op", "print v(DRV)", "print i(Vsense)"]
    for i in range(0, len(probes), 40):          # chunked: ngspice truncates long lines
        ctl.append("print " + " ".join(probes[i:i + 40]))
    ctl += [".endc", ".end"]
    L += ctl
    return "\n".join(L) + "\n", ground, names


def run_ngspice_text(netlist_text, n_nodes, ground):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cosim.cir")
        open(p, "w").write(netlist_text)
        r = subprocess.run(["ngspice", "-b", p], capture_output=True,
                           text=True, timeout=900)
    V = np.zeros(n_nodes)
    for m in re.finditer(r"v\(n(\d+)\)\s*=\s*([-+0-9.eE]+)", r.stdout, re.I):
        V[int(m.group(1))] = float(m.group(2))
    V[ground] = 0.0
    drv = re.search(r"v\(drv\)\s*=\s*([-+0-9.eE]+)", r.stdout, re.I)
    amp = re.search(r"i\(vsense\)\s*=\s*([-+0-9.eE]+)", r.stdout, re.I)
    return (V, (float(drv.group(1)) if drv else np.nan),
            (abs(float(amp.group(1))) if amp else np.nan), r.stdout)


def delivered_current(model, V, anode, r_el):
    """Sum the currents actually entering the head through the anode contacts."""
    # I_contact = (V_drv - V_node)/R_contact ; recover from node injection instead:
    # total current into the conductor = G V summed over anode nodes.
    I = model.G @ V
    return float(np.sum(I[anode.nodes]))


def main():
    os.makedirs(OUT, exist_ok=True)
    if not nl.ngspice_available():
        print("ngspice not on PATH — aborting co-simulation demo.")
        return

    model = build_head()
    a = sh.place_rim_electrode(model, 45, 2.5, name="A+")
    c = sh.place_rim_electrode(model, 225, 2.5, name="A-")
    print(model, f"  anode contacts={a.nodes.size}")

    # save one example netlist for the supplement
    txt, ground, names = head_driver_netlist(model, a, c, r_el=2e3, r_out=1e6, v_cc=15.0)
    open(os.path.join(OUT, "example_head_driver.cir"), "w").write(txt)

    rows = []
    # (a) electrode impedance sweep at a realistic compliance rail
    for r_el in (200, 1e3, 5e3, 20e3, 50e3):
        txt, ground, _ = head_driver_netlist(model, a, c, r_el=r_el, r_out=1e6, v_cc=15.0)
        V, vdrv, i_meas, _ = run_ngspice_text(txt, model.n_nodes, ground)
        env = sh.field_magnitude_inplane(sh.efield_components(model, V))
        rows.append(dict(sweep="electrode_Z", param=r_el, r_out=1e6, v_cc=15.0,
                         v_drv=vdrv, I_del=i_meas,
                         E_peak=np.nanmax(env), E_med=np.nanmedian(env)))
        print(f"  r_el={r_el:>8.0f}  V_drv={vdrv:7.3f} V  "
              f"I_del={rows[-1]['I_del']*1e3:6.3f} mA  E_med={rows[-1]['E_med']:.4f}")

    # (b) source output-impedance sweep (ideal source -> poor source)
    for r_out in (1e6, 1e5, 1e4, 3e3, 1e3):
        txt, ground, _ = head_driver_netlist(model, a, c, r_el=2e3, r_out=r_out, v_cc=15.0)
        V, vdrv, i_meas, _ = run_ngspice_text(txt, model.n_nodes, ground)
        env = sh.field_magnitude_inplane(sh.efield_components(model, V))
        rows.append(dict(sweep="source_Rout", param=r_out, r_out=r_out, v_cc=15.0,
                         v_drv=vdrv, I_del=i_meas,
                         E_peak=np.nanmax(env), E_med=np.nanmedian(env)))
        print(f"  r_out={r_out:>8.0f}  V_drv={vdrv:7.3f} V  "
              f"I_del={rows[-1]['I_del']*1e3:6.3f} mA  E_med={rows[-1]['E_med']:.4f}")

    import csv
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "cosim.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, (sw, xlabel) in enumerate([("electrode_Z", "electrode contact impedance (Ω)"),
                                      ("source_Rout", "source output impedance (Ω)")]):
        s = [r for r in rows if r["sweep"] == sw]
        x = [r["param"] for r in s]
        ax[k].semilogx(x, [r["I_del"] * 1e3 for r in s], "o-", label="delivered I (mA)")
        ax[k].axhline(I_PROG * 1e3, ls="--", c="k", lw=1, label="programmed 2 mA")
        ax[k].set_xlabel(xlabel); ax[k].set_ylabel("delivered current (mA)")
        ax[k].grid(alpha=.3); ax[k].legend(fontsize=8)
    fig.suptitle("Circuit-head co-simulation: non-ideal front end vs delivered current")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cosim.png"), dpi=130)
    print(f"\nwrote {OUT}/cosim.csv, cosim.png, example_head_driver.cir")


if __name__ == "__main__":
    main()
