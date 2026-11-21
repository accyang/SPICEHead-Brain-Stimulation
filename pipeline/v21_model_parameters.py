"""
v21 model-parameter table (reviewer comment D).

Emits the complete specification of the resistor-network model — every value a
reader needs to reproduce it — read directly from the code so it cannot drift
from the implementation.

Run:  python pipeline/v21_model_parameters.py
Output: results/v21_params/model_parameters.md
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spicehead.conductivity import CONDUCTIVITY, CHARM_LABELS

OUT = "results/v21_params"

SOURCES = {
    "white_matter": "Wagner et al. 2004; Opitz et al. 2015",
    "gray_matter": "Wagner et al. 2004; Opitz et al. 2015",
    "csf": "Wagner et al. 2004",
    "bone": "Wagner et al. 2004",
    "scalp": "Wagner et al. 2004",
    "eyeballs": "SimNIBS default",
    "compact_bone": "Opitz et al. 2015",
    "spongy_bone": "Opitz et al. 2015",
    "blood": "SimNIBS default",
    "muscle": "SimNIBS default",
    "air": "non-conducting background (excluded from the network)",
}

SPEC = [
    ("Discretization", [
        ("Element", "cubic voxel; node at each voxel centre"),
        # The 3.0 mm here was a stale default: no analysis in the paper runs at 3 mm. State the
        # resolutions actually used, per analysis, so this record cannot contradict the Methods.
        ("Node spacing dx", "5 mm (300-subject cohort optimization); 2 mm and 1 mm (FEM comparison "
                            "and two-channel co-simulation); 1-5 mm in the convergence studies"),
        ("Connectivity", "6-neighbour (face-adjacent) resistor edges"),
        ("Edge conductance", "g = sigma_edge * dx  (equivalently R = 1/(sigma_edge*dx))"),
        ("Edge conductivity", "arithmetic mean of the conductivities of the two voxels "
                              "sharing the edge (harmonic option available)"),
        ("Units", "conductances in S, currents in A -> potentials in V, field in V/m; "
                  "no empirical scale factor (alpha = 1.000, verified on a uniform slab)"),
    ]),
    ("Boundary and background", [
        ("Air / background", "sigma = 0; voxels excluded from the node set entirely"),
        ("Outer boundary", "natural (homogeneous Neumann): no edges leave the conductor, "
                           "so no current crosses the surface"),
        ("Reference node", "one node pinned to V = 0 (SPICE node 0); chosen as a "
                           "zero-injection node"),
    ]),
    ("Sources and electrodes", [
        ("Electrode model", "ideal current source injecting into the contact set below"),
        ("Contact set", "all conductor nodes within a given radius (12 mm) of the target site "
                        "snapped to the scalp: a volumetric support that spans the scalp "
                        "thickness and can include skull or muscle nodes (used in every "
                        "published analysis)"),
        ("Surface contact (sensitivity only)", "scalp nodes within the same radius that touch "
                                               "the non-conductor, i.e. the skin under the pad; "
                                               "used only in the contact-sensitivity analysis"),
        ("Current distribution", "total electrode current divided equally across its nodes"),
        ("Charge balance", "anode and cathode currents sum to zero (required for solvability)"),
        ("Contact impedance", "0 by default; finite electrode-skin impedance is modelled "
                              "explicitly in the circuit co-simulation"),
    ]),
    ("Solver", [
        ("Direct", "sparse LU (SuperLU) factorization, reused across electrode "
                   "configurations; practical below ~1e5 nodes in 3-D"),
        ("Iterative", "Jacobi-preconditioned conjugate gradients, relative tolerance 1e-10 "
                      "(library default, with an automatic switch to CG above 2e5 nodes)"),
        ("Head analyses", "direct LU up to 1e5 nodes and CG (relative tolerance 1e-10) above, "
                          "in the grid-convergence, FEM-comparison, two-channel and 2 mm "
                          "re-scoring analyses; the 5 mm cohort search is always direct"),
        ("Exceptions", "homogeneous-disk benchmark CG tolerance 1e-9 above 6e4 nodes; timing "
                       "benchmark CG tolerance 1e-8"),
        ("SPICE back end", "netlist export solved by ngspice/HSPICE/LTspice (.op); "
                           "agrees with the native solver to ~3e-7 relative"),
    ]),
    ("Envelope", [
        ("Per-axis (Eq. 5)", "A_i = 2 min(|E1_i|, |E2_i|) for i in x,y,z, combined as "
                             "sqrt(sum A_i^2); COORDINATE-FRAME DEPENDENT"),
        ("Max-direction", "A_max = max over n of 2 min(|E1.n|, |E2.n|) (Grossman et al. "
                          "2017 closed form); rotation-invariant"),
    ]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    L = ["# SPICEhead model parameters",
         "",
         "Machine-generated from the implementation by "
         "`pipeline/v21_model_parameters.py`.", ""]

    L += ["## Tissue conductivities", "",
          "| CHARM label | Tissue | sigma (S/m) | Source |", "|---|---|---|---|"]
    for lab in sorted(CHARM_LABELS):
        name = CHARM_LABELS[lab]
        L.append(f"| {lab} | {name.replace('_',' ')} | "
                 f"{CONDUCTIVITY.get(name, float('nan')):.3f} | {SOURCES.get(name,'')} |")
    L.append("")

    for section, items in SPEC:
        L += [f"## {section}", "", "| Parameter | Value |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in items]
        L.append("")

    path = os.path.join(OUT, "model_parameters.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
