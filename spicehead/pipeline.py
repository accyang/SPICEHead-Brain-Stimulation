"""
End-to-end temporal-interference simulation.

Given a head/phantom model and two electrode pairs, solve the two quasi-static
fields and combine them into the TI envelope (Manuscript Sec. 2.6). The two
pairs are solved with a *single* shared factorization of G, illustrating the
intended workflow: build/factorize the head once, then evaluate many electrode
configurations by changing only the right-hand side.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .electrodes import pair_injection
from .solve import make_solver, solve_potentials
from .field import (
    efield_components,
    ti_envelope_components,
    ti_envelope_magnitude,
)


@dataclass
class TIResult:
    model: object
    V1: np.ndarray
    V2: np.ndarray
    E1: tuple
    E2: tuple
    envelope_components: tuple  # (Ax, Ay, Az) grids, Eq. 5 per axis
    envelope: np.ndarray        # magnitude grid
    ground_node: int

    def envelope_on_nodes(self):
        """Return the envelope magnitude sampled at node locations (N,)."""
        return self.model.grid_to_nodes(np.nan_to_num(self.envelope))


def simulate_ti(model, pair1, pair2, current1=2e-3, current2=2e-3,
                k=1.0, ground_node=None, solver=None, method="auto"):
    """Simulate a two-pair TI montage.

    Parameters
    ----------
    model : SPICEHeadModel
    pair1, pair2 : (anode Electrode, cathode Electrode)
        The two carrier pairs (e.g. f1 = 2 kHz and f2 = 2.01 kHz).
    current1, current2 : float
        Pair currents in amperes (default 2 mA each).
    k : float
        Field normalization factor (1.0 with physical conductances).
    ground_node : int, optional
        Reference node reused for both solves. Defaults to a cathode node.
    solver : callable, optional
        A prefactorized solver from ``make_solver`` (for many-montage sweeps on a
        fixed head). If given, ``method`` is ignored.
    method : {'auto', 'direct', 'cg'}
        Used when ``solver`` is None. 'auto' factorizes+reuses a direct LU below
        60k nodes and switches to the iterative CG solver above (whole-head
        scale), where direct factorization is prohibitively slow.
    """
    (a1, c1), (a2, c2) = pair1, pair2
    if ground_node is None:
        ground_node = int(c1.nodes[0])

    inj1 = pair_injection(model, a1, c1, current1)
    inj2 = pair_injection(model, a2, c2, current2)

    if solver is not None:
        V1, V2 = solver(inj1), solver(inj2)
    else:
        use = method
        if use == "auto":
            use = "direct" if model.n_nodes <= 60_000 else "cg"
        if use == "direct":
            solver = make_solver(model.G, ground_node=ground_node)
            V1, V2 = solver(inj1), solver(inj2)
        else:
            V1 = solve_potentials(model.G, inj1, ground_node, method="cg")
            V2 = solve_potentials(model.G, inj2, ground_node, method="cg")

    E1 = efield_components(model, V1, k=k)
    E2 = efield_components(model, V2, k=k)
    env_comp = ti_envelope_components(E1, E2)
    # Envelope definition. 'maxdir' (default) is the rotation-invariant maximum-over-direction
    # form; 'peraxis' is the superseded coordinate-dependent combination and is kept only to
    # reproduce pre-2026-08 outputs. envelope_components stays per-axis either way -- it is the
    # per-axis decomposition by definition and is not used for scoring.
    import os as _os
    if _os.environ.get("SPICEHEAD_ENVELOPE", "maxdir") == "maxdir":
        from .envelope_fast import envelope_max_direction_flat
        _sh = np.asarray(E1[0]).shape
        env_mag = envelope_max_direction_flat(
            *[np.asarray(c).ravel() for c in E1], *[np.asarray(c).ravel() for c in E2]
        ).reshape(_sh)
    else:
        env_mag = ti_envelope_magnitude(E1, E2)

    return TIResult(
        model=model, V1=V1, V2=V2, E1=E1, E2=E2,
        envelope_components=env_comp, envelope=env_mag,
        ground_node=ground_node,
    )
