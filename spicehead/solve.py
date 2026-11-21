"""
Native solver for the resistor network: G V = I.

This is the in-Python equivalent of the SPICE .op analysis. Because the network
is purely resistive (justified for the 1-10 kHz TI band, Manuscript Sec. 2.4),
the "circuit" is just a sparse symmetric-positive-semidefinite linear system.
Solving it directly with a sparse factorization (or conjugate gradients for very
large models) sidesteps the SPICE-solver bottleneck reported in the manuscript
(Ngspice's KLU/Sparse direct solvers failing to converge on ~1e5 nodes).

The Laplacian G is singular (one null vector: a constant potential offset), so a
reference node is grounded (V = 0) before solving, exactly as a SPICE netlist
needs node 0. The injected-current vector must sum to zero (charge conservation);
a balanced anode/cathode pair satisfies this automatically.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, cg, factorized


def solve_potentials(G, current, ground_node=0, method="auto", tol=1e-10,
                     maxiter=None, check_balance=True):
    """Solve G V = I with one node grounded.

    Parameters
    ----------
    G : scipy.sparse matrix (N, N)
        Conductance / Laplacian matrix.
    current : (N,) array
        Injected current at each node (A). Should sum to ~0.
    ground_node : int
        Index of the node pinned to V = 0 (SPICE node 0).
    method : {'auto', 'direct', 'cg'}
        'direct' uses a sparse LU factorization; 'cg' uses preconditioned
        conjugate gradients (Jacobi preconditioner); 'auto' picks 'direct'
        below ~200k nodes and 'cg' above.
    tol, maxiter : CG controls.
    check_balance : bool
        Warn (raise) if the injected currents are not balanced.

    Returns
    -------
    V : (N,) array of node potentials (V), with V[ground_node] = 0.
    """
    current = np.asarray(current, dtype=float)
    N = G.shape[0]
    if current.shape != (N,):
        raise ValueError("current must have length N = G.shape[0]")
    if check_balance:
        imbalance = float(current.sum())
        scale = float(np.abs(current).sum()) or 1.0
        if abs(imbalance) > 1e-9 * scale:
            raise ValueError(
                f"Injected currents are not balanced (sum={imbalance:.3e}); "
                "anode and cathode currents must cancel."
            )

    keep = np.ones(N, dtype=bool)
    keep[ground_node] = False
    idx = np.flatnonzero(keep)

    G = csr_matrix(G)
    Gkk = G[idx][:, idx].tocsc()
    ik = current[idx]

    if method == "auto":
        method = "direct" if N <= 200_000 else "cg"

    if method == "direct":
        vk = spsolve(Gkk, ik)
    elif method == "cg":
        # Jacobi (diagonal) preconditioner.
        diag = Gkk.diagonal()
        diag[diag == 0] = 1.0
        M = csr_matrix((1.0 / diag, (np.arange(len(diag)), np.arange(len(diag)))))
        vk, info = cg(Gkk, ik, rtol=tol, maxiter=maxiter, M=M)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info}).")
    else:
        raise ValueError("method must be 'auto', 'direct', or 'cg'")

    V = np.zeros(N)
    V[idx] = vk
    return V


def make_solver(G, ground_node=0):
    """Return a reusable direct solver for a *fixed* G with one ground node.

    This is the key to the manuscript's intended workflow (Sec. 3.5): the head
    model is factorized once, then many electrode configurations are solved
    cheaply by back-substitution, because only the right-hand side (the current
    vector) changes when electrodes move.

    Returns
    -------
    solve(current) -> V : callable mapping an (N,) current vector to potentials.
    """
    N = G.shape[0]
    keep = np.ones(N, dtype=bool)
    keep[ground_node] = False
    idx = np.flatnonzero(keep)
    Gkk = csr_matrix(G)[idx][:, idx].tocsc()
    lu = factorized(Gkk)  # SuperLU factorization, reused across right-hand sides

    def solve(current):
        current = np.asarray(current, dtype=float)
        V = np.zeros(N)
        V[idx] = lu(current[idx])
        return V

    return solve
