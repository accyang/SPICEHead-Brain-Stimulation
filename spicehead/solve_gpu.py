"""GPU conjugate-gradient solver for the resistor network (PyTorch, CUDA).

Same contract as ``pipeline/v21_head_convergence.make_solver``: one ground node per connected
component (so isolated conductor islands do not make the system singular), Jacobi-preconditioned
CG in float64 to a relative residual of ``rtol``. The matrix stays on the GPU; each call moves one
right-hand side (or a block of them) across and returns the full-length potential vector.

    from spicehead.solve_gpu import make_gpu_solver
    solve, ncomp = make_gpu_solver(model.G)        # same (solve, ncomp) pair as the CPU version
    V = solve(current)                             # (N,) float64, V = 0 at the grounds
    Vs = solve.many(currents)                      # (N, k) for k right-hand sides at once

Selected by the pipeline when SPICEHEAD_SOLVER=gpu (see v21_head_convergence.make_solver). The
residual is checked on the GPU every ``check_every`` iterations; convergence failure raises, as the
CPU solver does. Memory: a 1 mm head (3.7 M nodes, ~26 M nonzeros) needs about 0.6 GB for the
matrix plus a few vectors; a 10 GB desktop card is enough, an H200 trivially so.
"""
from __future__ import annotations

import os
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def make_gpu_solver(G, rtol=1e-10, maxiter=20000, check_every=10, device=None, verbose=True):
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = G.shape[0]
    Gc = csr_matrix(G)
    ncomp, comp = connected_components(csgraph=Gc, directed=False)
    grounds = [int(np.flatnonzero(comp == c)[0]) for c in range(ncomp)]
    keep = np.ones(N, bool)
    keep[grounds] = False
    idx = np.flatnonzero(keep)
    A = Gc[idx][:, idx].tocsr()
    A.sort_indices()
    d = A.diagonal().copy()
    d[d == 0] = 1.0

    # torch CSR in float64; int64 indices (cuSPARSE spmm handles both int32 and int64)
    At = torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int64)),
        torch.from_numpy(A.indices.astype(np.int64)),
        torch.from_numpy(A.data.astype(np.float64)),
        size=A.shape, device=device)
    dinv = torch.from_numpy((1.0 / d).astype(np.float64)).to(device)
    idx_t = torch.from_numpy(idx)

    def pcg(B):
        """B: (n, k) float64 on device. Returns X (n, k), iterations, relative residuals."""
        X = torch.zeros_like(B)
        R = B.clone()
        Z = R * dinv[:, None]
        P = Z.clone()
        rz = (R * Z).sum(0)                                   # (k,)
        bnorm = torch.linalg.vector_norm(B, dim=0)
        bnorm = torch.where(bnorm == 0, torch.ones_like(bnorm), bnorm)
        it = 0
        while it < maxiter:
            AP = torch.sparse.mm(At, P)
            pAp = (P * AP).sum(0)
            alpha = rz / pAp
            X += alpha * P
            R -= alpha * AP
            it += 1
            if it % check_every == 0:
                rel = torch.linalg.vector_norm(R, dim=0) / bnorm
                if bool((rel <= rtol).all()):
                    break
            Z = R * dinv[:, None]
            rz_new = (R * Z).sum(0)
            beta = rz_new / rz
            rz = rz_new
            P = Z + beta * P
        rel = (torch.linalg.vector_norm(B - torch.sparse.mm(At, X), dim=0) / bnorm)
        return X, it, rel

    def many(currents):
        C = np.asarray(currents, dtype=np.float64)
        if C.ndim == 1:
            C = C[:, None]
        t0 = time.time()
        B = torch.from_numpy(np.ascontiguousarray(C[idx])).to(device)
        X, it, rel = pcg(B)
        relmax = float(rel.max())
        if verbose:
            print(f"      GPU-CG {C.shape[1]} rhs, {it} it, max relres {relmax:.2e} in {time.time()-t0:.1f}s",
                  flush=True)
        if relmax > 1e-6:
            raise RuntimeError(f"GPU CG failed to converge (relres={relmax:.2e})")
        V = np.zeros((N, C.shape[1]))
        V[idx] = X.cpu().numpy()
        return V

    def solve(current):
        return many(current)[:, 0]

    solve.many = many
    solve.device = str(device)
    solve.n_unknowns = int(A.shape[0])
    solve.nnz = int(A.nnz)
    return solve, ncomp
