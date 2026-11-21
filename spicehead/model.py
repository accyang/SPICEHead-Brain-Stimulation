"""
Core resistor-network model for SPICEhead.

The head (or phantom) volume conductor is discretized on a regular 3-D grid.
Every in-mask grid point is a circuit *node*; every pair of face-adjacent
in-mask nodes is joined by a resistor. This is the finite-difference /
resistor-network form of the quasi-static conduction equation

    div( sigma grad V ) = 0                                    (Manuscript Eq. 1)

Enforcing Kirchhoff's current law at every node,

    sum_q  g_pq (V_p - V_q)  =  I_p          (injected current at node p),

is *identical* to the discretized PDE, and is exactly what a SPICE DC operating
point (.op) computes. In matrix form:

    G V = I

where G is the weighted graph Laplacian (conductance matrix).

Conductance convention
----------------------
For a cubic voxel of side dx, a single edge represents a prism of medium of
length dx and cross-section dx^2, so

    R = length / (sigma * area) = dx / (sigma * dx^2) = 1 / (sigma * dx)
    g = 1 / R = sigma * dx.

Using this *physical* conductance (rather than the manuscript's relative 1 Ohm
resistors) makes the pipeline self-calibrating: node voltages come out in volts
and the field in V/m, so the manuscript's normalization factor k is ~1 by
construction instead of ~98 (see field.py / README).

The edge conductivity ``sigma_edge`` is the average conductivity of the voxels
adjacent to the edge (Manuscript Sec. 2.5, "taken as their average"), which
smooths tissue boundaries. Arithmetic (default) and harmonic averaging are
provided; they coincide for a homogeneous medium such as the validation
cylinder.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


class SPICEHeadModel:
    """A voxel-wise resistor network derived from a conductivity volume.

    Parameters
    ----------
    sigma : (nx, ny, nz) float array
        Per-voxel electrical conductivity in S/m.
    dx : float
        Node spacing in metres (default 3e-3 = 3 mm, the manuscript resolution).
    mask : (nx, ny, nz) bool array, optional
        Which grid points are conductor nodes. Defaults to ``sigma > 0``.
    conductance_mean : {'arithmetic', 'harmonic'}
        How the two voxel conductivities on an edge are combined.
        'arithmetic' matches the manuscript wording ("average"); 'harmonic' is
        the physically exact series-resistor combination.
    """

    def __init__(self, sigma, dx=3e-3, mask=None, conductance_mean="arithmetic"):
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim != 3:
            raise ValueError("sigma must be a 3-D array (nx, ny, nz)")
        if mask is None:
            mask = sigma > 0
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != sigma.shape:
            raise ValueError("mask and sigma must have the same shape")
        if conductance_mean not in ("arithmetic", "harmonic"):
            raise ValueError("conductance_mean must be 'arithmetic' or 'harmonic'")

        self.sigma = sigma
        self.mask = mask
        self.dx = float(dx)
        self.shape = sigma.shape
        self.conductance_mean = conductance_mean

        # Contiguous node numbering: node_id[i,j,k] = index in [0, N), or -1.
        self.n_nodes = int(mask.sum())
        node_id = np.full(self.shape, -1, dtype=np.int64)
        node_id[mask] = np.arange(self.n_nodes)
        self.node_id = node_id

        # (i, j, k) grid coordinates for each node index, for mapping V back
        # onto the grid and for placing electrodes.
        ii, jj, kk = np.nonzero(mask)
        self.grid_coords = (ii, jj, kk)

        self.G = self._build_laplacian()

    # ------------------------------------------------------------------
    def _edge_conductance(self, sig_a, sig_b):
        if self.conductance_mean == "arithmetic":
            sig_edge = 0.5 * (sig_a + sig_b)
        else:  # harmonic (series resistors); guard against zeros
            denom = sig_a + sig_b
            sig_edge = np.where(denom > 0, 2.0 * sig_a * sig_b / denom, 0.0)
        return sig_edge * self.dx  # g = sigma_edge * dx  [siemens]

    def _build_laplacian(self):
        """Assemble the sparse conductance (Laplacian) matrix G."""
        rows, cols, vals = [], [], []
        nid = self.node_id

        for axis in range(3):
            # Face-adjacent node pairs along this axis where BOTH ends are nodes.
            a = tuple(slice(None, -1) if d == axis else slice(None) for d in range(3))
            b = tuple(slice(1, None) if d == axis else slice(None) for d in range(3))
            both = self.mask[a] & self.mask[b]
            if not both.any():
                continue
            ids_a = nid[a][both]
            ids_b = nid[b][both]
            g = self._edge_conductance(self.sigma[a][both], self.sigma[b][both])

            # Off-diagonal -g (both directions) and diagonal +g contributions.
            rows.append(ids_a); cols.append(ids_b); vals.append(-g)
            rows.append(ids_b); cols.append(ids_a); vals.append(-g)
            rows.append(ids_a); cols.append(ids_a); vals.append(g)
            rows.append(ids_b); cols.append(ids_b); vals.append(g)

        if not rows:
            raise ValueError("No resistor edges were created; check mask/sigma.")

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        G = sparse.coo_matrix(
            (vals, (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        ).tocsr()
        G.sum_duplicates()
        return G

    # ------------------------------------------------------------------
    @property
    def n_edges(self):
        # Off-diagonal stored entries are 2 per edge.
        return (self.G.nnz - self.n_nodes) // 2

    def grid_to_nodes(self, values_on_grid):
        """Sample a (nx,ny,nz) array at node locations -> (N,) vector."""
        ii, jj, kk = self.grid_coords
        return np.asarray(values_on_grid)[ii, jj, kk]

    def nodes_to_grid(self, values, fill=np.nan):
        """Scatter an (N,) node vector back onto a (nx,ny,nz) grid."""
        out = np.full(self.shape, fill, dtype=float)
        ii, jj, kk = self.grid_coords
        out[ii, jj, kk] = values
        return out

    def __repr__(self):
        return (
            f"SPICEHeadModel(shape={self.shape}, dx={self.dx*1e3:.3g} mm, "
            f"nodes={self.n_nodes}, edges={self.n_edges}, "
            f"mean='{self.conductance_mean}')"
        )
