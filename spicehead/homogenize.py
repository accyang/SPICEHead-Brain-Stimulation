"""
Homogenized coarse network (model v23): edge conductances from the fine segmentation.

The v21/v22 network at a working resolution dx first resamples the CHARM labels (0.5 mm) to dx
by nearest neighbour, so a skull or CSF layer thinner than dx is either lost or thickened. v23
keeps the fine structure inside each coarse edge instead:

  * the coarse grid is the fine label grid in blocks of f x f x f voxels (f = dx / fine voxel
    size, an integer), so the coarse voxel axes are the CHARM volume's axes (not world axes);
  * the conductance between two face-adjacent coarse voxels is the parallel sum of f^2
    conduction tubes; each tube is the series combination of the f fine voxels between the two
    block centres (resistance 1 / (sigma_i * dx_f) per fine segment, dx_f = fine voxel size);
  * a coarse voxel is a node when at least MIN_FILL (0.5) of its fine voxels conduct, and only
    the largest connected component of the network is kept;
  * each node keeps the majority label of its conducting fine voxels (used by the skin-surface
    contacts) and the mean conductivity of those voxels (diagnostic only; G is built from tubes).

For uniform blocks the tube conductance reduces exactly to sigma * dx, so the v22 network on the
same grid is the special case of uniform labels (tests/test_homogenize.py).

Validated on 23 October 2026 against SimNIBS FEM in all 300 heads of the cohort
(results_v22/v22_fem_fullsearch_300_homog): at 1 mm the field correlation with FEM rose from
0.89 to 0.98 and the FEM winner was chosen in 70% instead of 56% of head x target combinations.
This module is the package form of pipeline/nchc/homog_experiment.homogenized_model, with the
same arithmetic (checked G-for-G against it on 23 October 2026).

Because the coarse grid follows the CHARM voxel axes, the model carries ``axes_to_world`` (unit
column directions of the coarse affine) and spicehead.field.efield_components rotates the field
into world axes, so field components compare with v22 and FEM.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

MIN_FILL = 0.5


def block_views(a, f):
    """Crop to multiples of f and return the (n0, n1, n2, f, f, f) block view of a fine array."""
    n = [(s // f) * f for s in a.shape]
    a = a[:n[0], :n[1], :n[2]]
    return a.reshape(n[0] // f, f, n[1] // f, f, n[2] // f, f).transpose(0, 2, 4, 1, 3, 5)


def block_factor(fine_mm, dx_mm):
    """Integer block size f with f * fine_mm == dx_mm; raises if dx is not a multiple of the fine grid."""
    f = int(round(dx_mm / fine_mm))
    if f < 1 or abs(f * fine_mm - dx_mm) > 1e-6:
        raise ValueError(f"v23 needs dx ({dx_mm} mm) to be a whole multiple of the segmentation's "
                         f"voxel size ({fine_mm} mm)")
    return f


def homogenized_model(fine_labels, fine_affine, dx_mm, min_fill=MIN_FILL, label_map=None, cond=None):
    """Coarse SPICEHeadModel at dx_mm with edge conductances from the fine label volume.

    Parameters
    ----------
    fine_labels : (n0, n1, n2) int array, the segmentation at its own (isotropic) resolution.
    fine_affine : (4, 4) voxel-to-world affine of fine_labels (mm).
    dx_mm       : working resolution, a whole multiple of the fine voxel size.

    Returns (model, coarse_affine). The model carries ``labels`` (majority label per coarse
    voxel, 0 outside the network), ``affine``, ``axes_to_world``, ``dropped_nodes`` (nodes
    outside the largest component), ``block`` (f), ``fine_mm`` and ``network = "homogenized"``.
    """
    from .conductivity import label_volume_to_sigma, CHARM_LABELS, CONDUCTIVITY
    from .model import SPICEHeadModel
    label_map = CHARM_LABELS if label_map is None else label_map
    cond = CONDUCTIVITY if cond is None else cond
    fine_labels = np.asarray(fine_labels).astype(np.int16)
    fine_affine = np.asarray(fine_affine, dtype=float)
    vox = np.sqrt((fine_affine[:3, :3] ** 2).sum(0))
    if not np.allclose(vox, vox[0], atol=1e-4):
        raise ValueError(f"v23 needs an isotropic segmentation, got voxel sizes {vox}")
    fine_mm = float(vox[0])
    f = block_factor(fine_mm, dx_mm)

    # float64: a float32 Laplacian is not reliably positive semidefinite at these contrasts
    sig_f = label_volume_to_sigma(fine_labels, label_map=label_map, cond=cond).astype(np.float64)
    B = block_views(sig_f, f)                                 # (N0, N1, N2, f, f, f)
    Lb = block_views(fine_labels, f)
    N = B.shape[:3]
    # node rule: at least min_fill of the block conducts (blocks with a single conducting fine voxel
    # would be nodes attached by ~1e-6 S tubes, which made CG diverge in the prototype)
    mask = (B > 0).mean(axis=(3, 4, 5)) >= min_fill
    mean_sigma = np.where(mask, B.sum(axis=(3, 4, 5)) / np.maximum((B > 0).sum(axis=(3, 4, 5)), 1), 0.0)
    # majority label among the conducting fine voxels (used by the skin-surface contacts)
    lab_c = np.zeros(N, np.int16)
    flat = Lb.reshape(N + (f ** 3,))
    for v in np.unique(fine_labels):
        if v == 0:
            continue
        cnt = (flat == v).sum(-1)
        better = cnt > np.where(lab_c > 0, (flat == lab_c[..., None]).sum(-1), 0)
        lab_c[better] = v
    dx_f = (dx_mm / f) * 1e-3                                 # metres, as model.py (g = sigma * dx in siemens)
    h = f // 2
    with np.errstate(divide="ignore"):
        R = np.where(B > 0, 1.0 / (B * dx_f), np.inf)       # resistance of each fine segment
    g = []
    for axis in range(3):
        fa = 3 + axis                                         # fine axis inside the block
        # second half of block A (h..f-1) and first half of block B (0..h-1) along the fine axis
        slA = [slice(None)] * 6; slA[fa] = slice(h, f)
        slB = [slice(None)] * 6; slB[fa] = slice(0, h)
        RA = R[tuple(slA)].sum(axis=fa)                       # (N0, N1, N2, f, f): per tube, A's half
        RB = R[tuple(slB)].sum(axis=fa)
        sa = [slice(None)] * 3; sa[axis] = slice(0, -1)
        sb = [slice(None)] * 3; sb[axis] = slice(1, None)
        Rtube = RA[tuple(sa)] + RB[tuple(sb)]
        g.append(np.where(np.isfinite(Rtube), 1.0 / Rtube, 0.0).sum(axis=(-2, -1)))
    del R, B, Lb, flat

    def assemble(m):
        model = SPICEHeadModel(mean_sigma, dx=dx_mm * 1e-3, mask=m, conductance_mean="harmonic")
        nid = model.node_id
        rows, cols, vals = [], [], []
        for axis in range(3):
            a = tuple(slice(None, -1) if d == axis else slice(None) for d in range(3))
            b = tuple(slice(1, None) if d == axis else slice(None) for d in range(3))
            # only neighbour pairs joined by at least one conducting tube are edges: a zero conductance stored as an
            # edge made a block linked only through non-conducting tubes look connected, so the giant-component
            # filter kept it and the matrix was singular (27 Oct 2026: h062 at 5 mm, a two-node island)
            both = m[a] & m[b] & (g[axis] > 0)
            ga = g[axis][both]
            ia, ib = nid[a][both], nid[b][both]
            rows += [ia, ib, ia, ib]; cols += [ib, ia, ia, ib]; vals += [-ga, -ga, ga, ga]
        G = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(model.n_nodes,) * 2).tocsr()
        G.sum_duplicates()
        model.G = G
        return model

    # nodes without a conducting edge, and islands not connected to the main conductor, cannot be
    # solved (zero diagonal; no current path to the reference): keep the largest component only
    from scipy.sparse.csgraph import connected_components
    model = assemble(mask)
    deg = np.asarray(model.G.diagonal()).ravel()
    keep = deg > 0
    ncomp, comp = connected_components(model.G[keep][:, keep], directed=False)
    giant = np.bincount(comp).argmax()
    keep_idx = np.flatnonzero(keep)[comp == giant]
    mask2 = np.zeros_like(mask)
    ii, jj, kk = model.grid_coords
    mask2[ii[keep_idx], jj[keep_idx], kk[keep_idx]] = True
    dropped = int(mask.sum() - mask2.sum())
    model = assemble(mask2)

    aff_c = fine_affine.copy()
    aff_c[:3, :3] = fine_affine[:3, :3] * f
    aff_c[:3, 3] = fine_affine[:3, 3] + fine_affine[:3, :3] @ np.full(3, (f - 1) / 2.0)   # centre of the first block
    model.labels = np.where(mask2, lab_c, 0).astype(np.int16)
    model.affine = aff_c
    model.axes_to_world = aff_c[:3, :3] / np.sqrt((aff_c[:3, :3] ** 2).sum(0))
    model.dropped_nodes = dropped
    model.block = f
    model.fine_mm = fine_mm
    model.network = "homogenized"
    return model, aff_c
