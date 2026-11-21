# -*- coding: utf-8 -*-
"""MNI -> subject mapping for a CHARM folder that lacks toMNI/MNI2Conform_nonl.nii.gz.

h088 is the only such head in the 300-head set (18 Oct 2026). Its charm run of 22 Jul wrote only
Conform2MNI_nonl. That field lies on the subject T1 grid and holds each voxel's MNI coordinate.
  pull_labels      an MNI atlas read at each voxel's MNI coordinate (nearest neighbour). Validated: identical
                   labels to SimNIBS warp_volume(mni2subject, order 0) on h001, h002 and h150 (100.00 % of
                   GM voxels).
  points_to_subject  MNI points -> subject world coordinates: the voxel whose MNI coordinate is nearest,
                   refined by a least-squares affine fit of the field over the surrounding voxels.
                   Validated against mni2subject_coords; see check_points().
"""
import os

import numpy as np


def _field(m2m):
    import nibabel as nib
    fld = nib.load(os.path.join(m2m, "toMNI", "Conform2MNI_nonl.nii.gz"))
    t1 = nib.load(os.path.join(m2m, "T1.nii.gz"))
    if fld.shape[:3] != t1.shape[:3] or not np.allclose(fld.affine, t1.affine, atol=1e-3):
        raise ValueError("Conform2MNI_nonl is not on the T1 grid")
    return np.asarray(fld.dataobj, np.float32), t1.affine, t1.shape[:3]


def pull_labels(m2m, atlas_nii, gm_idx):
    import nibabel as nib
    F, _, _ = _field(m2m)
    gi = np.asarray(gm_idx, np.int64)
    mni = F[gi[:, 0], gi[:, 1], gi[:, 2]].astype(float)
    at = nib.load(atlas_nii)
    A = np.asarray(at.dataobj).astype(np.int32)
    ijk = np.rint((np.linalg.inv(at.affine) @ np.c_[mni, np.ones(len(mni))].T)[:3].T).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(A.shape)), 1) & np.any(mni != 0, 1)
    out = np.zeros(len(gi), np.int32)
    out[ok] = A[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    return out


def points_to_subject(m2m, pts, half=3):
    from scipy.spatial import cKDTree
    F, aff, shape = _field(m2m)
    valid = np.any(F != 0, axis=3)
    idx = np.argwhere(valid)
    tree = cKDTree(F[valid])
    out = []
    for p in np.asarray(pts, float):
        _, j = tree.query(p)
        c = idx[j]
        lo = np.maximum(c - half, 0); hi = np.minimum(c + half + 1, shape)
        sub = np.argwhere(valid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) + lo
        M = F[sub[:, 0], sub[:, 1], sub[:, 2]].astype(float)          # MNI coords of the neighbourhood
        X = np.c_[M, np.ones(len(M))]
        coef, *_ = np.linalg.lstsq(X, sub.astype(float), rcond=None)  # local affine MNI -> voxel index
        v = np.r_[p, 1.0] @ coef
        out.append((aff @ np.r_[v, 1.0])[:3])
    return np.array(out)


def check_points(m2m, pts):
    """Maximum distance (mm) between points_to_subject and SimNIBS mni2subject_coords on a complete head."""
    from simnibs.utils.transformations import mni2subject_coords
    a = points_to_subject(m2m, pts)
    b = np.asarray(mni2subject_coords(np.asarray(pts, float), m2m))
    return float(np.max(np.linalg.norm(a - b, axis=1)))
