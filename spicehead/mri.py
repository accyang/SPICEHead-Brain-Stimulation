"""
MRI ingestion: segmented volume -> SPICEHeadModel.

Implements the manuscript's image path (Sec. 2.2, 2.5): load a CHARM-segmented
label volume, optionally resample to the working resolution (3 mm), map labels
to conductivities, and build the resistor network. Requires nibabel.

Electrode placement from the EEG 10-10 / 10-20 system (e.g. FC3-C1, C3-CP3 in
Sec. 3.4) is left to the caller: pass world-space coordinates and use
``world_to_voxel`` + ``spicehead.electrodes.nodes_in_sphere``.
"""

from __future__ import annotations

import numpy as np

from .model import SPICEHeadModel
from .conductivity import label_volume_to_sigma, CHARM_LABELS, CONDUCTIVITY


def load_label_volume(path, dx_mm=None):
    """Load a segmentation NIfTI. Returns (labels[int], affine, zooms).

    If ``dx_mm`` is given and the image is not already at that isotropic
    resolution, the label volume is resampled with **nearest-neighbour**
    interpolation (order=0) so tissue labels are never blended into invalid
    intermediate values. This lets you run CHARM at its native resolution and
    downsample to the SPICEhead working resolution here (equivalent to the
    manuscript's "resample to 3 mm then segment", Sec. 2.2).
    """
    import nibabel as nib

    img = nib.load(str(path))
    if dx_mm is not None:
        zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
        if not np.allclose(zooms, dx_mm, atol=1e-3):
            from nibabel.processing import resample_to_output
            img = resample_to_output(img, voxel_sizes=(dx_mm,) * 3, order=0)
    labels = np.rint(np.asarray(img.dataobj)).astype(np.int64)
    zooms = img.header.get_zooms()[:3]
    return labels, img.affine, zooms


def model_from_labels(labels, zooms, dx_mm=3.0, label_map=None, cond=None,
                      conductance_mean=None):
    """Build a SPICEHeadModel from a label volume.

    Assumes ``labels`` is already at (near-)isotropic ``dx_mm`` resolution
    (the manuscript resamples to 3 mm before segmentation). ``zooms`` is used
    only to sanity-check isotropy.

    ``conductance_mean`` defaults to the model version in spicehead.config (harmonic for v22,
    arithmetic for v21). The label volume is kept on the model as ``model.labels`` because the
    v22 skin-surface contacts need it.
    """
    from . import config
    if config.NETWORK == "homogenized":
        raise ValueError(f"SPICEHEAD_MODEL={config.MODEL} builds its network from the fine segmentation; "
                         "use spicehead.mri.head_model(seg_path, dx_mm), not resampled labels")
    label_map = CHARM_LABELS if label_map is None else label_map
    cond = CONDUCTIVITY if cond is None else cond
    sigma = label_volume_to_sigma(labels, label_map=label_map, cond=cond)
    mask = sigma > 0
    model = SPICEHeadModel(sigma, dx=dx_mm * 1e-3, mask=mask,
                           conductance_mean=conductance_mean or config.CONDUCTANCE_MEAN)
    model.labels = labels
    return model


def head_model(path, dx_mm, label_map=None, cond=None, conductance_mean=None):
    """Build the head network of the active model version from a segmentation file.

    Returns (model, affine); ``model.labels`` holds the label volume on the model grid and
    ``model.affine`` the grid's voxel-to-world affine.

    v21/v22: the labels are resampled to dx_mm by nearest neighbour (load_label_volume) and the
    network is built from them (model_from_labels, with an optional interface-rule override);
    the grid is world-aligned. This is exactly the two-call sequence the pipelines used before
    23 October 2026.
    v23: the labels are read at their own resolution and the network is homogenized from them
    (spicehead.homogenize); the grid follows the segmentation's voxel axes and the model carries
    ``axes_to_world``, which spicehead.field.efield_components uses to return world-axis fields.
    v23 has no interface-rule choice; any override other than "harmonic" is refused.
    """
    from . import config
    if config.NETWORK == "homogenized":
        if conductance_mean not in (None, "harmonic"):
            raise ValueError(f"SPICEHEAD_MODEL={config.MODEL} has no interface-rule option "
                             f"(conductance_mean={conductance_mean!r} requested)")
        from .homogenize import homogenized_model
        labels, affine, _ = load_label_volume(path)
        return homogenized_model(labels, affine, dx_mm, min_fill=config.HOMOG_MIN_FILL,
                                 label_map=label_map, cond=cond)
    labels, affine, zooms = load_label_volume(path, dx_mm=dx_mm)
    model = model_from_labels(labels, zooms, dx_mm=dx_mm, label_map=label_map, cond=cond,
                              conductance_mean=conductance_mean)
    model.affine = affine
    return model, affine


def head_grid(path, dx_mm):
    """The active model's grid without solving: returns (labels, affine, zooms, model_or_None).

    For scripts that need the labels and affine of the model grid (masks, electrode voxels, reading
    fields saved on that grid) but not always the network.
    v21/v22: load_label_volume only (nearest-label resampling, no network built); model is None,
    zooms as loaded. The returned labels are exactly those head_model would attach.
    v23: which coarse voxels are nodes is decided by the network (node rule, largest component),
    so the homogenized network is built and returned for reuse; labels are its majority labels and
    zooms is (dx_mm,) * 3.
    """
    from . import config
    if config.NETWORK == "homogenized":
        model, affine = head_model(path, dx_mm)
        return model.labels, affine, (float(dx_mm),) * 3, model
    labels, affine, zooms = load_label_volume(path, dx_mm=dx_mm)
    return labels, affine, zooms, None


def world_to_voxel(affine, xyz):
    """Map world coordinates (mm) to fractional voxel indices (i,j,k)."""
    inv = np.linalg.inv(affine)
    xyz1 = np.append(np.asarray(xyz, dtype=float), 1.0)
    ijk = inv @ xyz1
    return tuple(ijk[:3])
