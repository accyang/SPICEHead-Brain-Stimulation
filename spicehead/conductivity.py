"""
Tissue conductivity values and label -> conductivity mapping.

Values are the SimNIBS/CHARM standard conductivities (S/m), consistent with the
references cited in the manuscript (Wagner et al. 2004; Opitz et al. 2015). The
label ordering matches the ten CHARM tissues listed in Manuscript Sec. 3.4:
white matter, gray matter, CSF, bone, scalp, eyeballs, compact bone, spongy
bone, blood, muscle.
"""

from __future__ import annotations

import numpy as np

# Conductivity in S/m.
CONDUCTIVITY = {
    "white_matter": 0.126,
    "gray_matter": 0.275,
    "csf": 1.654,
    "bone": 0.010,
    "scalp": 0.465,
    "eyeballs": 0.500,
    "compact_bone": 0.008,
    "spongy_bone": 0.025,
    "blood": 0.600,
    "muscle": 0.160,
    "air": 0.0,          # background -> excluded from the conductor
}

# CHARM final-tissue integer labels (SimNIBS convention).
CHARM_LABELS = {
    0: "air",
    1: "white_matter",
    2: "gray_matter",
    3: "csf",
    4: "bone",
    5: "scalp",
    6: "eyeballs",
    7: "compact_bone",
    8: "spongy_bone",
    9: "blood",
    10: "muscle",
}


def label_volume_to_sigma(labels, label_map=None, cond=None):
    """Map an integer label volume to a conductivity volume (S/m).

    Parameters
    ----------
    labels : int array
        Per-voxel tissue label.
    label_map : dict[int, str], optional
        Label -> tissue-name map (default CHARM_LABELS).
    cond : dict[str, float], optional
        Tissue-name -> conductivity map (default CONDUCTIVITY).

    Returns
    -------
    sigma : float array, same shape as ``labels``. Unknown labels -> 0.
    """
    label_map = CHARM_LABELS if label_map is None else label_map
    cond = CONDUCTIVITY if cond is None else cond
    labels = np.asarray(labels)
    sigma = np.zeros(labels.shape, dtype=float)
    for lab, name in label_map.items():
        sigma[labels == lab] = cond.get(name, 0.0)
    return sigma
