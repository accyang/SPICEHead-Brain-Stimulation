# -*- coding: utf-8 -*-
"""Exploratory: optimal 10-20 TI montages for one-sided insula and hippocampus targets.

Figure S1 (pipeline/fig_ti_targets_mni.py) scores the insula and hippocampus as bilateral targets
(left + right AAL parcels together). A single two-channel montage has essentially one main focus,
so a bilateral target with its halves 50-80 mm apart favours broad fields. This script repeats the
same search (MNI152 template, SPICEhead v22, 2 mm, 18 basis solves relative to Cz, all 11,628
two-pair montages, F = median target GM envelope / p99 of GM more than 20 mm from the target) with
each side as its own target. The contralateral homolog then counts as off-target tissue.

Writes the same files as fig_ti_targets_mni.py (envelope and target NIfTIs at 1 mm and a summary
JSON/CSV) to `Paper_A_SPICEhead/Brain Stimulation/figures/TI_targets_MNI_unilateral/`; the bilateral results are not
touched. Exploratory only: not part of the submitted analyses.

  SPICEHEAD_MODEL=v22 python pipeline/ti_template_unilateral.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)
import fig_ti_targets_mni as M                                       # noqa: E402

M.TARGETS = {
    "insula_L": dict(title="Left insula", parcels=["Insula_L"]),
    "insula_R": dict(title="Right insula", parcels=["Insula_R"]),
    "hippocampus_L": dict(title="Left hippocampus", parcels=["Hippocampus_L"]),
    "hippocampus_R": dict(title="Right hippocampus", parcels=["Hippocampus_R"]),
}
M.OUT = os.path.join(os.path.dirname(M.OUT), "TI_targets_MNI_unilateral")
M.render = lambda: None                          # the 2-D slice figure belongs to the bilateral set

if __name__ == "__main__":
    M.main()
