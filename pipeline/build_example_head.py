# -*- coding: utf-8 -*-
"""Assemble the redistributable example head for the code deposit (plan item 6, 2 October 2026).

The deposit promised "an example head model" but shipped none. This writes examples/mni152_head/:

  labels_1mm.nii.gz        the CHARM ten-tissue segmentation of the MNI152 template, resampled
                           from 0.5 mm to 1 mm (nearest neighbour), uint8
  electrodes_1020_mni.json the 19 10-20 positions in MNI space (SimNIBS cap EEG10-10_UI_Jurak_2007)
  roi_coords_mni.json      the six target centres in MNI space (place_rois_all.ROIS_MNI)
  README.md                provenance and licences

The MNI152 template (ICBM 2009) may be redistributed with attribution; no subject data is used.
examples/mni152_full_search.py runs the paper's full two-pair montage search on this head.

  python pipeline/build_example_head.py
"""
import json
import os
import sys

import numpy as np
import nibabel as nib

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)
from spicehead import mri                                            # noqa: E402
from fig_ti_targets_mni import M2M, CAP, LABELS_1020, read_cap        # noqa: E402

OUT = os.path.join(ROOT, "examples", "mni152_head")


def rois_mni():
    """place_rois_all.ROIS_MNI, read from the source: that module imports simnibs at import time,
    which the project interpreter does not have."""
    import ast
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)       # that module's docstring has a "\s"
        tree = ast.parse(open(os.path.join(_here, "place_rois_all.py"), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ROIS_MNI"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("ROIS_MNI not found in place_rois_all.py")

README = """# Example head: CHARM-segmented MNI152 template

`labels_1mm.nii.gz` is the ten-tissue segmentation of the MNI152 (ICBM 2009) T1 template produced
by SimNIBS 4.6 CHARM at 0.5 mm and resampled here to 1 mm isotropic by nearest neighbour. Label
values follow CHARM (see `spicehead/conductivity.py`, `CHARM_LABELS`): 1 white matter, 2 grey
matter, 3 CSF, 4 bone, 5 scalp, 6 eyeballs, 7 compact bone, 8 spongy bone, 9 blood, 10 muscle;
0 is air.

`electrodes_1020_mni.json` gives the 19 positions of the 10-20 system in MNI space, taken from the
cap file `EEG10-10_UI_Jurak_2007.csv` distributed with SimNIBS (Jurcak V, Tsuzuki D, Dan I.
NeuroImage 2007;34:1600-11). `roi_coords_mni.json` gives the six target centres used in the paper
(mm, MNI); each target is a 10 mm sphere about its centre(s).

This is a template, not a subject. No participant data is included. The MNI152 template is
copyright McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University, and
is redistributed under its licence (free use and redistribution with attribution).

Run the paper's search on it:

    python examples/mni152_full_search.py --dx 3      (a few minutes on an idle workstation)
    python examples/mni152_full_search.py --dx 2      (tens of minutes)
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    seg = os.path.join(M2M, "label_prep", "tissue_labeling_upsampled.nii.gz")
    labels, aff, zooms = mri.load_label_volume(seg, dx_mm=1.0)
    assert labels.max() <= 255
    img = nib.Nifti1Image(labels.astype(np.uint8), aff)
    img.header.set_xyzt_units("mm")
    nib.save(img, os.path.join(OUT, "labels_1mm.nii.gz"))
    cap = read_cap()
    json.dump({k: [round(float(x), 2) for x in cap[k]] for k in LABELS_1020},
              open(os.path.join(OUT, "electrodes_1020_mni.json"), "w"), indent=1)
    json.dump(rois_mni(), open(os.path.join(OUT, "roi_coords_mni.json"), "w"), indent=1)
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8", newline="\n").write(README)
    vals, cnt = np.unique(labels, return_counts=True)
    print(f"labels {labels.shape} at {[round(float(z), 2) for z in zooms]} mm: "
          + ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnt)))
    for f in sorted(os.listdir(OUT)):
        print(f"  {f:28s} {os.path.getsize(os.path.join(OUT, f))/1e6:6.2f} MB")


if __name__ == "__main__":
    main()
