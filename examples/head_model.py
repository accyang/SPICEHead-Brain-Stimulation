"""
Whole-head TI simulation from a segmented MRI.

This is the entry point for the MRI path (Manuscript Sec. 2.2, 2.5, 3.4). It
reuses the validated building blocks (mri.model_from_labels, nodes_in_sphere,
simulate_ti); only argument parsing and I/O are added here.

Expected input: a CHARM / SimNIBS ``final_tissues.nii.gz`` (integer tissue
labels), ideally already at ~3 mm. See spicehead/conductivity.py CHARM_LABELS
for the label convention; use --list-tissues to check yours.

Examples
--------
# 1) Inspect the segmentation labels present in your file:
python examples/head_model.py /path/final_tissues.nii.gz --list-tissues

# 2) Run a two-pair TI montage. Electrode contacts are given as voxel indices
#    (i j k) of the four scalp locations (anode1 cathode1 anode2 cathode2).
python examples/head_model.py /path/final_tissues.nii.gz \
    --pair1 34 80 40  34 20 40 \
    --pair2 60 80 40  10 80 40 \
    --current1 2e-3 --current2 2e-3 --radius 4 \
    --out env_motor

To convert EEG-10-10 world coordinates (mm) to voxel indices, use
spicehead.mri.world_to_voxel(affine, (x, y, z)).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spicehead as sh
from spicehead import mri
from spicehead.conductivity import CHARM_LABELS, CONDUCTIVITY
from spicehead.electrodes import Electrode, nodes_in_sphere


def parse_args():
    p = argparse.ArgumentParser(description="SPICEhead whole-head TI simulation")
    p.add_argument("segmentation", help="CHARM/SimNIBS label NIfTI (.nii/.nii.gz)")
    p.add_argument("--list-tissues", action="store_true",
                   help="print the labels present and exit")
    p.add_argument("--dx", type=float, default=3.0, help="voxel size in mm")
    p.add_argument("--pair1", type=float, nargs=6, metavar="I",
                   help="anode1 i j k  cathode1 i j k (voxel indices)")
    p.add_argument("--pair2", type=float, nargs=6, metavar="I",
                   help="anode2 i j k  cathode2 i j k (voxel indices)")
    p.add_argument("--current1", type=float, default=2e-3)
    p.add_argument("--current2", type=float, default=2e-3)
    p.add_argument("--radius", type=float, default=4.0,
                   help="electrode radius in voxels")
    p.add_argument("--out", default="head_envelope",
                   help="output basename (.npy and .nii.gz written)")
    return p.parse_args()


def list_tissues(labels):
    vals, counts = np.unique(labels, return_counts=True)
    print(f"{'label':>6}  {'tissue':<14} {'voxels':>10}  {'sigma (S/m)':>11}")
    for v, c in zip(vals, counts):
        name = CHARM_LABELS.get(int(v), "?")
        sig = CONDUCTIVITY.get(name, float("nan"))
        print(f"{int(v):>6}  {name:<14} {c:>10}  {sig:>11.3f}")


def main():
    args = parse_args()
    labels, affine, zooms = mri.load_label_volume(args.segmentation)
    zt = tuple(round(float(z), 2) for z in zooms)
    print(f"loaded {args.segmentation}  shape={labels.shape}  zooms={zt} mm")

    if args.list_tissues:
        list_tissues(labels)
        return

    if not (args.pair1 and args.pair2):
        raise SystemExit("Provide --pair1 and --pair2 (or use --list-tissues).")

    # Build the network on the working grid of the active model version (SPICEHEAD_MODEL):
    # v21/v22 resample the labels to dx by nearest neighbour; v23 homogenizes the conductances
    # from the labels at their own resolution (dx must then be a whole multiple of the voxel size).
    # NOTE: electrode voxel indices below refer to THIS working grid.
    model, affine = mri.head_model(args.segmentation, args.dx)
    labels = model.labels
    print(f"working grid: shape={labels.shape}  dx={args.dx} mm")
    print(model)

    def elec(coord, name):
        nodes = nodes_in_sphere(model, tuple(coord), radius_nodes=args.radius)
        if nodes.size == 0:
            raise SystemExit(
                f"electrode '{name}' at voxel {coord} selected no conductor "
                f"nodes; check it lies on tissue (scalp), not air."
            )
        return Electrode(nodes, name)

    a1 = elec(args.pair1[0:3], "P1+"); c1 = elec(args.pair1[3:6], "P1-")
    a2 = elec(args.pair2[0:3], "P2+"); c2 = elec(args.pair2[3:6], "P2-")
    print(f"electrode node counts: {a1.nodes.size}/{c1.nodes.size}/"
          f"{a2.nodes.size}/{c2.nodes.size}")

    res = sh.simulate_ti(model, (a1, c1), (a2, c2),
                         current1=args.current1, current2=args.current2,
                         method="auto")
    env = np.nan_to_num(res.envelope)

    np.save(args.out + ".npy", env)
    try:
        import nibabel as nib
        nib.save(nib.Nifti1Image(env.astype(np.float32), affine),
                 args.out + ".nii.gz")
        print(f"wrote {args.out}.npy and {args.out}.nii.gz")
    except Exception as e:  # nibabel present for load, but be defensive
        print(f"wrote {args.out}.npy (NIfTI save skipped: {e})")

    tissue_nodes = res.envelope_on_nodes()
    print(f"envelope: max={env.max():.4f} V/m  "
          f"mean(in-tissue)={tissue_nodes[tissue_nodes>0].mean():.4f} V/m")


if __name__ == "__main__":
    main()
