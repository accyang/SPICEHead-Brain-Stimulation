"""
Run SPICEhead TI-envelope simulations for one subject, for both montages, from
pre-computed EEG-10-10 subject-space positions (see place_electrodes.py).

  python run_cohort_envelopes.py <m2m_dir> <electrodes.json> <out_dir>

Montages (manuscript Sec. 3.4 / 3.6):
  motor    : FC3-C1 / C3-CP3, 2 mA each
  thalamic : F7-PO7 / F8-PO8, 1 mA each

For each montage writes <out_dir>/<montage>_env.{npy,nii.gz} and a summary.json with the
snapped contact voxels, the scalp-snap distance per electrode (QC), and envelope stats.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spicehead as sh
from spicehead import mri, config
from spicehead.electrodes import Electrode, contact_nodes

SCALP = 5          # CHARM scalp label
DX = 3.0           # working-grid voxel size (mm)
RADIUS = 4.0       # electrode radius (voxels)

MONTAGES = {
    "motor":    {"pairs": [("FC3", "C1"), ("C3", "CP3")], "current": 2e-3},
    "thalamic": {"pairs": [("F7", "PO7"), ("F8", "PO8")], "current": 1e-3},
}


def snap_to_scalp(scalp_vox, ijk):
    """Nearest scalp voxel (int idx) to a fractional voxel ijk; returns (vox, dist_vox)."""
    d = np.linalg.norm(scalp_vox - np.asarray(ijk, float), axis=1)
    k = int(np.argmin(d))
    return scalp_vox[k], float(d[k])


def main(m2m, ejson, outdir):
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    if not os.path.exists(seg):
        raise SystemExit(f"no segmentation: {seg}")
    model, affine = mri.head_model(seg, DX)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    scalp_vox = np.argwhere(labels == SCALP)
    if scalp_vox.size == 0:
        raise SystemExit("no scalp voxels in segmentation")

    elec = json.load(open(ejson))
    contacts, snap_mm = {}, {}
    for lab, world in elec.items():
        ijk = mri.world_to_voxel(affine, world)
        vox, dvox = snap_to_scalp(scalp_vox, ijk)
        contacts[lab] = tuple(int(v) for v in vox)
        snap_mm[lab] = round(dvox * DX, 1)

    if config.MODEL not in ("v21", "v22") and config.MODEL not in os.path.basename(os.path.normpath(outdir)):
        raise SystemExit(f"model {config.MODEL}: the envelope folder name must carry the model tag ({outdir})")
    os.makedirs(outdir, exist_ok=True)

    def elec_obj(lab):
        nodes = contact_nodes(model, contacts[lab], radius_nodes=RADIUS)
        if nodes.size == 0:
            raise SystemExit(f"electrode {lab} at {contacts[lab]} selected no conductor nodes")
        return Electrode(nodes, lab)

    stats = {}
    for mname, m in MONTAGES.items():
        (a1, c1), (a2, c2) = m["pairs"]
        res = sh.simulate_ti(model, (elec_obj(a1), elec_obj(c1)),
                             (elec_obj(a2), elec_obj(c2)),
                             current1=m["current"], current2=m["current"], method="auto")
        env = np.nan_to_num(res.envelope)
        np.save(os.path.join(outdir, f"{mname}_env.npy"), env)
        try:
            import nibabel as nib
            nib.save(nib.Nifti1Image(env.astype(np.float32), affine),
                     os.path.join(outdir, f"{mname}_env.nii.gz"))
        except Exception as e:
            print(f"  ({mname}: NIfTI save skipped: {e})")
        tn = res.envelope_on_nodes()
        in_tissue = tn[tn > 0]
        stats[mname] = {
            "max_Vpm": round(float(env.max()), 4),
            "mean_in_tissue_Vpm": round(float(in_tissue.mean()), 4),
            "p99_Vpm": round(float(np.percentile(in_tissue, 99)), 4),
        }
        print(f"  {mname:>8}: max={stats[mname]['max_Vpm']:.3f}  "
              f"mean(tissue)={stats[mname]['mean_in_tissue_Vpm']:.3f} V/m")

    summary = {
        "m2m": m2m, "model": config.MODEL, "grid_shape": list(labels.shape), "n_nodes": int(model.n_nodes),
        "contacts_ijk": contacts, "scalp_snap_mm": snap_mm, "stats": stats,
    }
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    worst = max(snap_mm.values())
    print(f"  scalp-snap max = {worst} mm  (large => placement suspect)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
