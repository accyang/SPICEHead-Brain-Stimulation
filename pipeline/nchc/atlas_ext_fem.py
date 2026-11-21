# -*- coding: utf-8 -*-
"""FEM reference for the cerebellum and brainstem extension of the 300-head atlas (24 Oct 2026; PI: "build the FEM
pipeline for all 300 heads"). SimNIBS interpreter.

The 300-head FEM leadfields (v22_fem_fullsearch.stage_fem_leadfield) stored grey matter only. Brainstem nuclei lie in
CHARM white matter, so this step reruns the same leadfield with white and grey matter stored, on the same mesh and
with the same settings:
  - all 19 pads meshed (ellipse 24 x 24 mm, 2 mm gel), reference Cz, SimNIBS default conductivities (checked
    against spicehead.conductivity);
  - mapping with ElementData.interpolate_to_grid(method="linear") on the T1 scoring grid, 1 A -> 1 mA.
It then maps twice:
  - grey-matter points: from the leadfield cropped to grey-matter elements, i.e. exactly the published procedure. This
    reproduces the published FEM basis (checked per head on the desktop) and serves the 26 cerebellar regions;
  - brainstem points (prep_brainstem.py): from the white- plus grey-matter leadfield.
Output: <out>/<code>__fem_ext.npz with basis_gm (18 x 3 x n_gm) and basis_bs (18 x 3 x n_bs), float32, labs, meta.
    "<SimNIBS>/python.exe" pipeline/nchc/atlas_ext_fem.py --head h001 --m2m <m2m dir with <code>.msh> --targets <code>__exttargets.npz --out DIR [--work DIR]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline"), HERE]
WM_LABEL, GM_LABEL = 1, 2
REF = "Cz"
ELEC_RADIUS_MM = 12.0
LF_METHOD = "linear"            # as v22_fem_fullsearch.LF_METHOD


def main():
    a = sys.argv[1:]
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    code, m2m, tfile, out = arg("--head"), arg("--m2m"), arg("--targets"), arg("--out")
    work = arg("--work", os.path.join(out, f"{code}_leadfield"))
    import h5py
    from simnibs import sim_struct, run_simnibs
    from simnibs.mesh_tools import mesh_io
    from spicehead.conductivity import CONDUCTIVITY
    import nibabel as nib
    p_out = os.path.join(out, f"{code}__fem_ext.npz")
    if os.path.exists(p_out):
        print(f"{code}: cached"); return
    probe = sim_struct.SESSION().add_tdcslist()
    ours = [CONDUCTIVITY[k] for k in ("white_matter", "gray_matter", "csf", "bone", "scalp", "eyeballs", "compact_bone", "spongy_bone", "blood", "muscle")]
    for i, (x, y) in enumerate(zip(ours, [probe.cond[i].value for i in range(10)]), 1):
        if x is None or y is None or abs(x - y) > 1e-9:
            raise SystemExit(f"conductivity mismatch at tissue {i}: spicehead {x} vs simnibs {y}")
    zt = np.load(tfile)
    gm_idx = zt["gm_idx"].astype(np.int64); bs_idx = zt["bs_idx"].astype(np.int64)
    t1 = nib.load(os.path.join(m2m, "T1.nii.gz")); aff, shape = t1.affine, t1.shape[:3]
    if not np.allclose(aff, np.asarray(zt["affine"], float)) or tuple(shape) != tuple(int(s) for s in zt["shape"]):
        raise SystemExit(f"{code}: target file grid differs from the m2m T1 grid")
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs = [REF] + sorted(l for l in ej if l != REF)
    os.makedirs(work, exist_ok=True)
    cap = os.path.join(work, "cap_1020_subject.csv")
    with open(cap, "w", newline="\n") as fh:
        for lab in labs:
            x, y, z = ej[lab]
            fh.write(f"Electrode,{x},{y},{z},{lab}\n")
    t0 = time.time()
    hdf = [f for f in os.listdir(work) if f.endswith(".hdf5")]
    if not hdf:
        lf = sim_struct.TDCSLEADFIELD()
        lf.subpath, lf.pathfem, lf.eeg_cap = m2m, work, cap
        lf.electrode.shape = "ellipse"
        lf.electrode.dimensions = [2 * ELEC_RADIUS_MM, 2 * ELEC_RADIUS_MM]
        lf.electrode.thickness = [2.0]
        lf.interpolation, lf.tissues, lf.field = None, [WM_LABEL, GM_LABEL], "E"
        lf.open_in_gmsh = False
        run_simnibs(lf)
        hdf = [f for f in os.listdir(work) if f.endswith(".hdf5")]
    t_lf = time.time() - t0
    fn = os.path.join(work, hdf[0])
    t0 = time.time()
    msh = mesh_io.Msh.read_hdf5(fn, "mesh_leadfield/")
    sel_gm = msh.elm.tag1 == GM_LABEL
    msh_gm = msh.crop_mesh(elements=np.flatnonzero(sel_gm) + 1)
    assert msh_gm.elm.nr == int(sel_gm.sum())
    with h5py.File(fn, "r") as h:
        ds = h["mesh_leadfield/leadfields/tdcs_leadfield"]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in ds.attrs["electrode_names"]]
        ref = ds.attrs["reference_electrode"]; ref = ref.decode() if isinstance(ref, bytes) else str(ref)
        if ref != REF:
            raise RuntimeError(f"{code}: leadfield reference is {ref}, expected {REF}")
        others = [n for n in names if n != ref]
        bgm = np.zeros((len(others), 3, len(gm_idx)), np.float32)
        bbs = np.zeros((len(others), 3, len(bs_idx)), np.float32)
        nan_bs = 0
        for i in range(len(others)):
            v = ds[i] * 1e-3                                              # 1 A -> 1 mA
            G = np.nan_to_num(mesh_io.ElementData(v[sel_gm], name="E", mesh=msh_gm).interpolate_to_grid(shape, aff, method=LF_METHOD))
            bgm[i] = G[gm_idx[:, 0], gm_idx[:, 1], gm_idx[:, 2]].T
            Gf = mesh_io.ElementData(v, name="E", mesh=msh).interpolate_to_grid(shape, aff, method=LF_METHOD)
            sb = Gf[bs_idx[:, 0], bs_idx[:, 1], bs_idx[:, 2]]
            nan_bs = max(nan_bs, int(np.isnan(sb).any(1).sum()))
            bbs[i] = np.nan_to_num(sb).T
            print(f"  {code}: mapped {i+1}/{len(others)} ({(time.time()-t0)/(i+1):.0f} s/field)", flush=True)
    meta = dict(code=code, n_gm=int(len(gm_idx)), n_bs=int(len(bs_idx)), n_elements=int(msh.elm.nr), n_elements_gm=int(msh_gm.elm.nr),
                bs_points_outside_mesh=nan_bs, t_leadfield_s=round(t_lf, 1), t_map_s=round(time.time() - t0, 1), interpolation=LF_METHOD,
                solver="SimNIBS 4.6 TDCSLEADFIELD, 19 pads meshed, reference Cz, tissues WM+GM; GM mapped from the GM-cropped leadfield")
    np.savez(p_out + ".part.npz", basis_gm=bgm, basis_bs=bbs, labs=np.array(others), meta=json.dumps(meta))
    os.replace(p_out + ".part.npz", p_out)
    print(f"{code}: FEM extension done ({json.dumps(meta)})", flush=True)


if __name__ == "__main__":
    main()
