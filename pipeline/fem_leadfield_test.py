# -*- coding: utf-8 -*-
"""Time a SimNIBS TDCSLEADFIELD for our 19 10-20 electrodes on one head, and inspect its output.

A leadfield meshes every electrode once and solves one right-hand side per electrode against the
reference (the first entry of the cap file), so superposition across electrodes is exact by
construction. If it is fast enough, it replaces the 18 separate runs per head of
v22_fem_fullsearch.py --fem-basis. Fields are requested per grey-matter element (no surface
interpolation); the comparison script will map them to the T1 grid.

  "<SimNIBS>\\simnibs_env\\python.exe" pipeline/fem_leadfield_test.py
"""
import json
import os
import time

M2M = r"L:\BrainProbe\SPICEhead\charm\m2m_SUBJ01"
OUT = r"C:\spicehead_v22_logs\fem_tests\leadfield"


def main():
    from simnibs import sim_struct, run_simnibs
    os.makedirs(OUT, exist_ok=True)
    ej = json.load(open(os.path.join(M2M, "electrodes_1020.json")))
    labs = ["Cz"] + sorted(l for l in ej if l != "Cz")          # Cz first = reference
    cap = os.path.join(OUT, "cap_1020_subject.csv")
    with open(cap, "w", newline="\n") as fh:
        for lab in labs:
            x, y, z = ej[lab]
            fh.write(f"Electrode,{x},{y},{z},{lab}\n")
    lf = sim_struct.TDCSLEADFIELD()
    lf.subpath, lf.pathfem = M2M, OUT
    lf.eeg_cap = cap
    lf.electrode.shape = "ellipse"
    lf.electrode.dimensions = [24.0, 24.0]
    lf.electrode.thickness = [2.0]
    lf.interpolation = None            # keep the field in the volume elements ...
    lf.tissues = [2]                   # ... of grey matter
    lf.field = "E"
    lf.open_in_gmsh = False
    t0 = time.time()
    run_simnibs(lf)
    print(f"leadfield OK ({(time.time()-t0)/60:.1f} min)", flush=True)
    import h5py
    fn = [f for f in os.listdir(OUT) if f.endswith(".hdf5")][0]
    with h5py.File(os.path.join(OUT, fn), "r") as h:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: {obj.shape} {obj.dtype}")
            for k, v in obj.attrs.items():
                print(f"    attr {name}@{k} = {str(v)[:120]}")
        h.visititems(show)
    print("file:", os.path.join(OUT, fn), f"{os.path.getsize(os.path.join(OUT, fn))/1e6:.0f} MB")


if __name__ == "__main__":
    main()
