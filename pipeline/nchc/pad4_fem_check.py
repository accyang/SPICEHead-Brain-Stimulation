# -*- coding: utf-8 -*-
"""Four-pad FEM check of the cohort montages (31 Oct 2026; review decision D4, PI: "agree all").

The FEM reference of Paper A is a SimNIBS leadfield with all 19 pads meshed: every montage is a superposition of
Cz-referenced fields in a geometry where the 15 unused pads are present as passive conductors. Eight-head evidence
(v22_fem_fullsearch.fem_superposition_check) showed that the unused pads alter the field. This script measures that
effect on every head for the six reference targets: for each ref6 cohort montage it runs SimNIBS directly with ONLY the
four active pads (two TDCS runs, one per channel, same pad geometry, currents and conductivities as the leadfield),
maps the fields to the T1 grid, forms the TI envelope on the grey-matter voxels of the scoring grid and scores T, O
and F with the grid, masks and definitions of the 300-head comparison (v22_fem_fullsearch.grid_geometry). The 19-pad
values of the same montages come from the stored <code>__fem.json on the desktop at collection time.
Runs under the SimNIBS interpreter (CPU). Output: <out>/<code>__pad4.json; the SimNIBS mesh outputs are deleted.
    python pipeline/nchc/pad4_fem_check.py --head h002 --m2m DIR --montages results_v23/v23_uq/uq_fixed_montages.json --out DIR [--threads 8]

--leadfield (1 Nov 2026, Nano 4 run): charm meshing is not bit-reproducible (h002 rebuilt on Nano 4: 678,266 nodes vs
681,947 in the archived mesh), so when the mesh has to be rebuilt the 19-pad reference is recomputed on the SAME mesh:
one TDCSLEADFIELD (all 19 pads, reference Cz, the leadfield code of v22_fem_fullsearch.stage_fem_leadfield) mapped
to the scoring grid, and the six montages scored by superposition -> "targets19". Then 4-pad vs 19-pad on one mesh is
the passive-pad effect alone, and targets19 vs the archived <code>__fem.json is the re-mesh effect. The JSON also
records the mesh SHA-256, node and element counts. --lf-first runs the leadfield before the direct runs (testing).
"""
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "pipeline")); sys.path.insert(0, HERE)


def main():
    a = sys.argv[1:]
    code = a[a.index("--head") + 1]
    m2m = a[a.index("--m2m") + 1]
    mj = a[a.index("--montages") + 1]
    out = a[a.index("--out") + 1]
    threads = int(a[a.index("--threads") + 1]) if "--threads" in a else 8
    os.makedirs(out, exist_ok=True)
    final = os.path.join(out, f"{code}__pad4.json")
    if os.path.exists(final):
        print(f"{code}: done before"); return
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import v22_fem_fullsearch as fs
    from v21_simnibs_compare import load_simnibs_E
    from spicehead.envelope_fast import envelope_max_direction_flat
    from simnibs import sim_struct, run_simnibs
    t0 = time.time()
    geo = fs.grid_geometry(code, m2m)
    gm_idx = np.asarray(geo["gm_idx"]).astype(np.int64)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    monts = json.load(open(mj))["montages"]                      # target -> "A+B / C+D"
    pairs = sorted({tuple(p.split("+")) for m in monts.values() for p in m.split(" / ")})
    work = os.path.join(out, f"{code}__runs")
    fields, times = {}, {}
    lf = leadfield_scores(code, m2m, ej, monts, geo, gm_idx, work) if "--leadfield" in a and "--lf-first" in a else None
    for pa, pb in pairs:
        key = f"{pa}+{pb}"
        fem = os.path.join(work, key)
        vol = os.path.join(fem, "subject_volumes")
        if not (os.path.isdir(vol) and any(f.endswith("_E.nii.gz") for f in os.listdir(vol))):
            shutil.rmtree(fem, ignore_errors=True); os.makedirs(fem)
            S = sim_struct.SESSION()
            S.subpath, S.pathfem = m2m, fem
            S.fields = "E"; S.map_to_vol = True; S.open_in_gmsh = False
            t = S.add_tdcslist()
            t.currents = [fs.CURRENT, -fs.CURRENT]
            for name, chn in ((pa, 1), (pb, 2)):               # the same pad model as the leadfield: 24 mm ellipse, 2 mm thick
                e = t.add_electrode()
                e.channelnr = chn; e.centre = ej[name]; e.shape = "ellipse"
                e.dimensions = [2 * fs.ELEC_RADIUS_MM, 2 * fs.ELEC_RADIUS_MM]; e.thickness = [2.0]
            t1 = time.time()
            run_simnibs(S)
            times[key] = round(time.time() - t1, 1)
            for f in os.listdir(fem):                           # the mesh results are not read by anything
                if f.endswith((".msh", ".geo")):
                    try: os.remove(os.path.join(fem, f))
                    except OSError: pass
        (Ex, Ey, Ez), aff, shape = load_simnibs_E(fem)
        if tuple(shape) != tuple(geo["shape"]) or not np.allclose(aff, geo["aff"], atol=1e-3):
            raise SystemExit(f"{code}: SimNIBS volume grid differs from the scoring grid")
        fields[key] = np.stack([c[gm_idx[:, 0], gm_idx[:, 1], gm_idx[:, 2]] for c in (Ex, Ey, Ez)]).astype(np.float32)
        print(f"{code} {key}: field on {len(gm_idx):,} GM voxels ({times.get(key, 'cached')} s)", flush=True)
    res = {}
    for tgt, m in monts.items():
        A, C = (fields[p] for p in m.split(" / "))
        env = envelope_max_direction_flat(A[0], A[1], A[2], C[0], C[1], C[2])
        T = float(np.median(env[geo["roi_sel"][tgt]])); O = float(np.percentile(env[geo["out_sel"][tgt]], 99))
        res[tgt] = dict(montage=m, T=T, O=O, F=T / O if O > 0 else 0.0, n_roi=int(geo["roi_sel"][tgt].sum()),
                        env_gm_max=float(env.max()), env_gm_p999=float(np.percentile(env, 99.9)))
    doc = dict(code=code, model="SimNIBS direct runs, four active pads only (two per channel), 1 mA per channel",
               pads="24 mm ellipse, 2 mm, SimNIBS default conductivities (= leadfield)", grid="T1 grid of the 300-head comparison",
               n_pairs=len(pairs), run_seconds=times, minutes=round((time.time() - t0) / 60, 1), targets=res)
    json.dump(doc, open(final + ".pad4part.json", "w"), indent=1)          # the direct-run part survives a leadfield failure
    if "--leadfield" in a:
        if lf is None:
            lf = leadfield_scores(code, m2m, ej, monts, geo, gm_idx, work)
        doc.update(targets19=lf["targets"], leadfield=lf["meta"], minutes=round((time.time() - t0) / 60, 1))
    mp = os.path.join(m2m, f"{code}.msh")
    if os.path.exists(mp):
        import hashlib
        h = hashlib.sha256()
        with open(mp, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        doc["mesh_sha256"] = h.hexdigest()
    json.dump(doc, open(final + ".part", "w"), indent=1)
    os.replace(final + ".part", final)
    try: os.remove(final + ".pad4part.json")
    except OSError: pass
    shutil.rmtree(work, ignore_errors=True)
    print(f"{code}: {len(res)} targets, {len(pairs)} direct runs{', 19-pad leadfield' if lf else ''}, {(time.time() - t0) / 60:.1f} min", flush=True)


def leadfield_scores(code, m2m, ej, monts, geo, gm_idx, work):
    """19-pad TDCSLEADFIELD on the head's current mesh (reference Cz), mapped to the scoring grid; the six montages
    scored by superposition exactly as v22_fem_fullsearch.score_head does for the 300-head comparison."""
    import h5py
    import v22_fem_fullsearch as fs
    from simnibs import sim_struct, run_simnibs
    from simnibs.mesh_tools import mesh_io
    from spicehead.envelope_fast import envelope_max_direction_flat
    REF = fs.REF
    out = os.path.join(work, "leadfield"); os.makedirs(out, exist_ok=True)
    labs = [REF] + sorted(l for l in ej if l != REF)
    cap = os.path.join(out, "cap_1020_subject.csv")
    with open(cap, "w", newline="\n") as fh:
        for lab in labs:
            x, y, z = ej[lab]
            fh.write(f"Electrode,{x},{y},{z},{lab}\n")
    t0 = time.time()
    hdf = [f for f in os.listdir(out) if f.endswith(".hdf5")]
    if not hdf:
        L = sim_struct.TDCSLEADFIELD()
        L.subpath, L.pathfem, L.eeg_cap = m2m, out, cap
        L.electrode.shape = "ellipse"
        L.electrode.dimensions = [2 * fs.ELEC_RADIUS_MM, 2 * fs.ELEC_RADIUS_MM]
        L.electrode.thickness = [2.0]
        L.interpolation, L.tissues, L.field = None, [fs.GM_LABEL], "E"
        L.open_in_gmsh = False
        run_simnibs(L)
        hdf = [f for f in os.listdir(out) if f.endswith(".hdf5")]
    t_lf = time.time() - t0
    fn = os.path.join(out, hdf[0])
    t1 = time.time()
    msh = mesh_io.Msh.read_hdf5(fn, "mesh_leadfield/")
    shape, aff = geo["shape"], geo["aff"]
    with h5py.File(fn, "r") as h:
        ds = h["mesh_leadfield/leadfields/tdcs_leadfield"]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in ds.attrs["electrode_names"]]
        ref = ds.attrs["reference_electrode"]
        ref = ref.decode() if isinstance(ref, bytes) else str(ref)
        if ref != REF:
            raise RuntimeError(f"{code}: leadfield reference is {ref}, expected {REF}")
        others = [n for n in names if n != ref]
        basis = np.zeros((len(others), 3, len(gm_idx)), np.float32)
        for i in range(len(others)):
            ed = mesh_io.ElementData(ds[i] * 1e-3, name="E", mesh=msh)          # 1 A -> 1 mA
            G = np.nan_to_num(ed.interpolate_to_grid(shape, aff, method=fs.LF_METHOD))
            basis[i] = G[gm_idx[:, 0], gm_idx[:, 1], gm_idx[:, 2]].T
    idx = {lab: i for i, lab in enumerate(others)}
    pf = lambda x, y: (basis[idx[x]] if x != REF else 0.0) - (basis[idx[y]] if y != REF else 0.0)
    res = {}
    for tgt, m in monts.items():
        (pa, pb), (pc, pd) = [tuple(p.split("+")) for p in m.split(" / ")]
        A, C = pf(pa, pb), pf(pc, pd)
        env = envelope_max_direction_flat(A[0], A[1], A[2], C[0], C[1], C[2])
        T = float(np.median(env[geo["roi_sel"][tgt]])); O = float(np.percentile(env[geo["out_sel"][tgt]], 99))
        res[tgt] = dict(montage=m, T=T, O=O, F=T / O if O > 0 else 0.0, n_roi=int(geo["roi_sel"][tgt].sum()),
                        env_gm_max=float(env.max()), env_gm_p999=float(np.percentile(env, 99.9)))
    meta = dict(solver="SimNIBS 4.6 TDCSLEADFIELD (hypre), 19 pads meshed, reference Cz", interpolation=fs.LF_METHOD,
                n_fields=len(others), leadfield_mesh_nodes_gm_crop=int(msh.nodes.nr), leadfield_mesh_elements_gm_crop=int(msh.elm.nr),
                t_leadfield_s=round(t_lf, 1), t_map_s=round(time.time() - t1, 1))
    print(f"{code} 19-pad leadfield: solve {t_lf/60:.1f} min, map {(time.time()-t1)/60:.1f} min, mesh {msh.nodes.nr:,} nodes", flush=True)
    shutil.rmtree(out, ignore_errors=True)
    return dict(targets=res, meta=meta)


if __name__ == "__main__":
    main()
