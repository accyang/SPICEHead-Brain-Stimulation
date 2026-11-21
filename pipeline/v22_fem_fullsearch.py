# -*- coding: utf-8 -*-
"""Matched full-space comparison: SimNIBS FEM against SPICEhead over all 11,628 montages.

Plan of 2 October 2026 (v22/PLAN_TI_TOOLBOX_COMPARISON_2026-10-02.md, item 1): in the eight
FEM-comparison heads, both models evaluate every two-pair 10-20 montage from 18 single-electrode
basis fields (each electrode against Cz; E(a,b) = E(a,Cz) - E(b,Cz) holds for FEM exactly as for
the network), and both are scored with the same code on the same grid:

  grid     the SimNIBS output grid (subject T1, 1 mm); SPICEhead fields are sampled there
  GM       the CHARM grey-matter label resampled to that grid (> 0.5), as v21_simnibs_compare
  targets  10 mm spheres about the warped centres in roi_coords.json; O = p99 of GM beyond 20 mm
  set      171 pairs at least 40 mm apart by world coordinates; 11,628 montages, one list for all
  models   FEM, SP2 (SPICEhead 2 mm), SP1 (SPICEhead 1 mm)

What is NOT matched, and is reported as such: the electrode boundary condition (SimNIBS solves
the current through a 24 mm, 2 mm thick conductive pad; SPICEhead injects uniformly over the skin
contact nodes of a 12 mm radius disc) and the discretization (tetrahedral mesh vs voxel network).

Stages, each resumable and shardable (--shard k/n), restricted with --heads A,B,...:
  --fem-basis          18 SimNIBS runs per head (SimNIBS interpreter); meshes deleted, volumes kept
  --sp-basis <dx>      18 SPICEhead solves per head at dx, sampled on the T1-grid GM voxels, cached
  --score <model>      F, T, O for every montage and target; model = fem | sp2 | sp1
  --collect            decisions, rank correlation, cross-evaluated regret, timings, checks

  "<SimNIBS>\\simnibs_env\\python.exe" pipeline/v22_fem_fullsearch.py --fem-basis [--shard k/n]
  python pipeline/v22_fem_fullsearch.py --sp-basis 2.0 [--shard k/n]     (then --score sp2)
  python pipeline/v22_fem_fullsearch.py --score fem
  python pipeline/v22_fem_fullsearch.py --collect

Outputs: <results>/v22_fem_fullsearch/<head>__<model>.json (scores, 7 significant digits, plus
stage times), fullsearch_decisions.csv, fullsearch_cohort.csv, fullsearch_timing.csv,
fullsearch_summary.json. FEM volumes live outside Dropbox (SPICEHEAD_FEM_BASIS_ROOT); the
SPICEhead basis cache on SPICEHEAD_FULLSEARCH_CACHE.
"""
import csv
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)

CH = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")
FEM_BASIS = os.environ.get("SPICEHEAD_FEM_BASIS_ROOT", r"L:\BrainProbe\SPICEhead\fem_basis")
CACHE = os.environ.get("SPICEHEAD_FULLSEARCH_CACHE", r"C:\spicehead_v22_logs\fullsearch_cache")
FEM_REF = os.path.join(ROOT, "results", "v21_simnibs_validation")   # the 64 existing direct runs


def sp_basis_path(cid, tag):
    """SPICEhead basis cache of one head and grid. v22 keeps its historical file names; any other model
    version (v23 since 23 Oct 2026) carries its tag, so a shared cache folder can never mix versions."""
    from spicehead import config
    sfx = "" if config.MODEL == "v22" else f"_{config.MODEL}"
    return os.path.join(CACHE, f"{cid}__{tag}{sfx}_basis.npz")

REF = "Cz"
CURRENT = 1e-3
ELEC_RADIUS_MM = 12.0
MIN_DIST_MM = 40.0
ROI_RADIUS_MM = 10.0
EXCLUDE_MM = 20.0
GM_LABEL, SCALP_LABEL = 2, 5
ROI_ORDER = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]
CHECK_PAIRS = (("P8", "T8"), ("F7", "T7"))
MODELS = {"fem": None, "sp2": 2.0, "sp1": 1.0}
# the directly simulated montages of the existing comparison, for the FEM superposition check
# (channel = (+electrode, -electrode); the inferior chain uses F9/T9/F10/T10 and is not in the set)
DIRECT = {"THAL": (("T8", "P8"), ("F7", "T7"), ("ch1", "ch2")),
          "M1": (("F3", "F4"), ("Pz", "O1"), ("M1_ch1", "M1_ch2")),
          "HIPP": (("Fp2", "F7"), ("P8", "O2"), ("HIPP_ch1", "HIPP_ch2"))}


def out_dir():
    from spicehead import config
    # SPICEHEAD_FULLSEARCH_OUT redirects the outputs, e.g. for the clean one-head benchmark
    d = os.environ.get("SPICEHEAD_FULLSEARCH_OUT") or os.path.join(config.RESULTS, "v22_fem_fullsearch")
    os.makedirs(d, exist_ok=True)
    return d


def heads():
    """The eight FEM-comparison heads: the same selection as v21_simnibs_validation.subjects(8).

    SPICEHEAD_FS_HEADS=<csv with a CaseID column> (added 17 Oct 2026 for the NCHC batches) replaces
    that selection with a pinned list; those heads need no existing direct FEM runs, and their
    scoring grid comes from the m2m T1 image (see fem_grid)."""
    pin = os.environ.get("SPICEHEAD_FS_HEADS")
    if pin:
        from run_montage_cohort import resolve_m2m
        ids = [r["CaseID"] for r in csv.DictReader(open(pin, encoding="utf-8-sig"))]
        return [(c, resolve_m2m(CH, c)) for c in ids]
    from v21_simnibs_validation import subjects
    subs = subjects(8)
    have = {d for d in os.listdir(FEM_REF) if os.path.isdir(os.path.join(FEM_REF, d))}
    missing = [c for c, *_ in subs if c not in have]
    if missing:
        raise SystemExit(f"FEM heads without existing reference runs: {missing}")
    return [(c, m2m) for c, m2m, *_ in subs]


def select(args, subs):
    if "--heads" in args:
        want = args[args.index("--heads") + 1].split(",")
        subs = [s for s in subs if s[0] in want]
    if "--shard" in args:
        k, n = (int(x) for x in args[args.index("--shard") + 1].split("/"))
        subs = [s for i, s in enumerate(subs) if i % n == k]
    return subs


def g7(x):
    return float(f"{x:.7g}")


def label_of(pa, pb):
    """Order-independent montage label, identical to values_v22.label / dx2 canon()."""
    return " / ".join(sorted("+".join(sorted(p)) for p in (pa, pb)))


def candidate_set(ej):
    """Pairs >= 40 mm apart by WORLD coordinates (one list for every model) and the montages."""
    labs = sorted(ej)
    pairs = [(a, b) for a, b in combinations(labs, 2)
             if np.linalg.norm(np.subtract(ej[a], ej[b])) >= MIN_DIST_MM]
    montages = [(pa, pb) for pa, pb in combinations(pairs, 2) if len(set(pa) | set(pb)) == 4]
    return labs, pairs, montages


# ---------------------------------------------------------------- scoring grid (shared)
def fem_grid(cid, m2m=None):
    """Affine and shape of the SimNIBS output grid, from the head's existing ch1 run.

    Heads without direct runs (NCHC batches) use the m2m T1 image instead: SimNIBS maps volumes to
    that grid, so FEM and SPICEhead are still scored on one common grid."""
    import nibabel as nib
    vol = os.path.join(FEM_REF, cid, "ch1", "subject_volumes")
    if os.path.isdir(vol):
        p = [os.path.join(vol, f) for f in sorted(os.listdir(vol)) if f.endswith("_E.nii.gz")][0]
    elif m2m and os.path.exists(os.path.join(m2m, "T1.nii.gz")):
        p = os.path.join(m2m, "T1.nii.gz")
    else:
        raise FileNotFoundError(f"{cid}: no direct FEM run and no m2m T1.nii.gz for the scoring grid")
    img = nib.load(p)
    return img.affine, img.shape[:3]


def grid_geometry(cid, m2m):
    """GM voxels of the T1 grid with their world coordinates and the per-target masks (cached)."""
    import nibabel as nib
    from montage_search import roi_mask
    from v21_simnibs_compare import resample_to
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{cid}__grid.npz")
    aff, shape = fem_grid(cid, m2m)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return dict(aff=aff, shape=shape, gm_idx=z["gm_idx"], gm_world=z["gm_world"],
                    roi_sel=z["roi_sel"].item(), out_sel=z["out_sel"].item())
    lab_img = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    lab = np.asanyarray(lab_img.dataobj)
    gm = resample_to((lab == GM_LABEL).astype(np.float32), lab_img.affine, aff, shape) > 0.5
    gm_idx = np.argwhere(gm)
    gm_world = (aff[:3, :3] @ gm_idx.T).T + aff[:3, 3]
    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    roi_sel = {r: roi_mask(shape, gm_idx, gm_world, rois[r], ROI_RADIUS_MM)[gm] for r in ROI_ORDER}
    out_sel = {r: ~roi_mask(shape, gm_idx, gm_world, rois[r], EXCLUDE_MM)[gm] for r in ROI_ORDER}
    # atomic: the 2 mm and 1 mm jobs may start the same head at the same time
    np.savez(p + f".{os.getpid()}.part.npz", gm_idx=gm_idx.astype(np.int16),
             gm_world=gm_world.astype(np.float32),
             roi_sel=np.array(roi_sel, dtype=object), out_sel=np.array(out_sel, dtype=object))
    os.replace(p + f".{os.getpid()}.part.npz", p)
    return dict(aff=aff, shape=shape, gm_idx=gm_idx, gm_world=gm_world, roi_sel=roi_sel,
                out_sel=out_sel)


# ---------------------------------------------------------------- stage: FEM basis (SimNIBS)
def stage_fem_basis(args):
    """Run under the SimNIBS interpreter: one TDCS run per electrode against Cz, mapped to volume."""
    from simnibs import sim_struct, run_simnibs
    from spicehead.conductivity import CONDUCTIVITY
    probe = sim_struct.SESSION().add_tdcslist()
    ours = [CONDUCTIVITY[k] for k in ("white_matter", "gray_matter", "csf", "bone", "scalp",
                                      "eyeballs", "compact_bone", "spongy_bone", "blood", "muscle")]
    for i, (a, b) in enumerate(zip(ours, [probe.cond[i].value for i in range(10)]), 1):
        if a is None or b is None or abs(a - b) > 1e-9:
            raise SystemExit(f"conductivity mismatch at tissue {i}: spicehead {a} vs simnibs {b}")
    print("conductivities match SimNIBS defaults for all 10 tissues", flush=True)
    subs = select(args, heads())
    for hi, (cid, m2m) in enumerate(subs, 1):
        ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
        labs, pairs, montages = candidate_set(ej)
        tfile = os.path.join(FEM_BASIS, cid, "timings.json")
        times = json.load(open(tfile)) if os.path.exists(tfile) else {}
        for li, lab in enumerate([x for x in labs if x != REF], 1):
            fem = os.path.join(FEM_BASIS, cid, lab)
            vol = os.path.join(fem, "subject_volumes")
            if os.path.isdir(vol) and any(f.endswith("_E.nii.gz") for f in os.listdir(vol)):
                print(f"  [{hi}/{len(subs)}] {cid} {lab} (cached)", flush=True)
                continue
            os.makedirs(fem, exist_ok=True)
            S = sim_struct.SESSION()
            S.subpath, S.pathfem = m2m, fem
            S.fields = "E"                   # the vector field only; load_simnibs_E reads *_E
            S.map_to_vol = True
            S.open_in_gmsh = False
            t = S.add_tdcslist()
            t.currents = [CURRENT, -CURRENT]
            for name, chn in ((lab, 1), (REF, 2)):
                e = t.add_electrode()
                e.channelnr = chn
                e.centre = ej[name]
                e.shape = "ellipse"
                e.dimensions = [2 * ELEC_RADIUS_MM, 2 * ELEC_RADIUS_MM]
                e.thickness = [2.0]
            t0 = time.time()
            try:
                run_simnibs(S)
            except Exception as ex:
                print(f"  [{hi}/{len(subs)}] {cid} {lab} FAILED {type(ex).__name__}: {ex}", flush=True)
                continue
            dt = time.time() - t0
            # the mesh results (~180 MB per run) are not read by anything; keep logs and volumes
            for f in os.listdir(fem):
                if f.endswith((".msh", ".geo")):
                    try:
                        os.remove(os.path.join(fem, f))
                    except OSError:
                        pass
            times[lab] = round(dt, 1)
            json.dump(times, open(tfile, "w"), indent=1)
            print(f"  [{hi}/{len(subs)}] {cid} {lab} OK ({dt/60:.1f} min) [{li}/18]", flush=True)


# ---------------------------------------------------------------- stage: FEM leadfield (SimNIBS)
LF_METHOD = "linear"      # ElementData.interpolate_to_grid method, as SimNIBS's own map_to_vol


def stage_fem_leadfield(args):
    """Run under the SimNIBS interpreter: one TDCSLEADFIELD per head with all 19 pads meshed
    (reference Cz), then the 18 fields mapped to the T1 grid and kept on the GM voxels.

    This replaces --fem-basis. Eighteen separate two-pad runs are NOT a valid basis: a pad with
    zero net current still shunts current, so E(a,Cz) - E(b,Cz) across runs with different pads
    missed the direct E(a,b) by 4-8% at the worst voxel (test of 3 October 2026). With every pad
    present in one geometry the superposition is exact by construction.
    """
    import h5py
    from simnibs import sim_struct, run_simnibs
    from simnibs.mesh_tools import mesh_io
    from spicehead.conductivity import CONDUCTIVITY
    probe = sim_struct.SESSION().add_tdcslist()
    ours = [CONDUCTIVITY[k] for k in ("white_matter", "gray_matter", "csf", "bone", "scalp",
                                      "eyeballs", "compact_bone", "spongy_bone", "blood", "muscle")]
    for i, (a, b) in enumerate(zip(ours, [probe.cond[i].value for i in range(10)]), 1):
        if a is None or b is None or abs(a - b) > 1e-9:
            raise SystemExit(f"conductivity mismatch at tissue {i}: spicehead {a} vs simnibs {b}")
    print("conductivities match SimNIBS defaults for all 10 tissues", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    subs = select(args, heads())
    for hi, (cid, m2m) in enumerate(subs, 1):
        cache = os.path.join(CACHE, f"{cid}__fem_basis.npz")
        if os.path.exists(cache):
            print(f"  [{hi}/{len(subs)}] {cid} fem leadfield (cached)", flush=True)
            continue
        gp = os.path.join(CACHE, f"{cid}__grid.npz")
        if not os.path.exists(gp):
            raise SystemExit(f"{cid}: scoring grid {gp} missing; run --sp-basis 2.0 for this head first")
        geo = np.load(gp, allow_pickle=True)
        gm_idx = geo["gm_idx"].astype(int)
        aff, shape = fem_grid(cid, m2m)
        ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
        labs = [REF] + sorted(l for l in ej if l != REF)
        out = os.path.join(FEM_BASIS, cid, "leadfield")
        os.makedirs(out, exist_ok=True)
        cap = os.path.join(out, "cap_1020_subject.csv")
        with open(cap, "w", newline="\n") as fh:
            for lab in labs:
                x, y, z = ej[lab]
                fh.write(f"Electrode,{x},{y},{z},{lab}\n")
        hdf = [f for f in os.listdir(out) if f.endswith(".hdf5")]
        t0 = time.time()
        if not hdf:
            lf = sim_struct.TDCSLEADFIELD()
            lf.subpath, lf.pathfem, lf.eeg_cap = m2m, out, cap
            lf.electrode.shape = "ellipse"
            lf.electrode.dimensions = [2 * ELEC_RADIUS_MM, 2 * ELEC_RADIUS_MM]
            lf.electrode.thickness = [2.0]
            lf.interpolation, lf.tissues, lf.field = None, [GM_LABEL], "E"
            lf.open_in_gmsh = False
            run_simnibs(lf)
            hdf = [f for f in os.listdir(out) if f.endswith(".hdf5")]
        t_lf = time.time() - t0
        fn = os.path.join(out, hdf[0])
        t0 = time.time()
        msh = mesh_io.Msh.read_hdf5(fn, "mesh_leadfield/")
        with h5py.File(fn, "r") as h:
            ds = h["mesh_leadfield/leadfields/tdcs_leadfield"]
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in ds.attrs["electrode_names"]]
            ref = ds.attrs["reference_electrode"]
            ref = ref.decode() if isinstance(ref, bytes) else str(ref)
            if ref != REF:
                raise RuntimeError(f"{cid}: leadfield reference is {ref}, expected {REF}")
            others = [n for n in names if n != ref]
            basis = np.zeros((len(others), 3, len(gm_idx)), np.float32)
            for i in range(len(others)):
                ed = mesh_io.ElementData(ds[i] * 1e-3, name="E", mesh=msh)      # 1 A -> 1 mA
                G = np.nan_to_num(ed.interpolate_to_grid(shape, aff, method=LF_METHOD))
                basis[i] = G[gm_idx[:, 0], gm_idx[:, 1], gm_idx[:, 2]].T
                if (i + 1) % 6 == 0 or i + 1 == len(others):
                    print(f"      mapped {i+1}/{len(others)} ({(time.time()-t0)/(i+1):.0f} s/field)", flush=True)
        t_map = time.time() - t0
        meta = dict(subject=cid, model="fem", dx_mm=None, n_gm=int(len(gm_idx)),
                    n_elements=int(msh.elm.nr), n_nodes_mesh=int(msh.nodes.nr),
                    t_basis_s=round(t_lf, 1), t_per_solve_s=round(t_lf / len(others), 1),
                    t_map_s=round(t_map, 1), leadfield=fn, interpolation=LF_METHOD,
                    solver="SimNIBS 4.6 TDCSLEADFIELD (hypre), 19 pads meshed, reference Cz")
        np.savez(cache + ".part.npz", basis=basis, labs=np.array(others), meta=json.dumps(meta))
        os.replace(cache + ".part.npz", cache)
        print(f"  [{hi}/{len(subs)}] {cid} fem leadfield OK (solve {t_lf/60:.1f} min, map {t_map/60:.1f} min)",
              flush=True)


# ---------------------------------------------------------------- stage: SPICEhead basis
def sample_on_gm(comp, aff_dx, gm_world):
    """Trilinear sample of one field component (on the SPICEhead grid) at the GM voxel centres."""
    from scipy.ndimage import map_coordinates
    inv = np.linalg.inv(aff_dx)
    vox = (inv[:3, :3] @ gm_world.T) + inv[:3, 3:4]
    return map_coordinates(np.nan_to_num(np.asarray(comp, np.float64)), vox, order=1, mode="nearest")


def stage_sp_basis(args, dx):
    from spicehead import mri, config
    from spicehead.electrodes import Electrode, contact_nodes, pair_injection
    from spicehead.field import efield_components
    from montage_search import snap
    from v21_head_convergence import make_solver
    subs = select(args, heads())
    tag = f"sp{dx:g}"
    for hi, (cid, m2m) in enumerate(subs, 1):
        p = sp_basis_path(cid, tag)
        if os.path.exists(p):
            print(f"  [{hi}/{len(subs)}] {cid} {tag} (cached)", flush=True)
            continue
        t_all = time.time()
        geo = grid_geometry(cid, m2m)
        seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
        t0 = time.time()
        # v22: nearest-label resampling + network (unchanged); v23: homogenized network (spicehead.mri.head_model)
        model, aff_dx = mri.head_model(seg, dx)
        labels = model.labels
        t_build = time.time() - t0
        ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
        labs, pairs, montages = candidate_set(ej)
        scalp_vox = np.argwhere(labels == SCALP_LABEL)
        elecs = {}
        for lab in labs:
            v = snap(scalp_vox, mri.world_to_voxel(aff_dx, ej[lab]))
            elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                                 radius_nodes=ELEC_RADIUS_MM / dx), lab)
        t0 = time.time()
        solver, ncomp = make_solver(model.G)
        t_setup = time.time() - t0
        print(f"  [{hi}/{len(subs)}] {cid} {tag}: {model.n_nodes:,} nodes ({ncomp} components), "
              f"{len(geo['gm_idx']):,} GM voxels on the T1 grid, {len(pairs)} pairs, "
              f"{len(montages)} montages", flush=True)

        def field_gm(inj):
            V = solver(inj)
            return np.stack([sample_on_gm(c, aff_dx, geo["gm_world"])
                             for c in efield_components(model, V)])          # float64

        others = [x for x in labs if x != REF]
        basis = np.zeros((len(others), 3, len(geo["gm_idx"])), np.float32)
        # the superposition check is done in float64: float32 storage alone would show ~1e-7
        keep64 = {}
        t0 = time.time()
        for i, lab in enumerate(others):
            fld = field_gm(pair_injection(model, elecs[lab], elecs[REF], CURRENT))
            basis[i] = fld
            if any(lab in pr for pr in CHECK_PAIRS):
                keep64[lab] = fld.astype(np.float64)
            if (i + 1) % 6 == 0 or i + 1 == len(others):
                print(f"      basis {i+1}/{len(others)} ({(time.time()-t0)/(i+1):.0f} s/solve)",
                      flush=True)
        t_basis = time.time() - t0

        def pair_field64(a, b):
            fa = keep64[a] if a != REF else 0.0
            fb = keep64[b] if b != REF else 0.0
            return fa - fb

        t0 = time.time()
        check = {}
        for a, b in CHECK_PAIRS:
            direct = field_gm(pair_injection(model, elecs[a], elecs[b], CURRENT)).astype(np.float64)
            check[f"{a}-{b}"] = float(np.abs(pair_field64(a, b) - direct).max() / np.abs(direct).max())
        t_check = time.time() - t0
        print(f"      superposition check: " + ", ".join(f"{k} {v:.1e}" for k, v in check.items()),
              flush=True)
        np.savez(p + ".part.npz", basis=basis, labs=np.array(others),
                 meta=json.dumps(dict(subject=cid, model=tag, spicehead_model=config.MODEL,
                                      network=getattr(model, "network", "nearest"), dx_mm=dx, n_nodes=int(model.n_nodes),
                                      n_components=int(ncomp), n_gm=int(len(geo["gm_idx"])),
                                      t_build_s=round(t_build, 1), t_setup_s=round(t_setup, 1),
                                      t_basis_s=round(t_basis, 1),
                                      t_per_solve_s=round(t_basis / len(others), 1),
                                      t_check_s=round(t_check, 1), basis_check=check,
                                      threads=os.environ.get("OPENBLAS_NUM_THREADS", "default"))))
        os.replace(p + ".part.npz", p)
        print(f"  [{hi}/{len(subs)}] {cid} {tag} basis OK ({(time.time()-t_all)/60:.1f} min)", flush=True)


# ---------------------------------------------------------------- stage: score
def load_fem_basis(cid, geo):
    """The 18 FEM leadfield fields on the GM voxels of the T1 grid (from --fem-leadfield)."""
    p = os.path.join(CACHE, f"{cid}__fem_basis.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}: run --fem-leadfield under the SimNIBS interpreter first")
    z = np.load(p, allow_pickle=True)
    return z["basis"], list(z["labs"]), json.loads(str(z["meta"]))


def fem_superposition_check(cid, basis, labs, geo):
    """Leadfield reconstruction of the existing directly simulated pair fields (GM voxels).

    Those direct runs meshed only their two pads, whereas the leadfield meshes all 19, so this
    difference is the passive-pad effect, not a superposition error (which the leadfield rules
    out by construction)."""
    from v21_simnibs_compare import load_simnibs_E
    idx = {lab: i for i, lab in enumerate(labs)}
    gi = geo["gm_idx"]
    out = {}
    for tag, (p1, p2, dirs) in DIRECT.items():
        for (a, b), d in zip((p1, p2), dirs):
            fem = os.path.join(FEM_REF, cid, d)
            if not os.path.isdir(os.path.join(fem, "subject_volumes")):
                continue
            (Ex, Ey, Ez), _, _ = load_simnibs_E(fem)
            direct = np.stack([c[gi[:, 0], gi[:, 1], gi[:, 2]] for c in (Ex, Ey, Ez)])
            rec = (basis[idx[a]] if a != REF else 0.0) - (basis[idx[b]] if b != REF else 0.0)
            # the direct run's sign convention is (+a, -b); report the smaller of both signs
            rel = min(float(np.abs(rec - s * direct).max() / np.abs(direct).max()) for s in (1, -1))
            out[f"{tag}:{a}-{b}"] = rel
    return out


def score_head(cid, m2m, model):
    from spicehead.envelope_fast import envelope_max_direction_flat
    geo = grid_geometry(cid, m2m)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs_all, pairs, montages = candidate_set(ej)
    if model == "fem":
        basis, labs, meta = load_fem_basis(cid, geo)
        if len(labs) != 18:
            raise RuntimeError(f"{cid}: {len(labs)} of 18 FEM leadfield fields present")
        meta["fem_vs_direct_runs"] = fem_superposition_check(cid, basis, labs, geo)
        print(f"      FEM leadfield vs two-pad direct runs (passive-pad effect): " +
              ", ".join(f"{k} {v:.1e}" for k, v in meta["fem_vs_direct_runs"].items()), flush=True)
    else:
        z = np.load(sp_basis_path(cid, model), allow_pickle=True)
        basis, labs, meta = z["basis"], list(z["labs"]), json.loads(str(z["meta"]))
    idx = {lab: i for i, lab in enumerate(labs)}
    roi_sel, out_sel = geo["roi_sel"], geo["out_sel"]

    def pf(a, b):
        return (basis[idx[a]] if a != REF else 0.0) - (basis[idx[b]] if b != REF else 0.0)

    t1 = time.time()
    if os.environ.get("SPICEHEAD_SCORE", "").lower() == "gpu":
        # GPU scoring (17 Oct 2026, NCHC batches): same envelope, median and percentile definitions
        from spicehead.score_gpu import score_montages_gpu
        res = score_montages_gpu(basis, labs, REF, pairs, montages, roi_sel, out_sel, ROI_ORDER, label_of)
        meta.update(n_pairs=len(pairs), n_montages=len(montages), t_score_s=round(time.time() - t1, 1),
                    grid_shape=list(geo["shape"]), scoring="gpu", rois=res)
        return meta
    flat = {p: pf(*p) for p in pairs}
    res = {r: {} for r in ROI_ORDER}
    for i, (pa, pb) in enumerate(montages, 1):
        A, B = flat[pa], flat[pb]
        env = envelope_max_direction_flat(A[0], A[1], A[2], B[0], B[1], B[2])
        lab = label_of(pa, pb)
        for r in ROI_ORDER:
            rv = env[roi_sel[r]]
            if rv.size == 0:
                continue
            T = float(np.median(rv))
            O = float(np.percentile(env[out_sel[r]], 99))
            res[r][lab] = [g7(T / O if O > 0 else 0.0), g7(T), g7(O)]
        if i % 2000 == 0 or i == len(montages):
            print(f"      scored {i}/{len(montages)} ({time.time()-t1:.0f} s)", flush=True)
    meta.update(n_pairs=len(pairs), n_montages=len(montages), t_score_s=round(time.time() - t1, 1),
                grid_shape=list(geo["shape"]), rois=res)
    return meta


def stage_score(args, model):
    subs = select(args, heads())
    od = out_dir()
    for hi, (cid, m2m) in enumerate(subs, 1):
        p = os.path.join(od, f"{cid}__{model}.json")
        if os.path.exists(p):
            print(f"  [{hi}/{len(subs)}] {cid} {model} (cached)", flush=True)
            continue
        t = time.time()
        try:
            res = score_head(cid, m2m, model)
        except Exception as ex:
            print(f"  [{hi}/{len(subs)}] {cid} {model} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        with open(p + ".part", "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        os.replace(p + ".part", p)
        print(f"  [{hi}/{len(subs)}] {cid} {model} OK ({(time.time()-t)/60:.1f} min)", flush=True)


# ---------------------------------------------------------------- stage: collect
def stage_collect():
    from scipy.stats import spearmanr
    od = out_dir()
    subs = heads()
    data = {}
    for cid, _ in subs:
        for m in MODELS:
            p = os.path.join(od, f"{cid}__{m}.json")
            if os.path.exists(p):
                data[(cid, m)] = json.load(open(p, encoding="utf-8"))
    have = {m: sorted(c for c, mm in data if mm == m) for m in MODELS}
    print("heads scored:", {m: len(v) for m, v in have.items()})
    complete = [c for c, _ in subs if all((c, m) in data for m in MODELS)]
    pairs_of_models = [("fem", "sp2"), ("fem", "sp1"), ("sp2", "sp1")]

    _F = {}

    def F(cid, m, roi):
        """{montage: F}, built once per head, model and target (it is called inside loops)."""
        key = (cid, m, roi)
        if key not in _F:
            _F[key] = {k: v[0] for k, v in data[(cid, m)]["rois"][roi].items()}
        return _F[key]

    rows = []
    for cid in complete:
        for roi in ROI_ORDER:
            f = {m: F(cid, m, roi) for m in MODELS}
            keys = sorted(set.intersection(*(set(x) for x in f.values())))
            win = {m: max(keys, key=lambda k: f[m][k]) for m in MODELS}
            row = dict(subject=cid, roi=roi, n_montages=len(keys))
            for m in MODELS:
                row[f"winner_{m}"] = win[m]
                row[f"F_{m}"] = f[m][win[m]]
                row[f"T_{m}"] = data[(cid, m)]["rois"][roi][win[m]][1]
                row[f"O_{m}"] = data[(cid, m)]["rois"][roi][win[m]][2]
            for a, b in pairs_of_models:
                va = np.array([f[a][k] for k in keys])
                vb = np.array([f[b][k] for k in keys])
                row[f"agree_{a}_{b}"] = win[a] == win[b]
                row[f"rho_{a}_{b}"] = g7(float(spearmanr(va, vb).correlation))
                top = np.argsort(-va)[:100]
                row[f"rho_top100_{a}_{b}"] = g7(float(spearmanr(va[top], vb[top]).correlation))
                # regret of a's choice evaluated under b, and of b's choice under a (% of the optimum)
                row[f"regret_{a}winner_under_{b}_pct"] = round(100 * (1 - f[b][win[a]] / f[b][win[b]]), 3)
                row[f"regret_{b}winner_under_{a}_pct"] = round(100 * (1 - f[a][win[b]] / f[a][win[a]]), 3)
                ta = set(np.array(keys)[np.argsort(-va)[:10]])
                tb = set(np.array(keys)[np.argsort(-vb)[:10]])
                row[f"top10_overlap_{a}_{b}"] = len(ta & tb)
                ta = set(np.array(keys)[np.argsort(-va)[:25]])
                tb = set(np.array(keys)[np.argsort(-vb)[:25]])
                row[f"top25_overlap_{a}_{b}"] = len(ta & tb)
            rows.append(row)
    if rows:
        with open(os.path.join(od, "fullsearch_decisions.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # cohort level over the complete heads: median F per montage per model
    crow = []
    for roi in ROI_ORDER:
        if not complete:
            break
        med = {}
        for m in MODELS:
            keys = set.intersection(*(set(F(c, m, roi)) for c in complete))
            med[m] = {k: float(np.median([F(c, m, roi)[k] for c in complete])) for k in keys}
        keys = sorted(set.intersection(*(set(v) for v in med.values())))
        win = {m: max(keys, key=lambda k: med[m][k]) for m in MODELS}
        r = dict(roi=roi, n_heads=len(complete))
        for m in MODELS:
            r[f"winner_{m}"] = win[m]
            r[f"F_{m}"] = round(med[m][win[m]], 4)
            r[f"T_{m}"] = round(float(np.median([data[(c, m)]["rois"][roi][win[m]][1] for c in complete])), 5)
            r[f"O_{m}"] = round(float(np.median([data[(c, m)]["rois"][roi][win[m]][2] for c in complete])), 5)
        for a, b in pairs_of_models:
            r[f"agree_{a}_{b}"] = win[a] == win[b]
            r[f"regret_{a}winner_under_{b}_pct"] = round(100 * (1 - med[b][win[a]] / med[b][win[b]]), 3)
            r[f"regret_{b}winner_under_{a}_pct"] = round(100 * (1 - med[a][win[b]] / med[a][win[a]]), 3)
            va = np.array([med[a][k] for k in keys])
            vb = np.array([med[b][k] for k in keys])
            r[f"rho_{a}_{b}"] = g7(float(spearmanr(va, vb).correlation))
        crow.append(r)
    if crow:
        with open(os.path.join(od, "fullsearch_cohort.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(crow[0]))
            w.writeheader()
            w.writerows(crow)

    # timings and checks per head and model
    trow = []
    for (cid, m), d in sorted(data.items()):
        trow.append(dict(subject=cid, model=m, dx_mm=d.get("dx_mm"), n_nodes=d.get("n_nodes"),
                         n_gm=d.get("n_gm"), t_build_s=d.get("t_build_s"), t_setup_s=d.get("t_setup_s"),
                         t_basis_s=d.get("t_basis_s"), t_per_solve_s=d.get("t_per_solve_s"),
                         t_score_s=d.get("t_score_s"), threads=d.get("threads"),
                         basis_check_max=max(d["basis_check"].values()) if d.get("basis_check") else None,
                         fem_vs_direct_max=(max(d["fem_vs_direct_runs"].values())
                                            if d.get("fem_vs_direct_runs") else None),
                         t_map_s=d.get("t_map_s")))
    if trow:
        with open(os.path.join(od, "fullsearch_timing.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(trow[0]))
            w.writeheader()
            w.writerows(trow)

    summ = dict(heads_complete=complete, n_rows=len(rows))
    for a, b in pairs_of_models:
        summ[f"agree_{a}_{b}"] = f"{sum(r[f'agree_{a}_{b}'] for r in rows)}/{len(rows)}"
        for roi in ROI_ORDER:
            sub = [r for r in rows if r["roi"] == roi]
            summ[f"agree_{a}_{b}.{roi}"] = f"{sum(r[f'agree_{a}_{b}'] for r in sub)}/{len(sub)}"
    try:
        from v22_benchmark_heads import machine
        summ["machine"] = machine()
    except Exception as ex:                                         # psutil missing, etc.
        summ["machine"] = f"unavailable: {ex}"
    json.dump(summ, open(os.path.join(od, "fullsearch_summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "machine"}, indent=1))
    print("wrote", od)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "--fem-leadfield":
        stage_fem_leadfield(a)
    elif a[0] == "--fem-basis":
        raise SystemExit("--fem-basis (18 two-pad runs) is not a valid basis; use --fem-leadfield. "
                         "See stage_fem_leadfield.__doc__.")
    elif a[0] == "--sp-basis":
        stage_sp_basis(a, float(a[1]))
    elif a[0] == "--score":
        if a[1] not in MODELS:
            raise SystemExit(f"model must be one of {list(MODELS)}")
        stage_score(a, a[1])
    elif a[0] == "--collect":
        stage_collect()
    else:
        raise SystemExit(__doc__)
