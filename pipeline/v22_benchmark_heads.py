# -*- coding: utf-8 -*-
"""Stage-by-stage cost of a full montage evaluation in one real head, per grid spacing.

Review of 1 October, item 5: "Benchmark construction, setup, basis solves, checks, scoring, and
peak RAM on the same hardware." The earlier benchmark (v21_benchmark_stages.py) timed a
homogeneous cylinder; this times the actual pipeline of the full 11,628-montage search in an
MRI-derived head, with every stage separated:

    load        read the 0.5 mm CHARM labels and resample to the grid
    build       voxel network (conductance matrix) incl. contact supports for 19 electrodes
    setup       solver preparation: sparse LU factorization (<= 1e5 nodes) or the Jacobi
                preconditioner and per-component grounding (CG, above)
    basis       18 single-electrode solves against the common reference (Cz), with fields
    check       2 direct pair solves compared with their superposition
    score       all 11,628 two-pair montages x 6 targets: envelope, T, O, F

Peak resident memory is the process peak working set (psutil), sampled after each stage, so a
stage's figure includes everything allocated before it. The machine, library versions and the
other busy processes at start are recorded, because absolute times depend on contention.

  python pipeline/v22_benchmark_heads.py [subject] [dx list]     (default SUBJ01, 5,2,1)

Output: <results>/v22_benchmark/heads.csv and heads.json (model-versioned, spicehead.config).
"""
import csv
import json
import os
import platform
import sys
import time
from itertools import combinations

import numpy as np
import psutil

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)

from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components
from spicehead.envelope_fast import envelope_max_direction_flat
from montage_search import (snap, roi_mask, ELEC_RADIUS_MM, ROI_RADIUS_MM, EXCLUDE_MM,
                            MIN_DIST_MM, SCALP, SCORE, ROI_ORDER)
from run_montage_cohort import resolve_m2m
from v21_head_convergence import make_solver, DIRECT_MAX_NODES

CH = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")
OUT = os.path.join(config.RESULTS, "v22_benchmark")
REF = "Cz"
CHECK_PAIRS = (("P8", "T8"), ("F7", "T7"))


def peak_gb():
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 30, 3)


def machine():
    import scipy
    busy = []
    for p in psutil.process_iter(["name", "cpu_percent"]):
        pass
    time.sleep(1.0)
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
        try:
            c = p.info["cpu_percent"] or 0.0
            # pid 0 is Windows' "System Idle Process", whose share is idle time, not work
            if c >= 50 and p.pid not in (0, os.getpid()):
                busy.append(f"{p.info['name']}:{c:.0f}%")
        except Exception:
            continue
    return dict(cpu=platform.processor(), physical_cores=psutil.cpu_count(logical=False),
                logical_cores=psutil.cpu_count(), ram_gb=round(psutil.virtual_memory().total / 2 ** 30, 1),
                os=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, model=config.describe(),
                threads=os.environ.get("OPENBLAS_NUM_THREADS", "default"),
                busy_processes_at_start=busy)


def bench(cid, dx):
    rec = dict(subject=cid, dx_mm=dx)
    m2m = resolve_m2m(CH, cid)
    t = time.perf_counter()
    # v21/v22: labels only here (the network is timed below); v23: the homogenized network is built here
    labels, affine, _, model = mri.head_grid(
        os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"), dx)
    rec["t_load_s"] = time.perf_counter() - t
    rec["peak_gb_load"] = peak_gb()
    rec["t_load_includes_network_build"] = model is not None     # v23: t_build_s then covers contacts only

    t = time.perf_counter()
    if model is None:
        model = mri.model_from_labels(labels, None, dx_mm=dx)
    scalp_vox = np.argwhere(labels == SCALP)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs = sorted(ej)
    epos, elecs = {}, {}
    for lab in labs:
        v = snap(scalp_vox, mri.world_to_voxel(affine, ej[lab]))
        epos[lab] = v.astype(float)
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                             radius_nodes=ELEC_RADIUS_MM / dx), lab)
    rec["t_build_s"] = time.perf_counter() - t
    rec["n_nodes"] = int(model.n_nodes)
    rec["n_edges"] = int((model.G.nnz - model.n_nodes) // 2)
    rec["contact_nodes_median"] = int(np.median([e.nodes.size for e in elecs.values()]))
    rec["peak_gb_build"] = peak_gb()

    gm = np.isin(labels, SCORE)
    gm_idx = np.argwhere(gm)
    gm_world = (affine[:3, :3] @ gm_idx.T).T + affine[:3, 3]
    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    present = [r for r in ROI_ORDER if r in rois]
    roi_sel = {r: roi_mask(labels.shape, gm_idx, gm_world, rois[r], ROI_RADIUS_MM)[gm]
               for r in present}
    out_sel = {r: ~roi_mask(labels.shape, gm_idx, gm_world, rois[r], EXCLUDE_MM)[gm]
               for r in present}
    rec["n_gm_voxels"] = int(gm.sum())

    t = time.perf_counter()
    solver, ncomp = make_solver(model.G)
    rec["t_setup_s"] = time.perf_counter() - t
    rec["solver"] = "direct LU" if model.n_nodes <= DIRECT_MAX_NODES else "Jacobi-CG, rtol 1e-10"
    rec["n_components"] = int(ncomp)
    rec["peak_gb_setup"] = peak_gb()

    def field(inj):
        return np.stack([np.nan_to_num(c[gm]) for c in efield_components(model, solver(inj))])

    t = time.perf_counter()
    basis = {lab: field(pair_injection(model, elecs[lab], elecs[REF], 1e-3))
             for lab in labs if lab != REF}
    rec["n_basis"] = len(basis)
    rec["t_basis_s"] = time.perf_counter() - t
    rec["t_per_solve_s"] = rec["t_basis_s"] / len(basis)
    rec["peak_gb_basis"] = peak_gb()

    def pair_field(a, b):
        return (basis[a] if a != REF else 0.0) - (basis[b] if b != REF else 0.0)

    pairs = [(a, b) for a, b in combinations(labs, 2)
             if np.linalg.norm(epos[a] - epos[b]) * dx >= MIN_DIST_MM]
    t = time.perf_counter()
    worst = 0.0
    for a, b in [p for p in CHECK_PAIRS if p in pairs]:
        direct = field(pair_injection(model, elecs[a], elecs[b], 1e-3))
        worst = max(worst, float(np.abs(pair_field(a, b) - direct).max() / np.abs(direct).max()))
    rec["t_check_s"] = time.perf_counter() - t
    rec["superposition_worst"] = worst

    flatE = {p: tuple(c.astype(np.float32) for c in pair_field(*p)) for p in pairs}
    del basis
    montages = [(pa, pb) for pa, pb in combinations(pairs, 2) if len(set(pa) | set(pb)) == 4]
    t = time.perf_counter()
    best = {r: (-1.0, None) for r in present}
    for pa, pb in montages:
        a, b = flatE[pa], flatE[pb]
        env = envelope_max_direction_flat(a[0], a[1], a[2], b[0], b[1], b[2])
        for r in present:
            rv = env[roi_sel[r]]
            if rv.size == 0:
                continue
            O = float(np.percentile(env[out_sel[r]], 99))
            F = float(np.median(rv)) / O if O > 0 else 0.0
            if F > best[r][0]:
                best[r] = (F, (pa, pb))
    rec["n_pairs"] = len(pairs)
    rec["n_montages"] = len(montages)
    rec["t_score_s"] = time.perf_counter() - t
    rec["t_score_per_montage_ms"] = 1e3 * rec["t_score_s"] / len(montages)
    rec["peak_gb"] = peak_gb()
    rec["t_total_s"] = sum(rec[k] for k in ("t_load_s", "t_build_s", "t_setup_s", "t_basis_s",
                                            "t_check_s", "t_score_s"))
    rec["best_F"] = {r: round(v[0], 4) for r, v in best.items()}
    return rec


def main(cid="SUBJ01", dxs=(5.0, 2.0, 1.0)):
    os.makedirs(OUT, exist_ok=True)
    info = machine()
    print(json.dumps(info, indent=1), flush=True)
    rows = []
    for dx in dxs:
        print(f"\n== {cid} at {dx:g} mm", flush=True)
        r = bench(cid, dx)
        rows.append(r)
        print("  " + "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in r.items() if k != "best_F"), flush=True)
        flat = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()
                 if k != "best_F"} for x in rows]
        with open(os.path.join(OUT, "heads.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)
        json.dump(dict(machine=info, rows=rows), open(os.path.join(OUT, "heads.json"), "w",
                                                       encoding="utf-8"), indent=1, default=str)
    print("\nwrote", os.path.join(OUT, "heads.csv"))


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "SUBJ01"
    dxs = tuple(float(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (5.0, 2.0, 1.0)
    main(cid, dxs)
