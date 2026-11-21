# -*- coding: utf-8 -*-
"""Homogenized coarse network (prototype, 23 Oct 2026): are the coarse grids' errors against FEM caused by
nearest-label resampling of the 0.5 mm CHARM labels?

Standard SPICEhead at dx builds the network from labels resampled to dx by nearest neighbour, so a thin
skull or CSF layer is either lost or thickened. Here each coarse edge conductance is computed from the
0.5 mm labels directly: a coarse voxel is a block of f x f x f fine voxels (f = dx / 0.5); the edge
between two face-adjacent coarse voxels is the parallel sum of f^2 conduction tubes, each the series
combination of the f fine voxels between the two block centres (sigma_i * dx_f per fine segment). A
coarse voxel is a node if any fine voxel in its block conducts. For uniform sigma this reduces to
sigma * dx exactly, so the standard network is the special case of uniform blocks.

Electrodes keep the v22 skin-surface contact on the coarse majority-label volume. The 18 bases are
solved (GPU), sampled on the shared 1 mm grid and compared with the stored FEM basis of the 300-head
run (field statistics) and scored for the six Paper A targets (decisions), exactly as pad_experiment.py.
Output: results_v22/v22_homog_experiment/<code>__sp{dx}hom.json

    set SPICEHEAD_MODEL=v22 SPICEHEAD_SOLVER=gpu
    python pipeline/nchc/homog_experiment.py --heads h001 [--dx 2,1]
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "pipeline"))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; desktop only
CHARM = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")
OUT = os.environ.get("SPICEHEAD_EXP_OUT", os.path.join(ROOT, "results_v22", "v22_homog_experiment"))
FEMJ = os.environ.get("SPICEHEAD_FULLSEARCH_OUT", "")
CACHE = os.environ.setdefault("SPICEHEAD_FULLSEARCH_CACHE", r"C:\spicehead_v22_logs\pad_experiment\cache")
os.environ.setdefault("SPICEHEAD_MODEL", "v22")
os.environ["SPICEHEAD_CHARM_ROOT"] = CHARM


def codemap():
    if os.path.exists(CODEMAP):
        return {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    return None


def score_json(code, model):
    for q in (os.path.join(L, "heads", code, f"{code}__{model}.json"), os.path.join(FEMJ, f"{code}__{model}.json")):
        if os.path.exists(q):
            return json.load(open(q))["rois"]
    return None
DIST_BINS = [20, 40, 60]
MIN_FILL = float(os.environ.get("SPICEHEAD_HOMOG_MIN_FILL", "0.5"))


def block_views(a, f):
    """Crop to multiples of f and return (n0, n1, n2, f, f, f) view of the fine array."""
    n = [(s // f) * f for s in a.shape]
    a = a[:n[0], :n[1], :n[2]]
    return a.reshape(n[0] // f, f, n[1] // f, f, n[2] // f, f).transpose(0, 2, 4, 1, 3, 5)


def homogenized_model(fine_labels, fine_aff, f, dx_c, conductance_mean="harmonic"):
    """Coarse SPICEHeadModel with edge conductances from the fine labels; returns (model, coarse_aff, coarse_labels)."""
    from spicehead.conductivity import label_volume_to_sigma
    from spicehead.model import SPICEHeadModel
    sig_f = label_volume_to_sigma(fine_labels).astype(np.float64)        # float64: a float32 Laplacian is not reliably PSD
    B = block_views(sig_f, f)                                  # (N0, N1, N2, f, f, f)
    Lb = block_views(fine_labels, f)
    N = B.shape[:3]
    # node rule: at least half of the block conducts (blocks with a single conducting fine voxel would be nodes
    # attached by ~1e-6 S tubes, which made CG diverge on 23 Oct); MIN_FILL is a design parameter
    mask = (B > 0).mean(axis=(3, 4, 5)) >= MIN_FILL
    mean_sigma = np.where(mask, B.sum(axis=(3, 4, 5)) / np.maximum((B > 0).sum(axis=(3, 4, 5)), 1), 0.0)
    # majority label among conducting fine voxels (for electrode placement only)
    lab_c = np.zeros(N, np.int16)
    flat = Lb.reshape(N + (f ** 3,))
    for v in np.unique(fine_labels):
        if v == 0:
            continue
        cnt = (flat == v).sum(-1)
        better = cnt > np.where(lab_c > 0, (flat == lab_c[..., None]).sum(-1), 0)
        lab_c[better] = v
    dx_f = (dx_c / f) * 1e-3                                # metres, as model.py (g = sigma * dx in siemens)
    h = f // 2
    # edge conductances: tubes of f fine segments from block centre to block centre
    with np.errstate(divide="ignore"):
        R = np.where(B > 0, 1.0 / (B * dx_f), np.inf)        # resistance of each fine segment
    g = []
    for axis in range(3):
        fa = 3 + axis                                         # fine axis inside the block
        # second half of block A (indices h..f-1) and first half of block B (0..h-1) along the fine axis
        slA = [slice(None)] * 6; slA[fa] = slice(h, f)
        slB = [slice(None)] * 6; slB[fa] = slice(0, h)
        RA = R[tuple(slA)].sum(axis=fa)                       # (N0, N1, N2, f, f): per tube, A's half
        RB = R[tuple(slB)].sum(axis=fa)
        sa = [slice(None)] * 3; sa[axis] = slice(0, -1)
        sb = [slice(None)] * 3; sb[axis] = slice(1, None)
        Rtube = RA[tuple(sa)] + RB[tuple(sb)]                 # (N0-1, N1, N2, f, f) along axis 0, etc.
        g.append(np.where(np.isfinite(Rtube), 1.0 / Rtube, 0.0).sum(axis=(-2, -1)))
    def assemble(mask):
        model = SPICEHeadModel(mean_sigma, dx=dx_c * 1e-3, mask=mask, conductance_mean=conductance_mean)
        nid = model.node_id
        rows, cols, vals = [], [], []
        for axis in range(3):
            a = tuple(slice(None, -1) if d == axis else slice(None) for d in range(3))
            b = tuple(slice(1, None) if d == axis else slice(None) for d in range(3))
            both = mask[a] & mask[b]
            ga = g[axis][both]
            ia, ib = nid[a][both], nid[b][both]
            rows += [ia, ib, ia, ib]; cols += [ib, ia, ia, ib]; vals += [-ga, -ga, ga, ga]
        G = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(model.n_nodes,) * 2).tocsr()
        G.sum_duplicates()
        model.G = G
        return model
    # nodes with no conducting edge, and islands not connected to the main conductor, cannot be solved
    # (zero diagonal; current with no path to the reference): keep the giant component only
    from scipy.sparse.csgraph import connected_components
    model = assemble(mask)
    deg = np.asarray(model.G.diagonal()).ravel()
    keep = np.zeros(model.n_nodes, bool); keep[deg > 0] = True
    ncomp, comp = connected_components(model.G[keep][:, keep], directed=False)
    giant = np.bincount(comp).argmax()
    keep_idx = np.flatnonzero(keep)[comp == giant]
    mask2 = np.zeros_like(mask); ii, jj, kk = model.grid_coords; mask2[ii[keep_idx], jj[keep_idx], kk[keep_idx]] = True
    dropped = int(mask.sum() - mask2.sum())
    model = assemble(mask2)
    model.dropped_nodes = dropped
    lab_c = np.where(mask2, lab_c, 0)
    model.labels = lab_c
    aff_c = fine_aff.copy()
    aff_c[:3, :3] = fine_aff[:3, :3] * f
    aff_c[:3, 3] = fine_aff[:3, 3] + fine_aff[:3, :3] @ (np.full(3, (f - 1) / 2.0))   # centre of the first block
    return model, aff_c, lab_c


def run_head(code, cmap, dx):
    import nibabel as nib
    import v22_fem_fullsearch as fs
    from spicehead import mri
    from spicehead.electrodes import Electrode, contact_nodes, pair_injection
    from spicehead.field import efield_components
    from spicehead.score_gpu import score_montages_gpu
    from montage_search import snap
    from v21_head_convergence import make_solver
    grad = os.environ.get("SPICEHEAD_GRAD", "central")          # "tissue": same-tissue-neighbour differences
    tag = f"sp{dx:g}hom" + ("T" if grad == "tissue" else "")
    outp = os.path.join(OUT, f"{code}__{tag}.json")
    if os.path.exists(outp):
        return "cached"
    t0 = time.time()
    cid = cmap[code] if cmap else code
    m2m = os.path.join(CHARM, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)
    geo = fs.grid_geometry(cid, m2m)
    img = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    fine = np.asanyarray(img.dataobj).astype(np.int16)
    zf = float(img.header.get_zooms()[0]); f = int(round(dx / zf))
    assert abs(f * zf - dx) < 1e-6, (zf, dx)
    model, aff_c, lab_c = homogenized_model(fine, img.affine, f, dx)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs, pairs, montages = fs.candidate_set(ej)
    scalp_vox = np.argwhere(lab_c == fs.SCALP_LABEL)
    elecs = {}
    for lab in labs:
        v = snap(scalp_vox, mri.world_to_voxel(aff_c, ej[lab]))
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v), radius_nodes=fs.ELEC_RADIUS_MM / dx), lab)
    t1 = time.time()
    solver, ncomp = make_solver(model.G)
    others = [x for x in labs if x != fs.REF]
    basis = np.zeros((len(others), 3, len(geo["gm_idx"])), np.float32)
    # the fine CHARM volume is not RAS-aligned (the standard pipeline's resampled grid is): rotate the
    # voxel-axis field components into world axes before sampling, so vectors compare with FEM
    Rdir = aff_c[:3, :3] / np.sqrt((aff_c[:3, :3] ** 2).sum(0))
    if grad == "tissue":
        from field_definition_test import tissue_gradient
    for i, lab in enumerate(others):
        V = solver(pair_injection(model, elecs[lab], elecs[fs.REF], fs.CURRENT))
        comps = efield_components(model, V) if grad != "tissue" else tissue_gradient(model.nodes_to_grid(V, fill=np.nan), lab_c, model.dx, model.mask)
        world = [sum(Rdir[w, v] * comps[v] for v in range(3) if abs(Rdir[w, v]) > 1e-12) for w in range(3)]
        basis[i] = np.stack([fs.sample_on_gm(c, aff_c, geo["gm_world"]) for c in world])
    t_basis = time.time() - t1
    res = score_montages_gpu(basis, others, fs.REF, pairs, montages, geo["roi_sel"], geo["out_sel"], fs.ROI_ORDER, fs.label_of, verbose=False)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{code}__{tag}_basis.part.npz"), basis=basis, labs=np.array(others), meta=json.dumps(dict(model=tag, dx_mm=dx)))
    os.replace(os.path.join(OUT, f"{code}__{tag}_basis.part.npz"), os.path.join(OUT, f"{code}__{tag}_basis.npz"))
    fstats = None
    bp = os.path.join(L, "bundles", f"{code}.npz")
    if os.path.exists(bp):
        with np.load(bp) as z:
            Bf, labs_f = z["basis_fem"], list(z["labs_fem"]); gw = z["gm_world"].astype(float)
        assert labs_f == others
        fstats = {}
        for i, lab in enumerate(others):
            ef, es = Bf[i].astype(np.float64), basis[i].astype(np.float64)
            nf, ns = np.linalg.norm(ef, axis=0), np.linalg.norm(es, axis=0)
            ok = (nf > 1e-9 * nf.max()) & np.isfinite(ns)
            rel = np.linalg.norm(es - ef, axis=0)[ok] / nf[ok]
            d = np.linalg.norm(gw - np.asarray(ej[lab], float), axis=1)[ok]; b = np.digitize(d, DIST_BINS)
            fstats[lab] = dict(rel_median=float(np.median(rel)), ratio_median=float(np.median(ns[ok] / nf[ok])),
                               r=float(np.corrcoef(nf[ok], ns[ok])[0, 1]), bias_pct=float(100 * (ns[ok].mean() / nf[ok].mean() - 1)),
                               rel_by_dist=[float(np.median(rel[b == k])) if (b == k).any() else None for k in range(4)])
    out = dict(code=code, model=tag, gradient=grad, dx_mm=dx, fine_mm=zf, block=f, n_nodes=int(model.n_nodes), n_components=int(ncomp), dropped_nodes=int(model.dropped_nodes),
               t_build_s=round(t1 - t0, 1), t_basis_s=round(t_basis, 1), t_total_s=round(time.time() - t0, 1), rois=res, field_vs_fem=fstats)
    json.dump(out, open(outp + ".part", "w"), indent=0); os.replace(outp + ".part", outp)
    return f"ok ({model.n_nodes:,} nodes, {model.dropped_nodes:,} isolated dropped, build {t1-t0:.0f} s, basis {t_basis:.0f} s)"


def report(code, dx):
    import v22_fem_fullsearch as fs
    suf = "T" if os.environ.get("SPICEHEAD_GRAD", "central") == "tissue" else ""
    hom = json.load(open(os.path.join(OUT, f"{code}__sp{dx:g}hom{suf}.json")))
    fem, std = score_json(code, "fem"), score_json(code, f"sp{dx:g}")
    f = hom["field_vs_fem"]
    if f:
      print(f"{code} {dx:g} mm homogenized: field vs FEM rel {np.median([v['rel_median'] for v in f.values()]):.3f}, ratio "
          f"{np.median([v['ratio_median'] for v in f.values()]):.3f}, bias {np.median([v['bias_pct'] for v in f.values()]):+.1f}%, r {np.median([v['r'] for v in f.values()]):.3f}")
      print("  rel by distance:", [round(float(np.median([v['rel_by_dist'][k] for v in f.values() if v['rel_by_dist'][k] is not None])), 3) for k in range(4)])
    if not (fem and std):
        return
    for roi in fs.ROI_ORDER:
        keys = sorted(fem[roi]); Ff = np.array([fem[roi][k][0] for k in keys]); wf = Ff.argmax()
        ws = np.array([std[roi][k][0] for k in keys]).argmax(); wh = np.array([hom["rois"][roi][k][0] for k in keys]).argmax()
        print(f"  {roi:14s} FEM optimum kept: standard {100*Ff[ws]/Ff[wf]:5.1f}%  homogenized {100*Ff[wh]/Ff[wf]:5.1f}%")


def main():
    a = sys.argv[1:]
    cmap = codemap()
    codes = a[a.index("--heads") + 1].split(",")
    dxs = [float(x) for x in a[a.index("--dx") + 1].split(",")] if "--dx" in a else [2.0]
    os.makedirs(CACHE, exist_ok=True)
    for code in codes:
        for dx in dxs:
            try:
                msg = run_head(code, cmap, dx)
            except Exception as e:
                import traceback; traceback.print_exc(); msg = f"FAILED {type(e).__name__}: {e}"
            print(f"{code} {dx:g} mm: {msg}", flush=True)
            if not msg.startswith("FAILED"):
                report(code, dx)


if __name__ == "__main__":
    main()
