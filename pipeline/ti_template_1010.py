# -*- coding: utf-8 -*-
"""Exploratory: two-pair TI montages over the 76 positions of the 10-10 system, template head.

Same model, targets and objective as pipeline/ti_template_unilateral.py and the Figure S1 search
(MNI152 template, SPICEhead v22, 2 mm grid, 12 mm skin-surface pads, 1 mA per channel, F = median
target GM envelope / p99 of GM more than 20 mm from the target), but the electrodes may sit at
any of the 76 positions of the UI Jurak 2007 10-10 cap instead of the 19 10-20 positions.

Constraints: the two electrodes of a pair are at least 40 mm apart (as in the 10-20 search); the
two channels' pads do not overlap (centres at least 25 mm apart; pad radius 12 mm). That leaves
about three million montages, so the search runs in three stages:

  --basis    75 single-electrode solves relative to Cz, in parallel processes; brain-voxel fields
             cached in C:/spicehead_v22_logs/ti1010 (not in Dropbox)
  --screen   every montage scored on all target voxels plus a fixed random 30% of the other gray
             matter (seed 0), in parallel processes; all screening F values saved
  --exact    the top 400 screened montages per target rescored on all gray matter; the best is
             written out with its envelope, plus the best montage that avoids the inferior row
             (F9/F10, FT9/FT10, T9/T10, TP9/TP10, P9/P10, PO9/PO10, Iz, I1, I2), and the 10-20
             optimum rescored as a check (it must reproduce the 10-20 search exactly)

Outputs in `Paper_A_SPICEhead/Brain Stimulation/figures/TI_targets_MNI_1010/`. Exploratory only.

  SPICEHEAD_MODEL=v22 python pipeline/ti_template_1010.py --basis --screen --exact
"""
import json
import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import nibabel as nib                                                # noqa: E402
import numpy as np                                                   # noqa: E402
from nibabel.processing import resample_from_to                      # noqa: E402
from scipy.ndimage import distance_transform_edt                     # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)
import fig_ti_targets_mni as M                                       # noqa: E402
from spicehead import mri, config                                            # noqa: E402
from spicehead.electrodes import Electrode, contact_nodes, pair_injection   # noqa: E402
from spicehead.field import efield_components                       # noqa: E402
from spicehead.envelope_fast import envelope_max_direction_flat      # noqa: E402
from v21_head_convergence import make_solver                         # noqa: E402

CACHE = (os.environ.get("SPICEHEAD_TI1010_CACHE")                  # cluster runs (28 Oct 2026)
         or r"C:\spicehead_v22_logs\ti1010" + ("" if config.MODEL == "v22" else f"_{config.MODEL}"))   # model-tagged cache since v23
OUT = os.path.join(os.path.dirname(M.OUT), "TI_targets_MNI_1010" + ("" if config.MODEL == "v22" else f"_{config.MODEL}"))
TARGETS = {
    "insula": dict(title="Insula", parcels=["Insula_L", "Insula_R"]),
    "hippocampus": dict(title="Hippocampus", parcels=["Hippocampus_L", "Hippocampus_R"]),
    "insula_L": dict(title="Left insula", parcels=["Insula_L"]),
    "insula_R": dict(title="Right insula", parcels=["Insula_R"]),
    "hippocampus_L": dict(title="Left hippocampus", parcels=["Hippocampus_L"]),
    "hippocampus_R": dict(title="Right hippocampus", parcels=["Hippocampus_R"]),
}
INFERIOR = {"F9", "F10", "FT9", "FT10", "T9", "T10", "TP9", "TP10", "P9", "P10", "PO9", "PO10",
            "Iz", "I1", "I2"}
# --subregions: approximate geometric split of the AAL left insula (no subdivided atlas is
# installed): anterior = MNI y >= +5 mm, posterior = MNI y <= -10 mm; the band between (roughly the
# dysgranular middle insula) belongs to neither, so the two targets do not touch
SUBREGIONS = {
    "insula_ant_L": dict(title="Left anterior insula", parcels=["Insula_L"], ymin=5.0),
    "insula_post_L": dict(title="Left posterior insula", parcels=["Insula_L"], ymax=-10.0),
}
SUFFIX = ""                                      # cache/output names of the target set in use
CROSS_MIN_MM = 25.0
SCREEN_FRAC, TOP_K = 0.30, 400
N_BASIS_PROC, N_SCREEN_PROC = 4, 7


def region(aal, value, t, aff):
    """AAL parcels of target `t`, optionally limited to an MNI y range."""
    m = np.isin(aal, [value[p] for p in t["parcels"]])
    if "ymin" in t or "ymax" in t:
        ijk = np.indices(aal.shape).reshape(3, -1)
        y = (aff[1, :3] @ ijk + aff[1, 3]).reshape(aal.shape)
        m &= (y >= t.get("ymin", -np.inf)) & (y <= t.get("ymax", np.inf))
    return m


def read_cap_1010():
    d = {}
    for line in open(M.CAP):
        p = line.strip().split(",")
        if len(p) >= 5 and p[0].endswith("Electrode"):
            d[p[4]] = np.array([float(p[1]), float(p[2]), float(p[3])])
    return d


def setup():
    """Labels, masks, electrode voxels and the screening voxel set (no solves)."""
    seg = os.path.join(M.M2M, "label_prep", "tissue_labeling_upsampled.nii.gz")
    labels, aff, zooms, model = mri.head_grid(seg, M.DX)      # v23: the homogenized network is built here
    gm, brain = labels == M.GM, np.isin(labels, (M.WM, M.GM))
    tpl = nib.Nifti1Image(labels.astype(np.int16), aff)
    aal = np.rint(np.asarray(resample_from_to(nib.load(M.AAL), tpl, order=0).dataobj)).astype(int)
    value = {r["name"]: r["value"] for r in json.load(open(M.AAL_REGIONS))["regions"]}
    tmask = {k: region(aal, value, t, aff) & gm for k, t in TARGETS.items()}
    omask = {k: gm & (distance_transform_edt(~m, sampling=M.DX) > M.EXCLUDE_MM) for k, m in tmask.items()}
    cap = read_cap_1010()
    scalp = np.argwhere(labels == M.SCALP)
    inv = np.linalg.inv(aff)
    labs = list(cap)
    evox = {}
    for l in labs:
        ijk = (inv @ np.append(cap[l], 1))[:3]
        evox[l] = scalp[int(np.argmin(np.linalg.norm(scalp - ijk, axis=1)))]
    return dict(labels=labels, aff=aff, zooms=zooms, model=model, gm=gm, brain=brain, tmask=tmask, omask=omask,
                cap=cap, labs=labs, evox=evox)


# ------------------------------------------------------------------ stage 1: basis
def _basis_worker(chunk):
    S = setup()
    model = S["model"] if S["model"] is not None else mri.model_from_labels(S["labels"], S["zooms"], dx_mm=M.DX)
    solver, _ = make_solver(model.G)
    el = lambda l: Electrode(contact_nodes(model, tuple(int(x) for x in S["evox"][l]),
                                           radius_nodes=M.ELEC_RADIUS_MM / M.DX), l)
    ref = el(M.REF)
    bidx = np.flatnonzero(S["brain"].ravel())
    for l in chunk:
        f = os.path.join(CACHE, f"basis_{l}.npy")
        if os.path.exists(f):
            continue
        V = solver(pair_injection(model, el(l), ref, 1e-3))
        comps = [np.nan_to_num(c).ravel()[bidx].astype(np.float32) for c in efield_components(model, V)]
        np.save(f, np.stack(comps))
        print(f"  basis {l} done", flush=True)
    if chunk and chunk[0] == "__check__":
        pass
    return len(chunk)


def basis():
    os.makedirs(CACHE, exist_ok=True)
    S = setup()
    todo = [l for l in S["labs"] if l != M.REF and not os.path.exists(os.path.join(CACHE, f"basis_{l}.npy"))]
    print(f"basis: {len(todo)} solves to do in {N_BASIS_PROC} processes", flush=True)
    t0 = time.time()
    chunks = [todo[i::N_BASIS_PROC] for i in range(N_BASIS_PROC)]
    with Pool(N_BASIS_PROC) as pool:
        pool.map(_basis_worker, chunks)
    print(f"basis done in {(time.time() - t0) / 60:.1f} min", flush=True)


def load_basis(S, idx_in_brain):
    """Basis fields of all electrodes on a voxel subset (indices into the brain voxels); Cz = 0."""
    B = np.zeros((len(S["labs"]), 3, len(idx_in_brain)), np.float32)
    for i, l in enumerate(S["labs"]):
        if l != M.REF:
            B[i] = np.load(os.path.join(CACHE, f"basis_{l}.npy"), mmap_mode="r")[:, idx_in_brain]
    return B


def pairs_and_dist(S):
    labs = S["labs"]
    P = np.array([S["evox"][l] for l in labs], float) * M.DX
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    pairs = [(a, b) for a, b in combinations(range(len(labs)), 2) if D[a, b] >= M.MIN_DIST_MM]
    return pairs, D


# ------------------------------------------------------------------ stage 2: screening
_W = {}


def _screen_init(basis_file, sel_file):
    _W["B"] = np.load(basis_file)
    _W.update(np.load(sel_file, allow_pickle=True).item())


def _screen_task(i):
    B, pairs, D = _W["B"], _W["pairs"], _W["D"]
    a, b = pairs[i]
    A = B[a] - B[b]
    js, Fs = [], []
    for j in range(i + 1, len(pairs)):
        c, d = pairs[j]
        if len({a, b, c, d}) < 4 or min(D[a, c], D[a, d], D[b, c], D[b, d]) < CROSS_MIN_MM:
            continue
        C = B[c] - B[d]
        env = envelope_max_direction_flat(A[0], A[1], A[2], C[0], C[1], C[2])
        f = []
        for ts, os_ in zip(_W["tsel"], _W["osel"]):
            O = np.percentile(env[os_], 99)
            f.append(np.median(env[ts]) / O if O > 0 else 0.0)
        js.append(j)
        Fs.append(f)
    return i, np.array(js, np.int32), np.array(Fs, np.float32).reshape(-1, len(_W["tsel"]))


def screen():
    S = setup()
    pairs, D = pairs_and_dist(S)
    gm_b = S["gm"].ravel()[S["brain"].ravel()]                     # GM among brain voxels
    gidx = np.flatnonzero(gm_b)                                     # brain-voxel index of each GM voxel
    gm = S["gm"]
    keys = list(TARGETS)
    tsel_g = [S["tmask"][k][gm] for k in keys]                      # over GM voxels
    osel_g = [S["omask"][k][gm] for k in keys]
    rng = np.random.default_rng(0)
    keep = rng.random(len(gidx)) < SCREEN_FRAC
    for t in tsel_g:
        keep |= t
    os.makedirs(CACHE, exist_ok=True)
    np.save(os.path.join(CACHE, f"screen_basis{SUFFIX}.npy"), load_basis(S, gidx[keep]))
    np.save(os.path.join(CACHE, f"screen_sel{SUFFIX}.npy"),
            dict(pairs=pairs, D=D, tsel=[t[keep] for t in tsel_g], osel=[o[keep] for o in osel_g]),
            allow_pickle=True)
    n_mont = sum(1 for i in range(len(pairs)) for j in range(i + 1, len(pairs))
                 if len(set(pairs[i]) | set(pairs[j])) == 4
                 and min(D[pairs[i][0], pairs[j][0]], D[pairs[i][0], pairs[j][1]],
                         D[pairs[i][1], pairs[j][0]], D[pairs[i][1], pairs[j][1]]) >= CROSS_MIN_MM)
    print(f"screen: {len(S['labs'])} positions, {len(pairs)} pairs, {n_mont:,} montages; "
          f"{int(keep.sum()):,} of {len(gidx):,} GM voxels", flush=True)
    t0, done, I, J, F = time.time(), 0, [], [], []
    with Pool(N_SCREEN_PROC, initializer=_screen_init,
              initargs=(os.path.join(CACHE, f"screen_basis{SUFFIX}.npy"), os.path.join(CACHE, f"screen_sel{SUFFIX}.npy"))) as pool:
        for k, (i, js, fs) in enumerate(pool.imap_unordered(_screen_task, range(len(pairs)), chunksize=4)):
            I.append(np.full(len(js), i, np.int32)), J.append(js), F.append(fs)
            done += len(js)
            if (k + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {done:,}/{n_mont:,} montages ({el / 60:.1f} min, ~{el / done * (n_mont - done) / 60:.0f} min left)",
                      flush=True)
    np.savez(os.path.join(CACHE, f"screen_scores{SUFFIX}.npz"), I=np.concatenate(I), J=np.concatenate(J),
             F=np.concatenate(F), keys=np.array(keys))
    print(f"screen done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ stage 3: exact rescoring
def exact():
    S = setup()
    pairs, D = pairs_and_dist(S)
    labs = S["labs"]
    z = np.load(os.path.join(CACHE, f"screen_scores{SUFFIX}.npz"))
    I, J, Fsc, keys = z["I"], z["J"], z["F"], list(z["keys"])
    gm = S["gm"]
    bidx_gm = np.flatnonzero(S["gm"].ravel()[S["brain"].ravel()])
    B = load_basis(S, bidx_gm)
    name = lambda p: "+".join(labs[x] for x in p)
    inferior = np.array([labs[a] in INFERIOR or labs[b] in INFERIOR for a, b in pairs])

    def score(pa, pb, k):
        A, C = B[pa[0]] - B[pa[1]], B[pb[0]] - B[pb[1]]
        env = envelope_max_direction_flat(A[0], A[1], A[2], C[0], C[1], C[2])
        T = float(np.median(env[S["tmask"][keys[k]][gm]]))
        O = float(np.percentile(env[S["omask"][keys[k]][gm]], 99))
        return T / O, T, O

    prev = {}
    for folder in ("TI_targets_MNI", "TI_targets_MNI_unilateral"):
        f = os.path.join(os.path.dirname(M.OUT), folder, "TI_targets_MNI152_summary.json")
        for t in json.load(open(f, encoding="utf-8"))["targets"]:
            prev[t["target"]] = t
    t1img = nib.load(os.path.join(M.M2M, "T1.nii.gz"))
    lab05 = nib.load(os.path.join(M.M2M, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    lab1 = np.rint(np.asanyarray(resample_from_to(lab05, t1img, order=0).dataobj)).astype(int)
    brain1 = np.isin(lab1, (M.WM, M.GM))
    aal1 = np.rint(np.asanyarray(resample_from_to(nib.load(M.AAL), t1img, order=0).dataobj)).astype(int)
    value = {r["name"]: r["value"] for r in json.load(open(M.AAL_REGIONS))["regions"]}
    Bfull = load_basis(S, np.arange(int(S["brain"].sum())))
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for k, key in enumerate(keys):
        best = {}
        for tag, allow in (("all", np.ones(len(I), bool)), ("std", ~(inferior[I] | inferior[J]))):
            cand = np.flatnonzero(allow)
            order = cand[np.argsort(-Fsc[cand, k])[:TOP_K]]
            ex = [(score(pairs[I[m]], pairs[J[m]], k), r) for r, m in enumerate(order)]
            (F, T, O), rank = max(ex, key=lambda e: e[0][0])
            m = order[rank]
            best[tag] = dict(pa=pairs[I[m]], pb=pairs[J[m]], F=F, T=T, O=O, screen_rank=rank + 1,
                             screen_F=float(Fsc[m, k]))
        lab_idx = {l: i for i, l in enumerate(labs)}
        if key in prev:                                             # earlier 10-20 search: rescore it
            p = prev[key]
            p1 = tuple(lab_idx[x] for x in p["pair1"].split("+"))
            p2 = tuple(lab_idx[x] for x in p["pair2"].split("+"))
            F1020 = score(p1, p2, k)[0]
        else:                                                       # none (subregions): run it here
            i20 = [lab_idx[l] for l in M.LABELS_1020]
            pr20 = [(a, b_) for a, b_ in combinations(i20, 2) if D[a, b_] >= M.MIN_DIST_MM]
            F1020, p1, p2 = max(((score(pa, pb, k)[0], pa, pb) for pa, pb in combinations(pr20, 2)
                                 if len(set(pa) | set(pb)) == 4), key=lambda r_: r_[0])
            p = dict(montage=f"{name(p1)} / {name(p2)}", F=F1020)
        b = best["all"]
        A, C = Bfull[b["pa"][0]] - Bfull[b["pa"][1]], Bfull[b["pb"][0]] - Bfull[b["pb"][1]]
        envb = envelope_max_direction_flat(A[0], A[1], A[2], C[0], C[1], C[2])
        env2 = np.zeros(S["labels"].shape, np.float32)
        env2[S["brain"]] = envb
        env1 = M.to_1mm(env2, S["aff"], t1img, order=1).astype(np.float32)
        env1[~brain1] = 0.0
        tgt1 = region(aal1, value, TARGETS[key], t1img.affine) & (lab1 == M.GM)
        nib.save(nib.Nifti1Image(env1, t1img.affine), os.path.join(OUT, f"TI_envelope_{key}_MNI152_1mm.nii.gz"))
        nib.save(nib.Nifti1Image(tgt1.astype(np.uint8), t1img.affine), os.path.join(OUT, f"TI_target_{key}_MNI152_1mm.nii.gz"))
        s = best["std"]
        rows.append(dict(
            target=key, title=TARGETS[key]["title"], montage=f"{name(b['pa'])} / {name(b['pb'])}",
            pair1=name(b["pa"]), pair2=name(b["pb"]), F=round(b["F"], 3), T_V_per_m=round(b["T"], 4),
            O_V_per_m=round(b["O"], 4), screen_rank=b["screen_rank"],
            montage_1020=p["montage"], F_1020=round(p["F"], 3), F_1020_rescored=round(F1020, 3),
            gain_pct=round(100 * (b["F"] / p["F"] - 1), 1),
            montage_no_inferior=f"{name(s['pa'])} / {name(s['pb'])}", F_no_inferior=round(s["F"], 3),
            T_no_inferior=round(s["T"], 4), screen_rank_no_inferior=s["screen_rank"]))
        r_ = rows[-1]
        print(f"{r_['title']:<18} 10-20 {r_['montage_1020']:<16} F={r_['F_1020']:.3f} (rescored {r_['F_1020_rescored']:.3f}) | "
              f"10-10 {r_['montage']:<18} F={r_['F']:.3f} ({r_['gain_pct']:+.1f}%, T {r_['T_V_per_m']:.3f}, screen rank {r_['screen_rank']}) | "
              f"no inferior row {r_['montage_no_inferior']} F={r_['F_no_inferior']:.3f}", flush=True)
    json.dump(dict(model="SPICEhead template search over the 76 positions of the 10-10 cap; 1 mA per channel",
                   grid_mm=M.DX, n_positions=len(labs), n_pairs=len(pairs), n_montages=int(len(I)),
                   cross_min_mm=CROSS_MIN_MM, screen_fraction=SCREEN_FRAC, top_k=TOP_K, targets=rows),
              open(os.path.join(OUT, "TI_targets_MNI152_summary.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    if "--subregions" in sys.argv:                                 # anterior / posterior left insula
        TARGETS, SUFFIX = SUBREGIONS, "_sub"
        OUT = OUT + "_subregions"
    if "--basis" in sys.argv:
        basis()
    if "--screen" in sys.argv:
        screen()
    if "--exact" in sys.argv:
        exact()
