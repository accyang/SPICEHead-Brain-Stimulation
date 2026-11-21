# -*- coding: utf-8 -*-
"""Score every two-pair 10-20 montage for many targets per head on the GPU (all three models).

Written 18 Oct 2026 for the tier 1 + tier 2 target sets the PI chose:
  - tier 1: the 90 cerebral AAL regions;
  - tier 2: psychiatric targets, spheres about warped MNI centres;
  - ref6: the six Paper A targets, rescored on every head as a regression check.

Input per head is a bundle from prep_target_bundles.py (three float32 bases, grid, electrodes, AAL90
labels per GM voxel), plus an optional centres JSON {name: {"centres": [[x,y,z],...],
"radius_mm": r, "tier": "ref6"|"tier2"}} in subject world coordinates.

Target definitions follow the existing code exactly:
  - AAL region = labelled GM voxels (>= 5); off-target = GM farther than 20 mm from any region voxel
    (Euclidean distance transform, as montage_aal.py; all voxels, no subsampling);
  - sphere = GM within the radius of any centre; off-target = GM beyond 20 mm of every centre
    (v22_fem_fullsearch.grid_geometry). World coordinates are recomputed in float64 from the affine,
    as there.
The score per target is T = median envelope over the target, O = 99th percentile over the
off-target set, F = T/O. The functions are those of spicehead/score_gpu.py, validated against the
CPU scorer within 1e-6.

v2 (18 Oct): the models scored are those whose basis the bundle carries (basis_fem, basis_sp2, basis_sp1,
basis_sp5), so the same code scores the 5 mm bundles. Deployed as score_targets_v2.py beside v1, which the running
job uses.

Output: <out>/<code>__targets.npz with F, T, O (float32, shape models x targets x montages), the
target names, tiers and voxel counts, the model order, montage labels, and a ref6 check against the
collected score files when they are reachable.

    python score_targets.py <bundle.npz> <out_dir> [--centres <code>__centres.json] [--ref-json-dir DIR]
                            [--extra-labels <code>__t2labels.npz] [--aal 1,2 (test)]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline")):
    sys.path.insert(0, p)
CODE_ROOT = os.environ.get("SPICEHEAD_CODE_ROOT")              # cluster: $P/code
if CODE_ROOT:
    sys.path[:0] = [CODE_ROOT, os.path.join(CODE_ROOT, "pipeline")]

from spicehead.score_gpu import _envelope, _median_sorted, _percentile99        # noqa: E402
from v22_fem_fullsearch import candidate_set, label_of, REF, EXCLUDE_MM         # noqa: E402

ALL_MODELS = ("fem", "sp2", "sp1", "sp5")
MIN_REGION_VOX = 5


def build_targets(z, centres=None, aal_names=None, aal_subset=None, extra=None):
    from scipy.ndimage import distance_transform_edt
    gm_idx = z["gm_idx"].astype(np.int64)
    aff = np.asarray(z["affine"], float)
    shape = tuple(int(x) for x in z["shape"])
    gm_world = (aff[:3, :3] @ gm_idx.T).T + aff[:3, 3]
    vox = np.sqrt((aff[:3, :3] ** 2).sum(0))
    pad = int(np.ceil(EXCLUDE_MM / vox.min())) + 2
    T = []                                   # (name, tier, roi_bool, out_bool)
    if centres:
        for name, d in centres.items():
            cs = [np.asarray(c, float) for c in d["centres"]]
            roi = np.zeros(len(gm_idx), bool)
            near = np.zeros(len(gm_idx), bool)
            for c in cs:
                dist = np.linalg.norm(gm_world - c, axis=1)
                roi |= dist <= float(d["radius_mm"])
                near |= dist <= EXCLUDE_MM
            T.append((name, d.get("tier", "tier2"), roi, ~near))
    def region(name, tier, sel):
        if sel.sum() < MIN_REGION_VOX:
            T.append((name, tier, np.zeros(len(gm_idx), bool), np.zeros(len(gm_idx), bool)))
            return
        ri = gm_idx[sel]
        lo = np.maximum(ri.min(0) - pad, 0)
        hi = np.minimum(ri.max(0) + pad + 1, shape)
        box = np.ones(tuple(hi - lo), bool)
        box[tuple((ri - lo).T)] = False
        dist = distance_transform_edt(box, sampling=vox)
        inb = np.all((gm_idx >= lo) & (gm_idx < hi), axis=1)
        near = np.zeros(len(gm_idx), bool)
        g = gm_idx[inb] - lo
        near[inb] = dist[g[:, 0], g[:, 1], g[:, 2]] <= EXCLUDE_MM
        T.append((name, tier, sel, ~near))

    if extra is not None:                    # {"labels": int per GM voxel, "spec": {name: [atlas values]}}
        for name, vals in extra["spec"].items():
            region(name, "tier2", np.isin(extra["labels"], vals))
    lab = z["aal90"]
    for i in (aal_subset or range(1, 91)):
        region(aal_names[i - 1] if aal_names else f"AAL{i:02d}", "tier1", lab == i)
    return T


def score_model(basis, labs, pairs, montages, targets, device, batch=48):
    import torch
    idx = {lab: i for i, lab in enumerate(labs)}
    Bz = torch.from_numpy(np.ascontiguousarray(basis, dtype=np.float32)).to(device)
    zero = torch.zeros_like(Bz[0])
    P = torch.stack([(Bz[idx[a]] if a != REF else zero) - (Bz[idx[b]] if b != REF else zero) for a, b in pairs])
    del Bz
    pair_index = {p: j for j, p in enumerate(pairs)}
    ia = torch.tensor([pair_index[pa] for pa, _ in montages], device=device)
    ib = torch.tensor([pair_index[pb] for _, pb in montages], device=device)
    roi = [torch.from_numpy(np.flatnonzero(r)).to(device) for _, _, r, _ in targets]
    out = [torch.from_numpy(np.flatnonzero(o)).to(device) for _, _, _, o in targets]
    nt, nm = len(targets), len(montages)
    Tt = torch.zeros((nt, nm), dtype=torch.float64, device=device)
    Ot = torch.zeros((nt, nm), dtype=torch.float64, device=device)
    for s in range(0, nm, batch):
        e = min(s + batch, nm)
        env = _envelope(P[ia[s:e]], P[ib[s:e]])
        for t in range(nt):
            if roi[t].numel() == 0 or out[t].numel() == 0:
                continue
            Tt[t, s:e] = _median_sorted(env[:, roi[t]]).double()
            Ot[t, s:e] = _percentile99(env[:, out[t]]).double()
    T, O = Tt.cpu().numpy(), Ot.cpu().numpy()
    F = np.where(O > 0, T / np.where(O > 0, O, 1.0), 0.0)
    return F.astype(np.float32), T.astype(np.float32), O.astype(np.float32)


def ref6_check(code, names, F, T, O, labels, ref_dir, models):
    """Rescored ref6 targets against the collected per-head score files (7 significant digits)."""
    rep = {}
    for mi, m in enumerate(models):
        p = os.path.join(ref_dir, f"{code}__{m}.json")
        if not os.path.exists(p):
            continue
        rois = json.load(open(p))["rois"]
        for ti, n in enumerate(names):
            if not n.startswith("ref6:"):
                continue
            r = n.split(":", 1)[1]
            ref = np.array([rois[r][lab] for lab in labels], float)          # (nm, 3) = F, T, O
            new = np.stack([F[mi, ti], T[mi, ti], O[mi, ti]], 1).astype(float)
            rel = float(np.max(np.abs(new - ref) / np.maximum(np.abs(ref), 1e-12)))
            rep[f"{m}:{r}"] = dict(max_rel=rel, winner_same=bool(np.argmax(ref[:, 0]) == np.argmax(new[:, 0])))
    return rep


def main():
    a = sys.argv[1:]
    bundle, outd = a[0], a[1]
    centres = json.load(open(a[a.index("--centres") + 1])) if "--centres" in a else None
    ref_dir = a[a.index("--ref-json-dir") + 1] if "--ref-json-dir" in a else None
    names_file = os.path.join(os.path.dirname(bundle), "aal90_targets.json")
    aal_names = [d["name"] for d in json.load(open(names_file))] if os.path.exists(names_file) else None
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code = os.path.basename(bundle).split(".")[0]
    t0 = time.time()
    z = np.load(bundle)
    ej = json.loads(str(z["electrodes"]))
    _, pairs, montages = candidate_set(ej)
    labels = [label_of(pa, pb) for pa, pb in montages]
    sub = [int(x) for x in a[a.index("--aal") + 1].split(",")] if "--aal" in a else None   # testing only
    extra = None
    if "--extra-labels" in a:
        e = np.load(a[a.index("--extra-labels") + 1])
        extra = dict(labels=e["labels"], spec=json.loads(str(e["spec"])))
        if len(extra["labels"]) != len(z["gm_idx"]):
            raise SystemExit("extra labels do not match the grid")
    targets = build_targets(z, centres, aal_names, sub, extra)
    t1 = time.time()
    models = tuple(m for m in ALL_MODELS if f"basis_{m}" in z.files)
    res = {m: score_model(z[f"basis_{m}"], list(z[f"labs_{m}"]), pairs, montages, targets, device) for m in models}
    t2 = time.time()
    F = np.stack([res[m][0] for m in models]); T = np.stack([res[m][1] for m in models]); O = np.stack([res[m][2] for m in models])
    names = [n for n, *_ in targets]
    meta = dict(code=code, n_targets=len(targets), n_montages=len(montages), t_targets_s=round(t1 - t0, 1),
                t_score_s=round(t2 - t1, 1), device=str(torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"))
    if ref_dir:
        meta["ref6_check"] = ref6_check(code, names, F, T, O, labels, ref_dir, models)
    os.makedirs(outd, exist_ok=True)
    tmp = os.path.join(outd, f"{code}__targets.part.npz")
    np.savez(tmp, F=F, T=T, O=O, models=np.array(models), targets=np.array(names),
             tiers=np.array([t for _, t, _, _ in targets]),
             n_roi=np.array([int(r.sum()) for _, _, r, _ in targets]),
             n_out=np.array([int(o.sum()) for _, _, _, o in targets]),
             montages=np.array(labels), meta=np.array(json.dumps(meta)))
    os.replace(tmp, os.path.join(outd, f"{code}__targets.npz"))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
