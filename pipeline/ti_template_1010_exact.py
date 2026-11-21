# -*- coding: utf-8 -*-
"""Exhaustive two-pair TI search over the 10-10 cap on the MNI template: certifies the screened search (31 Oct 2026).

The 16 October review (N1/N4) rejected the screened 10-10 search of pipeline/ti_template_1010.py as evidence of a
global optimum: it scored every montage on a 30 % voxel sample and rescored only the top 400. Review decision D9
(31 Oct 2026; PI: "agree all") asked for the optimum to be certified rather than caveated. This script scores EVERY
eligible montage on ALL grey-matter voxels with the GPU scorer of the atlas runs (score_targets_v2.score_model), for
the same model, grid, targets, electrode set, pair rule (>= 40 mm) and channel rule (four distinct electrodes, cross
distance >= 25 mm) as the screened search, and reports, per target:
  - the exhaustive optimum (F, T, O, montage) and the best montage avoiding the inferior row;
  - the screened result and its exhaustive rank (equal => the screened search found the true optimum);
  - the 10-20 optimum from the same scorer (the 19 positions are a subset of the 76).
Full F arrays are stored so any later question about the 10-10 space needs no new solves.
Runs on a GPU node (basis from the cache of ti_template_1010.py --basis; SPICEHEAD_TI1010_CACHE).
    SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu python pipeline/ti_template_1010_exact.py [--subregions] [--out DIR]
"""
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here)); sys.path.insert(0, _here); sys.path.insert(0, os.path.join(_here, "nchc"))
import ti_template_1010 as TT                                        # noqa: E402
import fig_ti_targets_mni as M                                       # noqa: E402

ROOT = os.path.dirname(_here)
OUT = os.path.join(ROOT, "results_v23", "v23_template_design", "exhaustive_1010")


def main():
    import torch
    import score_targets_v2 as ST
    a = sys.argv[1:]
    sub = "--subregions" in a
    if sub:
        TT.TARGETS, TT.SUFFIX = TT.SUBREGIONS, "_sub"
        TT.OUT = TT.OUT + "_subregions"
    out = a[a.index("--out") + 1] if "--out" in a else OUT + ("_subregions" if sub else "")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    S = TT.setup()
    pairs, D = TT.pairs_and_dist(S)
    labs = S["labs"]
    gm = S["gm"]
    gidx = np.flatnonzero(gm.ravel()[S["brain"].ravel()])            # GM among brain voxels
    B = TT.load_basis(S, gidx)                                        # (76, 3, n_gm), Cz = 0
    assert M.REF in labs
    keys = list(TT.TARGETS)
    targets = [(k, "template", S["tmask"][k][gm], S["omask"][k][gm]) for k in keys]
    name = lambda p: "+".join(labs[x] for x in p)
    # the eligible montage set, exactly as ti_template_1010.screen enumerates it
    mont_idx = [(i, j) for i in range(len(pairs)) for j in range(i + 1, len(pairs))
                if len(set(pairs[i]) | set(pairs[j])) == 4
                and min(D[pairs[i][0], pairs[j][0]], D[pairs[i][0], pairs[j][1]],
                        D[pairs[i][1], pairs[j][0]], D[pairs[i][1], pairs[j][1]]) >= TT.CROSS_MIN_MM]
    lab_pairs = [(labs[a_], labs[b_]) for a_, b_ in pairs]
    montages = [(lab_pairs[i], lab_pairs[j]) for i, j in mont_idx]
    print(f"{len(labs)} positions, {len(pairs)} pairs, {len(montages):,} montages, {len(gidx):,} GM voxels, "
          f"{len(targets)} targets ({time.time() - t0:.0f} s)", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # scoring in slices of montages keeps the per-call tensors small; score_model builds P (pairs x 3 x n_gm) once per call,
    # so slices are large (200k montages) to amortize it
    F = np.zeros((len(targets), len(montages)), np.float32); T = np.zeros_like(F); O = np.zeros_like(F)
    step = 200_000
    for s in range(0, len(montages), step):
        e = min(s + step, len(montages))
        f_, t_, o_ = ST.score_model(B, labs, lab_pairs, montages[s:e], targets, dev, batch=64)
        F[:, s:e], T[:, s:e], O[:, s:e] = f_, t_, o_
        print(f"  {e:,}/{len(montages):,} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    inferior = np.array([any(l in TT.INFERIOR for l in (pa + pb)) for pa, pb in montages])
    is1020 = np.array([all(l in M.LABELS_1020 for l in (pa + pb)) for pa, pb in montages])
    # the screened search's answers (if present) and their exhaustive ranks
    prev = {}
    pj = os.path.join(TT.OUT, "TI_targets_MNI152_summary.json")
    if os.path.exists(pj):
        prev = {r["target"]: r for r in json.load(open(pj, encoding="utf-8"))["targets"]}
    mkey = {}                                                         # every spelling of a montage label -> index
    for m, (pa, pb) in enumerate(montages):
        mkey["+".join(pa) + " / " + "+".join(pb)] = m
        mkey["+".join(pb) + " / " + "+".join(pa)] = m
        mkey["+".join(pa[::-1]) + " / " + "+".join(pb)] = m; mkey["+".join(pa) + " / " + "+".join(pb[::-1])] = m
        mkey["+".join(pa[::-1]) + " / " + "+".join(pb[::-1])] = m
        mkey["+".join(pb[::-1]) + " / " + "+".join(pa)] = m; mkey["+".join(pb) + " / " + "+".join(pa[::-1])] = m
        mkey["+".join(pb[::-1]) + " / " + "+".join(pa[::-1])] = m
    rows = []
    for k, key in enumerate(keys):
        f = F[k]
        order = np.argsort(-f)
        rank = np.empty(len(f), np.int64); rank[order] = np.arange(1, len(f) + 1)
        b = int(order[0])
        s_ = int(np.flatnonzero(~inferior)[np.argmax(f[~inferior])])
        q = int(np.flatnonzero(is1020)[np.argmax(f[is1020])])
        r = dict(target=key, title=TT.TARGETS[key]["title"], n_montages=int(len(f)),
                 exhaustive_montage=f"{'+'.join(montages[b][0])} / {'+'.join(montages[b][1])}", F=round(float(f[b]), 4),
                 T_V_per_m=round(float(T[k, b]), 4), O_V_per_m=round(float(O[k, b]), 4),
                 n_within_1pct=int((f >= 0.99 * f[b]).sum()), n_within_3pct=int((f >= 0.97 * f[b]).sum()),
                 no_inferior_montage=f"{'+'.join(montages[s_][0])} / {'+'.join(montages[s_][1])}", F_no_inferior=round(float(f[s_]), 4),
                 montage_1020=f"{'+'.join(montages[q][0])} / {'+'.join(montages[q][1])}", F_1020=round(float(f[q]), 4),
                 gain_1010_over_1020_pct=round(100 * (float(f[b]) / float(f[q]) - 1), 2))
        if key in prev:
            p = prev[key]
            m = mkey.get(p["montage"])
            r.update(screened_montage=p["montage"], screened_F=p["F"],
                     screened_exhaustive_rank=int(rank[m]) if m is not None else None,
                     screened_F_exact=round(float(f[m]), 4) if m is not None else None,
                     screened_is_optimum=bool(m == b) if m is not None else None)
            m2 = mkey.get(p.get("montage_no_inferior", ""))
            r.update(screened_no_inferior_is_best=bool(m2 == s_) if m2 is not None else None)
        rows.append(r)
        print(f"{r['title']:<22} exhaustive {r['exhaustive_montage']:<20} F={r['F']:.3f} | 10-20 {r['montage_1020']:<18} F={r['F_1020']:.3f} "
              f"| screened: {r.get('screened_montage', '-')} rank {r.get('screened_exhaustive_rank', '-')}", flush=True)
    np.savez_compressed(os.path.join(out, "exhaustive_scores.npz"), F=F, T=T, O=O, keys=np.array(keys),
                        montages=np.array(["+".join(pa) + " / " + "+".join(pb) for pa, pb in montages]))
    json.dump(dict(model=f"SPICEhead {os.environ.get('SPICEHEAD_MODEL', '?')} template, exhaustive search over the 10-10 cap; 1 mA per channel",
                   grid_mm=M.DX, n_positions=len(labs), n_pairs=len(pairs), n_montages=len(montages), pair_min_mm=M.MIN_DIST_MM,
                   cross_min_mm=TT.CROSS_MIN_MM, gm_voxels=int(len(gidx)), scorer="score_targets_v2.score_model (GPU)",
                   screened_source=pj if prev else None, minutes=round((time.time() - t0) / 60, 1), targets=rows),
              open(os.path.join(out, "exhaustive_summary.json"), "w", encoding="utf-8"), indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


def name_pair(p):
    return "+".join(p)


if __name__ == "__main__":
    main()
