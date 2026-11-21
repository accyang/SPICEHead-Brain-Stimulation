# -*- coding: utf-8 -*-
"""Experiment 2 (23 Oct 2026): does the electrode model explain the SPICEhead-FEM gap?

SimNIBS's leadfield injects the current uniformly over the OUTER surface of a 2 mm gel pad (1.0 S/m)
on the scalp, with all 19 pads present; SPICEhead v22 injects uniformly over the skin-surface nodes
under the footprint, with no pads. Here SPICEhead is rebuilt with the same gel pads (spicehead/pads.py):
19 pads of 24 mm diameter and 2 mm thickness added to the label volume, injection at each pad's outer
layer, Cz as the reference. For each head and grid (2 and 1 mm) the 18 bases are solved (GPU), sampled
on the shared 1 mm grid, scored for the six Paper A targets over the 11,628 montages, and compared with
the stored FEM and standard-contact results of the 300-head run.

Runs on the desktop or on Nano 4 (env SPICEHEAD_CHARM_ROOT, SPICEHEAD_FULLSEARCH_CACHE,
SPICEHEAD_FULLSEARCH_OUT = where <code>__fem.json live, SPICEHEAD_EXP_OUT). On the cluster the head
folder is m2m_<code> and there is no code map; the FEM basis (for field statistics) exists only on the
desktop, so the variant basis is saved as <code>__sp{dx}pad_basis.npz and the field comparison is done
by --summary on the desktop after download.

Heads: the 8 heads with a combination below 90 % retention plus 22 others spread over the codes (30).

    python pipeline/nchc/pad_experiment.py [--heads h001,...] [--dx 2,1]
    python pipeline/nchc/pad_experiment.py --summary            (desktop)
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "pipeline"))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; desktop only
CHARM = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")
OUT = os.environ.get("SPICEHEAD_EXP_OUT", os.path.join(ROOT, "results_v22", "v22_pad_experiment"))
FEMJ = os.environ.get("SPICEHEAD_FULLSEARCH_OUT", "")
os.environ.setdefault("SPICEHEAD_MODEL", "v22")
os.environ.setdefault("SPICEHEAD_FULLSEARCH_CACHE", r"C:\spicehead_v22_logs\pad_experiment\cache")
os.environ["SPICEHEAD_CHARM_ROOT"] = CHARM
OUTLIERS = ["h067", "h041", "h063", "h071", "h134", "h053", "h009", "h098"]
PAD_MM, GEL_SIGMA = 2.0, 1.0
DIST_BINS = [20, 40, 60]


def codemap():
    if os.path.exists(CODEMAP):
        return {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    return None


def head_list(codes_all):
    others = [c for c in sorted(codes_all)[::14] if c not in OUTLIERS][:22]
    return OUTLIERS + others


def m2m_of(code, cmap):
    cid = cmap[code] if cmap else code
    return cid, os.path.join(CHARM, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)


def score_json(code, model):
    for p in (os.path.join(L, "heads", code, f"{code}__{model}.json"), os.path.join(FEMJ, f"{code}__{model}.json")):
        if os.path.exists(p):
            return json.load(open(p))["rois"]
    return None


def field_stats(basis, others, code, ej):
    """Per-basis statistics against the stored FEM basis (desktop only; None when the bundle is absent)."""
    bp = os.path.join(L, "bundles", f"{code}.npz")
    if not os.path.exists(bp):
        return None
    with np.load(bp) as z:
        Bf, labs_f = z["basis_fem"], list(z["labs_fem"]); gw = z["gm_world"].astype(float)
    assert labs_f == others
    out = {}
    for i, lab in enumerate(others):
        ef, es = Bf[i].astype(np.float64), basis[i].astype(np.float64)
        nf, ns = np.linalg.norm(ef, axis=0), np.linalg.norm(es, axis=0)
        ok = (nf > 1e-9 * nf.max()) & np.isfinite(ns)
        rel = np.linalg.norm(es - ef, axis=0)[ok] / nf[ok]
        d = np.linalg.norm(gw - np.asarray(ej[lab], float), axis=1)[ok]; b = np.digitize(d, DIST_BINS)
        out[lab] = dict(rel_median=float(np.median(rel)), ratio_median=float(np.median(ns[ok] / nf[ok])),
                        r=float(np.corrcoef(nf[ok], ns[ok])[0, 1]), bias_pct=float(100 * (ns[ok].mean() / nf[ok].mean() - 1)),
                        rel_by_dist=[float(np.median(rel[b == k])) if (b == k).any() else None for k in range(4)])
    return out


def run_head(code, cmap, dx):
    import v22_fem_fullsearch as fs
    from spicehead import mri
    from spicehead.conductivity import CHARM_LABELS, CONDUCTIVITY
    from spicehead.electrodes import Electrode, pair_injection
    from spicehead.field import efield_components
    from spicehead.pads import add_gel_pads, nodes_from_ijk, GEL_LABEL
    from spicehead.score_gpu import score_montages_gpu
    from montage_search import snap
    from v21_head_convergence import make_solver
    tag = f"sp{dx:g}pad"
    outp = os.path.join(OUT, f"{code}__{tag}.json")
    if os.path.exists(outp):
        return "cached"
    t0 = time.time()
    cid, m2m = m2m_of(code, cmap)
    geo = fs.grid_geometry(cid, m2m)
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    labels, aff_dx, _ = mri.load_label_volume(seg, dx_mm=dx)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs, pairs, montages = fs.candidate_set(ej)
    scalp_vox = np.argwhere(labels == fs.SCALP_LABEL)
    centres = {lab: tuple(int(x) for x in snap(scalp_vox, mri.world_to_voxel(aff_dx, ej[lab]))) for lab in labs}
    labels2, pads = add_gel_pads(labels, centres, fs.ELEC_RADIUS_MM / dx, PAD_MM / dx)
    lm = dict(CHARM_LABELS); lm[GEL_LABEL] = "electrode_gel"
    cond = dict(CONDUCTIVITY); cond["electrode_gel"] = GEL_SIGMA
    model = mri.model_from_labels(labels2, None, dx_mm=dx, label_map=lm, cond=cond)
    elecs = {lab: Electrode(nodes_from_ijk(model, pads[lab]["top"]), lab) for lab in labs}
    n_gel = {lab: int(len(pads[lab]["gel"])) for lab in labs}
    solver, ncomp = make_solver(model.G)
    others = [x for x in labs if x != fs.REF]
    basis = np.zeros((len(others), 3, len(geo["gm_idx"])), np.float32)
    t1 = time.time()
    for i, lab in enumerate(others):
        V = solver(pair_injection(model, elecs[lab], elecs[fs.REF], fs.CURRENT))
        basis[i] = np.stack([fs.sample_on_gm(c, aff_dx, geo["gm_world"]) for c in efield_components(model, V)])
    t_basis = time.time() - t1
    res = score_montages_gpu(basis, others, fs.REF, pairs, montages, geo["roi_sel"], geo["out_sel"], fs.ROI_ORDER, fs.label_of, verbose=False)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{code}__{tag}_basis.part.npz"), basis=basis, labs=np.array(others), meta=json.dumps(dict(model=tag, dx_mm=dx)))
    os.replace(os.path.join(OUT, f"{code}__{tag}_basis.part.npz"), os.path.join(OUT, f"{code}__{tag}_basis.npz"))
    out = dict(code=code, model=tag, dx_mm=dx, n_nodes=int(model.n_nodes), n_components=int(ncomp), pad_mm=PAD_MM, gel_sigma=GEL_SIGMA,
               gel_voxels=n_gel, t_build_s=round(t1 - t0, 1), t_basis_s=round(t_basis, 1), t_total_s=round(time.time() - t0, 1),
               rois=res, field_vs_fem=field_stats(basis, others, code, ej))
    json.dump(out, open(outp + ".part", "w"), indent=0); os.replace(outp + ".part", outp)
    return f"ok ({model.n_nodes:,} nodes, basis {t_basis:.0f} s, total {time.time()-t0:.0f} s)"


def summary(codes):
    """Desktop: decision and field comparison of standard vs pad contact against FEM; fills field statistics
    from the downloaded bases when a result JSON lacks them."""
    import v22_fem_fullsearch as fs
    from scipy.stats import spearmanr
    rows = []
    for code in codes:
        fem = score_json(code, "fem")
        for dx in (2, 1):
            std = score_json(code, f"sp{dx}")
            pp = os.path.join(OUT, f"{code}__sp{dx}pad.json")
            if not (fem and std and os.path.exists(pp)):
                continue
            padj = json.load(open(pp)); pad = padj["rois"]
            if padj.get("field_vs_fem") is None and os.path.exists(os.path.join(OUT, f"{code}__sp{dx}pad_basis.npz")):
                cmap = codemap(); _, m2m = m2m_of(code, cmap)
                ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
                with np.load(os.path.join(OUT, f"{code}__sp{dx}pad_basis.npz")) as z:
                    padj["field_vs_fem"] = field_stats(z["basis"], list(z["labs"]), code, ej)
                json.dump(padj, open(pp, "w"), indent=0)
            for roi in fs.ROI_ORDER:
                keys = sorted(fem[roi]); Ff = np.array([fem[roi][k][0] for k in keys]); wf = int(Ff.argmax())
                for name, sc in (("standard", std), ("pad", pad)):
                    Fs = np.array([sc[roi][k][0] for k in keys]); ws = int(Fs.argmax())
                    rows.append(dict(code=code, dx=dx, roi=roi, contact=name, same=ws == wf, keep=100 * Ff[ws] / Ff[wf],
                                     rho=float(spearmanr(Ff, Fs).correlation)))
            f = padj.get("field_vs_fem")
            if f:
                rows.append(dict(code=code, dx=dx, roi="_field", contact="pad", rel=float(np.median([v["rel_median"] for v in f.values()])),
                                 ratio=float(np.median([v["ratio_median"] for v in f.values()])), bias=float(np.median([v["bias_pct"] for v in f.values()])),
                                 r=float(np.median([v["r"] for v in f.values()])),
                                 rel_by_dist=[float(np.median([v["rel_by_dist"][k] for v in f.values() if v["rel_by_dist"][k] is not None])) for k in range(4)]))
    fe = os.path.join(L, "field_error_perhead")
    for code in codes:
        p = os.path.join(fe, f"{code}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            for dx, m in ((2, "sp2"), (1, "sp1")):
                e = r["by_electrode"]
                rows.append(dict(code=code, dx=dx, roi="_field", contact="standard", rel=float(np.median([e[k][m]["rel_median"] for k in e])),
                                 ratio=float(np.median([e[k][m]["ratio_median"] for k in e])), bias=float(np.median([e[k][m]["bias_pct"] for k in e])),
                                 r=float(np.median([e[k][m]["r"] for k in e])),
                                 rel_by_dist=[r["by_dist_this"].get(f"{m}:{k}", {}).get("rel") for k in range(4)]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "pad_experiment_rows.json"), "w"), indent=0)
    L_ = [f"# Pad-electrode experiment: {len(codes)} heads, 2 and 1 mm (vs FEM)", "",
          "SPICEhead rebuilt with SimNIBS-like gel pads (24 mm, 2 mm, 1.0 S/m, injection at the pad's outer surface, all 19 pads present) "
          "against the standard v22 skin-surface contact, both compared with the stored FEM leadfield of the 300-head run. Decisions over "
          "11,628 montages for the six Paper A targets; fields per basis on grey matter.", "",
          "| Grid | Contact | Same winner as FEM | FEM optimum kept, median [5th pct, min] | rho median | Field rel. error median | Magnitude ratio | Bias % | rel. error by distance to electrode (<20, 20-40, 40-60, >60 mm) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for dx in (2, 1):
        for name in ("standard", "pad"):
            d = [x for x in rows if x["dx"] == dx and x["contact"] == name and x["roi"] != "_field"]
            fld = [x for x in rows if x["dx"] == dx and x["contact"] == name and x["roi"] == "_field"]
            if not d:
                continue
            k = np.array([x["keep"] for x in d]); rho = np.array([x["rho"] for x in d])
            fl = (f"{np.median([x['rel'] for x in fld]):.3f} | {np.median([x['ratio'] for x in fld]):.3f} | {np.median([x['bias'] for x in fld]):+.1f} | "
                  + ", ".join(f"{np.median([x['rel_by_dist'][q] for x in fld if x['rel_by_dist'][q] is not None]):.3f}" for q in range(4))) if fld else "– | – | – | –"
            L_.append(f"| {dx} mm | {name} | {sum(x['same'] for x in d)}/{len(d)} | {np.median(k):.1f} [{np.percentile(k, 5):.1f}, {k.min():.1f}] | {np.median(rho):.3f} | {fl} |")
    L_ += ["", "Paired per head × target (1 mm): rows where the pad contact keeps more / less / the same share of the FEM optimum than the standard contact:"]
    d1 = {(x["code"], x["roi"]): x["keep"] for x in rows if x["dx"] == 1 and x["contact"] == "standard" and x["roi"] != "_field"}
    d2 = {(x["code"], x["roi"]): x["keep"] for x in rows if x["dx"] == 1 and x["contact"] == "pad" and x["roi"] != "_field"}
    both = [(d1[k], d2[k]) for k in d1 if k in d2]
    if both:
        diff = np.array([b - a for a, b in both])
        L_.append(f"{int((diff > 0.5).sum())} / {int((diff < -0.5).sum())} / {int((abs(diff) <= 0.5).sum())} of {len(both)}; median change {np.median(diff):+.2f} points.")
        low = [(a, b) for a, b in both if a < 90]
        if low:
            L_.append("Rows below 90% with the standard contact: " + "; ".join(f"{a:.1f} → {b:.1f}" for a, b in low))
    open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L_) + "\n")
    print("\n".join(L_))


def main():
    a = sys.argv[1:]
    cmap = codemap()
    if "--heads" in a:
        codes = a[a.index("--heads") + 1].split(",")
    else:
        codes = head_list(cmap) if cmap else sorted(d[4:] for d in os.listdir(CHARM) if d.startswith("m2m_h"))
    if "--summary" in a:
        if "--heads" not in a:                      # the heads actually computed (cluster list may differ)
            codes = sorted({f.split("__")[0] for f in os.listdir(OUT) if f.endswith("pad.json")})
        return summary(codes)
    dxs = [float(x) for x in a[a.index("--dx") + 1].split(",")] if "--dx" in a else [2.0, 1.0]
    os.makedirs(os.environ["SPICEHEAD_FULLSEARCH_CACHE"], exist_ok=True)
    t0 = time.time()
    for n, code in enumerate(codes, 1):
        for dx in dxs:
            try:
                msg = run_head(code, cmap, dx)
            except Exception as e:
                import traceback; traceback.print_exc()
                msg = f"FAILED {type(e).__name__}: {e}"
            print(f"[{n}/{len(codes)}] {code} {dx:g} mm: {msg} | elapsed {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
