# -*- coding: utf-8 -*-
"""Summarize a SPICEhead variant (pad / homogenized / 0.5 mm) against FEM and the standard contact, from the
per-head JSONs and variant bases downloaded from Nano 4 (23 Oct 2026).

For every head with <code>__<tag>.json in the variant folder:
  decisions  same winner as FEM; share of the FEM optimum's F kept; Spearman rho over 11,628 montages
             (variant, and the standard model of the same grid from the 300-head run)
  fields     per basis vs the FEM basis in L:/.../bundles: rel error, magnitude ratio, bias, r, angle
             (variant from <code>__<tag>_basis.npz; standard from the field-error map when present)

    python pipeline/nchc/variant_summary.py <variant_dir> <tag> <standard_tag> [<tag2> <standard_tag2> ...]
    e.g.  ... results_v22/v22_homog_experiment sp2hom sp2 sp1hom sp1
          ... results_v22/v22_fine_grid_experiment sp0.5 sp1
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "pipeline"))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"
CHARM = r"L:\BrainProbe\SPICEhead\charm"
ROIS = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]


def field_stats(basis, labs, code, cache):
    bp = os.path.join(L, "bundles", f"{code}.npz")
    if not os.path.exists(bp):
        return None
    with np.load(bp) as z:
        Bf, labs_f = z["basis_fem"], list(z["labs_fem"])
    assert labs_f == labs, (labs_f[:3], labs[:3])
    rel, ratio, bias, r, ang = [], [], [], [], []
    for i in range(len(labs)):
        ef, es = Bf[i].astype(np.float64), basis[i].astype(np.float64)
        nf, ns = np.linalg.norm(ef, axis=0), np.linalg.norm(es, axis=0)
        ok = (nf > 1e-9 * nf.max()) & np.isfinite(ns) & (ns > 0)
        rel.append(np.median(np.linalg.norm(es - ef, axis=0)[ok] / nf[ok])); ratio.append(np.median(ns[ok] / nf[ok]))
        bias.append(100 * (ns[ok].mean() / nf[ok].mean() - 1)); r.append(np.corrcoef(nf[ok], ns[ok])[0, 1])
        ang.append(np.median(np.degrees(np.arccos(np.clip((ef * es).sum(0)[ok] / (nf[ok] * ns[ok]), -1, 1)))))
    return dict(rel=float(np.median(rel)), ratio=float(np.median(ratio)), bias=float(np.median(bias)), r=float(np.median(r)), ang=float(np.median(ang)))


def main():
    d = sys.argv[1]; pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
    fe = os.path.join(L, "field_error_perhead")
    rows = []
    for tag, std_tag in pairs:
        codes = sorted(f.split("__")[0] for f in os.listdir(d) if f.endswith(f"__{tag}.json"))
        for code in codes:
            fem = json.load(open(os.path.join(L, "heads", code, f"{code}__fem.json")))["rois"]
            std = json.load(open(os.path.join(L, "heads", code, f"{code}__{std_tag}.json")))["rois"]
            var = json.load(open(os.path.join(d, f"{code}__{tag}.json")))
            for roi in ROIS:
                keys = sorted(fem[roi]); Ff = np.array([fem[roi][k][0] for k in keys]); wf = int(Ff.argmax())
                for name, sc in ((std_tag, std), (tag, var["rois"])):
                    Fs = np.array([sc[roi][k][0] for k in keys]); ws = int(Fs.argmax())
                    rows.append(dict(code=code, roi=roi, model=name, same=ws == wf, keep=100 * Ff[ws] / Ff[wf], rho=float(spearmanr(Ff, Fs).correlation)))
            bp = os.path.join(d, f"{code}__{tag}_basis.npz")
            if os.path.exists(bp):
                cache = os.path.join(d, f"{code}__{tag}_fieldstats.json")
                if os.path.exists(cache):
                    fs_ = json.load(open(cache))
                else:
                    with np.load(bp) as z:
                        fs_ = field_stats(z["basis"], list(z["labs"]), code, None)
                    json.dump(fs_, open(cache, "w"))
                if fs_:
                    rows.append(dict(code=code, roi="_field", model=tag, **fs_))
            p = os.path.join(fe, f"{code}.json")
            if os.path.exists(p) and std_tag in ("sp1", "sp2"):
                e = json.load(open(p))["by_electrode"]
                rows.append(dict(code=code, roi="_field", model=std_tag, rel=float(np.median([e[k][std_tag]["rel_median"] for k in e])),
                                 ratio=float(np.median([e[k][std_tag]["ratio_median"] for k in e])), bias=float(np.median([e[k][std_tag]["bias_pct"] for k in e])),
                                 r=float(np.median([e[k][std_tag]["r"] for k in e])), ang=float(np.median([e[k][std_tag]["ang_median"] for k in e]))))
    out = [f"# Variant summary for {os.path.basename(d)} ({time.strftime('%Y-%m-%d %H:%M')})", "",
           "| Model | Heads | Same winner as FEM | FEM optimum kept, median [5th pct, min] | rho median | Field rel. error | Magnitude ratio | Bias % | Angle ° | r |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    seen = []
    for tag, std_tag in pairs:
        for m in (std_tag, tag):
            if m in seen:
                continue
            seen.append(m)
            dec = [x for x in rows if x["model"] == m and x["roi"] != "_field"]; fld = [x for x in rows if x["model"] == m and x["roi"] == "_field"]
            if not dec:
                continue
            k = np.array([x["keep"] for x in dec]); heads = len({x["code"] for x in dec})
            f = (f"{np.median([x['rel'] for x in fld]):.3f} | {np.median([x['ratio'] for x in fld]):.3f} | {np.median([x['bias'] for x in fld]):+.1f} | "
                 f"{np.median([x['ang'] for x in fld]):.1f} | {np.median([x['r'] for x in fld]):.3f}") if fld else "– | – | – | – | –"
            out.append(f"| {m} | {heads} | {sum(x['same'] for x in dec)}/{len(dec)} | {np.median(k):.1f} [{np.percentile(k, 5):.1f}, {k.min():.1f}] | "
                       f"{np.median([x['rho'] for x in dec]):.3f} | {f} |")
    for tag, std_tag in pairs:
        a = {(x["code"], x["roi"]): x["keep"] for x in rows if x["model"] == std_tag and x["roi"] != "_field"}
        b = {(x["code"], x["roi"]): x["keep"] for x in rows if x["model"] == tag and x["roi"] != "_field"}
        both = [(a[k], b[k]) for k in a if k in b]
        if both:
            diff = np.array([y - x for x, y in both])
            out += ["", f"{tag} vs {std_tag}, paired head × target: better / worse / same (±0.5 pt): {int((diff > 0.5).sum())} / {int((diff < -0.5).sum())} / {int((abs(diff) <= 0.5).sum())} of {len(both)}; median change {np.median(diff):+.2f} points.",
                    "Rows below 90% with the standard model: " + ("; ".join(f"{x:.1f} → {y:.1f}" for x, y in both if x < 90) or "none")]
    json.dump(rows, open(os.path.join(d, "variant_rows.json"), "w"), indent=0)
    open(os.path.join(d, "SUMMARY.md"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
