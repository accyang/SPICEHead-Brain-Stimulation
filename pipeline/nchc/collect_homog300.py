# -*- coding: utf-8 -*-
"""Collect the 300-head FEM comparison of the homogenized coarse network (23 Oct 2026).

The PI asked to "re-run the 300-head FEM comparison with it on Nano 4". The homogenized network
(pipeline/nchc/homog_experiment.py; edge conductances from the 0.5 mm labels, MIN_FILL 0.5) was run on Nano 4
at 2 and 1 mm for every head by homog300_nano4.sbatch. wsl_pull_homog300.sh copied the outputs to
L:/BrainProbe/SPICEhead/nchc_fullsearch/homog300/ and verified them.

This script puts the homogenized models next to the published v22 models (2 mm and 1 mm) against the same
FEM score files, with the per-head metrics of v22_fem_fullsearch.stage_collect(): same winner, Spearman
rho, rho over the FEM top 100, regret both ways, top-10 and top-25 overlap, and the cohort-level winners.
Before anything else, it recomputes the published v22 rows and checks them against
results_v22/v22_fem_fullsearch_300/fullsearch_decisions.csv.
Field-level statistics (rel. error, magnitude ratio, bias, r, angle; median over the 18 single-electrode
bases) are computed for all four models with one function, from the FEM bundles and the basis files.

Output: results_v22/v22_fem_fullsearch_300_homog/ (h-codes only; the code map is never read).
    python pipeline/nchc/collect_homog300.py [--partial] [--workers 4]
"""
import csv
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
HEADS, HOM, BUNDLES = os.path.join(L, "heads"), os.path.join(L, "homog300"), os.path.join(L, "bundles")
OUT = os.path.join(ROOT, "results_v22", "v22_fem_fullsearch_300_homog")
PUB = os.path.join(ROOT, "results_v22", "v22_fem_fullsearch_300", "fullsearch_decisions.csv")
ROIS = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]
MODELS = ["sp2", "sp1", "sp2hom", "sp1hom"]
PAIRS = [("fem", m) for m in MODELS] + [("sp2", "sp2hom"), ("sp1", "sp1hom")]
NAME = {"sp2": "2 mm, standard (v22)", "sp1": "1 mm, standard (v22)", "sp2hom": "2 mm, homogenized", "sp1hom": "1 mm, homogenized"}


def g7(x):
    return float(f"{x:.7g}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def manifest(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) == 2:
            out[parts[1].lstrip("*").split("/")[-1]] = parts[0]
    return out


def score_file(code, m):
    return os.path.join(HOM if m.endswith("hom") else os.path.join(HEADS, code), f"{code}__{m}.json")


def verify(code):
    """sha256 of the five score JSONs against the per-head manifests (homog300/<code>.sha256, heads/<code>/MANIFEST)."""
    bad = []
    mh = manifest(os.path.join(HOM, f"{code}.sha256"))
    ms = manifest(os.path.join(HEADS, code, "MANIFEST.sha256"))
    for m in ["fem"] + MODELS:
        p = score_file(code, m); want = (mh if m.endswith("hom") else ms).get(os.path.basename(p))
        if want is None or sha256(p) != want:
            bad.append(m)
    return bad


def field_stats(ef_all, es_all):
    per = []
    for i in range(ef_all.shape[0]):
        ef, es = ef_all[i].astype(np.float64), es_all[i].astype(np.float64)
        nf, ns = np.linalg.norm(ef, axis=0), np.linalg.norm(es, axis=0)
        ok = (nf > 1e-9 * nf.max()) & np.isfinite(ns) & (ns > 0)
        nf, ns, ef, es = nf[ok], ns[ok], ef[:, ok], es[:, ok]
        per.append(dict(rel=float(np.median(np.linalg.norm(es - ef, axis=0) / nf)), ratio=float(np.median(ns / nf)),
                        bias=float(100 * (ns.mean() / nf.mean() - 1)), r=float(np.corrcoef(nf, ns)[0, 1]),
                        ang=float(np.median(np.degrees(np.arccos(np.clip((ef * es).sum(0) / (nf * ns), -1, 1)))))))
    return {k: float(np.median([p[k] for p in per])) for k in per[0]}


def head_fields(code):
    cache = os.path.join(HOM, f"{code}__fieldstats_all.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    with np.load(os.path.join(BUNDLES, f"{code}.npz")) as z:
        Bf, labs = z["basis_fem"], list(z["labs_fem"])
        out = {}
        for m in ("sp2", "sp1"):
            assert list(z[f"labs_{m}"]) == labs
            out[m] = field_stats(Bf, z[f"basis_{m}"])
    for m in ("sp2hom", "sp1hom"):
        with np.load(os.path.join(HOM, f"{code}__{m}_basis.npz")) as z:
            assert list(z["labs"]) == labs, (code, m)
            out[m] = field_stats(Bf, z["basis"])
    json.dump(out, open(cache + ".part", "w")); os.replace(cache + ".part", cache)
    return out


def head_rows(code):
    data = {m: json.load(open(score_file(code, m), encoding="utf-8")) for m in ["fem"] + MODELS}
    rows, Fs = [], {}
    for roi in ROIS:
        keys = sorted(data["fem"]["rois"][roi])
        for m in MODELS:
            assert sorted(data[m]["rois"][roi]) == keys, (code, m, roi)
        F = {m: np.array([data[m]["rois"][roi][k][0] for k in keys]) for m in ["fem"] + MODELS}
        Fs[roi] = {m: F[m].astype(np.float32) for m in F}
        win = {m: int(F[m].argmax()) for m in F}
        row = dict(subject=code, roi=roi, n_montages=len(keys))
        for m in ["fem"] + MODELS:
            k = keys[win[m]]
            row[f"winner_{m}"] = k; row[f"F_{m}"] = F[m][win[m]]
            row[f"T_{m}"] = data[m]["rois"][roi][k][1]; row[f"O_{m}"] = data[m]["rois"][roi][k][2]
        for a, b in PAIRS:
            va, vb = F[a], F[b]
            row[f"agree_{a}_{b}"] = win[a] == win[b]
            row[f"rho_{a}_{b}"] = g7(float(spearmanr(va, vb).correlation))
            top = np.argsort(-va)[:100]
            row[f"rho_top100_{a}_{b}"] = g7(float(spearmanr(va[top], vb[top]).correlation))
            row[f"regret_{a}winner_under_{b}_pct"] = round(100 * (1 - vb[win[a]] / vb[win[b]]), 3)
            row[f"regret_{b}winner_under_{a}_pct"] = round(100 * (1 - va[win[b]] / va[win[a]]), 3)
            for n in (10, 25):
                ta = set(np.array(keys)[np.argsort(-va)[:n]]); tb = set(np.array(keys)[np.argsort(-vb)[:n]])
                row[f"top{n}_overlap_{a}_{b}"] = len(ta & tb)
        rows.append(row)
    meta = {m: {k: data[m].get(k) for k in ("n_nodes", "n_components", "dropped_nodes", "t_build_s", "t_basis_s", "t_total_s", "block", "fine_mm")}
            for m in ("sp2hom", "sp1hom")}
    return rows, Fs, meta, sorted(data["fem"]["rois"][ROIS[0]])


def one(code):
    bad = verify(code)
    if bad:
        return code, None, None, None, None, bad
    rows, Fs, meta, keys = head_rows(code)
    return code, rows, Fs, meta, head_fields(code), []


def check_published(rows):
    pub = {(r["subject"], r["roi"]): r for r in csv.DictReader(open(PUB, encoding="utf-8"))}
    n = worst_rho = worst_reg = 0.0; mism = []
    for r in rows:
        p = pub[(r["subject"], r["roi"])]
        for m in ("sp2", "sp1"):
            n += 1
            if str(r[f"agree_fem_{m}"]) != p[f"agree_fem_{m}"] or r[f"winner_{m}"] != p[f"winner_{m}"] or r[f"top10_overlap_fem_{m}"] != int(p[f"top10_overlap_fem_{m}"]):
                mism.append((r["subject"], r["roi"], m))
            worst_rho = max(worst_rho, abs(r[f"rho_fem_{m}"] - float(p[f"rho_fem_{m}"])))
            worst_reg = max(worst_reg, abs(r[f"regret_{m}winner_under_fem_pct"] - float(p[f"regret_{m}winner_under_fem_pct"])))
    return dict(n_compared=int(n), mismatches=mism, max_abs_rho_diff=worst_rho, max_abs_regret_diff_pct=worst_reg)


def ret(rows, b, a="fem", roi=None):
    """retention (% of a's optimum kept by b's winner, judged by a)"""
    return np.array([100 - r[f"regret_{b}winner_under_{a}_pct"] for r in rows if roi in (None, r["roi"])])


def main():
    partial = "--partial" in sys.argv
    nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    codes = sorted(f[:-7] for f in os.listdir(HOM) if f.endswith(".sha256"))
    if not partial and len(codes) < 300:
        sys.exit(f"{len(codes)} heads pulled so far; use --partial for an interim table")
    t0 = time.time()
    rows, meta, fields, bad, Fall = [], {}, {}, {}, {roi: {m: [] for m in ["fem"] + MODELS} for roi in ROIS}
    keys0 = None
    with Pool(nw) as pool:
        for code, r, Fs, mt, fl, b in pool.imap(one, codes):
            if b:
                bad[code] = b; continue
            rows += r; meta[code] = mt; fields[code] = fl
            for roi in ROIS:
                for m in Fall[roi]:
                    Fall[roi][m].append(Fs[roi][m])
    heads_ok = sorted(meta)
    keys0 = head_rows(heads_ok[0])[3]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "decisions.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "fields_perhead.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["subject", "model", "rel", "ratio", "bias_pct", "r", "angle_deg"])
        for c in heads_ok:
            for m in MODELS:
                x = fields[c][m]; w.writerow([c, m, round(x["rel"], 5), round(x["ratio"], 5), round(x["bias"], 3), round(x["r"], 5), round(x["ang"], 3)])
    chk = check_published(rows)
    # cohort level: median F per montage across heads
    coh = []
    for roi in ROIS:
        med = {m: np.median(np.stack(Fall[roi][m]), axis=0) for m in Fall[roi]}
        win = {m: int(med[m].argmax()) for m in med}
        c = dict(roi=roi, n_heads=len(heads_ok))
        for m in med:
            c[f"winner_{m}"] = keys0[win[m]]
        for m in MODELS:
            c[f"retains_fem_{m}_pct"] = round(float(100 * med["fem"][win[m]] / med["fem"][win["fem"]]), 2)
            c[f"rho_fem_{m}"] = g7(float(spearmanr(med["fem"], med[m]).correlation))
        coh.append(c)
    with open(os.path.join(OUT, "cohort.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(coh[0])); w.writeheader(); w.writerows(coh)

    # ---------- summary ----------
    N = len(rows)
    S = dict(n_heads=len(heads_ok), n_rows=N, failed_verification=bad, check_published=chk, models={})
    for m in MODELS:
        k = ret(rows, m); fl = [fields[c][m] for c in heads_ok]
        S["models"][m] = dict(same=int(sum(r[f"agree_fem_{m}"] for r in rows)), rho_median=float(np.median([r[f"rho_fem_{m}"] for r in rows])),
                              rho_min=float(min(r[f"rho_fem_{m}"] for r in rows)), keep_median=float(np.median(k)), keep_p5=float(np.percentile(k, 5)),
                              keep_min=float(k.min()), below90=int((k < 90).sum()), below95=int((k < 95).sum()),
                              top10_median=float(np.median([r[f"top10_overlap_fem_{m}"] for r in rows])),
                              **{f"field_{q}": float(np.median([x[q] for x in fl])) for q in ("rel", "ratio", "bias", "ang", "r")},
                              by_roi={roi: dict(same=int(sum(r[f"agree_fem_{m}"] for r in rows if r["roi"] == roi)), keep_median=float(np.median(ret(rows, m, roi=roi))),
                                                keep_p5=float(np.percentile(ret(rows, m, roi=roi), 5)), keep_min=float(ret(rows, m, roi=roi).min()),
                                                rho_median=float(np.median([r[f"rho_fem_{m}"] for r in rows if r["roi"] == roi])),
                                                below90=int((ret(rows, m, roi=roi) < 90).sum())) for roi in ROIS})
    paired = {}
    for std, hom in (("sp2", "sp2hom"), ("sp1", "sp1hom")):
        d = ret(rows, hom) - ret(rows, std)
        low = [dict(subject=r["subject"], roi=r["roi"], standard=round(100 - r[f"regret_{std}winner_under_fem_pct"], 1), homogenized=round(100 - r[f"regret_{hom}winner_under_fem_pct"], 1))
               for r in rows if min(100 - r[f"regret_{std}winner_under_fem_pct"], 100 - r[f"regret_{hom}winner_under_fem_pct"]) < 90]
        fd = {q: [fields[c][hom][q] - fields[c][std][q] for c in heads_ok] for q in ("rel", "bias", "r")}
        paired[hom] = dict(better=int((d > 0.5).sum()), worse=int((d < -0.5).sum()), same=int((abs(d) <= 0.5).sum()), median_change=float(np.median(d)),
                           heads_rel_lower=int(sum(x < 0 for x in fd["rel"])), heads_r_higher=int(sum(x > 0 for x in fd["r"])),
                           heads_abs_bias_lower=int(sum(abs(fields[c][hom]["bias"]) < abs(fields[c][std]["bias"]) for c in heads_ok)), rows_below90=low)
    S["paired"] = paired
    S["network"] = {m: dict(nodes_median=float(np.median([meta[c][m]["n_nodes"] for c in heads_ok])), dropped_median=float(np.median([meta[c][m]["dropped_nodes"] for c in heads_ok])),
                            dropped_max=int(max(meta[c][m]["dropped_nodes"] for c in heads_ok)), components_max=int(max(meta[c][m]["n_components"] for c in heads_ok)),
                            t_total_median_s=float(np.median([meta[c][m]["t_total_s"] for c in heads_ok]))) for m in ("sp2hom", "sp1hom")}
    S["cohort"] = coh
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    Mx = S["models"]
    o = [f"# FEM vs SPICEhead with the homogenized coarse network: {len(heads_ok)} heads, 11,628 montages",
         "", f"Collected {time.strftime('%Y-%m-%d %H:%M')} by `pipeline/nchc/collect_homog300.py`" + (" (INTERIM: not all heads pulled yet)" if len(heads_ok) < 300 else "") + ".",
         "Homogenized network: `pipeline/nchc/homog_experiment.py` run on NCHC Nano 4 (H200, GPU solver) by `homog300_nano4.sbatch`, with the code and parameters of the 30-head test (MIN_FILL 0.5, central-difference field, v22 skin-surface contacts). "
         "FEM and the standard v22 models are the published 300-head score files (`results_v22/v22_fem_fullsearch_300`), unchanged. "
         "Metrics follow `v22_fem_fullsearch.stage_collect()`. Retention = share of the FEM optimum kept when the SPICEhead winner is judged by FEM (100 − regret). "
         "Field statistics: median over the 18 single-electrode bases of the per-voxel median on grey matter, then the median over heads; one function for all four models.",
         "", "This is a new result set. It is not in any Results record or manuscript.", "",
         "## Headline (per head × target, " + f"{N} combinations)", "",
         "| Model | Same winner as FEM | Retention median / 5th pct / min | Below 90% | Below 95% | ρ median [min] | Top-10 overlap | Field rel. error | Magnitude ratio | Bias | Angle | r |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        x = Mx[m]
        o.append(f"| {NAME[m]} | {x['same']}/{N} ({100*x['same']/N:.0f}%) | {x['keep_median']:.1f} / {x['keep_p5']:.1f} / {x['keep_min']:.1f}% | {x['below90']} | {x['below95']} | "
                 f"{x['rho_median']:.3f} [{x['rho_min']:.3f}] | {x['top10_median']:.0f} | {x['field_rel']:.3f} | {x['field_ratio']:.3f} | {x['field_bias']:+.1f}% | {x['field_ang']:.1f}° | {x['field_r']:.3f} |")
    for std, hom in (("sp1", "sp1hom"), ("sp2", "sp2hom")):
        p = paired[hom]
        o += ["", f"## {NAME[hom]} vs {NAME[std]}, by target", "",
              "| Target | Same winner: standard → homogenized | Retention 5th pct | Retention min | Below 90% | ρ median |", "|---|---|---|---|---|---|"]
        for roi in ROIS:
            a, b = Mx[std]["by_roi"][roi], Mx[hom]["by_roi"][roi]
            o.append(f"| {roi} | {a['same']} → {b['same']} of {len(heads_ok)} | {a['keep_p5']:.1f} → {b['keep_p5']:.1f}% | {a['keep_min']:.1f} → {b['keep_min']:.1f}% | {a['below90']} → {b['below90']} | {a['rho_median']:.3f} → {b['rho_median']:.3f} |")
        o += ["", f"Paired over {N} head × target combinations: {p['better']} keep more of the FEM optimum with the homogenized network, {p['worse']} keep less, "
                  f"{p['same']} are unchanged (±0.5 points); median change {p['median_change']:+.2f} points. Field level, per head: relative error lower in {p['heads_rel_lower']}/{len(heads_ok)}, "
                  f"r higher in {p['heads_r_higher']}/{len(heads_ok)}, |bias| lower in {p['heads_abs_bias_lower']}/{len(heads_ok)}.",
              "", "Combinations below 90% with either model (standard → homogenized): " + ("; ".join(f"{x['subject']} {x['roi']} {x['standard']} → {x['homogenized']}" for x in p["rows_below90"]) or "none") + "."]
    o += ["", "## Cohort level (median F per montage across heads)", "",
          "| Target | Winner FEM | Winner 1 mm standard | Winner 1 mm homogenized | Retains (FEM): standard / homogenized, 1 mm | 2 mm |", "|---|---|---|---|---|---|"]
    for c in coh:
        o.append(f"| {c['roi']} | {c['winner_fem']} | {c['winner_sp1']} | {c['winner_sp1hom']} | {c['retains_fem_sp1_pct']:.1f} / {c['retains_fem_sp1hom_pct']:.1f}% | {c['retains_fem_sp2_pct']:.1f} / {c['retains_fem_sp2hom_pct']:.1f}% |")
    nt = S["network"]
    o += ["", "## Checks", "",
          f"- Published rows reproduced: {chk['n_compared']} FEM-vs-standard comparisons recomputed from the score files; {len(chk['mismatches'])} differ in winner, agreement or top-10 overlap; "
          f"largest difference {chk['max_abs_rho_diff']:.1e} in ρ and {chk['max_abs_regret_diff_pct']:.3f} points in regret.",
          f"- Score files re-verified by SHA-256 before collection: {len(heads_ok)} heads passed" + (f"; failed: {', '.join(sorted(bad))}" if bad else "") + ".",
          f"- Homogenized network size: 1 mm median {nt['sp1hom']['nodes_median']/1e6:.2f} M nodes, 2 mm {nt['sp2hom']['nodes_median']/1e6:.2f} M; nodes outside the giant component dropped: median {nt['sp1hom']['dropped_median']:.0f} (max {nt['sp1hom']['dropped_max']}) at 1 mm, "
          f"{nt['sp2hom']['dropped_median']:.0f} (max {nt['sp2hom']['dropped_max']}) at 2 mm.",
          f"- Run time per head on one H200 (build, 18 solves, scoring): 1 mm median {nt['sp1hom']['t_total_median_s']:.0f} s, 2 mm {nt['sp2hom']['t_total_median_s']:.0f} s.",
          f"- Collection took {time.time()-t0:.0f} s."]
    open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8", newline="\n").write("\n".join(o) + "\n")
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M"), script="pipeline/nchc/collect_homog300.py", cluster_job_script="pipeline/nchc/homog300_nano4.sbatch",
                   experiment_code="pipeline/nchc/homog_experiment.py", sha256_experiment_code=sha256(os.path.join(HERE, "homog_experiment.py")),
                   inputs=dict(homogenized=HOM, standard_and_fem=HEADS, bundles=BUNDLES, published_table=os.path.relpath(PUB, ROOT)), heads=heads_ok),
              open(os.path.join(OUT, "PROVENANCE.json"), "w"), indent=1)
    print("\n".join(o))


if __name__ == "__main__":
    main()
