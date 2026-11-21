# -*- coding: utf-8 -*-
"""The 300-head full-space FEM comparison under v23, in the tables the Results record reads (30 Oct 2026).

Paper A's final results are v23 (PI, 26 Oct). The v23 networks of all 300 heads were scored on Nano 4 on 23 Oct
(homogenized network, 2 and 1 mm; collected by pipeline/nchc/collect_homog300.py next to the v22 models). The
package port reproduces those bases (results_v23/v23_package_check), so they are the v23 results. This script
writes them in the format of v22_fem_fullsearch.stage_collect(), with the v23 models in the SPICEhead columns:

    fem = SimNIBS leadfield (unchanged)    sp2 = v23 at 2 mm (sp2hom)    sp1 = v23 at 1 mm (sp1hom)

  fullsearch_decisions.csv, fullsearch_cohort.csv, fullsearch_summary.json, fullsearch_timing.csv
      same columns and formulas as stage_collect(), pairs fem-sp2, fem-sp1, sp2-sp1
  remesh_analysis.json
      the re-mesh check of 18 Oct (24 heads re-meshed and rerun in FEM) with the v23 1 mm choice; the FEM-only
      columns are recomputed too and checked against the v22 record
  checks.json
      every row's FEM-vs-v23 metrics against results_v22/v22_fem_fullsearch_300_homog/decisions.csv, and the
      score files' SHA-256 against the cluster manifests

Inputs: L:/BrainProbe/SPICEhead/nchc_fullsearch/{heads, homog300, remesh_check} (h-codes only; no case map read).
Output: results_v23/v23_fem_fullsearch_300/. Light (JSON reading); runs at below-normal priority on 4 CPUs.
    python pipeline/v23_fullsearch_collect.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "nchc"))
import collect_homog300 as CH                                   # noqa: E402  (score-file paths, SHA-256 manifests)

OUT = os.path.join(ROOT, "results_v23", "v23_fem_fullsearch_300")
HOMOG = os.path.join(ROOT, "results_v22", "v22_fem_fullsearch_300_homog", "decisions.csv")
REMESH = os.path.join(CH.L, "remesh_check")
REMESH_V22 = os.path.join(ROOT, "results_v22", "v22_fem_fullsearch_300", "remesh_analysis.json")
ROIS = CH.ROIS
SRC = {"fem": "fem", "sp2": "sp2hom", "sp1": "sp1hom"}         # record column -> score file
MODELS = list(SRC)
PAIRS = [("fem", "sp2"), ("fem", "sp1"), ("sp2", "sp1")]
# rows selected for the re-mesh check on 18 Oct (v22 1 mm kept < 90% of the FEM optimum); kept as selected
OUTL = {"h067": "insula", "h041": "insula", "h063": "insula", "h071": "insula", "h134": "M1_handknob",
        "h053": "M1_handknob", "h009": "M1_handknob", "h098": "M1_handknob"}


def g7(x):
    return float(f"{x:.7g}")


def low_footprint(cpus=4):
    """Below-normal priority and the first `cpus` logical CPUs (as run_v23_all.py --low-footprint)."""
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    h = k32.GetCurrentProcess()
    if not (k32.SetPriorityClass(h, 0x4000) and k32.SetProcessAffinityMask(h, (1 << cpus) - 1)):
        raise SystemExit(f"could not lower priority or affinity (Windows error {k32.GetLastError()})")


def load(code):
    return {m: json.load(open(CH.score_file(code, SRC[m]), encoding="utf-8")) for m in MODELS}


def head_rows(code, data):
    """stage_collect()'s per-head rows, from one head's three score files."""
    rows, Fh = [], {}
    for roi in ROIS:
        f = {m: {k: v[0] for k, v in data[m]["rois"][roi].items()} for m in MODELS}
        keys = sorted(set.intersection(*(set(x) for x in f.values())))
        # F, T and O of every montage, in key order (F is what the rows below use)
        Fh[roi] = {m: np.array([data[m]["rois"][roi][k][:3] for k in keys], np.float64).T for m in MODELS}
        win = {m: max(keys, key=lambda k: f[m][k]) for m in MODELS}
        row = dict(subject=code, roi=roi, n_montages=len(keys))
        for m in MODELS:
            row[f"winner_{m}"] = win[m]
            row[f"F_{m}"] = f[m][win[m]]
            row[f"T_{m}"] = data[m]["rois"][roi][win[m]][1]
            row[f"O_{m}"] = data[m]["rois"][roi][win[m]][2]
        for a, b in PAIRS:
            va, vb = Fh[roi][a][0], Fh[roi][b][0]
            row[f"agree_{a}_{b}"] = win[a] == win[b]
            row[f"rho_{a}_{b}"] = g7(float(spearmanr(va, vb).correlation))
            top = np.argsort(-va)[:100]
            row[f"rho_top100_{a}_{b}"] = g7(float(spearmanr(va[top], vb[top]).correlation))
            row[f"regret_{a}winner_under_{b}_pct"] = round(100 * (1 - f[b][win[a]] / f[b][win[b]]), 3)
            row[f"regret_{b}winner_under_{a}_pct"] = round(100 * (1 - f[a][win[b]] / f[a][win[a]]), 3)
            for n in (10, 25):
                ta = set(np.array(keys)[np.argsort(-va)[:n]])
                tb = set(np.array(keys)[np.argsort(-vb)[:n]])
                row[f"top{n}_overlap_{a}_{b}"] = len(ta & tb)
        rows.append(row)
    return rows, Fh, keys


def cohort(Fall, keys, n):
    """stage_collect()'s cohort table: median F per montage over the heads, then each model's winner
    (Fall[roi][m]: heads x (F, T, O) x montages)."""
    out = []
    for roi in ROIS:
        A = {m: np.stack(Fall[roi][m]) for m in MODELS}
        med = {m: np.median(A[m][:, 0], axis=0) for m in MODELS}
        win = {m: int(med[m].argmax()) for m in MODELS}
        r = dict(roi=roi, n_heads=n)
        for m in MODELS:
            r[f"winner_{m}"] = keys[win[m]]
            r[f"F_{m}"] = round(float(med[m][win[m]]), 4)
            r[f"T_{m}"] = round(float(np.median(A[m][:, 1, win[m]])), 5)
            r[f"O_{m}"] = round(float(np.median(A[m][:, 2, win[m]])), 5)
        for a, b in PAIRS:
            r[f"agree_{a}_{b}"] = win[a] == win[b]
            r[f"regret_{a}winner_under_{b}_pct"] = round(100 * (1 - med[b][win[a]] / med[b][win[b]]), 3)
            r[f"regret_{b}winner_under_{a}_pct"] = round(100 * (1 - med[a][win[b]] / med[a][win[a]]), 3)
            r[f"rho_{a}_{b}"] = g7(float(spearmanr(med[a], med[b]).correlation))
        out.append(r)
    return out


def check_homog(rows):
    """Every FEM-vs-v23 row against the 23 Oct collection (sp2hom/sp1hom columns)."""
    ref = {(r["subject"], r["roi"]): r for r in csv.DictReader(open(HOMOG, encoding="utf-8"))}
    n, mism, wr, wg = 0, [], 0.0, 0.0
    for r in rows:
        p = ref[(r["subject"], r["roi"])]
        for m in ("sp2", "sp1"):
            h = SRC[m]
            n += 1
            if (r[f"winner_{m}"] != p[f"winner_{h}"] or str(r[f"agree_fem_{m}"]) != p[f"agree_fem_{h}"]
                    or r[f"top10_overlap_fem_{m}"] != int(p[f"top10_overlap_fem_{h}"])):
                mism.append([r["subject"], r["roi"], m])
            wr = max(wr, abs(r[f"rho_fem_{m}"] - float(p[f"rho_fem_{h}"])))
            wg = max(wg, abs(r[f"regret_{m}winner_under_fem_pct"] - float(p[f"regret_{h}winner_under_fem_pct"])))
    return dict(n_compared=n, mismatches=mism, max_abs_rho_diff=wr, max_abs_regret_diff_pct=wg)


def remesh():
    """The re-mesh check with the v23 1 mm choice. Old FEM = the published 300-head FEM; new FEM = the re-meshed run."""
    roles = list(csv.DictReader(open(os.path.join(REMESH, "roles.csv"), encoding="utf-8")))
    out = []
    for r in roles:
        c = r["code"]
        fo = json.load(open(CH.score_file(c, "fem"), encoding="utf-8"))["rois"]
        fn = json.load(open(os.path.join(REMESH, "results", f"{c}__fem.json"), encoding="utf-8"))["rois"]
        s1 = json.load(open(CH.score_file(c, SRC["sp1"]), encoding="utf-8"))["rois"]
        for roi in ROIS:
            keys = sorted(fo[roi])
            Fo = np.array([fo[roi][k][0] for k in keys]); Fn = np.array([fn[roi][k][0] for k in keys])
            S1 = np.array([s1[roi][k][0] for k in keys])
            wo, wn, ws = int(Fo.argmax()), int(Fn.argmax()), int(S1.argmax())
            out.append(dict(code=c, role=r["role"], roi=roi, outlier_row=(OUTL.get(c) == roi),
                            fem_same_winner=wo == wn, rho_fem_old_new=float(spearmanr(Fo, Fn).correlation),
                            oldwin_keeps_newfem=100 * Fn[wo] / Fn[wn], Fmax_change=100 * (Fn[wn] / Fo[wo] - 1),
                            sp1_keeps_oldfem=100 * Fo[ws] / Fo[wo], sp1_keeps_newfem=100 * Fn[ws] / Fn[wn],
                            sp1_rerun_maxrel=None))       # the v22 rerun checked the v22 network; not repeated
    # the FEM-only columns must equal the v22 record's (same FEM files)
    v22 = {(x["code"], x["roi"]): x for x in json.load(open(REMESH_V22, encoding="utf-8"))}
    dev = max(max(abs(x[k] - v22[(x["code"], x["roi"])][k]) for k in ("rho_fem_old_new", "oldwin_keeps_newfem", "Fmax_change"))
              for x in out)
    same = all(x["fem_same_winner"] == v22[(x["code"], x["roi"])]["fem_same_winner"] for x in out)
    return out, dict(n_rows=len(out), fem_only_columns_max_abs_diff_vs_v22=dev, fem_same_winner_identical=same)


def main():
    if os.name == "nt":
        low_footprint()
    t0 = time.time()
    codes = sorted(f[:-7] for f in os.listdir(CH.HOM) if f.endswith(".sha256"))
    if len(codes) != 300:
        sys.exit(f"{len(codes)} heads in {CH.HOM}; expected 300")
    bad = {}
    for c in codes:
        b = [m for m in CH.verify(c) if m in SRC.values()]
        if b:
            bad[c] = b
    if bad:
        sys.exit(f"SHA-256 mismatch: {bad}")
    rows, trow, keys0 = [], [], None
    Fall = {roi: {m: [] for m in MODELS} for roi in ROIS}
    for i, c in enumerate(codes):
        data = load(c)
        r, Fh, keys = head_rows(c, data)
        if keys0 is None:
            keys0 = keys
        assert keys == keys0, c
        rows += r
        for roi in ROIS:
            for m in MODELS:
                Fall[roi][m].append(Fh[roi][m])
        for m in MODELS:
            d = data[m]
            trow.append(dict(subject=c, model=m, dx_mm=d.get("dx_mm"), n_nodes=d.get("n_nodes", d.get("n_nodes_mesh")),
                             n_elements=d.get("n_elements"), t_build_s=d.get("t_build_s"), t_basis_s=d.get("t_basis_s"),
                             t_map_s=d.get("t_map_s"), t_score_s=d.get("t_score_s"), t_total_s=d.get("t_total_s"),
                             n_components=d.get("n_components"), dropped_nodes=d.get("dropped_nodes")))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/300 heads, {time.time() - t0:.0f} s", flush=True)
    os.makedirs(OUT, exist_ok=True)

    def write(name, rs):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
    write("fullsearch_decisions.csv", rows)
    coh = cohort(Fall, keys0, len(codes))
    write("fullsearch_cohort.csv", coh)
    # cohort winners and the share of FEM's cohort optimum kept, against the 23 Oct collection
    ref = {r["roi"]: r for r in csv.DictReader(open(os.path.join(os.path.dirname(HOMOG), "cohort.csv"), encoding="utf-8"))}
    coh_chk = dict(winners_identical=all(r[f"winner_{m}"] == ref[r["roi"]][f"winner_{SRC[m]}"] for r in coh for m in MODELS),
                   max_abs_retention_diff_pct=max(abs(100 - r[f"regret_{m}winner_under_fem_pct"]
                                                      - float(ref[r["roi"]][f"retains_fem_{SRC[m]}_pct"]))
                                                  for r in coh for m in ("sp2", "sp1")))
    write("fullsearch_timing.csv", trow)
    summ = dict(heads_complete=codes, n_rows=len(rows), model="v23",
                columns={m: f"{SRC[m]} score files" for m in MODELS})
    for a, b in PAIRS:
        summ[f"agree_{a}_{b}"] = f"{sum(r[f'agree_{a}_{b}'] for r in rows)}/{len(rows)}"
        for roi in ROIS:
            sub = [r for r in rows if r["roi"] == roi]
            summ[f"agree_{a}_{b}.{roi}"] = f"{sum(r[f'agree_{a}_{b}'] for r in sub)}/{len(sub)}"
    summ["machine"] = "NCHC Nano 4, one H200 GPU per head (SPICEhead, GPU solver); FEM bases as published (v22 record)"
    json.dump(summ, open(os.path.join(OUT, "fullsearch_summary.json"), "w", encoding="utf-8"), indent=1)
    rm, rm_chk = remesh()
    json.dump(rm, open(os.path.join(OUT, "remesh_analysis.json"), "w", encoding="utf-8"), indent=1)
    chk = dict(score_files_sha256="300 heads x (fem, sp2hom, sp1hom) verified against the cluster manifests",
               vs_homog300_collection=check_homog(rows), cohort_vs_homog300_collection=coh_chk, remesh=rm_chk)
    json.dump(chk, open(os.path.join(OUT, "checks.json"), "w", encoding="utf-8"), indent=1)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M"), script="pipeline/v23_fullsearch_collect.py",
                   inputs=dict(fem=os.path.join(CH.HEADS, "<code>", "<code>__fem.json"),
                               v23=os.path.join(CH.HOM, "<code>__sp2hom.json / __sp1hom.json"),
                               remesh=REMESH, reference=os.path.relpath(HOMOG, ROOT)),
                   models=SRC, heads=codes), open(os.path.join(OUT, "PROVENANCE.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k.startswith("agree") and "." not in k}, indent=1))
    print(json.dumps(chk, indent=1)[:1500])
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
