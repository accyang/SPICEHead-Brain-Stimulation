# -*- coding: utf-8 -*-
"""Collect the 300-head NCHC full-space comparison (FEM vs SPICEhead 2 mm and 1 mm, 11,628 montages).

Written 18 Oct 2026 at the PI's request ("collect all 300 heads when done").

Input: the per-head score JSONs that pull_outputs.sh copied to L: and verified, under
L:/BrainProbe/SPICEhead/nchc_fullsearch/heads/<code>/. Each was computed on NCHC Nano 4 (H200):
  - SPICEhead bases with the GPU solver, rtol 1e-10;
  - FEM by the SimNIBS 4.6 19-pad leadfield on a cluster mesh;
  - scoring with the GPU kernel, within 1e-6 of the CPU scorer.

Steps:
 1. Re-check each head's three JSONs against its MANIFEST.sha256.
 2. Hard-link them into one flat staging folder on L: (no copies).
 3. Run v22_fem_fullsearch.stage_collect() unchanged on that folder.
 4. Copy the tables to results_v22/v22_fem_fullsearch_300/ with PROVENANCE.json and SUMMARY.md.

The 8-head desktop record in results_v22/v22_fem_fullsearch/ is read for a consistency check and is
never written. Per-head rows carry only the h-codes. head_codes.csv is used in memory to find the 8
desktop heads, and no file written here pairs an h-code with a CaseID.

    python pipeline/nchc/collect_300.py                         # collect on this machine
    python pipeline/nchc/collect_300.py --collected-on <job>    # tables already made by the same
                                                                # stage_collect on Nano 4, copied into the
                                                                # staging folder (needs ~20 GB RAM, which
                                                                # the desktop lacked on 18 Oct)
With --collected-on, the cluster's 900 score files must match the L: manifests. That check uses
L:/.../nchc_fullsearch/json_sha_cluster.txt (cluster sha256sum output).
"""
import csv
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L = r"L:\BrainProbe\SPICEhead\nchc_fullsearch"
HEADS = os.path.join(L, "heads")
STAGE = os.path.join(L, "collect300")
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"          # private; read only
OUT = os.path.join(ROOT, "results_v22", "v22_fem_fullsearch_300")
REF8 = os.path.join(ROOT, "results_v22", "v22_fem_fullsearch", "fullsearch_decisions.csv")
MODELS = ("fem", "sp2", "sp1")
ROIS = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def stage_inputs():
    codes = sorted(d for d in os.listdir(HEADS) if d.startswith("h") and not d.endswith(".part"))
    os.makedirs(STAGE, exist_ok=True)
    bad = []
    for c in codes:
        d = os.path.join(HEADS, c)
        man = dict(reversed(l.split(None, 1)) for l in open(os.path.join(d, "MANIFEST.sha256")).read().split("\n") if l.strip())
        man = {k.strip(): v for k, v in man.items()}
        for m in MODELS:
            f = f"{c}__{m}.json"
            if not os.path.exists(os.path.join(d, ".verified")) or sha256(os.path.join(d, f)) != man.get(f):
                bad.append(f)
                continue
            dst = os.path.join(STAGE, f)
            if os.path.exists(dst) and os.path.samefile(dst, os.path.join(d, f)):
                continue
            if os.path.exists(dst):
                os.remove(dst)
            os.link(os.path.join(d, f), dst)
    if bad:
        raise SystemExit(f"{len(bad)} score files fail their manifest: {bad[:5]}")
    with open(os.path.join(STAGE, "heads300.csv"), "w", newline="", encoding="utf-8") as fh:
        fh.write("CaseID\n" + "\n".join(codes) + "\n")
    return codes


def run_collect():
    os.environ["SPICEHEAD_MODEL"] = "v22"
    os.environ["SPICEHEAD_FS_HEADS"] = os.path.join(STAGE, "heads300.csv")
    os.environ["SPICEHEAD_FULLSEARCH_OUT"] = STAGE
    sys.path.insert(0, os.path.join(ROOT, "pipeline"))
    sys.path.insert(0, ROOT)
    import v22_fem_fullsearch as fs
    fs.stage_collect()


def num(r, k):
    return float(r[k])


def q(a, p):
    return float(np.percentile(np.asarray(a, float), p))


def summarize(codes):
    rows = list(csv.DictReader(open(os.path.join(OUT, "fullsearch_decisions.csv"), encoding="utf-8")))
    coh = list(csv.DictReader(open(os.path.join(OUT, "fullsearch_cohort.csv"), encoding="utf-8")))
    tim = list(csv.DictReader(open(os.path.join(OUT, "fullsearch_timing.csv"), encoding="utf-8")))
    nh = len({r["subject"] for r in rows})
    L_ = []
    L_.append(f"# FEM vs SPICEhead over all 11,628 montages: {nh} heads (NCHC Nano 4)\n")
    L_.append(f"Collected {time.strftime('%Y-%m-%d %H:%M')} by `pipeline/nchc/collect_300.py` from the verified per-head "
              "score files on L:. The tables beside this file come from `v22_fem_fullsearch.stage_collect()`, "
              "unchanged. This is a new results set: it is not in any Results record or manuscript yet. "
              "v22 model (skin-surface contacts, harmonic interfaces).\n")
    L_.append("Per-head regret: \"SP winner under FEM\" = % of the FEM optimum lost if the SPICEhead winner were "
              "used and judged by FEM. Retention = 100 − regret.\n")

    def block(a, b):
        L_.append(f"\n## {a.upper()} vs {b.upper()} (per head × target)\n")
        L_.append("| Target | Same winner | Spearman ρ median [min] | Retention of the "
                  f"{a.upper()} optimum by the {b.upper()} winner: median / 5th pct / min | Top-10 overlap median |")
        L_.append("|---|---|---|---|---|")
        for roi in ROIS + ["all"]:
            sub = [r for r in rows if roi == "all" or r["roi"] == roi]
            ag = sum(r[f"agree_{a}_{b}"] == "True" for r in sub)
            rho = [num(r, f"rho_{a}_{b}") for r in sub]
            ret = [100 - num(r, f"regret_{b}winner_under_{a}_pct") for r in sub]
            t10 = [num(r, f"top10_overlap_{a}_{b}") for r in sub]
            L_.append(f"| {roi} | {ag}/{len(sub)} ({100*ag/len(sub):.0f}%) | {np.median(rho):.3f} [{min(rho):.3f}] | "
                      f"{np.median(ret):.1f} / {q(ret, 5):.1f} / {min(ret):.1f}% | {np.median(t10):.0f} |")
    block("fem", "sp1")
    block("fem", "sp2")
    block("sp2", "sp1")

    L_.append("\n## Cohort level (median F per montage across heads)\n")
    L_.append("| Target | Winner FEM | Winner SP1 | Winner SP2 | SP1 = FEM | SP1 winner retains (FEM) | SP2 winner retains (FEM) |")
    L_.append("|---|---|---|---|---|---|---|")
    for r in coh:
        L_.append(f"| {r['roi']} | {r['winner_fem']} | {r['winner_sp1']} | {r['winner_sp2']} | {r['agree_fem_sp1']} | "
                  f"{100 - float(r['regret_sp1winner_under_fem_pct']):.1f}% | {100 - float(r['regret_sp2winner_under_fem_pct']):.1f}% |")

    # numerical checks carried in the score files
    bc = [float(t["basis_check_max"]) for t in tim if t.get("basis_check_max") not in (None, "", "None")]
    L_.append("\n## Checks\n")
    L_.append(f"- Score files: {3*len(codes)} JSONs re-verified against the per-head SHA-256 manifests before collection.")
    if bc:
        L_.append(f"- Basis self-check (max over heads and models): {max(bc):.1e}.")
    L_.append(f"- Score files per model: " + ", ".join(f"{m} {sum(1 for t in tim if t['model'] == m)}" for m in MODELS) + ".")
    return L_


def compare_ref8(L_):
    """The 8 desktop FEM heads (results_v22/v22_fem_fullsearch, desktop meshes, CPU solver/scoring) against
    the same heads in this run (cluster meshes, GPU solver/scoring). Aggregate only: no per-head identifiers."""
    if not (os.path.exists(REF8) and os.path.exists(CODEMAP)):
        L_.append("\n(8-head desktop comparison skipped: reference or code map not found.)")
        return
    cmap = {r["CaseID"]: r["code"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    ref = list(csv.DictReader(open(REF8, encoding="utf-8")))
    new = {(r["subject"], r["roi"]): r for r in csv.DictReader(open(os.path.join(OUT, "fullsearch_decisions.csv"), encoding="utf-8"))}
    pairs = [(r, new.get((cmap.get(r["subject"]), r["roi"]))) for r in ref]
    pairs = [(a, b) for a, b in pairs if b is not None]
    nhead = len({a["subject"] for a, _ in pairs})
    L_.append(f"\n## Consistency with the 8-head desktop record ({nhead} heads, {len(pairs)} head × target rows)\n")
    L_.append("Desktop: desktop meshes, CPU solver and scoring. Cluster: new meshes, GPU solver and scoring. "
              "The SPICEhead models use the same voxel data in both, so they should agree almost exactly. "
              "FEM differences include re-meshing.\n")
    L_.append("| Model | Same winner | Median relative change in the winner's F | Max |")
    L_.append("|---|---|---|---|")
    c8 = dict(n_heads=nhead, n_rows=len(pairs), note="eight desktop FEM heads: desktop meshes/CPU vs cluster meshes/GPU; aggregates only")
    for m in MODELS:
        same = sum(a[f"winner_{m}"] == b[f"winner_{m}"] for a, b in pairs)
        d = [abs(float(b[f"F_{m}"]) - float(a[f"F_{m}"])) / float(a[f"F_{m}"]) for a, b in pairs]
        L_.append(f"| {m.upper()} | {same}/{len(pairs)} | {np.median(d):.1e} | {max(d):.1e} |")
        c8[m] = dict(same_winner=int(same), n=len(pairs), rel_change_F_median=float(np.median(d)), rel_change_F_max=float(max(d)))
    json.dump(c8, open(os.path.join(OUT, "consistency_vs_8head.json"), "w", encoding="utf-8"), indent=1)
    for a_, b_ in (("fem", "sp1"), ("fem", "sp2")):
        da = [float(a[f"rho_{a_}_{b_}"]) for a, _ in pairs]
        db = [float(b[f"rho_{a_}_{b_}"]) for _, b in pairs]
        L_.append(f"\nρ({a_.upper()}, {b_.upper()}) median: desktop {np.median(da):.3f}, cluster {np.median(db):.3f}.")


def cluster_inputs_match():
    """The cluster's score files (input of a collect run there) equal the verified L: copies."""
    cl = os.path.join(L, "json_sha_cluster.txt")
    man = set()
    for c in os.listdir(HEADS):
        if c.startswith("h") and not c.endswith(".part"):
            for l in open(os.path.join(HEADS, c, "MANIFEST.sha256")):
                if l.strip().endswith(".json"):
                    man.add(tuple(l.split()))
    got = {tuple(l.split()) for l in open(cl) if l.strip()}
    got = {(h, os.path.basename(f)) for h, f in got}
    if man != got or len(man) != 900:
        raise SystemExit(f"cluster score files differ from the L: manifests: {len(man ^ got)} entries")
    return len(man)


# meshes made in the first production run on /work (17 Oct 20:35-22:24, the run that filled the disk);
# all other heads were meshed later in node-local scratch
EARLY_MESH = set(("h001 h031 h032 h033 h034 h035 h036 h037 h038 h039 h040 h041 h042 h043 h044 h045 h046 h047 h048 "
                  "h049 h050 h051 h052 h053 h054 h055 h056 h057 h058 h059 h060 h062 h063 h065 h067 h068 h069 h070 "
                  "h071 h072 h073 h075 h076 h077 h078 h081 h084 h085 h089 h091 h092 h093 h096 h097 h098 h101 h102 "
                  "h103 h104 h105 h106 h109 h110 h112 h115 h116 h117 h118 h119 h120 h121 h122 h124 h125 h126").split())


def outliers(L_):
    """Rows where the SP1 winner keeps < 90 % of the FEM optimum, and aggregates by mesh batch and site
    (site from the private code map, aggregated only)."""
    rows = list(csv.DictReader(open(os.path.join(OUT, "fullsearch_decisions.csv"), encoding="utf-8")))
    ret = lambda r: 100 - float(r["regret_sp1winner_under_fem_pct"])
    low = sorted((r for r in rows if ret(r) < 90), key=ret)
    L_.append(f"\n## Outliers: SP1 winner keeps < 90 % of the FEM optimum ({len(low)} of {len(rows)} rows)\n")
    L_.append("| Head | Target | Retention | ρ(FEM,SP1) | FEM winner | SP1 winner | SP2 winner keeps | Mesh batch |")
    L_.append("|---|---|---|---|---|---|---|---|")
    for r in low:
        L_.append(f"| {r['subject']} | {r['roi']} | {ret(r):.1f}% | {float(r['rho_fem_sp1']):.3f} | {r['winner_fem']} | "
                  f"{r['winner_sp1']} | {100 - float(r['regret_sp2winner_under_fem_pct']):.1f}% | "
                  f"{'first run' if r['subject'] in EARLY_MESH else 'scratch'} |")
    groups = [("mesh batch", lambda c: "first run on /work (75 heads)" if c in EARLY_MESH else "node-local scratch (225 heads)")]
    if os.path.exists(CODEMAP):
        cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
        site = lambda c: ("TAMI" if cmap[c].startswith("TAMI") else "BT-TSGH" if cmap[c].startswith("BT_")
                          else "TVGH-Bai" if cmap[c].startswith("NC") else "other")
        groups.append(("site", site))
    L_.append("\n| Group | Heads | Rows < 90 % | Rows < 95 % | ρ(FEM,SP1) median | Same winner |")
    L_.append("|---|---|---|---|---|---|")
    for name, key in groups:
        g = {}
        for r in rows:
            g.setdefault(key(r["subject"]), []).append(r)
        for k, rs in sorted(g.items()):
            rt = np.array([ret(r) for r in rs])
            L_.append(f"| {name}: {k} | {len({r['subject'] for r in rs})} | {int((rt < 90).sum())} | "
                      f"{int((rt < 95).sum())} ({100 * (rt < 95).mean():.1f} %) | "
                      f"{np.median([float(r['rho_fem_sp1']) for r in rs]):.4f} | "
                      f"{100 * np.mean([r['agree_fem_sp1'] == 'True' for r in rs]):.0f} % |")
    L_.append("\nThe bulk does not differ by mesh batch, but 6 of the 8 rows below 90 % come from first-run meshes. "
              "That batch over-represents BT-TSGH and TVGH-Bai heads, the two sites with the most outliers, "
              "so batch and site are confounded. Re-mesh check (18 Oct, REMESH_CHECK_2026-10-18.md): the 8 heads and 16 "
              "matched controls were re-meshed. The outliers persist with new meshes (insula 36-51 %, M1 88-90 %), while "
              "re-meshing moved control retention by at most 1.3 points. These are genuine FEM-vs-SPICEhead "
              "disagreements, not mesh defects.")


def main():
    t0 = time.time()
    codes = stage_inputs()
    print(f"{len(codes)} heads staged and verified ({time.time()-t0:.0f} s)")
    where = "this desktop"
    if "--collected-on" in sys.argv:
        where = f"NCHC Nano 4, Slurm job {sys.argv[sys.argv.index('--collected-on') + 1]} (dev partition)"
        print(f"cluster inputs = L: copies: {cluster_inputs_match()} files")
    else:
        run_collect()
    os.makedirs(OUT, exist_ok=True)
    for f in ("fullsearch_decisions.csv", "fullsearch_cohort.csv", "fullsearch_timing.csv", "fullsearch_summary.json"):
        shutil.copy2(os.path.join(STAGE, f), os.path.join(OUT, f))
    prov = dict(
        collected=time.strftime("%Y-%m-%d %H:%M:%S"), n_heads=len(codes), model="v22",
        compute="NCHC Nano 4 (晶創26), 8x H200 nodes, project <project>, 17-18 Oct 2026; Slurm arrays "
                "562922, 563112, 563149, 563178, 563599, 563676, 563724, 563761",
        solver="SPICEhead bases: GPU Jacobi-PCG (spicehead/solve_gpu.py), float64, rtol 1e-10",
        fem="SimNIBS 4.6.0 TDCS leadfield, 19 pads, reference Cz, cluster meshes (charm --mesh)",
        scoring="GPU kernel spicehead/score_gpu.py (validated within 1.0e-6 of the CPU scorer, identical winners)",
        inputs=r"L:\BrainProbe\SPICEhead\nchc_fullsearch\heads\<code>\ (verified copies)",
        collect_ran_on=where,
        note="fullsearch_summary.json 'machine' describes the machine that ran --collect, not the compute nodes; "
             "head codes h001-h300 are pseudonyms, the key stays in the lab's private staging folder")
    json.dump(prov, open(os.path.join(OUT, "PROVENANCE.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    L_ = summarize(codes)
    outliers(L_)
    compare_ref8(L_)
    open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(L_) + "\n")
    print("wrote", OUT, f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
