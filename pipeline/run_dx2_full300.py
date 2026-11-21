"""Full 2 mm montage sweep for all 300 cohort heads (launched 17 October 2026).

Extends the 20-head Table 7 sweep (v21_dx2_cohort_check.py --score-full) to the whole cohort, under
the v22 model. Every head is scored on all 11,628 two-pair montages at 2 mm from 18 basis solves;
the 20 heads already in results_v22/v21_dx2_check_full/ are reused (the stage skips cached JSONs).

  python pipeline/run_dx2_full300.py            launch NW detached shards (below-normal priority)
  python pipeline/run_dx2_full300.py --status   count finished heads, show the tail of each log
  python pipeline/run_dx2_full300.py --collect  cohort medians and 2 mm optima over all scored heads

Logs: C:\\spicehead_v22_logs\\dx2_full300_s<k>.log. Outputs: results_v22/v21_dx2_check_full/<head>__dx2_full.json
(one per head) and, after --collect, results_v22/v21_dx2_check_full/dx2_full300_*.{csv,json}.
"""
import csv, glob, json, os, subprocess, sys, time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = r"C:\Users\ALBERT YANG\AppData\Local\Programs\Python\Python312\python.exe"
CSV = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"
LOGDIR = r"C:\spicehead_v22_logs"
OUT = os.path.join(PROJ, "results_v22", "v21_dx2_check_full")
NW = 4
ENV = {"SPICEHEAD_MODEL": "v22", "SPICEHEAD_ENVELOPE": "maxdir", "PYTHONHASHSEED": "0",
       "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "SPICEHEAD_T7_SUBJECTS": CSV}          # pinned list = the whole cohort, in index order
BELOW_NORMAL = 0x4000
DETACHED = 0x00000008 | 0x00000200             # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def heads():
    return [r["CaseID"] for r in csv.DictReader(open(CSV, encoding="utf-8-sig"))]


def done():
    return [h for h in heads() if os.path.exists(os.path.join(OUT, f"{h}__dx2_full.json"))]


def launch():
    os.makedirs(LOGDIR, exist_ok=True)
    env = dict(os.environ, **ENV)
    pids = {}
    for k in range(NW):
        log = open(os.path.join(LOGDIR, f"dx2_full300_s{k}.log"), "a", encoding="utf-8")
        log.write(f"\n===== launched {time.strftime('%Y-%m-%d %H:%M:%S')} shard {k}/{NW} =====\n"); log.flush()
        p = subprocess.Popen([PY, os.path.join("pipeline", "v21_dx2_cohort_check.py"), "--score-full", "0",
                              "--shard", f"{k}/{NW}"],
                             cwd=PROJ, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             creationflags=BELOW_NORMAL | DETACHED)
        pids[k] = p.pid
    json.dump({"launched": time.strftime("%Y-%m-%d %H:%M:%S"), "pids": pids, "n_heads": len(heads()),
               "already_done": len(done())}, open(os.path.join(LOGDIR, "dx2_full300_pids.json"), "w"), indent=1)
    print("launched", pids, "| heads", len(heads()), "| already done", len(done()))


def status():
    d = done(); print(f"{len(d)}/{len(heads())} heads scored at 2 mm")
    for k in range(NW):
        p = os.path.join(LOGDIR, f"dx2_full300_s{k}.log")
        if os.path.exists(p):
            tail = [l.rstrip() for l in open(p, encoding="utf-8", errors="replace").read().splitlines() if l.strip()][-2:]
            print(f"  shard {k}:", " | ".join(tail))


def collect():
    sys.path.insert(0, os.path.join(PROJ, "pipeline"))
    import numpy as np
    files = sorted(os.path.join(OUT, f) for f in os.listdir(OUT) if f.endswith("__dx2_full.json"))   # not glob: "[New]" in the path is a glob class
    site = {r["CaseID"]: r["Site"] for r in csv.DictReader(open(CSV, encoding="utf-8-sig"))}
    data = [json.load(open(f, encoding="utf-8")) for f in files]
    ids = [d["subject"] for d in data]
    rois = list(data[0]["rois"].keys())
    out_rows, optima = [], {}
    for roi in rois:
        # per head: rois[roi] = {montage label: [F, T, O]} (v21_dx2_cohort_check.score_subject_full)
        labels = sorted({m for d in data for m in d["rois"].get(roi, {})})
        col = {m: j for j, m in enumerate(labels)}
        F = np.full((len(ids), len(labels)), np.nan, np.float32); T = F.copy(); O = F.copy()
        for s, d in enumerate(data):
            for m, (f, t, o) in d["rois"].get(roi, {}).items():
                j = col[m]; F[s, j], T[s, j], O[s, j] = f, t, o
        medF = np.nanmedian(F, 0); j = int(np.nanargmax(medF))
        optima[roi] = {"montage": labels[j], "median_F": float(medF[j]), "median_T": float(np.nanmedian(T[:, j])),
                       "median_O": float(np.nanmedian(O[:, j])), "n_heads": int(np.isfinite(F[:, j]).sum())}
        for rank, jj in enumerate(np.argsort(-medF)[:50], 1):
            out_rows.append({"roi": roi, "rank": rank, "montage": labels[jj], "median_F": float(medF[jj]),
                             "median_T": float(np.nanmedian(T[:, jj])), "median_O": float(np.nanmedian(O[:, jj]))})
    with open(os.path.join(OUT, "dx2_full300_top50.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    json.dump({"n_heads": len(ids), "heads": ids, "sites": {s: sum(1 for i in ids if site.get(i) == s) for s in set(site.values())},
               "optima": optima, "collected": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(OUT, "dx2_full300_summary.json"), "w", encoding="utf-8"), indent=1)
    for roi, o in optima.items():
        print(f"{roi:<16} {o['montage']:<24} F {o['median_F']:.3f}  T {o['median_T']:.4f}  O {o['median_O']:.4f}  (n={o['n_heads']})")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    {"--status": status, "--collect": collect}.get(arg, launch)()
