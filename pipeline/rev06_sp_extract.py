# -*- coding: utf-8 -*-
"""A2 of Rev0.5/10_ANALYSIS_REQUEST_v23_2026-11-02.md: gather the run files of nchc/uq_v23_sp_protocol.py into one array per head.

Read-only on the run files. Hashes each file against SHA256SUMS_sp.txt, written on the cluster before the transfer.
Output: C:/spicehead_v23_logs/rev06_extract_sp/<code>.npz (sp, fem: run x target x F, T, O as float32) and <code>.json.
    python pipeline/rev06_sp_extract.py
"""
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rev06_uq_extract import RUNS, BASE                               # noqa: E402

SRC = os.path.join(BASE, "runs_sp_protocol")
SUMS = os.path.join(BASE, "SHA256SUMS_sp.txt")
OUT = r"C:\spicehead_v23_logs\rev06_extract_sp"
PROTO = os.path.join(os.path.dirname(HERE), "results_v23", "v23_uq_recollect_2026-11-03", "uq_sp_protocols_28.json")


def one(job):
    code, rec, names, sp_lab, fem_lab = job
    out = os.path.join(OUT, f"{code}.npz")
    if os.path.exists(out):
        return code, 0
    nr, nt = len(RUNS), len(names)
    A = dict(sp=np.full((nr, nt, 3), np.nan, np.float32), fem=np.full((nr, nt, 3), np.nan, np.float32), n_roi_masked=np.full((nr, nt), -1, np.int32))
    bad, t_solve, t_run, nodes = [], [], [], []
    for i, r in enumerate(RUNS):
        rel = f"runs_sp_protocol/{code}/{r}.npz"
        p = os.path.join(SRC, code, f"{r}.npz")
        if not os.path.exists(p):
            bad.append(f"{rel}: missing")
            continue
        b = open(p, "rb").read()
        if rec.get(rel) != hashlib.sha256(b).hexdigest():
            bad.append(f"{rel}: checksum")
        with np.load(io.BytesIO(b)) as z:
            if [str(x) for x in z["targets"]] != names or [str(x) for x in z["sp_labels"]] != sp_lab or [str(x) for x in z["fem_labels"]] != fem_lab or [str(x) for x in z["cols"]] != ["F", "T", "O"]:
                bad.append(f"{rel}: labels")
            A["sp"][i], A["fem"][i] = z["sp"], z["fem"]
            prm = json.loads(str(z["params"]))
        if prm.get("code") != code or prm.get("run") != r:
            bad.append(f"{rel}: params")
        if "n_roi_masked" in prm:
            A["n_roi_masked"][i] = prm["n_roi_masked"]
        t_solve.append(prm.get("t_solve_s")); t_run.append(prm.get("t_run_s")); nodes.append(prm.get("n_nodes"))
    if not os.path.exists(os.path.join(SRC, code, "DONE")):
        bad.append(f"{code}: no DONE")
    tmp = out + ".tmp.npz"
    np.savez(tmp, **A, targets=np.array(names))
    os.replace(tmp, out)
    json.dump(dict(code=code, bad=bad, t_solve_s=t_solve, t_run_s=t_run, n_nodes=sorted(set(nodes))), open(os.path.join(OUT, f"{code}.json"), "w", encoding="utf-8"))
    return code, len(bad)


def main():
    os.makedirs(OUT, exist_ok=True)
    rec = {}
    for line in open(SUMS, encoding="utf-8"):
        s, _, rel = line.strip().partition("  ")
        rec[rel] = s
    assert len(rec) == 23400, len(rec)                                 # 300 x 77 run files + 300 DONE markers
    pr = json.load(open(PROTO, encoding="utf-8"))["targets"]
    names = sorted(pr)
    sp_lab, fem_lab = [pr[n]["spicehead"] for n in names], [pr[n]["fem"] for n in names]
    codes = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    assert len(codes) == 300
    jobs = [(c, {k: v for k, v in rec.items() if k.startswith(f"runs_sp_protocol/{c}/")}, names, sp_lab, fem_lab) for c in codes]
    nbad = 0
    with mp.Pool(int(os.environ.get("REV06_WORKERS", "3"))) as p:
        for k, (c, b) in enumerate(p.imap_unordered(one, jobs), 1):
            nbad += b
            if k % 50 == 0:
                print(k, "heads", nbad, "problems", flush=True)
    print("heads", len(codes), "problems", nbad)


if __name__ == "__main__":
    main()
