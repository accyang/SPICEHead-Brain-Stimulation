# -*- coding: utf-8 -*-
"""A0 of Rev0.5/10_ANALYSIS_REQUEST_v23_2026-11-02.md: read-only inventory and score extraction from the v23 uncertainty archive.

Every run file of both target sets is read once from L:, hashed, checked against the checksums recorded at fetch time and
parsed. The scores the later analyses need go to one compact file per head outside Dropbox. Nothing on L: is written, and no
field is solved. Montages are resolved by electrode label through the archived 11,628-label list, never by assumed position.
    python pipeline/rev06_uq_extract.py [--workers 3] [--heads h001,h002]
Output: C:/spicehead_v23_logs/rev06_extract/<code>.npz and <code>.json (h-codes only).
"""
import csv
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\uq_v23"
EXTMONT = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\atlas_ext\out\h001__ext.npz"
OUT = r"C:\spicehead_v23_logs\rev06_extract"
PARAMS = os.path.join(ROOT, "results_v23", "v23_atlas_parameters", "target_parameters.csv")
FIXEXT = os.path.join(ROOT, "results_v23", "v23_uq", "uq_atlas_fixed_montages_ext.json")
RUNS = (["nominal"] + [f"cond_{k:02d}" for k in range(40)] + [f"jit_{k:02d}" for k in range(20)] + [f"rot_{k:02d}" for k in range(10)]
        + ["seg_skull_plus", "seg_skull_minus", "seg_csf_plus", "seg_csf_minus", "mask_erode", "mask_dilate"])
SETS = ("runs_atlas", "runs_thal")
ROI6 = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]


def recorded_sums():
    """Checksums written when the runs were fetched: {relative path: sha256}."""
    rec = {}
    for p in [os.path.join(BASE, "SHA256SUMS_atlas.txt"), os.path.join(BASE, "SHA256SUMS_thal.txt")] + sorted(glob.glob(os.path.join(BASE, "sums_h*.txt"))):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                sha, _, rel = line.strip().partition("  ")
                if rel:
                    rec[rel.lstrip("./").replace("\\", "/")] = sha
    return rec


def montage_labels():
    with np.load(EXTMONT) as z:
        return [str(m) for m in z["montages"]]


def sp_fem_labels():
    sp, fem = {}, {}
    with open(PARAMS, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            (sp if r["model"] == "sp1v23" else fem if r["model"] == "fem" else {})[r["target"]] = r["cohort_choice"]
    return sp, fem


def one_head(job):
    code, rec, mi, sp, fem_ext = job
    t0 = time.time()
    out_npz = os.path.join(OUT, f"{code}.npz")
    out_json = os.path.join(OUT, f"{code}.json")
    if os.path.exists(out_npz) and os.path.exists(out_json):
        return code, "cached", 0.0
    arrays, inv, odd, meta = {}, [], [], {}
    for sub in SETS:
        d = os.path.join(BASE, sub, code)
        tag = sub.split("_")[1]
        names = sorted(os.listdir(d)) if os.path.isdir(d) else []
        expected = {f"{r}.npz" for r in RUNS} | {"targets.json", "DONE"}
        for f in names:
            if f not in expected:
                odd.append(dict(set=sub, head=code, file=f, issue="part_file" if ".part" in f else "unexpected_file"))
        for f in sorted(expected - set(names)):
            odd.append(dict(set=sub, head=code, file=f, issue="missing"))
        if "targets.json" not in names:
            continue
        tj = json.load(open(os.path.join(d, "targets.json"), encoding="utf-8"))
        meta[tag] = dict(targets=tj["targets"], tiers=tj["tiers"], n_roi=tj["n_roi"], n_out=tj["n_out"], new_targets=tj["new_targets"], fem_choice=tj["fem_choice"])
        nt, nr = len(tj["targets"]), len(RUNS)
        A = dict(idx=np.full((nr, nt, 3), -1, np.int32), vals=np.full((nr, nt, 3, 3), np.nan, np.float32), Tmax=np.full((nr, nt), np.nan, np.float32),
                 nband=np.full((nr, nt), -1, np.int32), fixed_idx=np.full((nr, nt, 4), -2, np.int32), fixed_vals=np.full((nr, nt, 4, 3), np.nan, np.float32),
                 n_roi_masked=np.full((nr, nt), -1, np.int32), done=np.zeros(nr, bool))
        new = [t for t in tj["new_targets"]]
        # float16 scores of the cerebellar and brainstem targets at their FEM and SPICEhead cohort montages
        m_fem = np.array([mi[fem_ext[tj["targets"][t]]] if tj["targets"][t] in fem_ext else -1 for t in new], np.int32)
        m_sp = np.array([mi[sp[tj["targets"][t]]] if tj["targets"][t] in sp else -1 for t in new], np.int32)
        A["F16"] = np.full((nr, len(new), 2), np.nan, np.float16)
        A["T16"] = np.full((nr, len(new), 2), np.nan, np.float16)
        A["f16_vs_vals_max"] = np.full((nr, 2), np.nan, np.float64)               # largest relative float16 / float32 gap: F, T
        for i, r in enumerate(RUNS):
            p = os.path.join(d, f"{r}.npz")
            rel = f"{sub}/{code}/{r}.npz"
            row = dict(set=sub, head=code, run=r, size=None, sha256=None, recorded="no record", scenario=None, n_nodes=None, t_solve_s=None, t_run_s=None, params_sha=None, shapes=None, status="missing")
            if os.path.exists(p):
                b = open(p, "rb").read()
                row.update(size=len(b), sha256=hashlib.sha256(b).hexdigest())
                if rel in rec:
                    row["recorded"] = "match" if rec[rel] == row["sha256"] else "MISMATCH"
                try:
                    with np.load(io.BytesIO(b)) as z:
                        prm = json.loads(str(z["params"]))
                        A["idx"][i] = z["idx"]; A["vals"][i] = z["vals"]; A["Tmax"][i] = z["Tmax"]; A["nband"][i] = z["nband"]
                        A["fixed_idx"][i] = z["fixed_idx"]; A["fixed_vals"][i] = z["fixed_vals"]
                        assert [str(x) for x in z["cols"]] == ["F", "T", "O"] and [str(x) for x in z["choices"]] == ["unconstrained", "constrained", "reported"]
                        assert [str(x) for x in z["fixed"]] == ["fem_cohort_choice", "fem_cohort_unconstrained", "nominal_unconstrained", "nominal_reported"]
                        if "n_roi_masked" in prm:
                            A["n_roi_masked"][i] = prm["n_roi_masked"]
                        if new:
                            assert [int(x) for x in z["new_targets"]] == new
                            Fn, Tn = z["F_new"], z["T_new"]
                            jj = np.arange(len(new))
                            for k, m in enumerate((m_fem, m_sp)):
                                ok = m >= 0
                                A["F16"][i, ok, k] = Fn[jj[ok], m[ok]]
                                A["T16"][i, ok, k] = Tn[jj[ok], m[ok]]
                            nw = np.array(new)
                            vc = z["vals"][nw, 1]                                 # constrained optimum, float32
                            c = z["idx"][nw, 1]
                            pos = vc[:, 0] > 0
                            A["f16_vs_vals_max"][i] = [np.abs(Fn[jj, c].astype(np.float64)[pos] / vc[pos, 0] - 1).max() if pos.any() else 0.0,
                                                       np.abs(Tn[jj, c].astype(np.float64)[pos] / vc[pos, 1] - 1).max() if pos.any() else 0.0]
                        row.update(scenario=prm.get("scenario"), n_nodes=prm.get("n_nodes"), t_solve_s=prm.get("t_solve_s"), t_run_s=prm.get("t_run_s"),
                                   params_sha=hashlib.sha256(json.dumps(prm, sort_keys=True).encode()).hexdigest()[:16],
                                   shapes=f"idx{z['idx'].shape} vals{z['vals'].shape} fixed{z['fixed_vals'].shape} {z['vals'].dtype}",
                                   status="ok" if prm.get("code") == code and prm.get("run") == r else "params_identity_mismatch")
                        if prm.get("scenario") == "conductivity":
                            meta.setdefault("log_factors", {})[r] = prm["log_factors"]
                    A["done"][i] = row["status"] == "ok"
                except Exception as e:                                            # recorded, never treated as a success
                    row["status"] = f"unreadable: {type(e).__name__}"
            inv.append(row)
        for k, v in A.items():
            arrays[f"{tag}_{k}"] = v
    # the six-target run of the first 35 heads holds every montage for ref6:hippocampus_R (one of the 34 differing protocols)
    six = os.path.join(BASE, "runs", code)
    if os.path.isdir(six):
        v6 = np.full((len(RUNS), 3), np.nan, np.float32)
        gap = 0.0
        tgt = mi[sp["ref6:hippocampus_R"]]
        for i, r in enumerate(RUNS):
            p6 = os.path.join(six, f"{r}.npz")
            if os.path.exists(p6):
                with np.load(p6) as z:
                    v6[i] = [z["F"][5, tgt], z["T"][5, tgt], z["O"][5, tgt]]
                    F6 = z["F"]
                ta = meta["atlas"]["targets"]
                for j, roi in enumerate(ROI6):
                    a = arrays["atlas_vals"][i, ta.index(f"ref6:{roi}"), 0, 0]
                    gap = max(gap, abs(float(a) - float(F6[j].max())) / float(F6[j].max()))
        arrays["six_hippR_sp"] = v6
        meta["six_max_rel_gap"] = gap
    tmp = out_npz + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, out_npz)
    json.dump(dict(code=code, meta=meta, inventory=inv, odd=odd), open(out_json + ".tmp", "w", encoding="utf-8"))
    os.replace(out_json + ".tmp", out_json)
    return code, "ok", time.time() - t0


def main():
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    workers = int(arg("--workers", "3"))
    os.makedirs(OUT, exist_ok=True)
    heads = arg("--heads", "")
    heads = heads.split(",") if heads else sorted(set(os.listdir(os.path.join(BASE, "runs_atlas"))) | set(os.listdir(os.path.join(BASE, "runs_thal"))))
    rec = recorded_sums()
    labs = montage_labels()
    assert len(labs) == 11628 and len(set(labs)) == 11628
    mi = {m: i for i, m in enumerate(labs)}
    sp, fem = sp_fem_labels()
    fx = json.load(open(FIXEXT, encoding="utf-8"))["targets"]
    fem_ext = {k: v["cohort_choice"] for k, v in fx.items()}
    jobs = [(c, rec, mi, sp, fem_ext) for c in heads]
    print(f"{len(heads)} heads, {len(rec)} recorded checksums, {workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(workers, maxtasksperchild=20) as pool:
        for n, (c, st, sec) in enumerate(pool.imap_unordered(one_head, jobs), 1):
            if st != "cached" and (n % 5 == 0 or n < 5):
                print(f"{n}/{len(jobs)} {c} {st} {sec:.0f} s; elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
