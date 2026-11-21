# -*- coding: utf-8 -*-
"""Recompute every analysis the revised paper reports, under the v22 model.

v22 = skin-surface electrode contacts + harmonic interface conductance (spicehead/config.py).
All outputs go to results_v22/ (and figures/TI_targets_MNI, <m2m>/envelopes_maxdir_v22); every
cache is tagged _v22, so nothing computed under v21 can be picked up. The v21 results stay in
results/ as the historical record.

  python pipeline/run_v22_all.py [--slots N] [--with-deep] [--rerun name,...] [--dry-run]

Jobs run in priority order as their dependencies finish, at most N processes at a time (default 5:
this machine is shared with other long jobs). Each job logs to LOGDIR/<name>.log; the driver logs
to LOGDIR/_driver.log and keeps LOGDIR/_status.json. A restart skips every job already recorded as
ok, and the sharded cohort jobs resume from their own per-subject caches, so an interruption costs
at most the subjects in flight. A failed job blocks only the jobs that depend on it.

Not run by default: the deep-target cohort (Table 4 of the v21 manuscript; about 180 CPU-hours at
4 mm). Add --with-deep to include it.
"""
import json
import os
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = r"C:\Users\ALBERT YANG\AppData\Local\Programs\Python\Python312\python.exe"
CSV = r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv"
CH = r"L:\BrainProbe\SPICEhead\charm"
RES = os.path.join(PROJ, "results_v22")
OPT = os.path.join(RES, "montage_optimization")
T3DIR = os.path.join(OPT, "cohort_gm300_dx5_maxdir")
T3 = os.path.join(T3DIR, "cohort_montage_optima.json")
T7 = os.path.join(PROJ, "Paper_A_SPICEhead", "Brain Stimulation", "verification_8subjects", "cohort_table7.csv")
LOGDIR = r"C:\spicehead_v22_logs"
NW = 4                                   # shards per sharded cohort job
HEADS = ["SUBJ01", "SUBJ02", "SUBJ03", "SUBJ04"]

BASE_ENV = {
    "SPICEHEAD_MODEL": "v22",
    "SPICEHEAD_ENVELOPE": "maxdir",
    "PYTHONHASHSEED": "0",
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    # one thread per process: the slot count is the parallelism, and the box is shared
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
}


def job(name, script, *args, deps=(), env=None):
    return {"name": name, "cmd": [PY, os.path.join("pipeline", script)] + [str(a) for a in args],
            "deps": list(deps), "env": env or {}}


def jobs(with_deep=False, with_1010=False):
    J = []
    # 1. the 300-head 5 mm search: feeds the cohort tables, LOSO, ratio, sensitivity and the
    #    5 mm side of the Table 7 comparison
    W = [f"cohort_w{i}" for i in range(NW)]
    J += [job(W[i], "run_montage_cohort.py", CSV, CH, T3DIR, "5.0", "0", "gm", i, NW)
          for i in range(NW)]
    J.append(job("cohort_agg", "run_montage_cohort.py", CSV, CH, T3DIR, "5.0", "0", "gm", deps=W))
    J.append(job("loso", "v21_loso_generalization.py", deps=["cohort_agg"]))
    J.append(job("learning_curve", "v21_learning_regret.py", "40", deps=["cohort_agg"]))
    # 2. Table 7: all 11,628 montages at 2 mm in the 20 pinned heads (scoring needs no 5 mm data)
    t7env = {"SPICEHEAD_T7_SUBJECTS": T7}
    S = [f"t7_s{k}" for k in range(NW)]
    J += [job(S[k], "v21_dx2_cohort_check.py", "--score-full", "5", "--shard", f"{k}/{NW}",
              env=t7env) for k in range(NW)]
    J.append(job("t7_collect", "v21_dx2_cohort_check.py", "--collect-full", "5",
                 deps=S + ["cohort_agg"], env=t7env))
    # 3. FEM comparison, 8 heads x 4 montages; the 1 mm pass starts after the 2 mm pass has
    #    written, because both merge into the same CSVs
    J.append(job("fem_compare2", "v21_multimontage_fem.py", "--compare", "8", "2.0"))
    J.append(job("fem_focality2", "v21_multimontage_fem.py", "--focality", "8", "2.0",
                 deps=["fem_compare2"]))
    J.append(job("fem_compare1", "v21_multimontage_fem.py", "--compare", "8", "1.0",
                 deps=["fem_focality2"]))
    J.append(job("fem_focality1", "v21_multimontage_fem.py", "--focality", "8", "1.0",
                 deps=["fem_compare1"]))
    J.append(job("fem_thal", "v21_simnibs_compare.py", "2.0,1.0", "8"))
    J.append(job("fem_raw", "v21_agreement_raw.py", "2.0,1.0", "8"))
    # 4. realistic-head grid study (SUBJ01 keeps the top-level layout of the v21 results)
    J.append(job("headconv_SUBJ01", "v21_head_convergence.py"))
    J += [job(f"headconv_{h}", "v21_head_convergence.py", h) for h in HEADS[1:]]
    # 5. circuit coupling (validation of delivered current; exact Kron-reduced co-simulation)
    # the published 2 mm sweep has eight contact conditions, bracketing the compliance boundary
    J.append(job("cosim_dx2", "v21_twochannel_exact.py", "8", "2.0", "1,3,5,7,10,15,20,50"))
    # 1 mm repeat: no linear-algebra checks (they run at 2 mm in every condition), heads split
    # over four processes and saved per head; the unsharded run only merges the saved heads
    C1 = [f"cosim_dx1_s{k}" for k in range(NW)]
    J += [job(C1[k], "v21_twochannel_exact.py", "8", "1.0", "1,20,50", "--nocheck",
              "--shard", f"{k}/{NW}") for k in range(NW)]
    J.append(job("cosim_dx1_merge", "v21_twochannel_exact.py", "8", "1.0", "1,20,50", "--nocheck",
                 deps=C1))
    J.append(job("cosim_sampling", "v21_twocarrier_sampling.py", deps=["cosim_dx2"]))
    J.append(job("cosim_dx1_gain", "v21_twocarrier_dx1_check.py",
                 deps=["cosim_dx2", "cosim_dx1_merge"]))
    # 6. cohort follow-ups that score the optimal montages
    J.append(job("ratio", "montage_ratio.py", CSV, CH, T3,
                 os.path.join(OPT, "montage_ratio_n300_maxdir.json"), "5.0", deps=["cohort_agg"]))
    J.append(job("sensitivity", "v21_focality_sensitivity.py", "5.0", deps=["cohort_agg"],
                 env={"SPICEHEAD_T3": T3}))
    # 7. MNI-template target maps
    J.append(job("mni_maps", "fig_ti_targets_mni.py"))
    # 8. age analysis: regenerate envelope volumes into <m2m>/envelopes_maxdir_v22, then analyse
    A = [f"aging_w{i}" for i in range(NW)]
    J += [job(A[i], "regen_envelopes_maxdir.py", CSV, CH, "envelopes_maxdir_v22", i, NW)
          for i in range(NW)]
    for t in ("brain", "gm"):
        J.append(job(f"aging_{t}", "analyze_rois.py", CSV, CH,
                     os.path.join(RES, f"aging_n300_{t}_maxdir"), t, deps=A,
                     env={"SPICEHEAD_ENVDIR": "envelopes_maxdir_v22"}))
    # 9. 10-10 electrode set (32 positions, 5 mm). Dropped from the v22 paper by the authors on
    #    2 October 2026: a 5 mm extension is a coarse-grid result of limited weight once the 2 mm
    #    sweep shows 5 mm choices changing. Kept behind --with-1010 for completeness.
    if with_1010:
        tenv = {"MONTAGE1010_CACHE": r"C:\spicehead_1010_cache_maxdir_v22"}
        out1010 = os.path.join(OPT, "montage_1010_n300_maxdir.json")
        X = [f"tenten_w{i}" for i in range(NW)]
        J += [job(X[i], "montage_1010.py", CSV, CH, out1010, "5.0", i, NW, env=tenv)
              for i in range(NW)]
        J.append(job("tenten_agg", "montage_1010.py", CSV, CH, out1010, "5.0", deps=X, env=tenv))
    # 10. network size of every head at 5 mm (cheap)
    J.append(job("scale_census", "v22_scale_census.py", "5"))
    J.append(job("scale_fem8", "v22_scale_census.py", "2,1", "--fem8"))
    if with_deep:
        D = [f"deep_w{i}" for i in range(NW)]
        out = os.path.join(OPT, "cohort_deep_gm300_maxdir")
        J += [job(D[i], "montage_deep_cohort.py", CH, "4.0", out, CSV, i, NW) for i in range(NW)]
        J.append(job("deep_agg", "montage_deep_cohort.py", CH, "4.0", out, CSV, deps=D))
    # 11. stage benchmark in one head at 5, 2 and 1 mm. Last, after every other job of this
    #     driver, so that none of them competes with it (other users' jobs are recorded instead)
    J.append(job("bench_heads", "v22_benchmark_heads.py", "SUBJ05", "5,2,1",
                 deps=[j["name"] for j in J]))
    return J


def pid_alive(pid):
    """True while the Windows process `pid` is running."""
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))        # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259               # STILL_ACTIVE


def adopted_ok(name, log):
    """Success of a job whose process this driver did not start (no exit code available).

    A restarted driver adopts jobs that are still running; when one ends, success is read from
    its log. Only the job kinds that may be adopted have a rule; anything else counts as failed,
    so it is rerun rather than trusted.
    """
    import re
    if "Traceback" in log:
        return False
    if name.startswith("cohort_w"):
        return re.search(rf"shard {name[len('cohort_w'):]}/{NW} done", log) is not None
    if name.startswith("t7_s"):
        m = re.search(r"shard \d+/\d+: (\d+) subjects", log)
        n_ok = len(re.findall(r"\] \S+ (OK \(|\(cached\))", log))
        return bool(m) and "FAILED" not in log and n_ok == int(m.group(1))
    if name.startswith("aging_w"):                 # regen_envelopes_maxdir final line
        m = re.search(r"shard \d+: (\d+) done, (\d+) skipped, (\d+) failed", log)
        return bool(m) and m.group(3) == "0"
    if name.startswith("fem_compare"):             # stage_compare ends with the agreement count
        return "best-montage agreement:" in log and "\nwrote " in log
    if name == "cosim_dx2":                        # v21_twochannel_exact summary and CSV
        return "=== dx = 2 mm, 8 subjects ===" in log and "\nwrote " in log
    if name.startswith("cosim_dx1_s"):             # sharded v21_twochannel_exact
        return re.search(r"shard \d+/\d+ done: \d+ heads", log) is not None
    return False


def note(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGDIR, "_driver.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    argv = sys.argv[1:]
    slots = int(argv[argv.index("--slots") + 1]) if "--slots" in argv else 5
    rerun = set(argv[argv.index("--rerun") + 1].split(",")) if "--rerun" in argv else set()
    J = jobs("--with-deep" in argv, "--with-1010" in argv)
    names = [j["name"] for j in J]
    assert len(set(names)) == len(names)
    for j in J:
        assert all(d in names for d in j["deps"]), j["name"]
    if "--dry-run" in argv:
        for j in J:
            print(f"{j['name']:<18} deps={','.join(j['deps']) or '-':<40} {' '.join(j['cmd'][1:])}")
        return

    os.makedirs(LOGDIR, exist_ok=True)
    spath = os.path.join(LOGDIR, "_status.json")
    status = json.load(open(spath, encoding="utf-8")) if os.path.exists(spath) else {}
    running = {}
    for n in list(status):
        st = status[n]
        if st.get("state") == "running" and n not in rerun and n in names and pid_alive(st["pid"]):
            # still running from an earlier driver: adopt it instead of starting a duplicate
            running[n] = (None, None, st.get("t0", time.time()))
            continue
        if n in rerun or st.get("state") != "ok":
            del status[n]

    def save():
        with open(spath + ".part", "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=1)
        os.replace(spath + ".part", spath)

    env0 = dict(os.environ, **BASE_ENV)
    note(f"driver start: {len(J)} jobs, {sum(1 for n in names if status.get(n, {}).get('state') == 'ok')}"
         f" already ok, {len(running)} adopted ({', '.join(running) or '-'}), {slots} slots")
    while True:
        # reap
        for n, (p, fh, t0) in list(running.items()):
            if p is None:                                   # adopted: no handle, no exit code
                if pid_alive(status[n]["pid"]):
                    continue
                log = open(os.path.join(LOGDIR, n + ".log"), encoding="utf-8",
                           errors="replace").read()
                rc = 0 if adopted_ok(n, log) else -1
            else:
                rc = p.poll()
                if rc is None:
                    continue
                fh.close()
            dt = (time.time() - t0) / 60
            status[n] = {"state": "ok" if rc == 0 else "failed", "rc": rc,
                         "minutes": round(dt, 1), "end": time.strftime("%m-%d %H:%M")}
            note(f"{'done  ' if rc == 0 else 'FAILED'} {n} (rc={rc}, {dt:.1f} min"
                 f"{', adopted' if p is None else ''})")
            del running[n]
            save()
        # block the dependents of failures
        changed = True
        while changed:
            changed = False
            for j in J:
                n = j["name"]
                if n in status or n in running:
                    continue
                bad = [d for d in j["deps"] if status.get(d, {}).get("state") in ("failed", "blocked")]
                if bad:
                    status[n] = {"state": "blocked", "by": bad}
                    note(f"BLOCKED {n} (by {', '.join(bad)})")
                    changed = True
                    save()
        # launch, in priority order
        for j in J:
            if len(running) >= slots:
                break
            n = j["name"]
            if n in status or n in running:
                continue
            if all(status.get(d, {}).get("state") == "ok" for d in j["deps"]):
                fh = open(os.path.join(LOGDIR, n + ".log"), "w", encoding="utf-8")
                fh.write(" ".join(j["cmd"]) + "\n" + json.dumps(j["env"]) + "\n\n")
                fh.flush()
                p = subprocess.Popen(j["cmd"], cwd=PROJ, env=dict(env0, **j["env"]),
                                     stdout=fh, stderr=subprocess.STDOUT)
                running[n] = (p, fh, time.time())
                status[n] = {"state": "running", "pid": p.pid, "t0": time.time(),
                             "start": time.strftime("%m-%d %H:%M")}
                save()
                note(f"start  {n} (pid {p.pid})")
        if not running:
            break
        time.sleep(15)
    ok = sum(1 for n in names if status.get(n, {}).get("state") == "ok")
    bad = [n for n in names if status.get(n, {}).get("state") != "ok"]
    note(f"ALL DONE: {ok}/{len(names)} ok" + (f"; not ok: {', '.join(bad)}" if bad else ""))


if __name__ == "__main__":
    main()
