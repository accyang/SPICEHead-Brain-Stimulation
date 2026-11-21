# -*- coding: utf-8 -*-
"""Recompute Paper A's analyses under v23 (26 Oct 2026; PI: "Switch to v23 for final version of results").

A thin wrapper around run_v22_all.py, which stays unchanged as the v22 record. Same jobs, dependencies, restart and
adoption logic; only these differ:
  - SPICEHEAD_MODEL=v23 and the GPU solver; results in results_v23/ (the ported scripts follow config.RESULTS);
  - logs and status in C:/spicehead_v23_logs, so the v22 status file is never read;
  - envelope volumes in <m2m>/envelopes_maxdir_v23 (analyze_rois refuses envelopes of another model);
  - no stage benchmark: its timings need a quiet desktop and are run separately
    (SPICEHEAD_MODEL=v23 python pipeline/v22_benchmark_heads.py SUBJ01 5,2,1).
Plan: Paper_A_SPICEhead/Brain Stimulation/v23/PLAN_V23_SWITCH_2026-10-26.md. For a low footprint (PI: "don't occupy
full cpu in this desktop"), --low-footprint sets below-normal priority and 4 logical CPUs before any job starts; jobs inherit both.
  python pipeline/run_v23_all.py --low-footprint --slots 3 [--rerun name,...] [--dry-run]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_v22_all as D                                                     # noqa: E402

D.RES = os.path.join(D.PROJ, "results_v23")
D.OPT = os.path.join(D.RES, "montage_optimization")
D.T3DIR = os.path.join(D.OPT, "cohort_gm300_dx5_maxdir")
D.T3 = os.path.join(D.T3DIR, "cohort_montage_optima.json")
D.LOGDIR = r"C:\spicehead_v23_logs"
D.BASE_ENV = dict(D.BASE_ENV, SPICEHEAD_MODEL="v23", SPICEHEAD_SOLVER="gpu")
_jobs = D.jobs


# 300-head jobs that run on Forerunner 1 (PI: "Use forerunner"), plus the local steps that need site or age and so wait
# for their fetched results; --desktop-part leaves all of them out
CLUSTER = ("cohort_", "t7_", "ratio", "sensitivity", "aging_", "scale_census", "loso", "learning_curve")


def jobs(with_deep=False, with_1010=False):
    J = []
    for j in _jobs(with_deep, with_1010):
        if j["name"] == "bench_heads":
            continue
        if "--desktop-part" in sys.argv and j["name"].startswith(CLUSTER):
            continue
        j["cmd"] = [c.replace("envelopes_maxdir_v22", "envelopes_maxdir_v23") for c in j["cmd"]]
        j["env"] = {k: v.replace("envelopes_maxdir_v22", "envelopes_maxdir_v23") for k, v in j["env"].items()}
        j["cmd"] = [c.replace("cache_maxdir_v22", "cache_maxdir_v23") for c in j["cmd"]]
        J.append(j)
    return J


D.jobs = jobs


def low_footprint(cpus=4):
    """Below-normal priority and the first `cpus` logical CPUs for this process; every job it starts inherits both."""
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p                         # 64-bit pseudo-handle, not a 32-bit int
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    h = k32.GetCurrentProcess()
    if not (k32.SetPriorityClass(h, 0x4000) and k32.SetProcessAffinityMask(h, (1 << cpus) - 1)):   # BELOW_NORMAL
        raise SystemExit(f"could not lower priority or affinity (Windows error {k32.GetLastError()})")


if __name__ == "__main__":
    if "--low-footprint" in sys.argv:
        low_footprint()
    D.main()
