# -*- coding: utf-8 -*-
"""Estimate the NCHC Nano 4 cost of a run before submitting it (2 Nov 2026).

Billing is 2.5 SU per allocated CPU core-hour. GPUs and memory are not billed, and an allocated CPU costs the same
whether it is busy or idle. Whole-node tasks keep billing until their last head ends, so the estimate needs the share
of allocated GPU time that actually holds a head. Run a pilot on one or two heads first, and use its measured time
per head (cpu_trial_nano4.sbatch shows how): never a guess.

    python pipeline/nchc/estimate_su.py --heads 265 --hours-per-head 9.1
    python pipeline/nchc/estimate_su.py --heads 265 --hours-per-head 9.1 --cpus-per-gpu 12

Check against the 300-head uncertainty run (array 660766): the pilot time of 9.1 h per head at 12 CPUs per GPU gives
93,955 SU; the run cost 98,526 SU. Occupancy 0.77 was measured on that same run, so it fits that run by construction
and other runs may differ. Report the estimate as a range: a head can take twice the median when GPUs are shared.
"""
import argparse

SU_PER_CORE_HOUR = 2.5


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--heads", type=int, required=True)
    a.add_argument("--hours-per-head", type=float, required=True, help="measured in a pilot, wall hours on one GPU")
    a.add_argument("--cpus-per-gpu", type=int, default=2)
    a.add_argument("--occupancy", type=float, default=0.77, help="share of allocated GPU time holding a head (0.77 measured)")
    a.add_argument("--gpus-per-node", type=int, default=8)
    a.add_argument("--nodes", type=int, default=16)
    p = a.parse_args()
    busy = p.heads * p.hours_per_head
    alloc = busy / p.occupancy
    core_h = alloc * p.cpus_per_gpu
    su = core_h * SU_PER_CORE_HOUR
    print(f"GPU-hours holding a head   {busy:10,.0f}")
    print(f"GPU-hours allocated        {alloc:10,.0f}   (occupancy {p.occupancy:.2f})")
    print(f"CPU core-hours billed      {core_h:10,.0f}   ({p.cpus_per_gpu} CPUs per GPU)")
    print(f"Estimated cost             {su:10,.0f} SU   (range {su * 0.9:,.0f} to {su * 1.25:,.0f})")
    print(f"Elapsed on {p.nodes} nodes       {alloc / (p.nodes * p.gpus_per_node):10.1f} h")


if __name__ == "__main__":
    main()
