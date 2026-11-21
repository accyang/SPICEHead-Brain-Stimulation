# -*- coding: utf-8 -*-
"""Precompute the multi-montage SPICEhead cache for some heads, in parallel with the main job.

The 1 mm pass of v21_multimontage_fem.py --compare walks its eight heads in order, one at a time,
and caches each head's envelopes atomically (<cid>__dx1.pkl) before comparing with FEM. On a
shared machine one head took about 2.5 h, so the serial pass is the critical path of the v22
recompute. This runs the SAME function, stage_compare, restricted to the heads named on the
command line (in that order), so the main job finds their caches and skips the solves. Nothing
about the computation changes; only which process does it.

  python pipeline/v22_multimontage_helper.py <dx> <cid> [<cid> ...]

It also merges its heads' rows into the result CSVs, as stage_compare always does; the main job
rewrites the same keys with identical values when it finishes.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import v21_multimontage_fem as M                                        # noqa: E402

if __name__ == "__main__":
    dx = float(sys.argv[1])
    wanted = sys.argv[2:]
    pool = {s[0]: s for s in M.subjects(8)}
    missing = [c for c in wanted if c not in pool]
    if missing:
        raise SystemExit(f"not among the eight FEM heads: {missing}")
    sel = [pool[c] for c in wanted]
    M.subjects = lambda n: sel            # stage_compare looks the list up at call time
    print(f"helper: {len(sel)} heads at {dx:g} mm: {', '.join(wanted)}", flush=True)
    M.stage_compare(len(sel), dx)
