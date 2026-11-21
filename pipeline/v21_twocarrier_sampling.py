# -*- coding: utf-8 -*-
"""Temporal-sampling convergence of the recovered carrier fundamental.

Final Brain Stimulation review, comment 2. The Methods claimed the fundamental was "recovered
exactly by quadrature projection". An integer number of carrier cycles removes spectral leakage,
but a clipped waveform sampled at finite resolution is still a discretization: the clip instants
fall between samples. So the claim of exactness is too strong, and the honest replacement is a
measured convergence rate.

This re-runs the two-carrier clamp solve at 25, 50, 100, 200, 400 and 800 samples per carrier
cycle on every subject/condition port matrix already computed at 2 mm, and reports how much the
recovered fundamental moves. No field solve is involved — the delivered amplitudes depend only on
the 2x2 port matrix, the programmed amplitude, the output impedance and the rail.

  python pipeline/v21_twocarrier_sampling.py
"""
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, ROOT)
from spicehead import config
from v21_twochannel_cosim import I_PROG, R_OUT, V_CC                     # noqa: E402
from v21_twochannel_timedomain import solve_timedomain, fundamental      # noqa: E402

OUT = os.path.join(config.RESULTS, "v21_twochannel")
F1, DF = 2000.0, 10.0
SPC = [25, 50, 100, 200, 400, 800]
REF = 800                      # the finest grid is the reference


def amps(R, spc):
    """Delivered fundamental amplitude of each carrier at `spc` samples per carrier cycle."""
    n = int(round(spc * F1 / DF))          # a whole beat period, integer cycles of both carriers
    t = np.arange(n) / (n * DF)
    w1, w2 = 2 * np.pi * F1, 2 * np.pi * (F1 + DF)
    s = np.vstack([I_PROG * np.sin(w1 * t), I_PROG * np.sin(w2 * t)])
    I, _ = solve_timedomain(R, s)
    return fundamental(I[0], t, w1, 0.0), fundamental(I[1], t, w2, 0.0)


def main():
    src = os.path.join(OUT, "twochannel_exact_dx2.csv")
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    out, worst = [], {spc: 0.0 for spc in SPC}
    for r in rows:
        R = np.array([[float(r["R11"]), float(r["R12"])],
                      [float(r["R21"]), float(r["R22"])]])
        ref = amps(R, REF)
        rec = {"subject": r["subject"], "r2_contact_ohm": r["r2_contact_ohm"]}
        for spc in SPC:
            a = amps(R, spc)
            # the starved channel is the one that clips, so it is the one that matters
            d = max(abs(a[k] - ref[k]) / ref[k] for k in (0, 1) if ref[k] > 0)
            rec[f"A2_spc{spc}_mA"] = f"{a[1] * 1e3:.6f}"
            rec[f"reldiff_spc{spc}"] = f"{d:.3e}"
            worst[spc] = max(worst[spc], d)
        out.append(rec)

    p = os.path.join(OUT, "twocarrier_sampling.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"{len(rows)} subject-conditions; reference = {REF} samples per carrier cycle\n")
    print(f"{'samples/cycle':>14}  {'worst relative change vs reference':>34}")
    for spc in SPC:
        print(f"{spc:>14}  {worst[spc]:>34.2e}")
    print(f"\nwrote {p}")
    print(f"\nAt the 100 samples/cycle used in the paper the recovered fundamental is within "
          f"{worst[100]:.1e} of the {REF}-sample value.")


if __name__ == "__main__":
    main()
