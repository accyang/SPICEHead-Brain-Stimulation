# -*- coding: utf-8 -*-
"""Is the two-carrier correction resolution-invariant?

The 2 mm experiment was re-run with two-carrier metrics; the 1 mm repeat was not, because the
field reconstruction at 1 mm is expensive. But the two-carrier correction is a property of the
*driver*, not of the grid: the delivered carrier amplitudes depend only on the 2x2 port-resistance
matrix, the programmed amplitude, the output impedance and the rail. Those port matrices are
already stored in both CSVs, so the question can be settled exactly without a single field solve.

This recomputes the delivered fundamental amplitudes from the 1 mm port matrices and compares the
gain (two-carrier / static) against the 2 mm value, subject by subject and condition by condition.

  python pipeline/v21_twocarrier_dx1_check.py
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
from v21_twochannel_cosim import I_PROG, R_OUT, V_CC          # noqa: E402
from v21_twochannel_timedomain import solve_timedomain, fundamental  # noqa: E402

OUT = os.path.join(config.RESULTS, "v21_twochannel")
F1, DF, SPC = 2000.0, 10.0, 100


def port(row):
    return np.array([[float(row["R11"]), float(row["R12"])],
                     [float(row["R21"]), float(row["R22"])]])


def carrier_amps(R):
    """Delivered fundamental amplitude of each carrier after compliance clipping."""
    f2 = F1 + DF
    n = int(round(SPC * F1 / DF))
    t = np.arange(n) / (n * DF)
    w1, w2 = 2 * np.pi * F1, 2 * np.pi * f2
    s = np.vstack([I_PROG * np.sin(w1 * t), I_PROG * np.sin(w2 * t)])
    I, _ = solve_timedomain(R, s)
    return fundamental(I[0], t, w1, 0.0), fundamental(I[1], t, w2, 0.0)


def load(name):
    p = os.path.join(OUT, name)
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def main():
    d2 = {(r["subject"], float(r["r2_contact_ohm"])): r for r in load("twochannel_exact_dx2.csv")}
    d1 = load("twochannel_exact_dx1.csv")
    if not d1:
        raise SystemExit("no 1 mm results to check")

    print(f"{'subject':10s} {'r2':>7s} {'gain 1mm':>9s} {'gain 2mm':>9s} {'Δ':>9s}"
          f" {'A2 1mm':>8s} {'A2 2mm':>8s}")
    rows, dg = [], []
    for r in d1:
        R = port(r)
        a1, a2 = carrier_amps(R)
        static2 = float(r["I2_mA"]) / 1e3
        g1 = a2 / static2 if static2 else float("nan")
        key = (r["subject"], float(r["r2_contact_ohm"]))
        g2 = float(d2[key]["td_over_static_I2"]) if key in d2 else float("nan")
        a2_2mm = float(d2[key]["A2_td_mA"]) if key in d2 else float("nan")
        print(f"{r['subject']:10s} {float(r['r2_contact_ohm'])/1e3:6g}k {g1:9.4f} {g2:9.4f}"
              f" {g1 - g2:+9.2e} {a2*1e3:8.4f} {a2_2mm:8.4f}")
        rows.append(dict(subject=r["subject"], r2_contact_ohm=r["r2_contact_ohm"],
                         A1_td_mA=f"{a1*1e3:.6f}", A2_td_mA=f"{a2*1e3:.6f}",
                         td_over_static_I2=f"{g1:.6f}",
                         gain_dx2=f"{g2:.6f}", gain_diff=f"{g1 - g2:.3e}"))
        if np.isfinite(g2):
            dg.append(abs(g1 - g2))

    p = os.path.join(OUT, "twocarrier_dx1_gain.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nlargest gain difference between resolutions: {max(dg):.3e}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
