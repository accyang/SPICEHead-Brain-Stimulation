# -*- coding: utf-8 -*-
"""Two-carrier, time-domain compliance analysis — Brain Stimulation review, comment 1.

The circuit result so far comes from a static `.op` solve: both channels held at their programmed
amplitude simultaneously. Real TIS drives two carriers at f1 and f2 = f1 + Δf, and compliance
clipping is an *instantaneous* nonlinearity, so a reviewer is right to ask what the static number
corresponds to. This answers that without a transistor-level transient simulation, using the port
matrices the exact reduction already produced.

Model. Each channel is the same non-ideal source (programmed amplitude I_prog, output impedance
R_out, ±V_cc rail) but now driven sinusoidally:

    s_k(t) = I_prog · sin(ω_k t + φ_k),      V(t) = R · I(t)

At every instant each channel is in one of three states — unclamped (I_k = s_k − V_k/R_out), or
pinned at +V_cc or −V_cc — giving nine combinations, solved exactly and selected by consistency.
The delivered **fundamental amplitude** of each carrier after clipping is then recovered by
quadrature projection over a whole beat period, which contains an integer number of cycles of both
carriers, so the projection is exact.

What this establishes:
  * what the static `.op` condition represents (both carriers at simultaneous positive peak — the
    worst instant of the beat cycle for voltage demand);
  * how the beat-averaged delivered carrier amplitude compares with it;
  * how sensitive clipping is to relative carrier phase (Δf = 0 sweep), which the beat cycle
    traverses automatically when Δf ≠ 0.

  python pipeline/v21_twochannel_timedomain.py [f1_Hz] [df_Hz] [samples_per_cycle]
"""
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, _here)
from v21_twochannel_cosim import I_PROG, R_OUT, V_CC

OUT = os.path.join(ROOT, "results", "v21_twochannel")
STATES = (0, +1, -1)          # unclamped, pinned high, pinned low


def solve_timedomain(R, s):
    """Delivered port currents for two clamped sinusoidal drivers on a linear 2-port.

    R : (2,2) port resistance.  s : (2,T) instantaneous source currents.
    Returns I (2,T) and V (2,T). Every one of the nine clamp-state combinations is solved for all
    time samples at once and then selected by consistency, so the solve is exact and vectorized.
    """
    T = s.shape[1]
    I = np.full((2, T), np.nan)
    V = np.full((2, T), np.nan)
    todo = np.ones(T, bool)
    for c0 in STATES:
        for c1 in STATES:
            c = (c0, c1)
            A = np.zeros((2, 2))
            b = np.zeros((2, T))
            for k in range(2):
                if c[k] == 0:                       # unclamped: I_k + (R I)_k / R_out = s_k
                    A[k, k] += 1.0
                    A[k] += R[k] / R_OUT
                    b[k] = s[k]
                else:                               # pinned: (R I)_k = ±V_cc
                    A[k] += R[k]
                    b[k] = c[k] * V_CC
            try:
                Ic = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            Vc = R @ Ic
            ok = np.ones(T, bool)
            for k in range(2):
                want = s[k] - Vc[k] / R_OUT         # current the source would deliver unclamped
                if c[k] == 0:
                    ok &= np.abs(Vc[k]) <= V_CC + 1e-9
                elif c[k] > 0:
                    ok &= (Vc[k] >= V_CC - 1e-9) & (Ic[k] <= want + 1e-12)
                else:
                    ok &= (Vc[k] <= -V_CC + 1e-9) & (Ic[k] >= want - 1e-12)
            sel = ok & todo
            if sel.any():
                I[:, sel] = Ic[:, sel]
                V[:, sel] = Vc[:, sel]
                todo &= ~sel
    if todo.any():
        raise RuntimeError(f"no consistent clamp state at {todo.sum()} of {T} samples")
    return I, V


def fundamental(x, t, w, phi):
    """Amplitude of the component of x at angular frequency w and phase phi (exact over an
    integer number of cycles): in-phase and quadrature projections, returned as magnitude."""
    a = 2.0 * np.mean(x * np.sin(w * t + phi))
    b = 2.0 * np.mean(x * np.cos(w * t + phi))
    return float(np.hypot(a, b))


def main(f1=2000.0, df=10.0, spc=100):
    src = os.path.join(OUT, "twochannel_exact_dx2.csv")
    rows_in = list(csv.DictReader(open(src, encoding="utf-8")))
    f2 = f1 + df
    T_beat = 1.0 / df                      # contains an integer number of cycles of BOTH carriers
    n = int(round(f1 * spc * T_beat))
    t = np.arange(n) / (n / T_beat)
    w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
    print(f"f1 = {f1:g} Hz, f2 = {f2:g} Hz, Δf = {df:g} Hz; beat {T_beat*1e3:g} ms, "
          f"{n} samples ({spc}/carrier cycle)\n")

    out = []
    for r in rows_in:
        R = np.array([[float(r["R11"]), float(r["R12"])],
                      [float(r["R21"]), float(r["R22"])]])
        Istat = np.array([float(r["I1_mA"]), float(r["I2_mA"])]) * 1e-3
        s = np.vstack([I_PROG * np.sin(w1 * t), I_PROG * np.sin(w2 * t)])
        I, V = solve_timedomain(R, s)
        A1 = fundamental(I[0], t, w1, 0.0)
        A2 = fundamental(I[1], t, w2, 0.0)
        clipped = float(np.mean(np.abs(V) >= V_CC - 1e-9, axis=1)[1])   # ch2 duty cycle
        out.append(dict(
            subject=r["subject"], r2_contact_ohm=float(r["r2_contact_ohm"]),
            I1_static_mA=round(1e3 * Istat[0], 5), I2_static_mA=round(1e3 * Istat[1], 5),
            A1_td_mA=round(1e3 * A1, 5), A2_td_mA=round(1e3 * A2, 5),
            ratio_static=round(Istat[1] / Istat[0], 5), ratio_td=round(A2 / A1, 5),
            A2_over_I2static=round(A2 / Istat[1], 5),
            ch2_clip_duty=round(clipped, 4),
            Vpk_ch1=round(float(np.abs(V[0]).max()), 3),
            Vpk_ch2=round(float(np.abs(V[1]).max()), 3)))

    with open(os.path.join(OUT, "twochannel_timedomain.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"{'r2':>6}{'I2 static':>11}{'A2 two-carrier':>16}{'A2/I2_static':>14}"
          f"{'clip duty':>11}{'ratio static':>14}{'ratio 2-carrier':>17}")
    for r2 in sorted({r["r2_contact_ohm"] for r in out}):
        g = [r for r in out if r["r2_contact_ohm"] == r2]
        f = lambda k: float(np.mean([r[k] for r in g]))
        print(f"{r2/1e3:>5g}k{f('I2_static_mA'):>11.4f}{f('A2_td_mA'):>16.4f}"
              f"{f('A2_over_I2static'):>14.4f}{f('ch2_clip_duty'):>11.3f}"
              f"{f('ratio_static'):>14.4f}{f('ratio_td'):>17.4f}")

    # ---- phase sensitivity: Δf = 0, sweep the fixed relative phase -----------------
    print("\nphase sensitivity at Δf = 0 (equal carriers, fixed relative phase φ):")
    print(f"{'r2':>6}" + "".join(f"{p:>10.0f}°" for p in (0, 45, 90, 135, 180)))
    Tc = 1.0 / f1
    tc = np.arange(spc * 20) / (spc * 20 / (20 * Tc))
    phase_rows = []
    for r2 in sorted({r["r2_contact_ohm"] for r in out}):
        vals = []
        for phi in np.deg2rad([0, 45, 90, 135, 180]):
            a2 = []
            for r in [x for x in rows_in if float(x["r2_contact_ohm"]) == r2]:
                R = np.array([[float(r["R11"]), float(r["R12"])],
                              [float(r["R21"]), float(r["R22"])]])
                s = np.vstack([I_PROG * np.sin(w1 * tc),
                               I_PROG * np.sin(w1 * tc + phi)])
                I, _ = solve_timedomain(R, s)
                a2.append(fundamental(I[1], tc, w1, phi))
            vals.append(1e3 * float(np.mean(a2)))
        phase_rows.append(dict(r2_contact_ohm=r2, **{f"phi_{d}": round(v, 5)
                                                     for d, v in zip((0, 45, 90, 135, 180), vals)}))
        print(f"{r2/1e3:>5g}k" + "".join(f"{v:>10.4f} " for v in vals))
    with open(os.path.join(OUT, "twochannel_phase.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(phase_rows[0]))
        w.writeheader()
        w.writerows(phase_rows)
    print("\n(mA delivered on channel 2, mean over subjects; φ = relative carrier phase)")
    print("wrote", os.path.join(OUT, "twochannel_timedomain.csv"))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 2000.0,
         float(sys.argv[2]) if len(sys.argv) > 2 else 10.0,
         int(sys.argv[3]) if len(sys.argv) > 3 else 100)
