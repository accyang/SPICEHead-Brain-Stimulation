# -*- coding: utf-8 -*-
"""Bench measurements of the stimulator on resistive loads against the paper's source model.

Protocol: Paper_A_SPICEhead/Brain Stimulation/v22/BENCH_PROTOCOL_source_model_2026-10-02.md (plan item 3). The
authors measure delivered current for a set of load resistors; this script predicts the same
quantities from the Methods' source model (Norton source I_PROG, output resistance R_OUT, ±V_CC
compliance rail, sharp knee; v21_twochannel_cosim.py) and reports the residuals.

Predictions per row (R_path = r_load_ohm + r_sense_ohm):
  unclamped amplitude   A = I_prog / (1 + R_path / R_out)
  clip level            c = V_cc / R_path             (current at the rail)
  peak current          min(A, c)
  fundamental           of the clipped sinusoid, from the time-domain solver of
                        v21_twochannel_timedomain.py (and the closed form, as a cross-check)
Fits from the measurements: the rail from clipped rows (median of i_peak * R_path) and the output
resistance from unclipped rows (slope of I_prog/I_fund - 1 against R_path).

Two-channel rows (test B) are predicted when the CSV carries r11_ohm, r22_ohm, r12_ohm (the
T-network built from a head's port matrix) and r_contact_ch1_ohm, r_contact_ch2_ohm (per
electrode; two in series per channel); channel 2 runs at f_hz + 10 Hz.

  python pipeline/v22_bench_source_check.py [measurements.csv]

Writes bench_source_summary.csv and bench_source.png beside the input.
"""
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)
from v21_twochannel_cosim import I_PROG, R_OUT, V_CC                  # noqa: E402
from v21_twochannel_timedomain import solve_timedomain, fundamental   # noqa: E402

DEFAULT = os.path.join(ROOT, "results_v22", "bench_source", "measurements.csv")
SPC = 400            # samples per carrier cycle
CYCLES = 20          # cycles solved (a beat period is not needed for a single channel)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def predict_single(r_path, i_prog, f_hz):
    """Peak and fundamental of the delivered current into a resistive path (sharp-knee model)."""
    A = i_prog / (1.0 + r_path / R_OUT)
    c = V_CC / r_path
    w = 2 * np.pi * f_hz
    t = np.arange(SPC * CYCLES) / (SPC * f_hz)
    s = np.vstack([i_prog * np.sin(w * t), np.zeros_like(t)])
    R = np.array([[r_path, 0.0], [0.0, 1.0]])            # channel 2 idle on a dummy 1 Ohm
    I, V = solve_timedomain(R, s)
    fund_td = fundamental(I[0], t, w, 0.0)
    # closed form for a sinusoid of amplitude A clipped at c
    if c >= A:
        fund_cf = A
    else:
        th = np.arcsin(c / A)
        fund_cf = (4 / np.pi) * (A * (th / 2 - np.sin(2 * th) / 4) + c * np.cos(th))
    if abs(fund_td - fund_cf) > 1e-3 * i_prog:
        raise RuntimeError(f"time-domain and closed-form fundamentals differ: {fund_td} vs {fund_cf}")
    return dict(pred_unclamped_ma=1e3 * A, pred_clip_level_ma=1e3 * c,
                pred_peak_ma=1e3 * min(A, c), pred_fund_ma=1e3 * fund_td,
                pred_clipped=c < A, pred_rail_v=min(A, c) * r_path)


def predict_two(row, i_prog, f_hz):
    r11, r22, r12 = f(row.get("r11_ohm")), f(row.get("r22_ohm")), f(row.get("r12_ohm"))
    s1, s2 = 2 * f(row.get("r_contact_ch1_ohm")), 2 * f(row.get("r_contact_ch2_ohm"))
    if not all(np.isfinite([r11, r22, r12, s1, s2])):
        return {}
    rs = f(row.get("r_sense_ohm"))
    R = np.array([[r11 + s1 + rs, r12], [r12, r22 + s2 + rs]])
    w1, w2 = 2 * np.pi * f_hz, 2 * np.pi * (f_hz + 10.0)
    t = np.arange(SPC * int(f_hz / 10.0)) / (SPC * f_hz)          # one beat period
    s = np.vstack([i_prog * np.sin(w1 * t), i_prog * np.sin(w2 * t)])
    I, V = solve_timedomain(R, s)
    return dict(pred_fund_ch1_ma=1e3 * fundamental(I[0], t, w1, 0.0),
                pred_fund_ch2_ma=1e3 * fundamental(I[1], t, w2, 0.0),
                pred_clip_duty_ch2=float(np.mean(np.abs(V[1]) >= V_CC - 1e-9)))


def main(path=DEFAULT):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    out = []
    for r in rows:
        i_prog = 1e-3 * f(r.get("i_prog_ma"))
        f_hz = f(r.get("f_hz"))
        r_path = f(r.get("r_load_ohm")) + f(r.get("r_sense_ohm"))
        rec = dict(r)
        if r.get("test", "").upper().startswith("B"):
            rec.update(predict_two(r, i_prog, f_hz))
        else:
            rec.update(predict_single(r_path, i_prog, f_hz))
            meas = f(r.get("i_fund_ma"))
            if np.isfinite(meas):
                rec["resid_fund_ma"] = meas - rec["pred_fund_ma"]
                rec["resid_fund_pct_of_prog"] = 100 * (meas - rec["pred_fund_ma"]) / (1e3 * i_prog)
            pk = f(r.get("i_peak_ma"))
            if np.isfinite(pk):
                rec["resid_peak_ma"] = pk - rec["pred_peak_ma"]
                rec["meas_rail_v"] = 1e-3 * pk * r_path       # meaningful only when clipped
        out.append(rec)

    single = [r for r in out if "pred_fund_ma" in r and np.isfinite(f(r.get("i_fund_ma")))]
    print(f"{len(rows)} rows, {len(single)} single-channel rows with a measured fundamental")
    if single:
        res = np.array([r["resid_fund_pct_of_prog"] for r in single])
        print(f"fundamental residual (measured - predicted), % of programmed: "
              f"mean {res.mean():+.2f}, largest |{np.abs(res).max():.2f}|")
        clipped = [r for r in single if str(r.get("clipped", "")).lower().startswith("y")
                   and np.isfinite(f(r.get("i_peak_ma")))]
        if clipped:
            rail = np.median([r["meas_rail_v"] for r in clipped])
            print(f"rail from {len(clipped)} clipped rows: {rail:.2f} V (model {V_CC:.1f} V)")
        unc = [r for r in single if not str(r.get("clipped", "")).lower().startswith("y")]
        if len(unc) >= 3:
            x = np.array([f(r["r_load_ohm"]) + f(r["r_sense_ohm"]) for r in unc])
            y = np.array([1e-3 * f(r["i_prog_ma"]) / (1e-3 * f(r["i_fund_ma"])) - 1 for r in unc])
            slope = np.polyfit(x, y, 1)[0]
            r_out = 1 / slope if slope > 0 else float("inf")
            print(f"output resistance from {len(unc)} unclipped rows: "
                  f"{r_out/1e6:.2f} MOhm (model {R_OUT/1e6:.1f} MOhm; droop at 15 kOhm "
                  f"{100*15e3/R_OUT:.1f}% predicted)")
        knee = {ip: V_CC / (1e-3 * ip - V_CC / R_OUT) for ip in sorted({f(r['i_prog_ma']) for r in single})}
        print("predicted knee (path resistance at which clipping begins): " +
              ", ".join(f"{ip:g} mA: {k/1e3:.1f} kOhm" for ip, k in knee.items()))

    od = os.path.dirname(os.path.abspath(path))
    keys = list(out[0]) if out else []
    for r in out:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(od, "bench_source_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print("wrote", os.path.join(od, "bench_source_summary.csv"))

    if single:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for ip, c in ((1.0, "#3182bd"), (2.0, "#e6550d")):
            rr = np.logspace(3, np.log10(6e4), 200)
            pred = [predict_single(x + 100.0, 1e-3 * ip, 2000.0)["pred_fund_ma"] for x in rr]
            ax.plot(rr, pred, color=c, lw=1, label=f"model, {ip:g} mA")
            m = [r for r in single if abs(f(r["i_prog_ma"]) - ip) < 1e-6]
            if m:
                ax.scatter([f(r["r_load_ohm"]) for r in m], [f(r["i_fund_ma"]) for r in m],
                           s=18, color=c, zorder=3, label=f"measured, {ip:g} mA")
        ax.set(xscale="log", xlabel="Load resistance (Ω)", ylabel="Fundamental amplitude (mA)",
               title="Delivered current on resistive loads")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(od, "bench_source.png"), dpi=200)
        print("wrote", os.path.join(od, "bench_source.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
