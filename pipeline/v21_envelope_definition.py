"""
v21 envelope-definition comparison (reviewer comment C).

The manuscript computes the TI envelope by applying Eq. 5, A = 2 min(|a|,|b|),
to each Cartesian component and combining them as sqrt(Ax^2+Ay^2+Az^2). The
reviewer asks precisely what this quantity is, since a realistic TI field is
vector-valued.

Two issues with the per-axis composite:

  1. It is **coordinate-frame dependent** — rotating the head changes the value.
  2. It is **not the envelope along any direction**, so it is not bounded by the
     physically maximal modulation amplitude; it can exceed it.

The rotation-invariant alternative is the maximum-modulation envelope of
Grossman et al. (2017), A_max = max over n of 2 min(|E1.n|, |E2.n|), implemented
in closed form as ``spicehead.ti_envelope_max_direction``.

This script quantifies the disagreement on a heterogeneous 3-D phantom so the
manuscript can state the definition precisely and report the consequence.

Run:  python pipeline/v21_envelope_definition.py
Output: results/v21_envelope/envelope_definition.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spicehead as sh

OUT = "results/v21_envelope"


def analytic_cases():
    """Closed-form checks of the max-direction envelope."""
    def g(e1, e2):
        f = lambda v: tuple(np.array([[[float(x)]]]) for x in v)
        return float(np.ravel(sh.ti_envelope_max_direction(f(e1), f(e2)))[0])
    return [
        ("collinear, equal", g([1, 0, 0], [1, 0, 0]), 2.0),
        ("collinear, 1 vs 0.4", g([1, 0, 0], [0.4, 0, 0]), 0.8),
        ("orthogonal, equal", g([1, 0, 0], [0, 1, 0]), 2 / np.sqrt(2)),
        ("antiparallel", g([1, 0, 0], [-1, 0, 0]), 2.0),
    ]


def rotation_invariance():
    """Per-axis composite vs max-direction under a rigid rotation."""
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    e1 = np.array([0.8, 0.3, 0.1]); e2 = np.array([0.2, 0.9, -0.2])
    f = lambda v: tuple(np.array([[[float(x)]]]) for x in v)

    def per_axis(a, b):
        return float(np.ravel(sh.ti_envelope_magnitude(f(a), f(b)))[0])

    def maxdir(a, b):
        return float(np.ravel(sh.ti_envelope_max_direction(f(a), f(b)))[0])

    return dict(
        per_axis_orig=per_axis(e1, e2), per_axis_rot=per_axis(R @ e1, R @ e2),
        maxdir_orig=maxdir(e1, e2), maxdir_rot=maxdir(R @ e1, R @ e2),
    )


def phantom_comparison():
    m = sh.cylinder_model(60, 12, 3e-3, 0.3)     # 3-D: all components defined
    p1 = (sh.place_rim_electrode(m, 45, 3), sh.place_rim_electrode(m, 135, 3))
    p2 = (sh.place_rim_electrode(m, 315, 3), sh.place_rim_electrode(m, 225, 3))
    res = sh.simulate_ti(m, p1, p2, 2e-3, 2e-3)

    per_axis = res.envelope
    maxdir = sh.ti_envelope_max_direction(res.E1, res.E2)
    mask = m.mask & np.isfinite(per_axis) & np.isfinite(maxdir)
    a, b = per_axis[mask], maxdir[mask]
    ratio = a / np.where(b > 0, b, np.nan)
    return dict(
        n=int(a.size),
        per_axis_median=float(np.median(a)), maxdir_median=float(np.median(b)),
        ratio_median=float(np.nanmedian(ratio)),
        ratio_p5=float(np.nanpercentile(ratio, 5)),
        ratio_p95=float(np.nanpercentile(ratio, 95)),
        pearson_r=float(np.corrcoef(a, b)[0, 1]),
        frac_per_axis_exceeds=float(np.mean(a > b)),
    )


def main():
    os.makedirs(OUT, exist_ok=True)

    print("Closed-form checks of max-direction envelope:")
    for name, got, want in analytic_cases():
        ok = "OK" if abs(got - want) < 1e-6 else "MISMATCH"
        print(f"  {name:22s} {got:.6f}  (expect {want:.6f})  {ok}")

    ri = rotation_invariance()
    print("\nRotation invariance (same fields, frame rotated 40 deg):")
    print(f"  per-axis composite : {ri['per_axis_orig']:.6f} -> {ri['per_axis_rot']:.6f}"
          f"   (changes by {100*abs(ri['per_axis_rot']/ri['per_axis_orig']-1):.2f}%)")
    print(f"  max-direction      : {ri['maxdir_orig']:.6f} -> {ri['maxdir_rot']:.6f}"
          f"   (invariant)")

    ph = phantom_comparison()
    print(f"\nPhantom comparison ({ph['n']} voxels):")
    print(f"  median per-axis {ph['per_axis_median']:.4f} vs "
          f"max-direction {ph['maxdir_median']:.4f} V/m")
    print(f"  ratio per-axis/max-direction: median {ph['ratio_median']:.4f}, "
          f"p5-p95 {ph['ratio_p5']:.4f}-{ph['ratio_p95']:.4f}")
    print(f"  Pearson r between definitions: {ph['pearson_r']:.4f}")
    print(f"  per-axis EXCEEDS the maximal modulation in "
          f"{100*ph['frac_per_axis_exceeds']:.1f}% of voxels")

    import csv
    with open(os.path.join(OUT, "envelope_definition.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["metric", "value"])
        for k, v in {**ri, **ph}.items():
            w.writerow([k, v])
    print(f"\nwrote {OUT}/envelope_definition.csv")


if __name__ == "__main__":
    main()
