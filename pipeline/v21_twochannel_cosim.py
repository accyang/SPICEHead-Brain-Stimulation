# -*- coding: utf-8 -*-
"""Two-channel circuit-head co-simulation in MRI-derived heads — fourth-round review, comment 6.

The circuit demonstration so far drove ONE channel into a homogeneous cylinder, so it could show
compliance-limited current droop but not the thing that justifies putting the head in the netlist:
circuit non-ideality -> unequal channel currents -> a spatially altered TI envelope in an
individual anatomy. This runs that experiment, simulation-only, as the reviewer prescribed.

Design
------
Eight MRI-derived heads (the FEM-validation subjects), dx = 2 mm, thalamic montage
T8+P8 / F7+T7. Each channel is driven by the non-ideal current-source model of Methods
(Norton source, programmed 1 mA, R_out = 1 MOhm, +/-15 V compliance rail) through per-electrode
contact resistance. Channel 1 contacts stay at 1 kOhm; channel 2 contacts are degraded to
1, 5, 20 and 50 kOhm.

Exactness. The head network is linear, so it is reduced EXACTLY to its 2x2 port resistance matrix
(Kron/Schur reduction, computed from the same two unit-current solves that give the fields); the
nonlinear driver pair is solved against that reduced network (piecewise-linear compliance clamp,
all four clamp states checked for consistency); and the in-head field is reconstructed by
superposition of the unit fields scaled by the delivered currents. Nothing is approximated by the
reduction. A reduced head+driver netlist is emitted per subject and, when ngspice is on PATH, run
as a cross-check on the delivered currents.

Reported per subject x scenario, against the ideal-current reference (both channels forced to
1 mA, which is what a field-only pipeline assumes):

  I1, I2 delivered (mA); which channel clamps
  thalamus median envelope and % change
  envelope center of mass over grey matter, displacement (mm); top-decile CoM displacement
  peak displacement (mm)
  focality F(thalamus) and change

  python pipeline/v21_twochannel_cosim.py [n_subjects]
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config
sys.path.insert(0, _here)

from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components
from spicehead.envelope_fast import envelope_max_direction_flat
from montage_search import (snap, roi_mask, ELEC_RADIUS_MM, ROI_RADIUS_MM, EXCLUDE_MM,
                            SCALP, SCORE)
from v21_head_convergence import make_solver
from v21_simnibs_validation import subjects

OUT = os.path.join(config.RESULTS, "v21_twochannel")
DX = 2.0
MONTAGE = (("T8", "P8"), ("F7", "T7"))     # the Table 3 thalamic optimum
I_PROG = 1e-3                              # programmed current per channel (A)
R_OUT = 1e6                                # source output impedance (Ohm)
V_CC = 15.0                                # compliance rail (V)
R1_EL = 1e3                                # channel-1 contact resistance, per electrode (Ohm)
SWEEP = (1e3, 5e3, 20e3, 50e3)             # channel-2 contact resistance, per electrode (Ohm)
TARGET = "thalamus"


def solve_drivers(R, S, i_prog=I_PROG, r_out=R_OUT, v_cc=V_CC):
    """Delivered port currents for two clamped Norton drivers on a linear 2-port.

    R : 2x2 head port-resistance matrix (V_port = R @ I).
    S : series contact resistance per channel (both electrodes of the pair summed).
    Driver terminal voltage: V_drv_k = S_k I_k + (R @ I)_k. Unclamped channels satisfy
    I_k = i_prog - V_drv_k / r_out; clamped channels sit at V_drv_k = v_cc. All four clamp
    states are solved; the consistent one (unclamped => V <= v_cc, clamped => I <= unclamped
    demand) is returned. The clamp is ideal, matching the sharp-knee numerical limiter of the
    Methods (diode N = 0.02).
    """
    R = np.asarray(R, float)
    S = np.asarray(S, float)
    for clamp in ((False, False), (False, True), (True, False), (True, True)):
        A = np.zeros((2, 2))
        b = np.zeros(2)
        for k in range(2):
            if clamp[k]:
                A[k, k] += S[k]
                A[k] += R[k]
                b[k] = v_cc
            else:
                A[k, k] += 1.0 + S[k] / r_out
                A[k] += R[k] / r_out
                b[k] = i_prog
        I = np.linalg.solve(A, b)
        V_drv = S * I + R @ I
        ok = True
        for k in range(2):
            if clamp[k]:
                ok &= I[k] <= i_prog - V_drv[k] / r_out + 1e-12
            else:
                ok &= V_drv[k] <= v_cc + 1e-9
        if ok:
            return I, V_drv, clamp
    raise RuntimeError("no consistent clamp state")


def reduced_netlist(R, S2, r1=R1_EL):
    """ngspice netlist: exact 2-port head (CCVS realization) + the two drivers."""
    L = ["* two-channel driver + Kron-reduced MRI head (exact for a linear network)",
         "* port k: V_k = R[k,0]*I(port0) + R[k,1]*I(port1), realized with H sources",
         f"Vs1 P1 M1 DC 0",                                 # ammeter, port 1
         f"Vs2 P2 M2 DC 0",                                 # ammeter, port 2
         f"H11 M1 X1 Vs1 {R[0,0]:.6f}",
         f"H12 X1 0  Vs2 {R[0,1]:.6f}",
         f"H21 M2 X2 Vs1 {R[1,0]:.6f}",
         f"H22 X2 0  Vs2 {R[1,1]:.6f}",
         f"Idrv1 0 D1 DC {I_PROG:.10g}",
         f"Rout1 D1 0 {R_OUT:.10g}",
         f"Rel1 D1 P1 {2*r1:.10g}",
         f"Idrv2 0 D2 DC {I_PROG:.10g}",
         f"Rout2 D2 0 {R_OUT:.10g}",
         f"Rel2 D2 P2 {S2:.10g}",
         f"Vcc VCC 0 DC {V_CC:.10g}",
         "D1c D1 VCC DCLAMP", "D2c D2 VCC DCLAMP",
         ".model DCLAMP D(Is=1e-15 N=0.02)",
         ".control", "op", "print i(Vs1) i(Vs2)", ".endc", ".end"]
    return "\n".join(L) + "\n"


def ngspice_currents(txt):
    if not shutil.which("ngspice"):
        return None
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "two.cir")
        open(p, "w").write(txt)
        r = subprocess.run(["ngspice", "-b", p], capture_output=True, text=True, timeout=120)
    import re
    out = {}
    for name in ("vs1", "vs2"):
        m = re.search(rf"i\({name}\)\s*=\s*([-+0-9.eE]+)", r.stdout, re.I)
        if m:
            out[name] = abs(float(m.group(1)))
    return (out.get("vs1"), out.get("vs2")) if len(out) == 2 else None


def com(env, world):
    wsum = env.sum()
    return (world * env[:, None]).sum(0) / wsum if wsum > 0 else np.full(3, np.nan)


def main(n=8):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for si, (cid, m2m, age, site) in enumerate(subjects(n), 1):
        t0 = time.time()
        seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
        model, affine = mri.head_model(seg, DX)   # v21/v22: nearest-label network; v23: homogenized
        labels = model.labels
        gm = np.isin(labels, SCORE)
        gm_idx = np.argwhere(gm)
        gm_world = (affine[:3, :3] @ gm_idx.T).T + affine[:3, 3]
        rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
        roi_sel = roi_mask(labels.shape, gm_idx, gm_world, rois[TARGET], ROI_RADIUS_MM)[gm]
        out_sel = ~roi_mask(labels.shape, gm_idx, gm_world, rois[TARGET], EXCLUDE_MM)[gm]

        scalp_vox = np.argwhere(labels == SCALP)
        ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
        pads = {}
        for lab in [l for p in MONTAGE for l in p]:
            v = snap(scalp_vox, mri.world_to_voxel(affine, ej[lab]))
            pads[lab] = Electrode(contact_nodes(
                model, tuple(int(x) for x in v), radius_nodes=ELEC_RADIUS_MM / DX), lab)

        solver, _ = make_solver(model.G)
        E, padV = [], []
        for (a, b) in MONTAGE:
            V = solver(pair_injection(model, pads[a], pads[b], I_PROG))
            Ex, Ey, Ez = efield_components(model, V)
            E.append(tuple(np.nan_to_num(c[gm]).astype(np.float32) for c in (Ex, Ey, Ez)))
            padV.append({lab: float(np.mean(V[pads[lab].nodes])) for lab in pads})
        # exact 2x2 port resistance from the same two solves: R[j,k] = dV across pair j
        # per unit current into pair k
        R = np.array([[(padV[k][MONTAGE[j][0]] - padV[k][MONTAGE[j][1]]) / I_PROG
                       for k in range(2)] for j in range(2)])

        def metrics(I1, I2):
            s1, s2 = I1 / I_PROG, I2 / I_PROG
            env = envelope_max_direction_flat(
                s1 * E[0][0], s1 * E[0][1], s1 * E[0][2],
                s2 * E[1][0], s2 * E[1][1], s2 * E[1][2])
            T = float(np.median(env[roi_sel]))
            O = float(np.percentile(env[out_sel], 99))
            return dict(env=env, T=T, F=T / O if O > 0 else 0.0,
                        com=com(env, gm_world),
                        com_top=com(np.where(env >= np.quantile(env, 0.9), env, 0.0), gm_world),
                        peak=gm_world[int(np.argmax(env))])

        ideal = metrics(I_PROG, I_PROG)
        if si == 1:
            open(os.path.join(OUT, "example_reduced_driver.cir"), "w").write(
                reduced_netlist(R, 2 * SWEEP[-1]))
        for r2 in SWEEP:
            S = np.array([2 * R1_EL, 2 * r2])
            I, V_drv, clamp = solve_drivers(R, S)
            ng = ngspice_currents(reduced_netlist(R, S[1]))
            m = metrics(I[0], I[1])
            d_com = float(np.linalg.norm(m["com"] - ideal["com"]))
            d_top = float(np.linalg.norm(m["com_top"] - ideal["com_top"]))
            d_peak = float(np.linalg.norm(m["peak"] - ideal["peak"]))
            # reciprocity: R12 and R21 come from the two INDEPENDENT solves (pair-1 injection
            # read at pair-2 pads and vice versa), so their agreement is a genuine numerical
            # check on the reduction, available even where ngspice is not installed.
            recip = 100 * abs(R[0, 1] - R[1, 0]) / abs(R[0, 1])
            rows.append(dict(
                subject=cid, site=site, r2_contact_ohm=r2,
                R11=round(R[0, 0], 1), R22=round(R[1, 1], 1),
                R12=round(R[0, 1], 4), R21=round(R[1, 0], 4),
                reciprocity_err_pct=round(recip, 4),
                I1_mA=round(1e3 * I[0], 4), I2_mA=round(1e3 * I[1], 4),
                imbalance=round(I[1] / I[0], 4),
                ch1_clamped=clamp[0], ch2_clamped=clamp[1],
                V1=round(V_drv[0], 2), V2=round(V_drv[1], 2),
                ng_I1_mA=(round(1e3 * ng[0], 4) if ng else None),
                ng_I2_mA=(round(1e3 * ng[1], 4) if ng else None),
                T_ideal=round(ideal["T"], 6), T=round(m["T"], 6),
                dT_pct=round(100 * (m["T"] - ideal["T"]) / ideal["T"], 2),
                F_ideal=round(ideal["F"], 4), F=round(m["F"], 4),
                dF_pct=round(100 * (m["F"] - ideal["F"]) / ideal["F"], 2),
                com_shift_mm=round(d_com, 2), com_top_shift_mm=round(d_top, 2),
                peak_shift_mm=round(d_peak, 2)))
            print(f"  [{si}] {cid} r2={r2/1e3:g}k: I=({1e3*I[0]:.3f},{1e3*I[1]:.3f}) mA "
                  f"clamp={clamp} dT={rows[-1]['dT_pct']:+.1f}% "
                  f"CoM {d_com:.1f} mm top {d_top:.1f} mm peak {d_peak:.1f} mm "
                  f"F {ideal['F']:.3f}->{m['F']:.3f}"
                  + (f" [ngspice ok]" if ng and abs(ng[1] - I[1]) < 2e-5 else ""), flush=True)
        print(f"  [{si}] {cid} done in {(time.time()-t0)/60:.1f} min", flush=True)

    with open(os.path.join(OUT, "twochannel.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n=== mean over subjects, by channel-2 contact resistance ===")
    print(f"{'r2':>6}{'I2 (mA)':>9}{'imbal':>8}{'dT %':>8}{'dF %':>8}"
          f"{'CoM mm':>8}{'top mm':>8}{'peak mm':>9}")
    for r2 in SWEEP:
        s = [r for r in rows if r["r2_contact_ohm"] == r2]
        f = lambda k: float(np.mean([r[k] for r in s]))
        print(f"{r2/1e3:>5g}k{f('I2_mA'):>9.3f}{f('imbalance'):>8.3f}{f('dT_pct'):>8.1f}"
              f"{f('dF_pct'):>8.1f}{f('com_shift_mm'):>8.1f}{f('com_top_shift_mm'):>8.1f}"
              f"{f('peak_shift_mm'):>9.1f}")
    print("wrote", os.path.join(OUT, "twochannel.csv"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
