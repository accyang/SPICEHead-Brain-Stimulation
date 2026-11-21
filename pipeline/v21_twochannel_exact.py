# -*- coding: utf-8 -*-
"""Exact two-channel co-simulation with the contact network inside the reduction.

Fifth-round review, comment 2. The previous implementation built the field basis from
`pair_injection`, which spreads each pad's current EQUALLY over its scalp nodes — a pattern fixed
independently of the contact resistance — and then treated the contacts as one lumped series
resistance per channel. The reviewer's objection is exact: if changing R_el redistributes current
among a pad's nodes, that basis is not a valid representation and "exact" is the wrong word.

This version removes the assumption. The distributed contact network is placed INSIDE the linear
system: each pad gets its own driver node, connected to every one of its n scalp nodes through
n*R_el (so the parallel combination is R_el, as the Methods state). The augmented network is then
reduced to the 2x2 matrix of its two differential ports, which are now genuinely the only external
connections — option (1) of the review. Because the ports are the only terminals, superposition of
the two unit-port solutions is exact for ANY delivered current pair, and the within-pad
distribution is whatever the physics dictates rather than whatever we assumed.

Three checks are run and reported (option (3) of the review):

  port voltages   inject the delivered currents into the UNREDUCED augmented network and compare
                  the resulting port voltages against R_port @ I
  field           compare the unreduced field against the superposition I1*E1 + I2*E2
  driver          confirm the delivered currents satisfy the Norton/clamp equations

and, separately, the previous fixed-uniform-injection model is evaluated on the same subjects so
the size of the old approximation is measured rather than asserted.

  python pipeline/v21_twochannel_exact.py [n_subjects] [dx_mm] [conditions_kOhm]
  python pipeline/v21_twochannel_exact.py 8 2.0
  python pipeline/v21_twochannel_exact.py 8 1.0 1,20,50
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

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
from v21_twochannel_cosim import solve_drivers, com, MONTAGE, I_PROG, R_OUT, V_CC, R1_EL, TARGET
from v21_twochannel_timedomain import solve_timedomain, fundamental

# Two-carrier operating point (Brain Stimulation review, comment 1). Compliance clipping is an
# instantaneous nonlinearity, and the TI envelope is formed by the FUNDAMENTAL amplitude of each
# carrier after clipping — which for a clipped sinusoid exceeds the static .op value. Both are
# computed; the two-carrier amplitudes are the ones the envelope is built from.
F1_HZ, DF_HZ, SPC = 2000.0, 10.0, 100

OUT = os.path.join(config.RESULTS, "v21_twochannel")


def augment(model, pads, r_el):
    """Head network + one driver node per pad, each tied to its n nodes through n*R_el.

    Returns (G_aug, driver_indices). r_el maps pad label -> contact resistance of the whole pad.
    """
    N = model.n_nodes
    G = csr_matrix(model.G).tocoo()
    rows = [G.row]
    cols = [G.col]
    data = [G.data]
    drv = {}
    for k, (lab, pad) in enumerate(pads.items()):
        d = N + k
        drv[lab] = d
        n = pad.nodes.size
        g = 1.0 / (n * r_el[lab])          # per-node conductance; n in parallel gives 1/r_el
        idx = pad.nodes.astype(np.int64)
        rows += [idx, np.full(n, d), idx, np.full(n, d)]
        cols += [idx, np.full(n, d), np.full(n, d), idx]
        data += [np.full(n, g), np.full(n, g), np.full(n, -g), np.full(n, -g)]
    A = coo_matrix((np.concatenate(data),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(N + 4, N + 4)).tocsr()
    return A, drv


def port_solve(solver, n_aug, dp, dm, current=1.0):
    inj = np.zeros(n_aug)
    inj[dp] += current
    inj[dm] -= current
    return solver(inj)


def main(n=8, dx=2.0, conds=(1e3, 5e3, 20e3, 50e3), checks=True, shard=(0, 1)):
    os.makedirs(OUT, exist_ok=True)
    rows = []   # NB: do not bind a local named checks here; it shadows the parameter
    labs = [l for p in MONTAGE for l in p]          # [ch1+, ch1-, ch2+, ch2-]
    # Each head's rows are saved as soon as it finishes, so an interrupted run resumes and the
    # heads can be split across processes (--shard k/n); an unsharded run then only merges.
    parts = os.path.join(OUT, f"parts_dx{dx:g}")
    os.makedirs(parts, exist_ok=True)
    k_sh, n_sh = shard
    for si, (cid, m2m, age, site) in enumerate(subjects(n), 1):
        if n_sh > 1 and (si - 1) % n_sh != k_sh:
            continue
        part = os.path.join(parts, f"{cid}.json")
        if os.path.exists(part):
            saved = json.load(open(part, encoding="utf-8"))
            if [r["r2_contact_ohm"] for r in saved] == list(conds):
                rows += saved
                print(f"  [{si}] {cid}: saved rows loaded", flush=True)
                continue
        n_before = len(rows)
        t0 = time.time()
        seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
        model, affine = mri.head_model(seg, dx)   # v21/v22: nearest-label network; v23: homogenized
        labels = model.labels
        N = model.n_nodes
        gm = np.isin(labels, SCORE)
        gm_idx = np.argwhere(gm)
        gm_world = (affine[:3, :3] @ gm_idx.T).T + affine[:3, 3]
        rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
        roi_sel = roi_mask(labels.shape, gm_idx, gm_world, rois[TARGET], ROI_RADIUS_MM)[gm]
        out_sel = ~roi_mask(labels.shape, gm_idx, gm_world, rois[TARGET], EXCLUDE_MM)[gm]

        scalp_vox = np.argwhere(labels == SCALP)
        ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
        pads = {}
        for lab in labs:
            v = snap(scalp_vox, mri.world_to_voxel(affine, ej[lab]))
            pads[lab] = Electrode(contact_nodes(
                model, tuple(int(x) for x in v), radius_nodes=ELEC_RADIUS_MM / dx), lab)
        print(f"  [{si}] {cid}: {N} nodes, pads " +
              ", ".join(f"{l}={pads[l].nodes.size}" for l in labs), flush=True)

        def metrics(env):
            T = float(np.median(env[roi_sel]))
            O = float(np.percentile(env[out_sel], 99))
            return dict(T=T, F=T / O if O > 0 else 0.0, com=com(env, gm_world),
                        com_top=com(np.where(env >= np.quantile(env, 0.9), env, 0.0), gm_world),
                        peak=gm_world[int(np.argmax(env))])

        ideal = None
        for r2 in conds:
            r_el = {labs[0]: R1_EL, labs[1]: R1_EL, labs[2]: r2, labs[3]: r2}
            A, drv = augment(model, pads, r_el)
            solver, _ = make_solver(A)
            # exact 2-port: the driver nodes are the ONLY external terminals
            V1 = port_solve(solver, N + 4, drv[labs[0]], drv[labs[1]])
            V2 = port_solve(solver, N + 4, drv[labs[2]], drv[labs[3]])
            R = np.array([[V1[drv[labs[0]]] - V1[drv[labs[1]]],
                           V2[drv[labs[0]]] - V2[drv[labs[1]]]],
                          [V1[drv[labs[2]]] - V1[drv[labs[3]]],
                           V2[drv[labs[2]]] - V2[drv[labs[3]]]]])
            I, V_drv, clamp = solve_drivers(R, np.zeros(2))   # contacts already inside R
            # two-carrier delivered fundamentals over one beat period
            T_beat = 1.0 / DF_HZ
            nt = int(round(F1_HZ * SPC * T_beat))
            tt = np.arange(nt) / (nt / T_beat)
            w1, w2 = 2 * np.pi * F1_HZ, 2 * np.pi * (F1_HZ + DF_HZ)
            Itd, _ = solve_timedomain(R, np.vstack([I_PROG * np.sin(w1 * tt),
                                                    I_PROG * np.sin(w2 * tt)]))
            Atd = np.array([fundamental(Itd[0], tt, w1, 0.0),
                            fundamental(Itd[1], tt, w2, 0.0)])

            E1 = tuple(np.nan_to_num(c[gm]).astype(np.float64)
                       for c in efield_components(model, V1[:N]))
            E2 = tuple(np.nan_to_num(c[gm]).astype(np.float64)
                       for c in efield_components(model, V2[:N]))
            env = envelope_max_direction_flat(*(I[0] * np.asarray(E1)),
                                              *(I[1] * np.asarray(E2)))
            m = metrics(env)
            env_td = envelope_max_direction_flat(*(Atd[0] * np.asarray(E1)),
                                                 *(Atd[1] * np.asarray(E2)))
            mtd = metrics(env_td)
            if r2 == conds[0] and ideal is None:
                pass
            # ideal-current reference uses the SAME network (contacts in place), both channels 1 mA
            env_id = envelope_max_direction_flat(*(I_PROG * np.asarray(E1)),
                                                 *(I_PROG * np.asarray(E2)))
            ref = metrics(env_id)

            # The verification block below is a statement about the linear algebra, not about
            # anatomy, so it is run at the working resolution and skipped at 1 mm (--nocheck),
            # where each extra solve costs minutes and would add nothing.
            dR = dV = dE = dres = dI = float("nan")
            # ---- check 1: the augmented network realizes the intended contact model.
            # Driving port 1 with unit current, the drop from the driver node to the pad's
            # mean scalp potential must equal R_el for that pad.
            if checks:
              dR = 0.0
              for lab, sgn in ((labs[0], +1), (labs[1], -1)):
                got = sgn * (V1[drv[lab]] - float(np.mean(V1[pads[lab].nodes])))
                dR = max(dR, abs(got - r_el[lab]) / r_el[lab])

            # ---- check 2: full unreduced netlist with the drivers attached.
            # Unclamped channels are Norton sources (R_OUT across the port), so the complete
            # circuit is linear and solvable directly, with no reduction anywhere; delivered
            # currents are then read back and compared. For a clamped channel the port voltage
            # is the unknown-free quantity, so we inject the reduced solution's current into the
            # unreduced network and verify the port actually sits at the compliance rail.
            if checks and not clamp[0] and not clamp[1]:
                # add R_OUT across each port in COO and sum; tolil() on a ~400k matrix is
                # minutes of pure overhead
                g = 1.0 / R_OUT
                rr, cc, vv = [], [], []
                for (dp, dm) in ((drv[labs[0]], drv[labs[1]]), (drv[labs[2]], drv[labs[3]])):
                    rr += [dp, dm, dp, dm]
                    cc += [dp, dm, dm, dp]
                    vv += [g, g, -g, -g]
                Afull = (A + coo_matrix((vv, (rr, cc)), shape=A.shape).tocsr()).tocsr()
                sf, _ = make_solver(Afull)
                inj = np.zeros(N + 4)
                for (dp, dm) in ((drv[labs[0]], drv[labs[1]]), (drv[labs[2]], drv[labs[3]])):
                    inj[dp] += I_PROG; inj[dm] -= I_PROG
                Vf = sf(inj)
                Vpf = np.array([Vf[drv[labs[0]]] - Vf[drv[labs[1]]],
                                Vf[drv[labs[2]]] - Vf[drv[labs[3]]]])
                If = I_PROG - Vpf / R_OUT
                dV = float(np.max(np.abs(If - I)) / I_PROG)
            elif checks:
                inj = np.zeros(N + 4)
                inj[drv[labs[0]]] += I[0]; inj[drv[labs[1]]] -= I[0]
                inj[drv[labs[2]]] += I[1]; inj[drv[labs[3]]] -= I[1]
                Vf = solver(inj)
                Vpf = np.array([Vf[drv[labs[0]]] - Vf[drv[labs[1]]],
                                Vf[drv[labs[2]]] - Vf[drv[labs[3]]]])
                dV = float(max(abs(Vpf[k] - V_drv[k]) / max(abs(V_drv[k]), 1e-30)
                               for k in range(2)))

            # ---- check 3: field superposition against a direct solve at the delivered
            # currents. The two channels carry different carriers, so each is solved on its own
            # and the envelope is formed from the two vector fields — not from a summed potential.
            if checks:
              Va = solver(np.eye(1, N + 4, drv[labs[0]])[0] * I[0]
                          - np.eye(1, N + 4, drv[labs[1]])[0] * I[0])
              Vb = solver(np.eye(1, N + 4, drv[labs[2]])[0] * I[1]
                          - np.eye(1, N + 4, drv[labs[3]])[0] * I[1])
              Ea = tuple(np.nan_to_num(c[gm]).astype(np.float64)
                         for c in efield_components(model, Va[:N]))
              Eb = tuple(np.nan_to_num(c[gm]).astype(np.float64)
                         for c in efield_components(model, Vb[:N]))
              envf = envelope_max_direction_flat(*Ea, *Eb)
              dE = float(np.max(np.abs(envf - env)) / max(float(np.max(envf)), 1e-30))

            # ---- check 4: driver equations satisfied ------------------------------
            dres = 0.0
            for k in range(2) if checks else ():
                if clamp[k]:
                    dres = max(dres, abs(V_drv[k] - V_CC) / V_CC)
                else:
                    dres = max(dres, abs(I[k] - (I_PROG - V_drv[k] / R_OUT)) / I_PROG)

            # ---- the superseded fixed-uniform-injection model, same subject -------
            Vu1 = None
            if checks:
                solver_h, _ = make_solver(model.G)
                Vu1 = solver_h(pair_injection(model, pads[labs[0]], pads[labs[1]], I_PROG))
                Vu2 = solver_h(pair_injection(model, pads[labs[2]], pads[labs[3]], I_PROG))
                padV = [{l: float(np.mean(V[pads[l].nodes])) for l in labs} for V in (Vu1, Vu2)]
                Ru = np.array([[(padV[k][MONTAGE[j][0]] - padV[k][MONTAGE[j][1]]) / I_PROG
                                for k in range(2)] for j in range(2)])
                Iu, _, _ = solve_drivers(Ru, np.array([2 * R1_EL, 2 * r2]))
                dI = float(np.max(np.abs(Iu - I)) / I_PROG)

            d_com = float(np.linalg.norm(m["com"] - ref["com"]))
            d_top = float(np.linalg.norm(m["com_top"] - ref["com_top"]))
            d_peak = float(np.linalg.norm(m["peak"] - ref["peak"]))
            rows.append(dict(
                subject=cid, site=site, dx_mm=dx, r2_contact_ohm=r2,
                R11=round(R[0, 0], 3), R22=round(R[1, 1], 3),
                R12=round(R[0, 1], 5), R21=round(R[1, 0], 5),
                reciprocity_err_pct=round(100 * abs(R[0, 1] - R[1, 0]) / max(abs(R[0, 1]), 1e-30), 5),
                I1_mA=round(1e3 * I[0], 5), I2_mA=round(1e3 * I[1], 5),
                imbalance=round(I[1] / I[0], 5),
                ch1_clamped=bool(clamp[0]), ch2_clamped=bool(clamp[1]),
                T_ideal=round(ref["T"], 6), T=round(m["T"], 6),
                dT_pct=round(100 * (m["T"] - ref["T"]) / ref["T"], 3),
                F_ideal=round(ref["F"], 5), F=round(m["F"], 5),
                dF_pct=round(100 * (m["F"] - ref["F"]) / ref["F"], 3),
                com_shift_mm=round(d_com, 3), com_top_shift_mm=round(d_top, 3),
                peak_shift_mm=round(d_peak, 3),
                A1_td_mA=round(1e3 * Atd[0], 5), A2_td_mA=round(1e3 * Atd[1], 5),
                td_over_static_I2=round(Atd[1] / I[1], 5),
                T_td=round(mtd["T"], 6),
                dT_td_pct=round(100 * (mtd["T"] - ref["T"]) / ref["T"], 3),
                F_td=round(mtd["F"], 5),
                dF_td_pct=round(100 * (mtd["F"] - ref["F"]) / ref["F"], 3),
                com_shift_td_mm=round(float(np.linalg.norm(mtd["com"] - ref["com"])), 3),
                com_top_shift_td_mm=round(float(np.linalg.norm(mtd["com_top"] - ref["com_top"])), 3),
                peak_shift_td_mm=round(float(np.linalg.norm(mtd["peak"] - ref["peak"])), 3),
                chk_contact_rel=dR, chk_fullnet_rel=dV, chk_field_rel=dE,
                chk_driver_rel=dres, uniform_model_dI_rel=dI))
            print(f"      r2={r2/1e3:g}k I=({1e3*I[0]:.3f},{1e3*I[1]:.3f})→2f({1e3*Atd[0]:.3f},"
                  f"{1e3*Atd[1]:.3f}) mA CoM {d_com:.1f}→{rows[-1]['com_shift_td_mm']:.1f} "
                  f"dT {rows[-1]['dT_pct']:+.0f}→{rows[-1]['dT_td_pct']:+.0f}% "
                  f"| checks Rel {dR:.1e} full {dV:.1e} E {dE:.1e} drv {dres:.1e} "
                  f"| uniform dI {dI:.1e}", flush=True)
        with open(part + ".part", "w", encoding="utf-8") as fh:
            json.dump(rows[n_before:], fh)
        os.replace(part + ".part", part)
        print(f"  [{si}] {cid} done in {(time.time()-t0)/60:.1f} min", flush=True)

    if n_sh > 1:
        print(f"shard {k_sh}/{n_sh} done: {len({r['subject'] for r in rows})} heads", flush=True)
        return
    tag = f"_dx{dx:g}"
    path = os.path.join(OUT, f"twochannel_exact{tag}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== dx = {dx:g} mm, {len({r['subject'] for r in rows})} subjects ===")
    print(f"{'r2':>6}{'I2 (mA)':>10}{'dT %':>9}{'dF %':>9}"
          f"{'CoM mm (mean±SD, range)':>34}{'top mm':>18}{'peak mm':>16}")
    for r2 in conds:
        s = [r for r in rows if r["r2_contact_ohm"] == r2]
        f = lambda k: np.array([r[k] for r in s], float)
        c, t, pk = f("com_shift_mm"), f("com_top_shift_mm"), f("peak_shift_mm")
        print(f"{r2/1e3:>5g}k{f('I2_mA').mean():>10.3f}{f('dT_pct').mean():>9.1f}"
              f"{f('dF_pct').mean():>9.1f}"
              f"{c.mean():>14.1f}±{c.std(ddof=1):<4.1f}[{c.min():.1f}-{c.max():.1f}]"
              f"{t.mean():>10.1f}±{t.std(ddof=1):<4.1f}"
              f"{pk.mean():>9.1f}±{pk.std(ddof=1):<4.1f}")
    import math

    def mx(k):
        """Max over rows, ignoring NaNs left where --nocheck skipped a check."""
        v = [r[k] for r in rows if isinstance(r[k], float) and not math.isnan(r[k])]
        return max(v) if v else float('nan')

    print(f"\nverification (max over all subject-conditions):")
    print(f"  contact model realized (R_el)       : {mx('chk_contact_rel'):.3e}")
    print(f"  full unreduced netlist vs reduced   : {mx('chk_fullnet_rel'):.3e}")
    print(f"  field, superposition vs direct solve: {mx('chk_field_rel'):.3e}")
    print(f"  driver equations residual           : {mx('chk_driver_rel'):.3e}")
    print(f"  reciprocity R12 vs R21 (%)          : {mx('reciprocity_err_pct'):.3e}")
    print(f"  superseded uniform-injection model, delivered-current error (rel. to 1 mA):")
    print(f"                                        {mx('uniform_model_dI_rel'):.3e}")
    print("wrote", path)


if __name__ == "__main__":
    a = sys.argv[1:]
    shard = (0, 1)
    if "--shard" in a:
        i = a.index("--shard")
        shard = tuple(int(x) for x in a[i + 1].split("/"))
        a = a[:i] + a[i + 2:]
    # --nocheck skips the linear-algebra verification block (intended for 1 mm, where it adds
    # about 20 solves per head and nothing new: the same checks pass at 2 mm in every condition)
    checks = "--nocheck" not in a
    a = [x for x in a if x != "--nocheck"]
    n = int(a[0]) if len(a) > 0 else 8
    dx = float(a[1]) if len(a) > 1 else 2.0
    conds = tuple(1e3 * float(x) for x in a[2].split(",")) if len(a) > 2 \
        else (1e3, 5e3, 20e3, 50e3)
    main(n, dx, conds, checks=checks, shard=shard)
