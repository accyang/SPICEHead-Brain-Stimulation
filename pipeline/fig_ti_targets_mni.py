# -*- coding: utf-8 -*-
"""Temporal-interference exposure maps for four targets on the MNI152 template, with figures.

Targets are the insula, hippocampus, anterior cingulate cortex (ACC) and orbitofrontal cortex
(OFC). Each is defined bilaterally from the AAL atlas (OFC = the AAL orbital parcels plus gyrus
rectus) and intersected with grey matter. On the CHARM-segmented MNI152 head the script scores
every two-pair montage of the 19 EEG 10-20 positions with the paper's focality metric, F = T/O:
T is the median envelope in the target and O the 99th percentile of grey-matter envelope more
than 20 mm from it. It then draws the best montage's envelope on the standard MNI152 1 mm T1.

Model: whichever SPICEhead version spicehead.config selects (v22 by default: skin-surface 12 mm
pad contacts and the harmonic interface rule; v21: volumetric contacts and the arithmetic rule),
1 mA per channel and the rotation-invariant maximum-over-direction envelope. Figures go to
TI_targets_MNI for v22 and TI_targets_MNI_v21 for v21. Fields are solved on a 2 mm grid;
resampling to 1 mm is for display only. All pair fields come from 18 single-electrode solves
against Cz, and one pair is checked against a direct solve.

These are simulated exposure maps for one standard head, not activation maps. At 2 mm,
SPICEhead magnitudes exceed FEM by about 15-30% (RESULTS_final_2026-10-01.md, section 3).

  python pipeline/fig_ti_targets_mni.py               (solve, search and draw)
  python pipeline/fig_ti_targets_mni.py --plot-only   (redraw from the saved maps)
"""
import csv
import json
import os
import sys
import textwrap
import time
from itertools import combinations

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt, map_coordinates

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)

from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components
from spicehead.envelope_fast import envelope_max_direction_flat
from v21_head_convergence import make_solver

M2M = os.environ.get("SPICEHEAD_MNI_M2M", r"L:\BrainProbe\SPICEhead\charm\m2m_MNItempl")
AAL = os.environ.get("SPICEHEAD_AAL_NII", r"C:\Users\ALBERT YANG\nilearn_data\aal_SPM12\aal\atlas\AAL.nii")
AAL_REGIONS = os.path.join(_here, "aal", "aal_regions.json")
CAP = os.environ.get("SPICEHEAD_CAP_CSV", r"C:\Users\ALBERT YANG\SimNIBS-4.6\simnibs_env\Lib\site-packages\simnibs\resources"
                     r"\ElectrodeCaps_MNI\EEG10-10_UI_Jurak_2007.csv")      # env overrides: cluster runs (28 Oct 2026)
from spicehead import config
# Paper A figures folder for v21/v22 only; other model versions (v23) write under their own results folder
OUT = (os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "figures",
                    "TI_targets_MNI" + ("" if config.MODEL == "v22" else f"_{config.MODEL}"))
       if config.MODEL in ("v21", "v22") else os.path.join(config.RESULTS, "figures", "TI_targets_MNI"))
MODEL_TEXT = {"v22": "skin-surface 12 mm pad contacts, harmonic interface rule",
              "v21": "volumetric 12 mm contacts, arithmetic interface rule",
              "v23": "skin-surface 12 mm pad contacts, network homogenized from the 0.5 mm segmentation"}[config.MODEL]

DX = 2.0
REF = "Cz"
ELEC_RADIUS_MM, MIN_DIST_MM, EXCLUDE_MM = 12.0, 40.0, 20.0
WM, GM, SCALP = 1, 2, 5
LABELS_1020 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
               "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
ORB = ["Frontal_Sup_Orb", "Frontal_Mid_Orb", "Frontal_Inf_Orb", "Frontal_Med_Orb", "Rectus"]
TARGETS = {
    "insula": dict(title="Insula", parcels=["Insula_L", "Insula_R"]),
    "hippocampus": dict(title="Hippocampus", parcels=["Hippocampus_L", "Hippocampus_R"]),
    "acc": dict(title="Anterior cingulate cortex",
                parcels=["Cingulum_Ant_L", "Cingulum_Ant_R"]),
    "ofc": dict(title="Orbitofrontal cortex",
                parcels=[f"{p}_{h}" for p in ORB for h in ("L", "R")]),
}


def read_cap():
    d = {}
    for line in open(CAP):
        p = line.strip().split(",")
        if len(p) >= 5 and p[0].endswith("Electrode") and p[4] in LABELS_1020:
            d[p[4]] = np.array([float(p[1]), float(p[2]), float(p[3])])
    missing = [l for l in LABELS_1020 if l not in d]
    if missing:
        raise SystemExit(f"cap file lacks {missing}")
    return d


def solve_template():
    """Model, targets, electrodes and all pair fields on the 2 mm template grid."""
    t0 = time.time()
    seg = os.path.join(M2M, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, aff = mri.head_model(seg, DX)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    gm = labels == GM
    brain = np.isin(labels, (WM, GM))

    tpl = nib.Nifti1Image(labels.astype(np.int16), aff)
    aal = np.rint(np.asarray(resample_from_to(nib.load(AAL), tpl, order=0).dataobj)).astype(int)
    value = {r["name"]: r["value"] for r in json.load(open(AAL_REGIONS))["regions"]}
    tmask, omask = {}, {}
    for key, t in TARGETS.items():
        m = np.isin(aal, [value[p] for p in t["parcels"]]) & gm
        tmask[key] = m
        omask[key] = gm & (distance_transform_edt(~m, sampling=DX) > EXCLUDE_MM)

    cap = read_cap()
    scalp = np.argwhere(labels == SCALP)
    inv = np.linalg.inv(aff)
    evox, elecs = {}, {}
    for lab in LABELS_1020:
        ijk = (inv @ np.append(cap[lab], 1))[:3]
        v = scalp[int(np.argmin(np.linalg.norm(scalp - ijk, axis=1)))]
        evox[lab] = v
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                               radius_nodes=ELEC_RADIUS_MM / DX), lab)
    pairs = [(a, b) for a, b in combinations(LABELS_1020, 2)
             if np.linalg.norm(evox[a] - evox[b]) * DX >= MIN_DIST_MM]
    solver, ncomp = make_solver(model.G)
    print(f"template: {model.n_nodes:,} nodes, {int(gm.sum()):,} GM voxels, {len(pairs)} valid "
          f"pairs; set up in {time.time()-t0:.0f}s", flush=True)

    basis = {}
    for i, lab in enumerate([l for l in LABELS_1020 if l != REF], 1):
        V = solver(pair_injection(model, elecs[lab], elecs[REF], 1e-3))
        basis[lab] = np.stack([np.nan_to_num(c).astype(np.float32)
                               for c in efield_components(model, V)])
        print(f"  basis {i}/18 ({time.time()-t0:.0f}s)", flush=True)

    def pair_field(a, b):
        return (basis[a] if a != REF else 0.0) - (basis[b] if b != REF else 0.0)

    a, b = "F7", "T8"
    direct = np.stack([np.nan_to_num(c) for c in efield_components(
        model, solver(pair_injection(model, elecs[a], elecs[b], 1e-3)))])
    check = float(np.abs(pair_field(a, b) - direct).max() / np.abs(direct).max())
    print(f"  superposition check {a}-{b}: {check:.1e}", flush=True)
    if check > 1e-5:
        raise SystemExit("superposition check failed")
    return dict(labels=labels, aff=aff, gm=gm, brain=brain, aal=aal, tmask=tmask, omask=omask,
                cap=cap, pairs=pairs, pair_field=pair_field, check=check,
                n_nodes=int(model.n_nodes))


def search(S):
    """Focality of every two-pair montage for every target; the top five per target."""
    t0 = time.time()
    gm = S["gm"]
    flat = {}
    for p in S["pairs"]:
        F = S["pair_field"](*p)
        flat[p] = tuple(F[c][gm] for c in range(3))
    tsel = {k: S["tmask"][k][gm] for k in TARGETS}
    osel = {k: S["omask"][k][gm] for k in TARGETS}
    montages = [(pa, pb) for pa, pb in combinations(S["pairs"], 2) if len(set(pa) | set(pb)) == 4]
    scores = {k: [] for k in TARGETS}
    for i, (pa, pb) in enumerate(montages, 1):
        A, B = flat[pa], flat[pb]
        env = envelope_max_direction_flat(A[0], A[1], A[2], B[0], B[1], B[2])
        for k in TARGETS:
            T = float(np.median(env[tsel[k]]))
            O = float(np.percentile(env[osel[k]], 99))
            scores[k].append((T / O if O > 0 else 0.0, T, O, pa, pb))
        if i % 2000 == 0 or i == len(montages):
            print(f"  scored {i}/{len(montages)} montages ({time.time()-t0:.0f}s)", flush=True)
    return {k: sorted(v, key=lambda x: -x[0])[:5] for k, v in scores.items()}, len(montages)


def label(pa, pb):
    return " / ".join("+".join(sorted(p)) for p in sorted((pa, pb), key=lambda p: sorted(p)))


def to_1mm(vol2, aff2, ref_img, order=1):
    """Resample a 2 mm grid volume onto the 1 mm template grid."""
    sh = ref_img.shape[:3]
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in sh], indexing="ij")
    ijk1 = np.stack([ii, jj, kk, np.ones_like(ii)]).reshape(4, -1)
    vox2 = (np.linalg.inv(aff2) @ ref_img.affine @ ijk1)[:3]
    return map_coordinates(vol2, vox2, order=order, mode="constant", cval=0.0).reshape(sh)


def slices(vol, c):
    """Axial, coronal and sagittal slices in neurological orientation (left on the left)."""
    i, j, k = c
    return (vol[::-1, :, k].T, vol[::-1, j, :].T, vol[i, :, :].T)


def cap_layout(cap):
    """2-D EEG layout: azimuthal equidistant projection about the vertex, nose up.

    A plain top-down view of the 3-D positions crowds the lateral sites onto the inner ring (C3
    lands on T7). Projecting each position's angle from the vertex, as EEG topographic plots do,
    keeps the rings apart.
    """
    P = np.array([cap[l] for l in LABELS_1020])
    A = np.c_[2 * P, np.ones(len(P))]
    sol = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]       # least-squares sphere fit
    c = sol[:3]
    xy = {}
    for lab in LABELS_1020:
        v = cap[lab] - c
        theta = np.degrees(np.arccos(v[2] / np.linalg.norm(v)))     # angle from the vertex
        h = np.hypot(v[0], v[1]) or 1.0
        xy[lab] = (theta * v[0] / h, theta * v[1] / h)
    return xy


def draw_cap(ax, cap, pa, pb):
    xy = cap_layout(cap)
    R = max(np.hypot(*p) for p in xy.values()) + 14
    ax.add_patch(Ellipse((0, 0), 2 * R, 2 * R, fill=False, edgecolor="0.45", lw=1.2))
    ax.add_patch(Polygon([[-9, R - 1], [9, R - 1], [0, R + 13]], closed=True, fill=False,
                         edgecolor="0.45", lw=1.2))
    for pr, col in ((pa, "#d62728"), (pb, "#1f77b4")):
        ax.plot([xy[pr[0]][0], xy[pr[1]][0]], [xy[pr[0]][1], xy[pr[1]][1]], "-", color=col,
                lw=2.4, alpha=0.75, zorder=2)
    for lab in LABELS_1020:
        on = "#d62728" if lab in pa else "#1f77b4" if lab in pb else None
        ax.scatter(*xy[lab], s=300, color=on or "0.9", edgecolor="black" if on else "0.6",
                   lw=0.8, zorder=3)
        ax.text(*xy[lab], lab, ha="center", va="center", fontsize=6.8,
                color="white" if on else "black", zorder=4)
    ax.text(-R - 5, 0, "L", ha="right", va="center", fontsize=9, color="0.3")
    ax.text(R + 5, 0, "R", ha="left", va="center", fontsize=9, color="0.3")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-R - 22, R + 22); ax.set_ylim(-R - 10, R + 18)


def panel_row(axs, t1n, env1, tgt1, brain1, centre, vmax, row_title, info, cap, pa, pb):
    ext = [(-91, 90, -126, 91), (-91, 90, -72, 109), (-126, 91, -72, 109)]
    names = ("Axial", "Coronal", "Sagittal")
    world = centre["world"]
    coord = (f"z = {world[2]:+.0f} mm", f"y = {world[1]:+.0f} mm", f"x = {world[0]:+.0f} mm")
    ov = np.where(brain1 & (env1 >= 0.05 * vmax), env1, np.nan)
    im = None
    for a, bg, o, tg, e, nm, cs in zip(axs[:3], slices(t1n, centre["ijk"]),
                                       slices(ov, centre["ijk"]),
                                       slices(tgt1.astype(float), centre["ijk"]), ext, names, coord):
        a.imshow(bg, cmap="gray", origin="lower", extent=e, vmin=0, vmax=1)
        im = a.imshow(o, cmap="inferno", origin="lower", extent=e, vmin=0, vmax=vmax, alpha=0.82)
        if tg.any():
            a.contour(tg, levels=[0.5], colors="#00e5ff", linewidths=1.1, origin="lower", extent=e)
        a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
        a.set_title(f"{nm}  ({cs})", fontsize=9)
        if nm != "Sagittal":
            a.text(0.03, 0.04, "L", transform=a.transAxes, color="white", fontsize=9)
            a.text(0.93, 0.04, "R", transform=a.transAxes, color="white", fontsize=9)
        else:
            a.text(0.03, 0.04, "P", transform=a.transAxes, color="white", fontsize=9)
            a.text(0.94, 0.04, "A", transform=a.transAxes, color="white", fontsize=9)
    draw_cap(axs[3], cap, pa, pb)
    axs[3].set_title(info, fontsize=8.4, loc="left")
    axs[0].text(-0.08, 0.5, row_title, transform=axs[0].transAxes, rotation=90, ha="center",
                va="center", fontsize=11, weight="bold")
    return im


FOOTNOTE = ("F = T / O.  T = median envelope in the target (AAL, bilateral, grey matter); "
            "O = 99th percentile of grey-matter envelope more than 20 mm from the target.  "
            "Colour scale per row: 0 to the 99.5th percentile in grey matter.  The envelope is "
            "overlaid on grey matter only, the tissue F is scored in; white matter, where the field "
            "runs higher because of its lower conductivity, is shown as anatomy (full maps in the "
            f"NIfTI files).\nSPICEhead {config.MODEL} ({MODEL_TEXT}), 2 mm solve resampled to "
            "1 mm for display, 1 mA per channel, maximum-over-direction envelope; one standard "
            "head.  Simulated exposure, not neural activation; magnitudes depend on grid "
            "spacing and differ from FEM (see the results file).")


def render():
    """Draw every figure from the saved NIfTI maps and summary, without re-solving."""
    summ = json.load(open(os.path.join(OUT, "TI_targets_MNI152_summary.json"), encoding="utf-8"))
    t1img = nib.load(os.path.join(M2M, "T1.nii.gz"))
    t1 = np.asanyarray(t1img.dataobj).astype(float)
    lo, hi = np.percentile(t1[t1 > 0], [1, 99.5])
    t1n = np.clip((t1 - lo) / (hi - lo), 0, 1)
    lab05 = nib.load(os.path.join(M2M, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    lab1 = np.rint(np.asanyarray(resample_from_to(lab05, t1img, order=0).dataobj)).astype(int)
    brain1, gm1 = np.isin(lab1, (WM, GM)), lab1 == GM
    cap = read_cap()

    maps = []
    for r in summ["targets"]:
        key = r["target"]
        env1 = np.asanyarray(nib.load(os.path.join(
            OUT, f"TI_envelope_{key}_MNI152_1mm.nii.gz")).dataobj).astype(np.float32)
        tgt1 = np.asanyarray(nib.load(os.path.join(
            OUT, f"TI_target_{key}_MNI152_1mm.nii.gz")).dataobj) > 0
        # axial and coronal through the target centroid; sagittal through its left half, so a
        # bilateral lateral target is cut rather than the midline between its two parts
        tix = np.argwhere(tgt1)
        wx = (t1img.affine @ np.c_[tix, np.ones(len(tix))].T)[0]
        c = tix.mean(0).round().astype(int)
        if (wx < 0).any():
            c[0] = int(round(tix[wx < 0, 0].mean()))
        world = (t1img.affine @ np.append(c, 1))[:3]
        vmax = float(np.percentile(env1[gm1], 99.5))
        pa, pb = tuple(r["pair1"].split("+")), tuple(r["pair2"].split("+"))
        info = (f"Montage {r['montage']}\nF = {r['F']:.2f}  (T {r['T_V_per_m']:.3f}, "
                f"O {r['O_V_per_m']:.3f} V/m)")
        maps.append(dict(r=r, env1=env1, tgt1=tgt1, centre=dict(ijk=c, world=world), vmax=vmax,
                         pa=pa, pb=pb, info=info))

    widths = dict(width_ratios=[1.0, 1.0, 1.15, 0.95])
    for m in maps:
        fig, axs = plt.subplots(1, 4, figsize=(17, 5.0), gridspec_kw=widths)
        im = panel_row(axs, t1n, m["env1"], m["tgt1"], gm1, m["centre"], m["vmax"],
                       m["r"]["title"], m["info"], cap, m["pa"], m["pb"])
        cb = fig.colorbar(im, ax=axs[:3], fraction=0.025, pad=0.01)
        cb.set_label("TI envelope (V/m), 1 mA per channel")
        fig.suptitle(f"Simulated TI exposure: {m['r']['title']} on the MNI152 template "
                     f"(target outlined in cyan; red = pair 1, blue = pair 2)", fontsize=12)
        fig.text(0.01, -0.04, textwrap.fill(FOOTNOTE.replace("\n", " "), 205), fontsize=7.4,
                 color="0.25", va="top")
        fig.savefig(os.path.join(OUT, f"TI_{m['r']['target']}_MNI152.png"), dpi=200,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axs = plt.subplots(len(maps), 4, figsize=(17, 4.5 * len(maps)), gridspec_kw=widths)
    fig.subplots_adjust(top=0.955, bottom=0.05, hspace=0.22, wspace=0.22)
    for ax_row, m in zip(axs, maps):
        im = panel_row(ax_row, t1n, m["env1"], m["tgt1"], gm1, m["centre"], m["vmax"],
                       m["r"]["title"], m["info"], cap, m["pa"], m["pb"])
        cb = fig.colorbar(im, ax=ax_row[:3], fraction=0.025, pad=0.01)
        cb.set_label("V/m", fontsize=9)
    fig.suptitle("Simulated temporal-interference exposure on the MNI152 template, best two-pair "
                 "10-20 montage per target (target outlined in cyan; red = pair 1, blue = pair 2)",
                 fontsize=12.5, y=0.985)
    fig.text(0.02, 0.035, textwrap.fill(FOOTNOTE.replace("\n", " "), 190), fontsize=8.6,
             color="0.25", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"TI_four_targets_MNI152.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figures written to", OUT)


def main():
    os.makedirs(OUT, exist_ok=True)
    S = solve_template()
    best, n_mont = search(S)

    t1img = nib.load(os.path.join(M2M, "T1.nii.gz"))
    lab05 = nib.load(os.path.join(M2M, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    lab1 = np.rint(np.asanyarray(resample_from_to(lab05, t1img, order=0).dataobj)).astype(int)
    brain1 = np.isin(lab1, (WM, GM))
    aal1 = np.rint(np.asanyarray(resample_from_to(nib.load(AAL), t1img, order=0).dataobj)).astype(int)
    value = {r["name"]: r["value"] for r in json.load(open(AAL_REGIONS))["regions"]}

    rows = []
    for key, t in TARGETS.items():
        Fb, Tb, Ob, pa, pb = best[key][0]
        A, B = S["pair_field"](*pa), S["pair_field"](*pb)
        env2 = envelope_max_direction_flat(*[c.ravel() for c in A],
                                           *[c.ravel() for c in B]).reshape(S["labels"].shape)
        env2 = np.where(S["brain"], np.nan_to_num(env2), 0.0).astype(np.float32)
        env1 = to_1mm(env2, S["aff"], t1img, order=1).astype(np.float32)
        env1[~brain1] = 0.0
        tgt1 = np.isin(aal1, [value[p] for p in t["parcels"]]) & (lab1 == GM)
        nib.save(nib.Nifti1Image(env1, t1img.affine),
                 os.path.join(OUT, f"TI_envelope_{key}_MNI152_1mm.nii.gz"))
        nib.save(nib.Nifti1Image(tgt1.astype(np.uint8), t1img.affine),
                 os.path.join(OUT, f"TI_target_{key}_MNI152_1mm.nii.gz"))
        pk = np.unravel_index(int(np.argmax(np.where(brain1, env1, 0))), env1.shape)
        rows.append(dict(target=key, title=t["title"], aal_parcels="+".join(t["parcels"]),
                         montage=label(pa, pb), pair1="+".join(pa), pair2="+".join(pb),
                         F=round(Fb, 3), T_V_per_m=round(Tb, 4), O_V_per_m=round(Ob, 4),
                         peak_mni=[round(float(x)) for x in (t1img.affine @ np.append(pk, 1))[:3]],
                         peak_in_target=bool(tgt1[pk]),
                         runner_up=label(*best[key][1][3:5]), F_runner_up=round(best[key][1][0], 3),
                         top5=[dict(montage=label(m[3], m[4]), F=round(m[0], 3),
                                    T=round(m[1], 4), O=round(m[2], 4)) for m in best[key]]))
        print(f"{t['title']:<26} {label(pa, pb):<18} F={Fb:.3f} T={Tb:.4f} O={Ob:.4f}", flush=True)

    with open(os.path.join(OUT, "TI_targets_MNI152_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(model=f"SPICEhead {config.MODEL}: {MODEL_TEXT}",
                       grid_mm=DX, current_mA_per_channel=1.0, envelope="maximum over direction",
                       n_nodes=S["n_nodes"], n_montages=n_mont,
                       superposition_check=S["check"], targets=rows), fh, indent=1)
    with open(os.path.join(OUT, "TI_targets_MNI152_summary.csv"), "w", newline="",
              encoding="utf-8") as fh:
        keys = [k for k in rows[0] if k != "top5"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)
    render()


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        render()
    else:
        main()
