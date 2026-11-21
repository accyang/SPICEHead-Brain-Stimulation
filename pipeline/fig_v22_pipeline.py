# -*- coding: utf-8 -*-
"""Figure 1 of the v22 paper: from MRI to resistor network to TI envelope, on the MNI152 head.

The template head (CHARM segmentation of MNI152) is used rather than a participant, so the figure
shows no study data. One montage (bilateral temporal, F7+T7 / P8+T8, 1 mA per channel) is solved
directly at 2 mm under the active model (spicehead.config; v22 = skin-surface contacts, harmonic
interface rule).

  A  T1 (axial slice through the thalamus)
  B  ten-tissue segmentation, same slice
  C  network close-up at electrode T8: nodes coloured by tissue, contact nodes marked
  D  |E| of channel 1 and of channel 2
  E  TI envelope (maximum over directions)

  python pipeline/fig_v22_pipeline.py

Output: Paper_A_SPICEhead/Brain Stimulation/figures/v22/fig1_pipeline.{png,pdf}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm, BoundaryNorm
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
os.environ.setdefault("SPICEHEAD_MODEL", "v22")   # v22 record script; the package default is v23 since 2 Nov 2026
from spicehead import config
sys.path.insert(0, _here)
from spicehead import mri
from spicehead.electrodes import Electrode, contact_nodes, pair_injection
from spicehead.field import efield_components
from spicehead.envelope_fast import envelope_max_direction_flat
from fig_ti_targets_mni import M2M, read_cap, DX, ELEC_RADIUS_MM, SCALP
from v21_head_convergence import make_solver

# Paper A figures folder for v21/v22 only; other model versions (v23) write under their own results folder
OUT = (os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "figures", "v22" if config.MODEL == "v22"
                    else f"{config.MODEL}_draft")
       if config.MODEL in ("v21", "v22") else os.path.join(config.RESULTS, "figures", "pipeline"))
MONTAGE = (("F7", "T7"), ("P8", "T8"))
ZOOM_ELEC = "T8"
TISSUE = ["White matter", "Gray matter", "CSF", "Bone", "Scalp", "Eyes", "Compact bone",
          "Spongy bone", "Blood", "Muscle"]
COLORS = ["#f2f2f2", "#9e9e9e", "#5aa7d6", "#e8d9a8", "#f0b7a1", "#c9e3f2", "#d9c27a",
          "#efe4bd", "#c0392b", "#b57b6b"]


def main():
    os.makedirs(OUT, exist_ok=True)
    seg = os.path.join(M2M, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, aff = mri.head_model(seg, DX)   # v21/v22: nearest-label network; v23: homogenized
    labels = model.labels
    cap = read_cap()
    scalp = np.argwhere(labels == SCALP)
    inv = np.linalg.inv(aff)
    elecs, evox = {}, {}
    for lab in sorted({e for p in MONTAGE for e in p}):
        ijk = (inv @ np.append(cap[lab], 1))[:3]
        v = scalp[int(np.argmin(np.linalg.norm(scalp - ijk, axis=1)))]
        evox[lab] = v
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v),
                                             radius_nodes=ELEC_RADIUS_MM / DX), lab)
    cache = os.path.join(config.RESULTS, "figure_cache", "fig1_fields.npz")
    if "--plot-only" in sys.argv and os.path.exists(cache):
        z = np.load(cache)
        E, env = [z["E1"], z["E2"]], z["env"]
        brain = np.isin(labels, (1, 2))
    else:
        solver, _ = make_solver(model.G)
        E = []
        for a, b in MONTAGE:
            V = solver(pair_injection(model, elecs[a], elecs[b], 1e-3))
            E.append(np.stack([np.nan_to_num(c) for c in efield_components(model, V)]).astype(np.float32))
        brain = np.isin(labels, (1, 2))
        env = np.zeros(labels.shape, np.float32)
        a, b = E
        env[brain] = envelope_max_direction_flat(a[0][brain], a[1][brain], a[2][brain],
                                                 b[0][brain], b[1][brain], b[2][brain])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, E1=E[0], E2=E[1], env=env)
    mag = [np.sqrt((x ** 2).sum(0)) for x in E]

    # slice through the thalamus (MNI z = +5 mm) and the T1 resampled to the analysis grid
    k = int(round((inv @ np.array([0.0, -15.0, 5.0, 1.0]))[2]))
    grid = nib.Nifti1Image(labels.astype(np.int16), aff)
    t1 = np.asarray(resample_from_to(nib.load(os.path.join(M2M, "T1.nii.gz")), grid,
                                     order=1).dataobj)

    def ax_img(axh, img, **kw):
        axh.imshow(np.rot90(img[:, :, k]), **kw)
        axh.set_xticks([]), axh.set_yticks([])

    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(2, 4, wspace=0.08, hspace=0.18)
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2:])
    axD1, axD2, axE = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    ax_img(axA, t1, cmap="gray")
    cm = ListedColormap(["white"] + COLORS)
    ax_img(axB, labels, cmap=cm, norm=BoundaryNorm(np.arange(-0.5, 11.5), cm.N),
           interpolation="nearest")

    # C: close-up at the zoom electrode, in the axial plane through its centre
    ci, cj, ck = evox[ZOOM_ELEC]
    half = 9
    sl = labels[ci - half:ci + half + 1, cj - half:cj + half + 1, ck]
    ii, jj = np.nonzero(sl > 0)
    node_set = set(zip(ii, jj))
    for (x, y) in node_set:
        for dx_, dy_ in ((1, 0), (0, 1)):
            if (x + dx_, y + dy_) in node_set:
                axC.plot([x, x + dx_], [y, y + dy_], color="0.75", lw=0.5, zorder=1)
    axC.scatter(ii, jj, s=18, c=[COLORS[int(sl[x, y]) - 1] for x, y in zip(ii, jj)],
                edgecolors="0.35", linewidths=0.3, zorder=2)
    gi, gj, gk = model.grid_coords
    cn = elecs[ZOOM_ELEC].nodes
    on = (gk[cn] == ck) & (np.abs(gi[cn] - ci) <= half) & (np.abs(gj[cn] - cj) <= half)
    axC.scatter(gi[cn][on] - ci + half, gj[cn][on] - cj + half, s=46, facecolors="none",
                edgecolors="#d62728", linewidths=1.4, zorder=3)
    axC.set_xlim(-0.8, 2 * half + 0.8), axC.set_ylim(-0.8, 2 * half + 0.8)
    axC.set_aspect("equal"), axC.set_xticks([]), axC.set_yticks([])
    axC.set_title(f"Network at {ZOOM_ELEC}; red: skin contact nodes", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="0.35", label=t)
               for t, c in zip(TISSUE, COLORS) if (sl == TISSUE.index(t) + 1).any()]
    axC.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False)

    vmax = float(np.percentile(np.concatenate([m[brain] for m in mag]), 99.5))
    for axh, m, (p, q) in ((axD1, mag[0], MONTAGE[0]), (axD2, mag[1], MONTAGE[1])):
        img = np.where(labels > 0, m, np.nan)
        hD = axh.imshow(np.rot90(img[:, :, k]), cmap="viridis",
                        norm=LogNorm(vmin=vmax / 100, vmax=vmax * 10))
        axh.set_xticks([]), axh.set_yticks([])
        axh.set_title(f"|E|, channel {p}+{q}", fontsize=9)
    # horizontal, under both channel maps, so nothing crowds panel E
    fig.colorbar(hD, ax=[axD1, axD2], orientation="horizontal", fraction=0.05, pad=0.04,
                 aspect=40, label="|E| (V/m), log scale")
    im = np.where(brain, env, np.nan)
    h = axE.imshow(np.rot90(im[:, :, k]), cmap="inferno", vmin=0,
                   vmax=float(np.percentile(env[brain], 99.5)))
    axE.set_xticks([]), axE.set_yticks([])
    axE.set_title("TI envelope, brain (V/m)", fontsize=9)
    fig.colorbar(h, ax=axE, fraction=0.046, pad=0.02)
    for axh, t in ((axA, "A"), (axB, "B"), (axC, "C"), (axD1, "D"), (axE, "E")):
        axh.text(-0.12 if axh is axC else -0.06, 1.04, t, transform=axh.transAxes, fontsize=13,
                 fontweight="bold")
    axA.set_title("T1, MNI152", fontsize=9)
    axB.set_title("Ten-tissue segmentation", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig1_pipeline.{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(OUT, "fig1_pipeline.png"), "|", config.describe())


if __name__ == "__main__":
    main()
