# -*- coding: utf-8 -*-
"""Figures 1-5 of the Rev0.4 manuscript (v23, 300 heads), drawn from result files (31 Oct 2026).

Panel plans: Rev0.4_2026-10-30/08_FIGURE_INSTRUCTIONS_v23.md; captions: 07_figure_legends.docx (build_rev04.LEGENDS).
  Figure 1  network and region-wise outputs (MNI152 template, solved at 1 mm under v23; fields cached)
  Figure 2  analytical verification (disk, layered slab; synthetic geometries)
  Figure 3  agreement with FEM in 300 heads (field per head, six reference targets, 163 regions)
  Figure 4  whole-brain targeting parameters (AAL / AAL3 regions on MNI152 slices; cohort montages)
  Figure 5  individual variability and demographic determinants (163 regions)

Style of make_submission_figures.py: 180 mm wide, Arial, text at least 8 pt, PDF (vector text, rasters at 600 dpi) and
600 dpi PNG; SPICEhead 1 mm vermilion circles, 2 mm blue squares (Okabe-Ito); the FEM reference appears as gray
reference lines only. Regions within a head are correlated: no error bar treats regions as independent samples.

  SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu python pipeline/fig_v23_rev04.py [1 2 3 4 5] [--docx]

Figure 1 solves two pair fields on the template at 1 mm on the desktop GPU the first time (cache
results_v23/figure_cache/fig1_fields_1mm.npz); every other panel reads saved results only.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter
import nibabel as nib
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)
from spicehead import config                                         # noqa: E402
import rev04_numbers as RN                                           # noqa: E402

PKG = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "Rev0.4_2026-10-30")
OUT = os.environ.get("SPICEHEAD_FIG_OUT") or os.path.join(PKG, "figures")
R23 = os.path.join(ROOT, "results_v23")
AAL3_NII = r"C:\Users\ALBERT YANG\atlases\AAL3\AAL3\AAL3v1_1mm.nii"
MM = 1 / 25.4
W2 = 180 * MM
FS = 8
PDF_DPI = 600
C1, C2, GRAY = "#D55E00", "#0072B2", "#404040"          # SPICEhead 1 mm, 2 mm; reference lines
CH1, CH2 = "#009E73", "#CC79A7"                          # montage channels (not the 1 mm / 2 mm colours)
SP, SP2, FE = "sp1v23", "sp2v23", "fem"
SETS = [k for k, _ in RN.SETS]
SET_NAME = dict(RN.SETS)
SHORT6 = {"M1_handknob": "M1", "thalamus": "Thalamus", "insula": "Insula", "ACC": "ACC",
          "hippocampus_L": "Hipp. L", "hippocampus_R": "Hipp. R"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
    "legend.handletextpad": 0.4, "legend.columnspacing": 1.0, "legend.handlelength": 1.5,
    "legend.borderaxespad": 0.2, "axes.titlepad": 4, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")                     # 0.01, 0.1, 1 instead of small exponents
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]


def plabel(ax, t, x=-0.16, y=1.06):
    ax.text(x, y, t, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def clean(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def save(fig, name, png_dpi=600):
    os.makedirs(OUT, exist_ok=True)
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    over = {k_: round(v_ * 25.4, 1) for k_, v_ in (("left", -bb.x0), ("right", bb.x1 - fig.get_figwidth()),
                                                     ("bottom", -bb.y0), ("top", bb.y1 - fig.get_figheight()))
            if v_ > 0.005}
    if over:
        print(f"  {name}: content outside the canvas (mm): {over}")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), dpi=PDF_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=png_dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", os.path.join(OUT, f"{name}.pdf"), "+ .png")


_N = {}


def N():
    """Every Rev0.4 number (rev04_numbers.compute), loaded once; N()['A'] is the per-region table."""
    if not _N:
        _N.update(RN.compute())
    return _N


def val(t, m, k):
    return RN.f(N()["A"][(t, m)][k])


def jitter(n, width, seed=0):
    """Deterministic jitter (same layout on every run)."""
    return (np.random.default_rng(seed).random(n) - 0.5) * width


# ================================================================== Figure 1
TISSUE = ["White matter", "Gray matter", "CSF", "Bone", "Scalp", "Eyes", "Compact bone",
          "Spongy bone", "Blood", "Muscle"]
TCOLORS = ["#f2f2f2", "#9e9e9e", "#5aa7d6", "#e8d9a8", "#f0b7a1", "#c9e3f2", "#d9c27a",
           "#efe4bd", "#c0392b", "#b57b6b"]
MONTAGE = (("F7", "T7"), ("P8", "T8"))
DX1 = 1.0                                                     # Figure 1 grid: the atlas grid


def fig1_fields():
    """Template network at 1 mm (v23), the contact nodes at T8 and the two channel fields of F7+T7 / P8+T8 in the axial
    plane through the thalamus. Cached after the first (GPU) solve."""
    from spicehead import mri
    from spicehead.electrodes import Electrode, contact_nodes, pair_injection
    from fig_ti_targets_mni import M2M, read_cap, ELEC_RADIUS_MM, SCALP
    assert config.MODEL == "v23", "Figure 1 of Rev0.4 is a v23 solve: set SPICEHEAD_MODEL=v23"
    cache = os.path.join(R23, "figure_cache", "fig1_fields_1mm.npz")
    seg = os.path.join(M2M, "label_prep", "tissue_labeling_upsampled.nii.gz")
    model, aff = mri.head_model(seg, DX1)
    labels = model.labels
    inv = np.linalg.inv(aff)
    k = int(round((inv @ np.array([0.0, -15.0, 5.0, 1.0]))[2]))          # axial z = +5 mm (thalamus)
    cap = read_cap()
    scalp = np.argwhere(labels == SCALP)
    elecs, evox = {}, {}
    for lab in sorted({e for p in MONTAGE for e in p}):
        v = scalp[int(np.argmin(np.linalg.norm(scalp - (inv @ np.append(cap[lab], 1))[:3], axis=1)))]
        evox[lab] = v
        elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v), radius_nodes=ELEC_RADIUS_MM / DX1), lab)
    if os.path.exists(cache):
        z = dict(np.load(cache))
    else:
        from spicehead.field import efield_components
        from spicehead.envelope_fast import envelope_max_direction_flat
        from v21_head_convergence import make_solver
        solver, _ = make_solver(model.G)
        E = []
        for a, b in MONTAGE:
            V = solver(pair_injection(model, elecs[a], elecs[b], 1e-3))
            E.append(np.stack([np.nan_to_num(c) for c in efield_components(model, V)]).astype(np.float32))
        brain = np.isin(labels, (1, 2))
        env = np.zeros(labels.shape, np.float32)
        env[brain] = envelope_max_direction_flat(*[E[0][i][brain] for i in range(3)], *[E[1][i][brain] for i in range(3)])
        mag = [np.sqrt((x ** 2).sum(0)) for x in E]
        z = dict(mag1=mag[0][:, :, k], mag2=mag[1][:, :, k], env=env[:, :, k], k=k,
                 mag_vmax=float(np.percentile(np.concatenate([m[brain] for m in mag]), 99.5)),
                 env_vmax=float(np.percentile(env[brain], 99.5)), nodes=model.n_nodes if hasattr(model, "n_nodes") else model.G.shape[0],
                 model=config.MODEL, network=config.NETWORK, montage=json.dumps(MONTAGE))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **z)
        print("solved and cached", cache)
    t1img = nib.load(os.path.join(M2M, "T1.nii.gz"))
    return model, aff, labels, k, elecs, evox, z, t1img


def zigzag(ax, x0, x1, y, amp=0.08, n=4, **kw):
    """A resistor symbol between x0 and x1 at height y."""
    xs = np.linspace(x0 + 0.12 * (x1 - x0), x1 - 0.12 * (x1 - x0), 2 * n + 1)
    ys = y + amp * np.array([0] + [(-1) ** i for i in range(2 * n - 1)] + [0])
    ax.plot([x0, xs[0]], [y, y], **kw)
    ax.plot(xs, ys, **kw)
    ax.plot([xs[-1], x1], [y, y], **kw)


def fig1():
    model, aff, labels, k, elecs, evox, z, t1img = fig1_fields()
    fig = plt.figure(figsize=(W2, 196 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.08, 0.62], hspace=0.28, left=0.01, right=0.985, top=0.93, bottom=0.01)

    # ---- row 1: A T1, B segmentation, C homogenization and contact nodes
    g1 = gs[0].subgridspec(1, 4, width_ratios=[1, 1, 1.05, 1.15], wspace=0.12)
    aA, aB, aC1, aC2 = (fig.add_subplot(g1[i]) for i in range(4))
    t1 = np.asanyarray(t1img.dataobj).astype(float)
    k1 = int(round((np.linalg.inv(t1img.affine) @ np.array([0.0, -15.0, 5.0, 1.0]))[2]))
    lo, hi = np.percentile(t1[t1 > 0], [1, 99.5])
    t1s = np.clip((t1[:, :, k1] - lo) / (hi - lo), 0, 1)
    if t1img.affine[0, 0] < 0:                                   # neurological display: left on the left
        t1s = t1s[::-1]
    aA.imshow(np.rot90(t1s), cmap="gray")
    lab = labels[:, :, k]
    if aff[0, 0] < 0:
        lab = lab[::-1]
    cm = ListedColormap(["white"] + TCOLORS)
    aB.imshow(np.rot90(lab), cmap=cm, norm=BoundaryNorm(np.arange(-0.5, 11.5), cm.N), interpolation="nearest")
    for a, t in ((aA, "T1, MNI152"), (aB, "Tissue labels (1 mm blocks)")):
        clean(a)
        a.set_title(t)
        a.text(0.03, 0.03, "L", transform=a.transAxes, color="white" if a is aA else "black")
        a.text(0.93, 0.03, "R", transform=a.transAxes, color="white" if a is aA else "black")
    present = [i for i in range(10) if (labels == i + 1).any()]
    pA, pB = aA.get_position(), aB.get_position()
    fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=TCOLORS[i], mec="0.4", mew=0.3, ms=5.5, label=TISSUE[i])
                        for i in present], loc="upper center", bbox_to_anchor=((pA.x0 + pB.x1) / 2, pA.y0 - 0.003), ncol=3,
               frameon=False, handletextpad=0.2, columnspacing=1.0, fontsize=FS)

    # C1: two neighbouring 1 mm blocks of 0.5 mm voxels; f^2 parallel tubes of series fine voxels between the centres
    fine = [[5, 5, 7, 3], [5, 5, 7, 3]]                          # tissue per fine voxel (rows): scalp | compact bone | CSF
    for r in range(2):
        for c in range(4):
            aC1.add_patch(Rectangle((c, r), 1, 1, facecolor=TCOLORS[fine[r][c] - 1], edgecolor="white", lw=0.8))
    for x0 in (0, 2):
        aC1.add_patch(Rectangle((x0, 0), 2, 2, fill=False, edgecolor="black", lw=1.0))
    for r in range(2):
        zigzag(aC1, 1, 2, r + 0.5, color=C1, lw=0.9)
        zigzag(aC1, 2, 3, r + 0.5, color=C1, lw=0.9)
        aC1.plot([1, 1], [1, r + 0.5], color=C1, lw=0.9)
        aC1.plot([3, 3], [1, r + 0.5], color=C1, lw=0.9)
    aC1.scatter([1, 3], [1, 1], s=22, color="black", zorder=4)
    aC1.text(1, 2.12, "block a", ha="center", va="bottom")
    aC1.text(3, 2.12, "block b", ha="center", va="bottom")
    aC1.annotate("", xy=(4, -0.18), xytext=(3, -0.18), arrowprops=dict(arrowstyle="<->", lw=0.6))
    aC1.text(3.5, -0.32, "0.5 mm", ha="center", va="top")
    aC1.annotate("", xy=(2, -0.18), xytext=(0, -0.18), arrowprops=dict(arrowstyle="<->", lw=0.6))
    aC1.text(1, -0.32, "1 mm", ha="center", va="top")
    aC1.text(2, -0.95, r"$g_{ab}=\sum_{\mathrm{tubes}}\left(\sum_i \dfrac{\ell_i}{\sigma_i A}\right)^{-1}$",
             ha="center", va="top", fontsize=11, color=C1)
    aC1.set_xlim(-0.2, 4.2), aC1.set_ylim(-2.1, 2.6), aC1.set_aspect("equal"), aC1.axis("off")
    aC1.set_title("Homogenized conductance\n(2-D section; f² tubes in 3-D)")

    # C2: the template network near T8 at 1 mm, contact nodes highlighted; the window is padded because T8 lies
    # near the grid edge, and drawn with the subject's right on the right (as panels A, B)
    from fig_ti_targets_mni import ELEC_RADIUS_MM
    ci, cj, ck = evox["T8"]
    half = 12
    sx = -1 if aff[0, 0] < 0 else 1
    pad = np.pad(labels[:, :, ck], half)
    sl = pad[ci:ci + 2 * half + 1, cj:cj + 2 * half + 1]
    px = lambda i: half + sx * (i - half)                       # window column -> plotted x
    ii, jj = np.nonzero(sl > 0)
    for (x, y) in zip(ii, jj):
        for dx_, dy_ in ((1, 0), (0, 1)):
            if x + dx_ < sl.shape[0] and y + dy_ < sl.shape[1] and sl[x + dx_, y + dy_] > 0:
                aC2.plot([px(x), px(x + dx_)], [y, y + dy_], color="0.8", lw=0.3, zorder=1)
    aC2.scatter(px(ii), jj, s=5, c=[TCOLORS[int(sl[x, y]) - 1] for x, y in zip(ii, jj)], edgecolors="0.35",
                linewidths=0.2, zorder=2)
    gi, gj, gk = model.grid_coords
    cn = elecs["T8"].nodes
    on = (gk[cn] == ck) & (np.abs(gi[cn] - ci) <= half) & (np.abs(gj[cn] - cj) <= half)
    aC2.scatter(px(gi[cn][on] - ci + half), gj[cn][on] - cj + half, s=18, facecolors="none", edgecolors="#d62728",
                linewidths=0.9, zorder=3)
    aC2.set_xlim(-0.8, 2 * half + 0.8), aC2.set_ylim(-0.8, 2 * half + 0.8), aC2.set_aspect("equal")
    clean(aC2)
    aC2.set_title(f"Network in the plane through T8;\nred: contact nodes ({ELEC_RADIUS_MM:.0f} mm radius)")
    fig.canvas.draw()                                           # panel letters A-C level with the tallest title
    ytop = max(a.get_position().y1 for a in (aA, aB, aC1, aC2)) + 0.045
    for a, t in ((aA, "A"), (aB, "B"), (aC1, "C")):
        fig.text(a.get_position().x0 - 0.005, ytop, t, fontsize=11, fontweight="bold", va="bottom")

    # ---- row 2: D channel magnitudes (log), E envelope
    g2 = gs[1].subgridspec(1, 4, width_ratios=[1, 1, 1.18, 0.17], wspace=0.08)     # last column: room for E's colour bar
    aD1, aD2, aE = (fig.add_subplot(g2[i]) for i in range(3))
    flip = (lambda a: a[::-1]) if aff[0, 0] < 0 else (lambda a: a)
    head = labels[:, :, k] > 0
    brain = np.isin(labels[:, :, k], (1, 2))
    vmax = float(z["mag_vmax"])
    for a, m, (p, q) in ((aD1, z["mag1"], MONTAGE[0]), (aD2, z["mag2"], MONTAGE[1])):
        hD = a.imshow(np.rot90(flip(np.where(head, np.maximum(m, vmax / 1000), np.nan))), cmap="viridis",
                      norm=LogNorm(vmin=vmax / 100, vmax=vmax * 10))
        clean(a)
        a.set_title(f"|E|, channel {p}+{q}")
    cb = fig.colorbar(hD, ax=[aD1, aD2], orientation="horizontal", fraction=0.05, pad=0.03, aspect=45, extend="max")
    cb.set_label("|E| (V/m per mA), logarithmic scale; arrow: values above the scale")
    cb.ax.xaxis.set_major_formatter(PLAIN)
    hE = aE.imshow(np.rot90(flip(np.where(brain, z["env"], np.nan))), cmap="inferno", vmin=0, vmax=float(z["env_vmax"]))
    clean(aE)
    aE.set_title("TI envelope in brain tissue\n(template, F7+T7 / P8+T8, z = +5 mm)")
    cbE = fig.colorbar(hE, ax=aE, fraction=0.046, pad=0.02, extend="max")
    cbE.set_label("Envelope (V/m, 1 mA per channel)")
    cbE.ax.locator_params(nbins=5)
    plabel(aD1, "D", x=-0.04, y=1.04)
    plabel(aE, "E", x=-0.04, y=1.04)

    # ---- row 3: F region-wise outputs
    aF = fig.add_subplot(gs[2])
    aF.axis("off"), aF.set_xlim(0, 1), aF.set_ylim(0, 1)
    top = [("300 heads: 0.5 mm\nCHARM labels →\n1 mm network", "#dbe9f6"),
           ("18 basis solves →\n171 pair fields\n(superposition)", "#dbe9f6"),
           ("163 regions ×\n11,628 montages:\nT, O, F = T / O", "#dbe9f6"),
           ("Rule: T ≥ 0.8 max T;\nhighest F; 3% band →\nmost edge electrodes", "#fbe3d2"),
           ("Individual and\ncohort montages\n(Figs. 4–5, Data S1)", "#fbe3d2")]
    w, h, y0 = 0.17, 0.44, 0.53
    xs = [0.012 + i * 0.2 for i in range(5)]
    for (t, c), x in zip(top, xs):
        aF.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.006", facecolor=c, edgecolor="0.5", lw=0.5))
        aF.text(x + w / 2, y0 + h / 2, t, ha="center", va="center", linespacing=1.15)
    for x in xs[:-1]:
        aF.annotate("", xy=(x + 0.2 - 0.002, y0 + h / 2), xytext=(x + w + 0.002, y0 + h / 2),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"))
    xb, wb = xs[1], xs[3] + w - xs[1]
    aF.add_patch(FancyBboxPatch((xb, 0.02), wb, 0.34, boxstyle="round,pad=0.006", facecolor="#ececec", edgecolor="0.5", lw=0.5))
    aF.text(xb + wb / 2, 0.19, "FEM (SimNIBS, 19-pad leadfield): same heads, regions, montages,\nmasks and scoring code → "
            "choice agreement and retention (Fig. 3)", ha="center", va="center", linespacing=1.2)
    aF.annotate("", xy=(xs[4] + w / 2, y0 - 0.005), xytext=(xb + wb + 0.004, 0.19),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3", connectionstyle="angle,angleA=0,angleB=90"))
    aF.text(xs[0] + w / 2, 0.19, "Envelope: exposure\ndescriptor, not a\nneural response", ha="center", va="center",
            color="0.3", linespacing=1.15)
    plabel(aF, "F", x=-0.005, y=1.02)
    save(fig, "Figure1")


# ================================================================== Figure 2
def fig2():
    disk = RN.rows(os.path.join(R23, "v21_validation", "disk_analytic_gradient.csv"))
    slab = RN.rows(os.path.join(R23, "v23_layered_slab", "layered_slab.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 62 * MM), gridspec_kw=dict(wspace=0.32, left=0.08, right=0.99, top=0.9,
                                                                               bottom=0.2))
    dx = np.array([RN.f(r["dx_mm"]) for r in disk])
    for key, name, col, mk, ls in (("model_vs_exact_nmae_pct", "Homogenized network", C1, "o", "-"),
                                   ("discrete_vs_exact_nmae_pct", "Exact potential, same field stencil", "black", "", ":")):
        y = np.array([RN.f(r[key]) for r in disk])
        a.plot(dx, y, marker=mk or None, color=col, ms=4.5, ls=ls, label=name)
    a.set_yscale("log")
    a.yaxis.set_major_formatter(PLAIN)
    a.set_xticks([5, 4, 3, 2, 1]), a.set_xlim(5.3, 0.7)
    a.set_xlabel("Grid spacing (mm)")
    a.set_ylabel("Disk nMAE vs exact gradient (%)")
    a.legend(frameon=False, loc="lower left")
    for x, y in zip(dx, [RN.f(r["model_vs_exact_nmae_pct"]) for r in disk]):
        a.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -11), ha="center", color=C1)
    plabel(a, "A", x=-0.2)
    for stack, ls, tag in (("aligned", "-", "whole-mm layers"), ("thin", "--", "thin layers")):
        for model, name, col, mk in (("v23", "Homogenized", C1, "o"), ("v22", "Label resampling", GRAY, "D")):
            s = [r for r in slab if r["stack"] == stack and r["model"] == model]
            b.plot([RN.f(r["dx_mm"]) for r in s], [RN.f(r["R_ratio"]) for r in s], ls=ls, marker=mk, color=col, ms=4.5,
                   mfc=col if stack == "aligned" else "white", label=f"{name}, {tag}")
    b.axhline(1.0, color="0.6", lw=0.6, zorder=0)
    b.set_xticks([3, 2, 1]), b.set_xlim(3.25, 0.75)
    b.set_xlabel("Grid spacing (mm)")
    b.set_ylabel("Slab resistance / exact series value")
    b.legend(frameon=False, loc="upper right", fontsize=FS, handlelength=2.4)
    b.set_ylim(0.85, 1.42)
    for r in slab:
        if r["stack"] == "thin" and r["model"] == "v22":
            b.annotate(f"{RN.f(r['R_ratio']):.3f}", (RN.f(r["dx_mm"]), RN.f(r["R_ratio"])), textcoords="offset points",
                       xytext=(10, 6) if RN.f(r["dx_mm"]) == 3 else (0, 8), va="center",
                       ha="left" if RN.f(r["dx_mm"]) == 3 else "center", color=GRAY)
    plabel(b, "B", x=-0.2)
    a.set_title("Analytical disk (0.5 mm voxelization)", loc="left")
    b.set_title("Scalp / bone / CSF / GM slab (thin: 5.5 / 4.5 / 2.5 mm)", loc="left")
    save(fig, "Figure2", png_dpi=1000)


# ================================================================== Figure 3
def fig3():
    fr = RN.rows(RN.FIELDS)
    dec = RN.rows(os.path.join(R23, "v23_fem_fullsearch_300", "fullsearch_decisions.csv"))
    fig = plt.figure(figsize=(W2, 128 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.2], height_ratios=[1, 1], wspace=0.32, hspace=0.55,
                          left=0.09, right=0.99, top=0.93, bottom=0.08)
    # A: per head r and mean-magnitude bias (the metric of the Results text; the voxel-median ratio is in Table S5)
    a = fig.add_subplot(gs[0, 0])
    for m, col, mk, g in (("sp2hom", C2, "s", "2 mm"), ("sp1hom", C1, "o", "1 mm")):
        s = [x for x in fr if x["model"] == m]
        r = np.array([RN.f(x["r"]) for x in s]); q = np.array([RN.f(x["bias_pct"]) for x in s])
        a.scatter(r, q, s=5, marker=mk, color=col, alpha=0.45, lw=0, label=f"SPICEhead {g} (n = {len(s)})")
        a.scatter([np.median(r)], [np.median(q)], s=40, marker=mk, color=col, edgecolor="black", lw=0.7, zorder=4)
        a.annotate(f"median r {np.median(r):.3f}\nbias {np.median(q):+.1f}%", (np.median(r), np.median(q)),
                   textcoords="axes fraction", xytext=(0.98, 0.69) if m == "sp1hom" else (0.62, 0.99),
                   ha="right" if m == "sp1hom" else "left", va="bottom" if m == "sp1hom" else "top", path_effects=HALO,
                   arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
    a.axhline(0.0, color="0.6", lw=0.6, ls="--", zorder=0)
    a.set_xlabel("Correlation with FEM, |E| in GM")
    a.set_ylabel("Mean |E| bias vs FEM (%)")
    a.legend(frameon=False, loc="lower left", markerscale=2)
    plabel(a, "A", x=-0.27)
    # B: six reference targets, retention of the unconstrained choice, full range
    b = fig.add_subplot(gs[0, 1])
    rois = RN.ROI6
    for i, roi in enumerate(rois):
        s = [r for r in dec if r["roi"] == roi]
        for m, col, mk, off in (("sp1", C1, "o", -0.19), ("sp2", C2, "s", 0.19)):
            k = np.array([100 - RN.f(r[f"regret_{m}winner_under_fem_pct"]) for r in s])
            x = i + off + jitter(len(k), 0.22, seed=i)
            b.scatter(x, k, s=3, marker=mk, color=col, alpha=0.35, lw=0)
            low = k < 90
            if m == "sp1" and low.any():
                b.scatter(x[low], k[low], s=16, marker=mk, color=col, edgecolor="black", lw=0.6, zorder=4)
            b.plot([i + off - 0.13, i + off + 0.13], [np.median(k)] * 2, color="black", lw=1.0, zorder=5)
            b.text(i + off, 3, f"{int(low.sum())}", ha="center", va="bottom", color=col)
    b.axhline(90, color="0.6", lw=0.6, ls="--", zorder=0)
    b.set_xticks(range(6)), b.set_xticklabels([SHORT6[r] for r in rois])
    b.set_xlim(-0.6, 5.6), b.set_ylim(0, 103)
    b.set_ylabel("FEM retention (%)")
    b.scatter([], [], s=12, marker="o", color=C1, label="1 mm"); b.scatter([], [], s=12, marker="s", color=C2, label="2 mm")
    b.legend(frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 0.1))
    b.set_title("Six reference targets, unconstrained F, 300 heads each\nbars: medians; numbers: combinations below 90%",
                loc="left")
    plabel(b, "B", x=-0.12)
    # C: per region, median and 5th percentile of retention under the selection rule
    c = fig.add_subplot(gs[1, :])
    T = N()["targets"]
    top = 100.0
    for m, col, mk, g in ((SP2, C2, "s", "2 mm"), (SP, C1, "o", "1 mm")):
        p5 = np.array([val(t, m, "fem_optimum_kept_p5") for t in T])
        p50 = np.array([val(t, m, "fem_optimum_kept_median") for t in T])
        o = np.argsort(p5)
        x = np.arange(1, len(T) + 1)
        c.scatter(x, p5[o], s=7, marker=mk, color=col, lw=0, label=f"{g}: 5th percentile across heads")
        c.plot(x, np.sort(p50), color=col, lw=1.0, label=f"{g}: median across heads")
        top = max(top, float(np.nanmax(p50)))
        if m == SP:
            for j in np.flatnonzero(p5[o] < 90):
                name = {"brainstem:VTA_L": "VTA L", "brainstem:VTA_R": "VTA R"}.get(T[o[j]], __import__("build_rev04").label(T[o[j]]))
                c.annotate(f"{name}, {p5[o][j]:.1f}%", (x[j], p5[o][j]), xytext=(26, 91.2 + 1.8 * j), va="center",
                           fontsize=FS, color=col, path_effects=HALO, arrowprops=dict(arrowstyle="-", lw=0.5, color=col))
    c.axhline(90, color="0.6", lw=0.6, ls="--", zorder=0)
    c.set_xlim(0, len(T) + 1)
    c.set_xlabel(f"Regions, ranked separately for each series (n = {len(T)})")
    c.set_ylabel("FEM retention (%)")
    c.set_ylim(min(70, c.get_ylim()[0]), max(100.8, top + 0.5))
    c.legend(frameon=False, loc="lower right", ncol=2)
    c.set_title("163 regions under the selection rule, 300 heads per region; retention relative to FEM's constrained "
                "optimum", loc="left")
    plabel(c, "C", x=-0.055)
    save(fig, "Figure3")


# ================================================================== Figure 4
SLICES = [-40, -24, -15, -8, 2, 14, 34, 56]          # chosen so that every drawn AAL / AAL3 region appears
XS, YS = np.arange(-74, 75), np.arange(-110, 76)
REPS = ["DLPFC_L_Fox2012_optimized", "sgACC_L", "NAcc_L", "MDthal_L", "Temporal_Mid_L", "Cerebellum_Crus1_L"]
REP_NAME = {"DLPFC_L_Fox2012_optimized": "DLPFC L (Fox opt.)", "sgACC_L": "Subgenual ACC L", "NAcc_L": "N. accumbens L",
            "MDthal_L": "MD thalamus L", "Temporal_Mid_L": "Mid. temporal L", "Cerebellum_Crus1_L": "Cerebellar crus I L"}
AAL3_TIER2 = {"MDthal_L": [135, 137], "MDthal_R": [136, 138], "Pulvinar_L": [143, 145, 147, 149],
              "Pulvinar_R": [144, 146, 148, 150], "NAcc_L": [157], "NAcc_R": [158]}


def sample(img, z, order0=True):
    """Nearest-neighbour values of a NIfTI image on the world grid (XS, YS) at height z; rows = y (anterior up)."""
    arr = np.asanyarray(img.dataobj)
    X, Y = np.meshgrid(XS, YS)
    ijk = np.linalg.inv(img.affine) @ np.stack([X.ravel(), Y.ravel(), np.full(X.size, z), np.ones(X.size)])
    i, j, kk = (np.rint(ijk[n]).astype(int) for n in range(3))
    ok = (i >= 0) & (j >= 0) & (kk >= 0) & (i < arr.shape[0]) & (j < arr.shape[1]) & (kk < arr.shape[2])
    out = np.zeros(X.size, arr.dtype)
    out[ok] = arr[i[ok], j[ok], kk[ok]]
    return out.reshape(X.shape)


def region_maps():
    """For each slice: the T1 background, the drawn region of every voxel and the brainstem mask."""
    from fig_ti_targets_mni import M2M, AAL, AAL_REGIONS
    v1 = {r["name"]: r["value"] for r in json.load(open(AAL_REGIONS))["regions"]}
    lut3 = {}
    for line in open(AAL3_NII + ".txt", encoding="utf-8", errors="replace"):
        p = line.split()
        if len(p) >= 2 and p[0].isdigit():
            lut3[p[1]] = int(p[0])
    T = N()["targets"]
    tier = {t: N()["A"][(t, FE)]["tier"] for t in T}
    # AAL thalamus is left out: the AAL3 nuclei cover it, and only atlas-mismatch rims would remain
    layers = [(t, "aal", [v1[t]]) for t in T if tier[t] == "tier1" and not t.startswith("Thalamus")]
    cortex = [v1[t] for t in T if tier[t] == "tier1"]
    layers += [(t, "aal3", AAL3_TIER2[t]) for t in T if t in AAL3_TIER2]
    layers += [(t, "aal3", [lut3[t.replace("brainstem:", "")]]) for t in T if tier[t] in ("cerebellum", "thalamus", "brainstem")]
    t1, aal, aal3 = nib.load(os.path.join(M2M, "T1.nii.gz")), nib.load(AAL), nib.load(AAL3_NII)
    out = []
    for z in SLICES:
        a1, a3 = sample(aal, z).astype(int), sample(aal3, z).astype(int)
        bg = sample(t1, z).astype(float)
        reg = np.full(a1.shape, -1)
        for n, (t, src, vals) in enumerate(layers):
            m = np.isin(a1 if src == "aal" else a3, vals)
            if tier[t] == "cerebellum":
                m &= ~np.isin(a1, cortex)                    # AAL3 cerebellum does not overpaint AAL cortex
            reg[m] = n
        bs = np.isin(reg, [n for n, (t, _, _) in enumerate(layers) if tier[t] == "brainstem"])
        out.append(dict(z=z, bg=bg, reg=reg, bs=bs))
    drawn = sorted({layers[n][0] for s in out for n in np.unique(s["reg"]) if n >= 0})
    return layers, out, drawn


def mini_cap(ax, montage, xy, R):
    p1, p2 = [p.split("+") for p in montage.split(" / ")]
    ax.add_patch(Circle((0, 0), R, fill=False, edgecolor="0.5", lw=0.7))
    ax.add_patch(Polygon([[-7, R - 1], [7, R - 1], [0, R + 10]], closed=True, fill=False, edgecolor="0.5", lw=0.7))
    for lab, (x, y) in xy.items():
        ax.scatter(x, y, s=5, color="0.75", lw=0, zorder=2)
    for pair, col in ((p1, CH1), (p2, CH2)):
        (xa, ya), (xb, yb) = xy[pair[0]], xy[pair[1]]
        ax.plot([xa, xb], [ya, yb], color=col, lw=1.4, alpha=0.8, zorder=3)
        for lab in pair:
            x, y = xy[lab]
            ax.scatter(x, y, s=26, color=col, edgecolor="black", lw=0.5, zorder=4)
            side = abs(x) > 15                                  # lateral sites: label beside, outward; midline: above
            ax.text(x + (9 if x > 0 else -9) if side else x, y if side else y + 13, lab, va="center", fontsize=FS,
                    ha=("left" if x > 0 else "right") if side else "center", zorder=5, path_effects=HALO)
    ax.set_aspect("equal"), ax.axis("off")
    ax.set_xlim(-R - 26, R + 26), ax.set_ylim(-R - 8, R + 14)


def fig4():
    from fig_ti_targets_mni import read_cap, cap_layout
    layers, sl, drawn = region_maps()
    T = N()["targets"]
    print(f"  Figure 4: {len(drawn)} of {len(T)} regions visible on {len(SLICES)} slices; not drawn:",
          sorted(set(T) - set(drawn)))
    fig = plt.figure(figsize=(W2, 140 * MM))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.45], hspace=0.12, left=0.035, right=0.89, top=0.955, bottom=0.01)
    rows = [("A", "F_choice_median", "Median F (T / O),\nlog scale", "coolwarm", LogNorm(vmin=0.625, vmax=1.6)),
            ("B", "T_choice_median_Vm_per_mA", "Median T (V/m per mA\nper channel)", "viridis", plt.Normalize(0.14, 0.34)),
            ("C", "heads_F_above_1_pct", "Heads with F > 1 (%)",
             ListedColormap(plt.get_cmap("Purples")(np.linspace(0.3, 1.0, 256))), plt.Normalize(0, 100))]
    for r, (tag, key, name, cmap, norm) in enumerate(rows):
        g = gs[r].subgridspec(1, len(SLICES), wspace=0.03)
        vals = np.array([val(t, SP, key) for t, _, _ in layers])
        im = None
        for c, s in enumerate(sl):
            ax = fig.add_subplot(g[c])
            bg = s["bg"]
            ax.imshow(bg, cmap="gray", origin="lower", vmin=0, vmax=2 * np.percentile(bg[bg > 0], 99.5) if (bg > 0).any() else 1,
                      extent=(XS[0], XS[-1], YS[0], YS[-1]))
            v = np.where(s["reg"] >= 0, vals[np.maximum(s["reg"], 0)], np.nan)
            im = ax.imshow(v, cmap=cmap, norm=norm, origin="lower", interpolation="nearest",
                           extent=(XS[0], XS[-1], YS[0], YS[-1]))
            edge = np.zeros(s["reg"].shape, bool)              # region borders, so neighbouring similar values stay apart
            edge[:, 1:] |= s["reg"][:, 1:] != s["reg"][:, :-1]
            edge[1:, :] |= s["reg"][1:, :] != s["reg"][:-1, :]
            edge &= s["reg"] >= 0
            ax.imshow(np.where(edge, 1.0, np.nan), cmap=ListedColormap(["#202020"]), alpha=0.45, origin="lower",
                      interpolation="nearest", extent=(XS[0], XS[-1], YS[0], YS[-1]))
            if s["bs"].any():
                ax.contour(XS, YS, s["bs"].astype(float), levels=[0.5], colors="black", linewidths=0.5)
            clean(ax)
            if r == 0:
                ax.set_title(f"z = {s['z']:+d} mm".replace("-", "−"), pad=2)
            if r == 0 and c == 0:
                ax.text(0.02, 0.02, "L", transform=ax.transAxes, color="white", fontsize=FS)
                ax.text(0.86, 0.02, "R", transform=ax.transAxes, color="white", fontsize=FS)
            if c == 0:
                plabel(ax, tag, x=-0.16, y=0.9)
        pos = ax.get_position()
        cax = fig.add_axes([0.895, pos.y0 + 0.08 * pos.height, 0.012, 0.84 * pos.height])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(name, fontsize=FS)
        if tag == "A":
            cb.set_ticks([0.625, 0.8, 1, 1.25, 1.6]), cb.set_ticklabels(["0.625", "0.8", "1", "1.25", "1.6"])
            cb.ax.minorticks_off()
        else:
            cb.ax.locator_params(nbins=4)
    # D: cohort montages of representative regions
    cap = read_cap()
    xy = cap_layout(cap)
    R = max(np.hypot(*p) for p in xy.values()) + 12
    g = gs[3].subgridspec(1, len(REPS), wspace=0.05)
    for c, t in enumerate(REPS):
        ax = fig.add_subplot(g[c])
        mont = N()["A"][(t, SP)]["cohort_choice"]
        femc = N()["A"][(t, FE)]["cohort_choice"]
        mini_cap(ax, mont, xy, R)
        ax.set_title(f"{REP_NAME[t]}\n{mont}\n{'same as FEM' if femc == mont else 'FEM: ' + femc}\n"
                     f"F {val(t, SP, 'p_F_cohort_median'):.2f}, T {val(t, SP, 'p_T_cohort_median'):.2f}",
                     fontsize=FS, pad=1, linespacing=1.1)
        if c == 0:
            ax0 = ax
    fig.canvas.draw()
    fig.text(0.005, ax0.get_position().y1 + 0.085, "D", fontsize=11, fontweight="bold", va="bottom")
    fig.text(0.902, 0.255, "Channel 1", color=CH1, fontsize=FS, fontweight="bold")
    fig.text(0.902, 0.23, "Channel 2", color=CH2, fontsize=FS, fontweight="bold")
    fig.text(0.902, 0.01, "1 mA each;\nT in V/m\nper mA;\nview from\nabove, nose\nup, left on\nthe left", color="0.3",
             fontsize=FS, va="bottom", linespacing=1.1)
    save(fig, "Figure4")
    return drawn


# ================================================================== Figure 5
def strips(ax, key, xlabel, filled=None, label_top=0, ref=None, fmt=None):
    """One horizontal strip per region set; one point per region (SPICEhead 1 mm); black tick: set median."""
    T = N()["targets"]
    tier = {t: N()["A"][(t, FE)]["tier"] for t in T}
    ypos = {}
    for i, s in enumerate(SETS):
        ts = [t for t in T if tier[t] == s]
        v = np.array([key(t) for t in ts])
        y = i + jitter(len(ts), 0.5, seed=i)
        ypos.update(zip(ts, y))
        f = np.ones(len(ts), bool) if filled is None else np.array([filled(t) for t in ts])
        ax.scatter(v[f], y[f], s=10, color=C1, lw=0, alpha=0.85)
        ax.scatter(v[~f], y[~f], s=10, facecolors="white", edgecolors=C1, lw=0.6)
        ax.plot([np.median(v)] * 2, [i - 0.34, i + 0.34], color="black", lw=1.1, zorder=5)
    if ref is not None:
        ax.axvline(ref, color="0.6", lw=0.6, ls="--", zorder=0)
    ax.set_yticks(range(len(SETS)))
    ax.set_yticklabels([f"{SET_NAME[s]} ({sum(N()['A'][(t, FE)]['tier'] == s for t in T)})" for s in SETS])
    ax.set_ylim(len(SETS) - 0.45, -0.55)
    ax.set_xlabel(xlabel)
    if label_top:
        top = sorted(T, key=lambda t: -key(t))[:label_top]
        for n, t in enumerate(top):
            name = __import__("build_rev04").label(t)
            ax.annotate(f"{name} ({key(t):.0f}%)", (key(t), ypos[t]), textcoords="offset points",
                        xytext=(-6, 10) if n == 0 else (-6, -11), ha="right", va="center", fontsize=FS,
                        path_effects=HALO, arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4"))
        ax.set_ylim(len(SETS) - 0.05, -0.55)                    # room for a label below the last row


def fig5():
    A = N()["A"]
    fig = plt.figure(figsize=(W2, 120 * MM))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.42, left=0.175, right=0.99, top=0.95, bottom=0.08)
    a = fig.add_subplot(gs[0, 0])
    strips(a, lambda t: val(t, SP, "heads_gain_ge_10pct"), "Heads gaining ≥ 10% in F individually (%)", label_top=3, ref=20)
    a.set_title("Benefit of an individual montage", loc="left")
    fig.text(0.005, a.get_position().y1 + 0.012, "A", fontsize=11, fontweight="bold")
    b = fig.add_subplot(gs[0, 1])
    strips(b, lambda t: val(t, SP, "p_T_choice_p95") / val(t, SP, "p_T_choice_p5"), "T spread across heads, p95 / p5")
    b.set_yticklabels([])
    b.set_title("Spread of T at the reported montage", loc="left")
    plabel(b, "B", x=-0.08, y=1.02)
    c = fig.add_subplot(gs[1, 0])
    strips(c, lambda t: val(t, SP, "p_age_T_pct_per_decade"), "Change of T per decade of age (%)",
           filled=lambda t: val(t, SP, "p_age_q") < 0.05, ref=0)
    c.scatter([], [], s=10, color=C1, label="q < 0.05"); c.scatter([], [], s=10, facecolors="white", edgecolors=C1, label="n.s.")
    c.legend(frameon=False, loc="lower right", handletextpad=0.1)
    c.set_title("Age, adjusted for sex and site", loc="left")
    fig.text(0.005, c.get_position().y1 + 0.012, "C", fontsize=11, fontweight="bold")
    d = fig.add_subplot(gs[1, 1])
    T = N()["targets"]
    m0 = np.array([100 * (np.exp(val(t, SP, "s_logT_sex_M0")) - 1) for t in T])
    m3 = np.array([100 * (np.exp(val(t, SP, "s_logT_sex_M3")) - 1) for t in T])
    q3 = np.array([val(t, SP, "s_logT_q_sex_M3") for t in T])
    d.scatter(m0[q3 >= 0.05], m3[q3 >= 0.05], s=10, facecolors="white", edgecolors=C1, lw=0.6, label="after adjustment n.s.")
    d.scatter(m0[q3 < 0.05], m3[q3 < 0.05], s=10, color=C1, lw=0, label=f"after adjustment q < 0.05 ({int((q3 < 0.05).sum())})")
    lim = (min(m0.min(), m3.min()) - 1, max(4, m3.max() + 1))
    d.plot(lim, lim, color="0.6", lw=0.6, ls="--", zorder=0)
    d.axhline(0, color="0.6", lw=0.6, zorder=0)
    d.set_xlim(lim), d.set_ylim(lim), d.set_aspect("equal", adjustable="box")
    d.set_xlabel("Men vs women, adjusted for age and site (%)")
    d.set_ylabel("Also adjusted for intracranial volume\nand extracranial thickness (%)")
    d.text(0.97, 0.3, f"medians {np.median(m0):+.1f}% → {np.median(m3):+.1f}%".replace("-", "−"),
           transform=d.transAxes, va="bottom", ha="right")
    d.legend(frameon=False, loc="lower right")
    d.set_title("Sex difference in T, before and after\nadjustment for head size",
                loc="left")
    plabel(d, "D", x=-0.42, y=1.02)
    save(fig, "Figure5")


def figures_doc():
    """09_figures_for_author_review.docx: each figure with its Rev0.4 legend."""
    import docx
    from docx.shared import Cm, Pt
    from build_rev04 import LEGENDS
    d = docx.Document()
    d.styles["Normal"].font.name = "Arial"
    d.styles["Normal"].font.size = Pt(10)
    d.add_paragraph().add_run("Rev0.4 figures (v23, 300 heads), for author review").bold = True
    d.add_paragraph("Drawn by pipeline/fig_v23_rev04.py from result files; legends as in 07_figure_legends.docx. "
                    "Journal files: figures/Figure1-5.pdf (vector) and .png (600 dpi).")
    for i, (head, text) in enumerate(LEGENDS, 1):
        d.add_page_break()
        d.add_picture(os.path.join(OUT, f"Figure{i}.png"), width=Cm(16.5))
        p = d.add_paragraph()
        p.add_run(head + ". ").bold = True
        p.add_run(text)
    from docx.oxml.ns import qn
    for z in d.settings.element.iter(qn("w:zoom")):            # the default template omits the required percent
        z.set(qn("w:percent"), z.get(qn("w:percent")) or "100")
    path = os.path.join(PKG, "09_figures_for_author_review.docx")
    d.save(path)
    print("wrote", path)


if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("--")] or ["1", "2", "3", "4", "5"]
    for w in want:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5}[w]()
    if "--docx" in sys.argv:
        figures_doc()
