# -*- coding: utf-8 -*-
"""Cohort processing-path figure: what the 300 cases go through in this study.

One box per stage, with the counts the manuscript reports. Vector PDF plus a 300-dpi PNG,
written to "Paper_A_SPICEhead/Brain Stimulation/figures/". Candidate for the Figure 4 cohort hand-off panel or a
Methods overview panel.

  python pipeline/fig_cohort_flow.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
OUT = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "figures")

SERIF = {"family": "serif", "serif": ["Times New Roman", "DejaVu Serif"]}
plt.rc("font", **SERIF)

# pad and rounding_size are in data units on a 0-1 axis, so they must be small; the first draft
# used pad=0.32, which rendered as arcs a third of the page wide.
BOX = dict(boxstyle="round,pad=0.012,rounding_size=0.014", linewidth=1.1,
           facecolor="white", edgecolor="black")
END = dict(boxstyle="round,pad=0.012,rounding_size=0.014", linewidth=1.1,
           facecolor="#f2f2f2", edgecolor="black")


def box(ax, xy, w, h, title, lines, style=BOX):
    x, y = xy
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, **style, mutation_scale=1)
    ax.add_patch(p)
    ax.text(x, y + h / 2 - 0.014, title, ha="center", va="top", fontsize=10.5, weight="bold")
    body = "\n".join(lines)
    ax.text(x, y + h / 2 - 0.046, body, ha="center", va="top", fontsize=9.2, linespacing=1.4)


def arrow(ax, a, b, label=None):
    ar = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.1,
                         color="black", shrinkA=2, shrinkB=2)
    ax.add_patch(ar)
    if label:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx + 0.03, my, label, ha="left", va="center", fontsize=8.8, style="italic")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cx = 0.5
    # ---- stage boxes, top to bottom -------------------------------------------------
    box(ax, (cx, 0.935), 0.86, 0.105, "300 T1-weighted MRI, four sites",
        ["Bai_TVGH 51  ·  BT_TSGH 50  ·  TAMI 130  ·  Vet_Home 69",
         "ages 20–88 (mean 54.4 ± 20.1) · 169 M / 131 F"])

    box(ax, (cx, 0.785), 0.86, 0.095, "CHARM ten-tissue segmentation (SimNIBS 4.6)",
        ["300 of 300 segmented · 4 required --forcesform",
         "native space, no template warp of the anatomy"])

    box(ax, (cx, 0.635), 0.86, 0.100, "Voxel resistor network, Δx = 5 mm",
        ["g = σ·Δx per edge, arithmetic interface rule",
         "≈3×10⁴ nodes per subject · factorized once per subject"])

    box(ax, (cx, 0.480), 0.86, 0.100, "171 electrode-pair unit fields per subject",
        ["19 EEG 10-20 positions → C(19,2) = 171 pair solves",
         "every further montage costs superposition only"])

    box(ax, (cx, 0.325), 0.86, 0.112, "≈11,600 two-pair montages scored per target",
        ["six 10-mm MNI targets warped to subject space",
         "focality F = T/O in gray matter · pairs ≥40 mm, no shared electrode",
         "extensions cost no extra solves: 9/10 chain, steering, 10-10 grid (S2.1–S2.3)"])

    # ---- outputs row ----------------------------------------------------------------
    yb = 0.115
    box(ax, (0.18, yb), 0.295, 0.150, "Cohort optima",
        ["Table 6:", "optimal montage,", "F per electrode set,", "in-ROI envelope"], END)
    box(ax, (0.50, yb), 0.295, 0.150, "Generalization",
        ["LOSO: 16/24 exact,", "median loss 0.00%", "learning curve:", "5–200 heads × 40 draws"],
        END)
    box(ax, (0.82, yb), 0.295, 0.150, "Resolution check",
        ["top 25 per target", "re-scored at 2 mm", "in 20 subjects", "(Table 7)"], END)

    # ---- arrows ----------------------------------------------------------------------
    arrow(ax, (cx, 0.882), (cx, 0.833))
    arrow(ax, (cx, 0.737), (cx, 0.686))
    arrow(ax, (cx, 0.585), (cx, 0.531))
    arrow(ax, (cx, 0.430), (cx, 0.376))
    arrow(ax, (cx, 0.269), (0.18, yb + 0.078))
    arrow(ax, (cx, 0.269), (0.50, yb + 0.078))
    arrow(ax, (cx, 0.269), (0.82, yb + 0.078))



    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig_cohort_flow.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print("wrote", os.path.relpath(p, ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
