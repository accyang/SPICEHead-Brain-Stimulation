# -*- coding: utf-8 -*-
"""Summary figure of the template montage-design exploration (insula and hippocampus).

Reads results_v22/v22_template_design/design_levels.csv (pipeline/template_design_collect.py) and
draws, for each target, the design levels side by side: (A) focality F, (B) the share of the
target inside the focus (brain voxels at or above 70% of the field scale), (C) target exposure T.
Equal-current two-pair 10-20 is the reference level; the no-inferior-row variants exclude pads
near the ears, face and neck. Style follows the submission figures (180 mm, Arial, 8 pt).

  SPICEHEAD_MODEL=v22 python pipeline/fig_v22_template_design.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
from spicehead import config                                         # noqa: E402

SRC = os.path.join(config.RESULTS, "v22_template_design", "design_levels.csv")
OUT = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "figures", "template_design")
if config.MODEL not in ("v21", "v22"):          # v23 (28 Oct 2026): the model's own collection and results folder
    SRC = os.path.join(config.RESULTS, f"{config.MODEL}_template_design", "design_levels.csv")
    OUT = os.path.join(config.RESULTS, "figures", "template_design")
MM = 1 / 25.4
TARGETS = ["insula", "insula_L", "insula_R", "insula_ant_L", "insula_post_L", "hippocampus", "hippocampus_L", "hippocampus_R"]
XLAB = {"insula": "Insula\n(bilateral)", "insula_L": "Insula\nL", "insula_R": "Insula\nR", "insula_ant_L": "Anterior\ninsula L",
        "insula_post_L": "Posterior\ninsula L", "hippocampus": "Hippo-\ncampus\n(bilateral)", "hippocampus_L": "Hippo-\ncampus L",
        "hippocampus_R": "Hippo-\ncampus R"}
LEVELS = [("1020", "two-pair 10-20, equal currents", "#8c8c8c"), ("ratio", "two-pair 10-20, unequal currents", "#E69F00"),
          ("1010", "two-pair 10-10", "#0072B2"), ("1010_noinf", "two-pair 10-10, no inferior row", "#56B4E9"),
          ("hdti", "multi-electrode steering", "#D55E00"), ("hdti_noinf", "steering, no inferior row", "#CC79A7")]

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})


def main():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    val = {}
    for r in rows:
        lvl = "1020" if r["level"] in ("s1", "uni") else r["level"]
        val[(r["target"], lvl)] = r
    fig, axs = plt.subplots(3, 1, figsize=(180 * MM, 150 * MM), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.12, hspace=0.25)
    x = np.arange(len(TARGETS))
    w = 0.13
    for a, (col, ylab, scale) in zip(axs, (("F", "Focality F", 1.0), ("target_in_focus_pct", "Target inside\nthe focus (%)", 1.0),
                                           ("T_V_per_m", "Target exposure\nT (V/m)", 1.0))):
        for j, (lvl, lab, c) in enumerate(LEVELS):
            ys = [float(val[(t, lvl)][col]) * scale if (t, lvl) in val and val[(t, lvl)][col] != "" else np.nan for t in TARGETS]
            a.bar(x + (j - 2.5) * w, ys, width=w * 0.95, color=c, label=lab, lw=0)
        a.set_ylabel(ylab)
        a.set_xlim(-0.6, len(TARGETS) - 0.4)
        if col == "F":
            a.axhline(1, color="0.4", lw=0.6, ls="--")
            a.text(len(TARGETS) - 0.45, 1.02, "F = 1", ha="right", va="bottom", fontsize=7, color="0.3")
            a.set_ylim(0, 1.4)
            plabel = "A"
        elif col == "target_in_focus_pct":
            a.set_ylim(0, 80)
            plabel = "B"
        else:
            a.set_ylim(0, 0.2)
            plabel = "C"
        a.text(-0.1, 1.02, plabel, transform=a.transAxes, fontsize=11, fontweight="bold", va="bottom")
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels([XLAB[t] for t in TARGETS])
    axs[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.04), columnspacing=1.2, handlelength=1.2)
    os.makedirs(OUT, exist_ok=True)
    for ext, dpi in (("pdf", 600), ("png", 1000)):
        fig.savefig(os.path.join(OUT, f"Figure_template_design.{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    print("wrote", os.path.join(OUT, "Figure_template_design.pdf/.png"))


if __name__ == "__main__":
    main()
