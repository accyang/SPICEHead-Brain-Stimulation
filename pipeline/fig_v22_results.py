# -*- coding: utf-8 -*-
"""Figures 2-4 of the v22 paper, drawn from the same result files as values_v22.py.

  Figure 2  numerical verification and grid sensitivity
            A disk: error against the exact gradient; B layered slab: resistance ratio by
            interface rule; C four heads: correlation with each head's 1 mm solution;
            D four heads: target-intensity excess over 1 mm
  Figure 3  comparison with finite-element fields
            A-C SPICEhead, SimNIBS and difference maps in one head (bilateral temporal montage,
            1 mm); D voxel-wise agreement in that head; E mean bias by montage and grid
  Figure 4  cost and outcome of exhaustive montage evaluation
            A stage times and peak working set at 5, 2 and 1 mm (one head); B subject-level
            regret at 2 mm of three coarse-grid choices; C cohort-median T and O of the same
            heads' 5 mm choice and of the 2 mm optimum
  Figure S1 template exposure maps (copied from fig_ti_targets_mni.py)

  python pipeline/fig_v22_results.py [2] [3] [4]        (default: all three)
  SPICEHEAD_VALUES_ROOT=results ... --outdir D           test on the v21 outputs

Output: Paper_A_SPICEhead/Brain Stimulation/figures/v22/fig{2,3,4}_*.{png,pdf}
"""
import csv
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)
os.environ.setdefault("SPICEHEAD_MODEL", "v22")   # v22 record script; the package default is v23 since 2 Nov 2026
from spicehead import config                                         # noqa: E402
import values_v22 as VV                                              # noqa: E402

OUT = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "figures", "v22")
ROIS, NICE = VV.ROIS, VV.NICE
SHORT = {"M1_handknob": "M1", "thalamus": "Thal", "insula": "Ins", "ACC": "ACC",
         "hippocampus_L": "Hipp L", "hippocampus_R": "Hipp R"}
MAP_SUBJECT = "SUBJ01"


def rows(*p):
    path = os.path.join(*p)
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []


def label(ax, t):
    ax.text(-0.14, 1.04, t, transform=ax.transAxes, fontsize=13, fontweight="bold")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, f"{name}.png"))


def fig2():
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    d = rows(VV.RES0, "v21_validation", "disk_analytic_gradient.csv")
    dx = [float(r["dx_mm"]) for r in d]
    # the network against the exact closed-form gradient (review of 2 October, R5.2)
    axs[0].plot(dx, [float(r["model_vs_exact_nmae_pct"]) for r in d], "o-", label="network vs exact")
    axs[0].plot(dx, [float(r["discrete_vs_exact_nmae_pct"]) for r in d], "s--", ms=4,
                label="discrete reference vs exact")
    axs[0].set(xscale="log", yscale="log", xlabel="Grid spacing (mm)",
               ylabel="nMAE vs exact gradient (%)", title="Homogeneous disk")
    axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].set_xticks([1, 2, 3, 4, 5]), axs[0].set_xticklabels(["1", "2", "3", "4", "5"])
    axs[0].invert_xaxis()
    axs[0].legend(fontsize=7, frameon=False)
    s = rows(VV.RES0, "v21_interface", "interface_rule.csv")
    for rule, mk in (("harmonic", "o-"), ("arithmetic", "s--")):
        q = [r for r in s if r["rule"] == rule]
        axs[1].plot([float(r["dx_mm"]) for r in q], [float(r["R_ratio"]) for r in q], mk, label=rule)
    axs[1].axhline(1, color="0.5", lw=0.8)
    axs[1].set(xlabel="Grid spacing (mm)", ylabel="Network / exact resistance",
               title="Layered slab", ylim=(0.4, 1.1))
    axs[1].invert_xaxis()
    axs[1].legend(fontsize=7, frameon=False)
    per = {}
    for h in VV.HEADS4:
        p = (os.path.join(VV.RES, "v21_head_convergence", "head_convergence.csv") if h == "SUBJ02"
             else os.path.join(VV.RES, "v21_head_convergence", h, "head_convergence.csv"))
        if os.path.exists(p):
            per[h] = {float(r["dx_mm"]): r for r in csv.DictReader(open(p, encoding="utf-8"))}
    for n, (h, rr) in enumerate(per.items(), 1):
        ds = sorted(d for d in rr if d > 1.0)
        axs[2].plot(ds, [float(rr[d]["r"]) for d in ds], "o-", ms=3, lw=1, label=f"Head {n}")
    axs[2].set(xlabel="Grid spacing (mm)", ylabel="r with the head's 1 mm solution",
               title="Four heads, gray matter", ylim=(0.4, 1.0))
    axs[2].invert_xaxis()
    axs[2].legend(fontsize=7, frameon=False)
    for roi in ROIS:
        ds = sorted(d for d in next(iter(per.values())) if d > 1.0) if per else []
        ex = [np.mean([100 * (float(per[h][d][f"T_{roi}"]) / float(per[h][1.0][f"T_{roi}"]) - 1)
                       for h in per]) for d in ds]
        axs[3].plot(ds, ex, "o-", ms=3, lw=1, label=SHORT[roi])
    axs[3].axhline(0, color="0.5", lw=0.8)
    axs[3].set(xlabel="Grid spacing (mm)", ylabel="Target T, excess over 1 mm (%)",
               title="Target intensity, mean of four heads")
    axs[3].invert_xaxis()
    axs[3].legend(fontsize=7, frameon=False, ncol=2)
    for a, t in zip(axs, "ABCD"):
        label(a, t)
    fig.tight_layout()
    save(fig, "fig2_verification")


def fig3():
    import nibabel as nib
    import v21_simnibs_compare as C
    from v21_multimontage_fem import fem_dir
    from v21_simnibs_validation import subjects
    from montage_search import SCORE
    cid, m2m = next((s[0], s[1]) for s in subjects(8) if s[0] == MAP_SUBJECT)
    cache = os.environ.get("SPICEHEAD_FIG3_CACHE") or os.path.join(
        os.environ.get("TEMP", "."), config.tagged("spicehead_femcmp_cache"))
    sp = pickle.load(open(os.path.join(cache, f"{cid}__dx1_env.pkl"), "rb"))
    E1, aff, sh = C.load_simnibs_E(fem_dir(cid, "THAL", 1))
    E2, _, _ = C.load_simnibs_E(fem_dir(cid, "THAL", 2))
    fem = np.nan_to_num(C.envelope(E1, E2))
    li = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    gm = C.resample_to(np.isin(np.asanyarray(li.dataobj), SCORE).astype(np.float32),
                       li.affine, aff, sh) > 0.5
    sp = np.asarray(sp, float).reshape(fem.shape)
    import json
    # axial slice through the head's own thalamus centre (mean of the two hemispheres)
    c = np.mean(np.atleast_2d(json.load(open(os.path.join(m2m, "roi_coords.json")))["thalamus"]), 0)
    k = int(round((np.linalg.inv(aff) @ np.append(c, 1.0))[2]))
    vmax = float(np.percentile(fem[gm], 99))
    ii, jj = np.nonzero(gm[:, :, k])
    box = (slice(max(ii.min() - 4, 0), ii.max() + 5), slice(max(jj.min() - 4, 0), jj.max() + 5))
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], wspace=0.28, hspace=0.32)
    for i, (img, t, cm, lim) in enumerate(((sp, "SPICEhead, 1 mm", "inferno", (0, vmax)),
                                           (fem, "SimNIBS FEM", "inferno", (0, vmax)),
                                           (sp - fem, "SPICEhead − FEM", "RdBu_r",
                                            (-vmax / 2, vmax / 2)))):
        ax = fig.add_subplot(gs[0, i])
        sl = np.where(gm[:, :, k], img[:, :, k], np.nan)[box]
        h = ax.imshow(np.rot90(sl), cmap=cm, vmin=lim[0], vmax=lim[1])
        ax.set_xticks([]), ax.set_yticks([]), ax.set_title(t, fontsize=9)
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02, label="V/m")
        label(ax, "ABC"[i])
    ax = fig.add_subplot(gs[1, 0])
    x, y = sp[gm], fem[gm]
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    ax.hexbin(x[ok], y[ok], gridsize=60, bins="log", cmap="Greys", mincnt=1)
    m = float(np.percentile(np.concatenate([x[ok], y[ok]]), 99.5))
    ax.plot([0, m], [0, m], color="#d62728", lw=1)
    ax.set(xlim=(0, m), ylim=(0, m), xlabel="SPICEhead (V/m)", ylabel="FEM (V/m)",
           )
    ax.set_title(f"Gray matter, r = {np.corrcoef(x[ok], y[ok])[0, 1]:.3f}", fontsize=9)
    label(ax, "D")
    ax = fig.add_subplot(gs[1, 1:])
    mm = rows(VV.RES, "v21_multimontage", "agreement_by_montage.csv")
    for j, dx in enumerate((2.0, 1.0)):
        for i, t in enumerate(VV.MM_ORDER):
            s = [r for r in mm if r["montage"] == t and float(r["dx_mm"]) == dx]
            rel = [100 * (float(r["mean_spicehead"]) / float(r["mean_fem"]) - 1) for r in s]
            xpos = i + (j - 0.5) * 0.35
            ax.bar(xpos, np.mean(rel), width=0.33, color=("#9ecae1", "#3182bd")[j],
                   label=f"{dx:g} mm" if i == 0 else None)
            ax.scatter(np.full(len(rel), xpos), rel, s=6, color="k", zorder=3)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(4)), ax.set_xticklabels(["Fronto-parietal", "Bilateral temporal",
                                                 "Fronto-occipital", "Inferior chain"], fontsize=8)
    ax.set_ylabel("Mean bias, % of FEM")
    ax.set_title("Bias by montage and SPICEhead grid, eight heads", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    label(ax, "E")
    save(fig, "fig3_fem")


def fig_sweep():
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.9), gridspec_kw=dict(width_ratios=[1, 1.7, 1.1]))
    b = rows(VV.RES, "v22_benchmark", "heads.csv")
    stages = [("t_load_s", "load"), ("t_build_s", "build"), ("t_setup_s", "setup"),
              ("t_basis_s", "18 basis solves"), ("t_check_s", "checks"),
              ("t_score_s", "scoring 11,628 montages")]
    if b:
        x = np.arange(len(b))
        bottom = np.zeros(len(b))
        for k_, lab in stages:
            v = np.array([float(r[k_]) / 60 for r in b])
            axs[0].bar(x, v, bottom=bottom, label=lab)
            bottom += v
        for i, r in enumerate(b):
            axs[0].text(i, bottom[i] * 1.03, f"{float(r['peak_gb']):.1f} GB", ha="center", fontsize=8)
        axs[0].set_xticks(x)
        axs[0].set_xticklabels([f"{float(r['dx_mm']):g} mm\n{int(float(r['n_nodes'])):,} nodes"
                                for r in b], fontsize=8)
        # one measured head; the bar is the sum of the instrumented stages, the label the
        # process's peak working set
        axs[0].set(ylabel="Sum of instrumented stages (min)",
                   title="One head, all montages, six targets")
        axs[0].legend(fontsize=7, frameon=False)
    else:
        axs[0].text(0.5, 0.5, "benchmark pending", ha="center", transform=axs[0].transAxes)
    label(axs[0], "A")
    q = rows(VV.RES, "v21_dx2_check_full", "dx2_full_subject_regret.csv")
    series = (("regret_5mm_cohort_winner_pct", "#3182bd", "300-head 5 mm choice"),
              ("regret_own_5mm_optimum_pct", "#e6550d", "head's own 5 mm optimum"),
              ("regret_best_shortlisted_pct", "#31a354", "best of the 5 mm top 25"))
    w = 0.26
    for i, roi in enumerate(ROIS):
        for j, (col, c, _) in enumerate(series):
            v = [float(r[col]) for r in q if r["roi"] == roi]
            xpos = i + (j - 1) * w
            axs[1].boxplot(v, positions=[xpos], widths=w * 0.85, showfliers=False,
                           medianprops=dict(color=c), boxprops=dict(color=c),
                           whiskerprops=dict(color=c), capprops=dict(color=c))
            axs[1].scatter(xpos + np.random.default_rng(i * 3 + j).uniform(-0.06, 0.06, len(v)),
                           v, s=6, color=c, alpha=0.7, zorder=3)
    axs[1].set_xticks(range(len(ROIS))), axs[1].set_xticklabels([SHORT[r] for r in ROIS])
    axs[1].set(ylabel="Regret: shortfall from the head's\nown 2 mm optimum (% of its F)",
               title="Twenty heads at 2 mm, all 11,628 montages")
    axs[1].legend(handles=[plt.Line2D([], [], color=c, marker="o", ls="", label=lab)
                           for _, c, lab in series], fontsize=7, frameon=False)
    label(axs[1], "B")
    # exposure behind the focality gain: cohort-median T and O at 2 mm of the 2 mm optimum and of
    # the same 20 heads' 5 mm choice, for the targets where the two differ
    V, _ = VV.values()
    ax = axs[2]
    for roi in ROIS:
        if f"sweep.{roi}.T2" not in V or V.get(f"sweep.{roi}.same_loss", 0) <= 0:
            continue
        o5, t5 = V[f"sweep.{roi}.O5s"], V[f"sweep.{roi}.T5s"]
        o2, t2 = V[f"sweep.{roi}.O2"], V[f"sweep.{roi}.T2"]
        ax.annotate("", xy=(o2, t2), xytext=(o5, t5),
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1))
        ax.scatter([o5], [t5], s=28, facecolors="white", edgecolors="#e6550d", zorder=3)
        ax.scatter([o2], [t2], s=28, color="#3182bd", zorder=3)
        ax.text(o2, t2, f"  {SHORT[roi]}", fontsize=8, va="center")
    lim = ax.get_xlim()[1] * 1.05, ax.get_ylim()[1] * 1.05
    m = max(lim)
    ax.plot([0, m], [0, m], color="0.75", lw=0.8, ls="--")
    ax.set(xlim=(0, m), ylim=(0, m), xlabel="O, cohort median at 2 mm (V/m)",
           ylabel="T, cohort median at 2 mm (V/m)", title="Exposure: 5 mm choice → 2 mm optimum")
    ax.legend(handles=[plt.Line2D([], [], color="#e6550d", marker="o", mfc="white", ls="",
                                  label="same heads' 5 mm choice"),
                       plt.Line2D([], [], color="#3182bd", marker="o", ls="", label="2 mm optimum")],
              fontsize=7, frameon=False, loc="upper left")
    label(ax, "C")
    fig.tight_layout()
    save(fig, "fig5_sweep")


def figS1():
    """Supplementary Figure S1 (formerly Figure 5) is drawn by fig_ti_targets_mni.py (the mni_maps
    job): template exposure maps. Moved to the supplement after the review of 4 October 2026."""
    import shutil
    src = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "figures",
                       "TI_targets_MNI" + ("" if config.MODEL == "v22" else f"_{config.MODEL}"))
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(src, f"TI_four_targets_MNI152.{ext}")
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found: run fig_ti_targets_mni.py first")
        shutil.copyfile(p, os.path.join(OUT, f"figS1_mni_targets.{ext}"))
    for ext in ("png", "pdf"):                       # the old main-text name is retired
        old = os.path.join(OUT, f"fig5_mni_targets.{ext}")
        if os.path.exists(old):
            os.remove(old)
    print("wrote", os.path.join(OUT, "figS1_mni_targets.png"))


def fig_decisions():
    """Figure 4 (record version): decisions against the FEM reference over all 11,628 montages in the
    300-head run (A, B); four-pad re-evaluation in the eight desktop heads (C)."""
    import json as _json
    d = os.path.join(VV.RES, "v22_fem_fullsearch_300")
    d8 = os.path.join(VV.RES, "v22_fem_fullsearch")
    dec = rows(d, "fullsearch_decisions.csv")
    per = rows(d8, "fem4pad_per_target.csv")
    qcp = os.path.join(d8, "fem4pad_qc.json")
    flagged = set(_json.load(open(qcp, encoding="utf-8"))["comparisons_flagged"]) if os.path.exists(qcp) else set()
    if not dec:
        raise SystemExit("run pipeline/nchc/collect_300.py first")
    nh = len({r["subject"] for r in dec})
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.0), gridspec_kw=dict(width_ratios=[1.6, 1.1, 1.0]))
    ax = axs[0]
    for i, roi in enumerate(ROIS):
        sub = [r for r in dec if r["roi"] == roi]
        for x_, key, c, lab in ((i - 0.17, "regret_sp2winner_under_fem_pct", "#e6550d", "2 mm"),
                                (i + 0.17, "regret_sp1winner_under_fem_pct", "#3182bd", "1 mm")):
            k = np.array([100 - float(r[key]) for r in sub])
            ax.boxplot([k], positions=[x_], widths=0.28, whis=(5, 95), showfliers=False, patch_artist=True,
                       medianprops=dict(color="white"), boxprops=dict(facecolor=c, edgecolor=c),
                       whiskerprops=dict(color=c), capprops=dict(color=c))
            low = k[k < np.percentile(k, 5)]
            ax.scatter(np.full(len(low), x_), low, s=9, color=c, zorder=3, label=lab if i == 0 else None)
    ax.axhline(100, color="0.5", lw=0.8)
    ax.set_xticks(range(len(ROIS))), ax.set_xticklabels([SHORT[r] for r in ROIS])
    ax.set(ylabel="SPICEhead choice scored by FEM\n(% of the FEM optimum's F)",
           title=f"All 11,628 montages, {nh} heads (box: quartiles; whiskers: 5th-95th pct)", ylim=(20, 102))
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    label(ax, "A")
    ax = axs[1]
    pairs = (("agree_fem_sp2", "FEM vs 2 mm", "#e6550d"), ("agree_fem_sp1", "FEM vs 1 mm", "#3182bd"),
             ("agree_sp2_sp1", "2 mm vs 1 mm", "0.55"))
    w = 0.26
    for j, (col, lab, c) in enumerate(pairs):
        v = [sum(r[col] == "True" for r in dec if r["roi"] == roi) for roi in ROIS]
        ax.bar(np.arange(len(ROIS)) + (j - 1) * w, v, width=w * 0.9, color=c, label=lab)
    ax.set_xticks(range(len(ROIS))), ax.set_xticklabels([SHORT[r] for r in ROIS], fontsize=8)
    ax.set(ylabel="Heads with the identical best montage", ylim=(0, nh * 1.1),
           title=f"Exact agreement (of {nh} heads)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    label(ax, "B")
    ax = axs[2]
    models = (("keep_fem", "FEM\nleadfield", "0.35"), ("keep_sp1", "SPICEhead\n1 mm", "#3182bd"),
              ("keep_sp2", "SPICEhead\n2 mm", "#e6550d"))
    rng = np.random.default_rng(1)
    for j, (col, lab, c) in enumerate(models):
        v = [(float(r[col]), f"{r['subject']}:{r['roi']}" in flagged) for r in per if r.get(col)]
        xs = j + rng.uniform(-0.12, 0.12, len(v))
        for x, (y, fl) in zip(xs, v):
            ax.scatter(x, y, s=16, facecolors="white" if fl else c, edgecolors=c, zorder=3)
    ax.set_xticks(range(3)), ax.set_xticklabels([m[1] for m in models], fontsize=8)
    ax.set(ylabel="Four-pad FEM F\n(% of the best selected candidate)", ylim=(30, 102),
           title="Four-pad re-evaluation (8 heads, 48 comparisons)")
    ax.scatter([], [], s=16, facecolors="white", edgecolors="0.35", label="comparison with a flagged run")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    label(ax, "C")
    fig.tight_layout()
    save(fig, "fig4_fem_decisions")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--outdir" in a:
        OUT = a[a.index("--outdir") + 1]
        a = [x for x in a if x not in ("--outdir", OUT)]
    which = [x for x in a if x in ("2", "3", "4", "5", "S1")] or ["2", "3", "4", "5", "S1"]
    for w in which:
        {"2": fig2, "3": fig3, "4": fig_decisions, "5": fig_sweep, "S1": figS1}[w]()
