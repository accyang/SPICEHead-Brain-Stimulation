# -*- coding: utf-8 -*-
"""Collect the template montage-design exploration (13-14 October 2026) into results_v22.

The exploration asked how far the focality of fixed two-pair 10-20 montages for deep targets
(insula, hippocampus) can be improved on the MNI152 template, within the declared model
(SPICEhead v22, 2 mm, 12 mm skin-surface pads, F = median target GM envelope / p99 of GM more than
20 mm from the target). Design levels, each from its own script:

  s1          two-pair 10-20, bilateral AAL target, 1 + 1 mA   (Figure S1; fig_ti_targets_mni.py)
  uni         two-pair 10-20, one-sided target                 (ti_template_unilateral.py; for the
                                                                insular subregions, the 10-20 search
                                                                inside ti_template_1010.py --subregions)
  ratio       two-pair 10-20, unequal channel currents, total 2 mA, montage and split chosen jointly
                                                               (ti_template_current_ratio.py)
  1010        two-pair over the 76 10-10 positions, 1 + 1 mA   (ti_template_1010.py)
  1010_noinf  the same without the inferior row (F9/F10, FT9/FT10, T9/T10, TP9/TP10, P9/P10,
              PO9/PO10, Iz, I1, I2)
  hdti        multi-electrode current steering, up to 6 electrodes per channel, 1 mA per channel
                                                               (ti_template_hdti.py)
  hdti_noinf  the same without the inferior row

This script reads the summaries those scripts wrote under Paper_A_SPICEhead/Brain Stimulation/figures/, adds the
focus measures of the 3-D figures (field scale = 99.9th percentile of the brain envelope; focus =
brain voxels at or above 70% of it), recomputes the steered montages' F independently from the
cached basis fields, runs the consistency checks across the chain of results, and writes

  results_v22/v22_template_design/design_levels.csv       one row per target x design level
                                 /current_ratio_sweep.csv  best F per current split
                                 /subregions.csv           the insular subregion definitions
                                 /checks.json              every consistency check, pass/fail
                                 /summary.json             search parameters
                                 /tables.md                the tables as Markdown (for the figure
                                                           legends file)

values_v22.template_design reads these; nothing here re-solves a field.

  SPICEHEAD_MODEL=v22 python pipeline/template_design_collect.py
"""
import csv
import json
import os
import sys

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)
from spicehead import config                                         # noqa: E402
from spicehead.envelope_fast import envelope_max_direction_flat      # noqa: E402
import fig_ti_targets_mni as M                                       # noqa: E402
import ti_template_1010 as X                                         # noqa: E402

FIG = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "figures")
DIRS = {"s1": "TI_targets_MNI", "uni": "TI_targets_MNI_unilateral", "ratio": "TI_targets_MNI_current_ratio",
        "1010": "TI_targets_MNI_1010", "1010sub": "TI_targets_MNI_1010_subregions", "hdti": "TI_targets_MNI_hdti"}
OUT = os.path.join(config.RESULTS, "v22_template_design")
if config.MODEL not in ("v21", "v22"):
    # v23 (PI 28 Oct 2026: rerun section 5 under v23): the design scripts write next to fig_ti_targets_mni's own
    # folder under the model's results, ti_template_1010 tags its folders with the model; v22 paths unchanged
    FIG = os.path.dirname(M.OUT)
    DIRS = dict(DIRS, s1=os.path.basename(M.OUT), **{"1010": f"TI_targets_MNI_1010_{config.MODEL}",
                                                       "1010sub": f"TI_targets_MNI_1010_{config.MODEL}_subregions"})
    OUT = os.path.join(config.RESULTS, f"{config.MODEL}_template_design")
FIELD_PCT, CORE_FRAC = 99.9, 0.7
LEVELS = {"s1": "Two-pair 10-20, bilateral target (Figure S1)",
          "uni": "Two-pair 10-20, one-sided target",
          "ratio": "Two-pair 10-20, unequal channel currents (total 2 mA)",
          "1010": "Two-pair 10-10 (76 positions)",
          "1010_noinf": "Two-pair 10-10, no inferior row",
          "hdti": "Multi-electrode steering, 10-10 (up to 6 per channel)",
          "hdti_noinf": "Multi-electrode steering, no inferior row"}
TITLES = {"insula": "Insula (bilateral)", "hippocampus": "Hippocampus (bilateral)",
          "insula_L": "Left insula", "insula_R": "Right insula", "hippocampus_L": "Left hippocampus",
          "hippocampus_R": "Right hippocampus", "insula_ant_L": "Left anterior insula",
          "insula_post_L": "Left posterior insula"}
ORDER = ["insula", "insula_L", "insula_R", "insula_ant_L", "insula_post_L",
         "hippocampus", "hippocampus_L", "hippocampus_R"]
COLS = ["target", "title", "level", "level_name", "montage", "currents_ch1_mA", "currents_ch2_mA",
        "n_electrodes", "F", "T_V_per_m", "O_V_per_m", "ref_level", "ref_F", "gain_pct", "map_dir",
        "map_key", "top_V_per_m", "T_over_top", "target_cm3", "focus_cm3", "target_in_focus_pct",
        "focus_in_target_pct"]


def jl(d):
    return json.load(open(os.path.join(FIG, DIRS[d], "TI_targets_MNI152_summary.json"), encoding="utf-8"))


def canon(s):
    """Canonical two-pair montage label: pairs sorted internally and against each other."""
    pairs = [tuple(sorted(p.strip().split("+"))) for p in s.split("/")]
    return " / ".join("+".join(p) for p in sorted(pairs))


def focus(d, key, brain1, gm1):
    """Focus measures of a saved 1 mm map, as drawn in the 3-D figures."""
    p = os.path.join(FIG, DIRS[d])
    env = np.asanyarray(nib.load(os.path.join(p, f"TI_envelope_{key}_MNI152_1mm.nii.gz")).dataobj).astype(np.float32)
    tgt = np.asanyarray(nib.load(os.path.join(p, f"TI_target_{key}_MNI152_1mm.nii.gz")).dataobj) > 0
    top = float(np.percentile(env[brain1], FIELD_PCT))
    core = brain1 & (env >= CORE_FRAC * top)
    return dict(map_dir=DIRS[d], map_key=key, top_V_per_m=round(top, 4),
                T_over_top=round(float(np.median(env[tgt & gm1])) / top, 3),
                target_cm3=round(tgt.sum() / 1000, 1), focus_cm3=round(core.sum() / 1000, 1),
                target_in_focus_pct=round(100 * (core & tgt).sum() / tgt.sum(), 1),
                focus_in_target_pct=round(100 * (core & tgt).sum() / max(core.sum(), 1), 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    checks = []

    def check(name, ok, detail):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))
        print(("PASS " if ok else "FAIL ") + name + ": " + detail, flush=True)

    t1img = nib.load(os.path.join(M.M2M, "T1.nii.gz"))
    lab05 = nib.load(os.path.join(M.M2M, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    lab1 = np.rint(np.asanyarray(resample_from_to(lab05, t1img, order=0).dataobj)).astype(int)
    brain1, gm1 = np.isin(lab1, (M.WM, M.GM)), lab1 == M.GM

    s1, uni, ratio, g1010, gsub, hdti = (jl(k) for k in ("s1", "uni", "ratio", "1010", "1010sub", "hdti"))
    by = lambda s: {t["target"]: t for t in s["targets"]}
    S1, UNI, RATIO, G1010, HD = by(s1), by(uni), by(ratio), {**by(g1010), **by(gsub)}, by(hdti)
    rows = []

    def add(target, level, F, T=None, O=None, montage="", ref_level="", ref_F=None, foc=None, gain=None, **extra):
        """`gain` is the source script's gain (computed before rounding) where it exists."""
        r = dict(target=target, title=TITLES[target], level=level, level_name=LEVELS[level], montage=montage,
                 F=round(F, 3), T_V_per_m="" if T is None else round(T, 4), O_V_per_m="" if O is None else round(O, 4),
                 ref_level=ref_level, ref_F="" if ref_F is None else round(ref_F, 3),
                 gain_pct="" if ref_F is None else (round(gain, 1) if gain is not None else round(100 * (F / ref_F - 1), 1)))
        r.update({k: "" for k in COLS if k not in r})
        if foc:
            r.update(foc)
        r.update(extra)
        rows.append(r)

    # two-pair 10-20: bilateral (Figure S1) and one-sided
    for k in ("insula", "hippocampus"):
        t = S1[k]
        add(k, "s1", t["F"], t["T_V_per_m"], t["O_V_per_m"], t["montage"], foc=focus("s1", k, brain1, gm1))
    for k in ("insula_L", "insula_R", "hippocampus_L", "hippocampus_R"):
        t = UNI[k]
        add(k, "uni", t["F"], t["T_V_per_m"], t["O_V_per_m"], t["montage"], foc=focus("uni", k, brain1, gm1))
    for k in ("insula_ant_L", "insula_post_L"):                       # 10-20 baseline from the subregion run
        t = G1010[k]
        add(k, "uni", t["F_1020"], montage=t["montage_1020"])
    base = {r["target"]: r for r in rows}                             # equal-current 10-20 optimum per target

    # unequal channel currents (montage and split chosen jointly)
    for k, t in RATIO.items():
        add(k, "ratio", t["F"], t["T_V_per_m"], t["O_V_per_m"], t["montage"], "10-20 equal currents",
            t["F_equal"], foc=focus("ratio", k, brain1, gm1), gain=t["gain_joint_pct"])
        check(f"ratio baseline = 10-20 optimum ({k})", abs(t["F_equal"] - base[k]["F"]) < 5e-4 and
              canon(t["equal_montage"]) == canon(base[k]["montage"]),
              f"F_equal {t['F_equal']} vs {base[k]['F']}; {t['equal_montage']} vs {base[k]['montage']}")
        check(f"same montage, best split = 1:1 ({k})", t["best_split_for_equal_montage"] == 1.0 and t["gain_fixed_montage_pct"] == 0.0,
              f"best split {t['best_split_for_equal_montage']}, gain {t['gain_fixed_montage_pct']}%")
    z = np.load(os.path.join(FIG, DIRS["ratio"], "scores.npz"))
    with open(os.path.join(OUT, "current_ratio_sweep.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "ratio_I1_I2", "I1_mA", "I2_mA", "best_F"])
        for kk, k in enumerate(z["targets"]):
            for j, r in enumerate(z["ratios"]):
                w.writerow([str(k), round(float(r), 4), round(2 * r / (1 + r), 3), round(2 / (1 + r), 3),
                            round(float(z["F"][:, j, kk].max()), 3)])
            j1 = list(np.round(z["ratios"], 6)).index(1.0)
            check(f"ratio sweep at 1:1 reproduces the 10-20 optimum ({k})",
                  abs(float(z["F"][:, j1, kk].max()) - base[str(k)]["F"]) < 5e-4,
                  f"{float(z['F'][:, j1, kk].max()):.3f} vs {base[str(k)]['F']}")

    # two-pair 10-10
    for k, t in G1010.items():
        d = "1010sub" if k in ("insula_ant_L", "insula_post_L") else "1010"
        add(k, "1010", t["F"], t["T_V_per_m"], t["O_V_per_m"], t["montage"], "10-20", t["F_1020"],
            foc=focus(d, k, brain1, gm1), gain=t["gain_pct"], screen_rank=t["screen_rank"])
        add(k, "1010_noinf", t["F_no_inferior"], t["T_no_inferior"], None, t["montage_no_inferior"], "10-20", t["F_1020"],
            screen_rank=t["screen_rank_no_inferior"])
        check(f"10-20 optimum rescored by the 10-10 code ({k})", abs(t["F_1020"] - t["F_1020_rescored"]) < 5e-4,
              f"{t['F_1020']} vs {t['F_1020_rescored']}")
        check(f"10-10 start of the exact stage within the screened top ({k})",
              max(t["screen_rank"], t["screen_rank_no_inferior"]) <= g1010["top_k"],
              f"screen ranks {t['screen_rank']}, {t['screen_rank_no_inferior']} of {g1010['top_k']}")
        if k in base and "montage_1020" in t and k not in ("insula_ant_L", "insula_post_L"):
            check(f"10-10 run's 10-20 baseline = earlier 10-20 optimum ({k})",
                  abs(t["F_1020"] - base[k]["F"]) < 5e-4 and canon(t["montage_1020"]) == canon(base[k]["montage"]),
                  f"{t['montage_1020']} {t['F_1020']} vs {base[k]['montage']} {base[k]['F']}")
    r1010 = {(r["target"], r["level"]): r for r in rows}

    # multi-electrode steering: baseline equality and an independent recomputation
    X.TARGETS = {k: (X.SUBREGIONS.get(k) or X.TARGETS[k]) for k in ("insula_L", "insula_ant_L", "insula_post_L", "hippocampus_L")}
    S = X.setup()
    labs = S["labs"]
    idx = {l: i for i, l in enumerate(labs)}
    gm = S["gm"]
    B = X.load_basis(S, np.flatnonzero(gm.ravel()[S["brain"].ravel()]))

    def F_of(C, key):
        E = [np.einsum("e,ecn->cn", c, B) for c in C]
        env = envelope_max_direction_flat(*E[0], *E[1])
        return float(np.median(env[S["tmask"][key][gm]])) / float(np.percentile(env[S["omask"][key][gm]], 99))

    for rid, t in HD.items():
        k, lvl = rid.replace("_noinf", ""), ("hdti_noinf" if rid.endswith("_noinf") else "hdti")
        ref = r1010[(k, "1010_noinf" if lvl == "hdti_noinf" else "1010")]
        add(k, lvl, t["F"], t["T_V_per_m"], t["O_V_per_m"], t["montage"], "two-pair 10-10 " + ("(no inferior row)" if lvl.endswith("noinf") else ""),
            t["F_two_pair"], foc=focus("hdti", rid, brain1, gm1), gain=t["gain_pct"], currents_ch1_mA=t["currents_ch1_mA"],
            currents_ch2_mA=t["currents_ch2_mA"], n_electrodes="+".join(map(str, t["n_electrodes"])))
        check(f"steering baseline = two-pair 10-10 optimum ({rid})", abs(t["F_two_pair"] - ref["F"]) < 5e-4,
              f"{t['F_two_pair']} vs {ref['F']}")
        C = np.zeros((2, len(labs)), np.float64)
        for ch, key_c in ((0, "currents_ch1"), (1, "currents_ch2")):
            for l, c in t[key_c]:
                C[ch, idx[l]] = c
        ok_budget = all(abs(C[ch].sum()) < 1e-4 and abs(np.abs(C[ch]).sum() - 2.0) < 1e-3 for ch in (0, 1))
        check(f"steering current budget (sum 0, 1 mA per channel) ({rid})", ok_budget,
              f"sums {C[0].sum():.1e}, {C[1].sum():.1e}; |I| {np.abs(C[0]).sum():.4f}, {np.abs(C[1]).sum():.4f} mA")
        Fr = F_of(C, k)
        check(f"steering F recomputed from the cached basis ({rid})", abs(Fr - t["F"]) < 1e-3, f"{Fr:.4f} vs {t['F']}")
        check(f"steering GPU and CPU F agree ({rid})", abs(t["F_gpu"] - t["F"]) < 1e-3, f"{t['F_gpu']} vs {t['F']}")
        Cr = np.round(C, 2)                                           # a stimulator's 0.01 mA resolution
        for ch in (0, 1):
            act = Cr[ch] != 0
            Cr[ch, act] -= Cr[ch].sum() / act.sum()
            Cr[ch] *= 2.0 / np.abs(Cr[ch]).sum()
        rows[-1]["F_currents_rounded_0p01mA"] = round(F_of(Cr, k), 3)

    # subregion definitions
    with open(os.path.join(OUT, "subregions.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "title", "aal_parcels", "mni_y_rule", "gm_voxels_2mm", "gm_cm3_2mm", "gm_cm3_1mm"])
        for k in ("insula_L", "insula_ant_L", "insula_post_L"):
            t = X.TARGETS[k]
            rule = "all" if "ymin" not in t and "ymax" not in t else (f"y >= {t['ymin']:+.0f} mm" if "ymin" in t else f"y <= {t['ymax']:+.0f} mm")
            n2 = int(S["tmask"][k].sum())
            tg1 = X.region(np.rint(np.asanyarray(resample_from_to(nib.load(M.AAL), t1img, order=0).dataobj)).astype(int),
                           {r["name"]: r["value"] for r in json.load(open(M.AAL_REGIONS))["regions"]}, t, t1img.affine) & gm1
            w.writerow([k, TITLES[k], "+".join(t["parcels"]), rule, n2, round(n2 * M.DX ** 3 / 1000, 1), round(tg1.sum() / 1000, 1)])

    rows.sort(key=lambda r: (ORDER.index(r["target"]), list(LEVELS).index(r["level"])))
    cols = COLS + ["F_currents_rounded_0p01mA", "screen_rank"]
    with open(os.path.join(OUT, "design_levels.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    json.dump(dict(checks=checks, n_checks=len(checks), n_failed=sum(not c["passed"] for c in checks)),
              open(os.path.join(OUT, "checks.json"), "w", encoding="utf-8"), indent=1)
    json.dump(dict(field_scale_percentile=FIELD_PCT, focus_fraction=CORE_FRAC,
                   s1={k: v for k, v in s1.items() if k != "targets"},
                   ratio={k: v for k, v in ratio.items() if k != "targets"},
                   cap_1010={k: v for k, v in g1010.items() if k != "targets"},
                   hdti={k: v for k, v in hdti.items() if k != "targets"},
                   inferior_row=sorted(X.INFERIOR), total_current_mA=2.0),
              open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=1)

    # Markdown tables for the figure-legend file
    md = ["# Template montage-design exploration: tables (generated)\n",
          f"Generated by `pipeline/template_design_collect.py`; the numbers are those of the Results record.\n",
          "| Target | Design level | Montage | F | T (V/m) | Gain vs reference | Target in focus | Focus (cm³) |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        g = f"{r['gain_pct']:+.1f}%" if r["gain_pct"] != "" else ""
        md.append(f"| {r['title']} | {r['level_name']} | {r['montage']} | {r['F']:.3f} | {r['T_V_per_m'] if r['T_V_per_m'] != '' else ''} | {g} | "
                  f"{str(r['target_in_focus_pct']) + '%' if r['target_in_focus_pct'] != '' else ''} | {r['focus_cm3']} |")
    md += ["", "Steered montages (signed currents in mA):", "", "| Target | Variant | Channel 1 | Channel 2 |", "|---|---|---|---|"]
    for r in rows:
        if r["level"].startswith("hdti"):
            md.append(f"| {r['title']} | {r['level_name']} | {r['currents_ch1_mA']} | {r['currents_ch2_mA']} |")
    md += ["", f"Consistency checks: {len(checks) - sum(not c['passed'] for c in checks)} of {len(checks)} passed."]
    open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print(f"\n{len(rows)} rows; checks {len(checks) - sum(not c['passed'] for c in checks)}/{len(checks)} passed; wrote {OUT}")


if __name__ == "__main__":
    main()
