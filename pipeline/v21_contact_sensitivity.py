# -*- coding: utf-8 -*-
"""Electrode contact support, volumetric versus outer-skin surface, compared against FEM.

Every published SPICEhead analysis injects each electrode's current uniformly into ALL conducting
nodes within 12 mm of the snapped scalp point (nodes_in_sphere). That support is volumetric: it
spans the scalp thickness and can include skull or muscle nodes. The FEM reference instead
models a 24 mm, 2 mm-thick pad on the skin. This script measures what the choice changes. For
the eight FEM heads and the four FEM montages it solves both supports at one resolution and
compares each with the same SimNIBS envelope, using the unadjusted statistics of the
multi-montage comparison (v21_agreement_raw.stats) and the same T, O and F definitions.

Published outputs are untouched: results go to results/v21_contact_sensitivity/. The volumetric
rows must reproduce results/v21_multimontage/agreement_by_montage.csv at the same resolution,
and --collect reports that check before anything else.

A second factor tests the tissue-interface rule. The published model averages the two voxel
conductivities on an edge arithmetically; --rule harmonic rebuilds the network with harmonic
averaging, which the layered-slab benchmark favours, and solves both contact supports again.
Arithmetic results are written as <subject>__dx<dx>.json, harmonic ones with a _harmonic suffix.

  python pipeline/v21_contact_sensitivity.py --score [dx] [--rule harmonic] [subject ...]
  python pipeline/v21_contact_sensitivity.py --collect [dx]
"""
import csv
import json
import os
import statistics as st
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)

from spicehead import mri
from spicehead.conductivity import CHARM_LABELS
from spicehead.electrodes import (Electrode, nodes_in_sphere, scalp_surface_nodes_in_sphere,
                                  pair_injection)
from spicehead.field import efield_components
from montage_search import (snap, roi_mask, ELEC_RADIUS_MM, ROI_RADIUS_MM, EXCLUDE_MM,
                            SCALP, SCORE, ROI_ORDER)
from v21_head_convergence import make_solver
from v21_simnibs_validation import subjects, CURRENT
from v21_simnibs_compare import load_simnibs_E, envelope, resample_to
from v21_multimontage_fem import MONTAGES, fem_dir
from v21_agreement_raw import stats

from spicehead import config                                         # noqa: E402

OUTC = os.path.join(ROOT, "results", "v21_contact_sensitivity")
PUBLISHED = os.path.join(ROOT, "results", "v21_multimontage", "agreement_by_montage.csv")
V23 = config.NETWORK == "homogenized"
if V23:
    # v23 (PI 28 Oct 2026: contact sensitivity ported and rerun under v23): homogenized network, so no interface
    # rule to vary; results and the reproduction check in the model's own results folder
    OUTC = os.path.join(config.RESULTS, "v21_contact_sensitivity")
    PUBLISHED = os.path.join(config.RESULTS, "v21_multimontage", "agreement_by_montage.csv")
MODES = ("volume", "surface")


def support_nodes(mode, model, labels, center, dx):
    r = ELEC_RADIUS_MM / dx
    if mode == "volume":
        return nodes_in_sphere(model, center, r)
    return scalp_surface_nodes_in_sphere(model, labels, center, r)


def score_subject(cid, m2m, site, dx, rule="arithmetic"):
    import nibabel as nib
    t0 = time.time()
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    if V23:
        model, saff = mri.head_model(seg, dx)
        labels = model.labels
    else:
        labels, saff, zooms = mri.load_label_volume(seg, dx_mm=dx)
        model = mri.model_from_labels(labels, zooms, dx_mm=dx, conductance_mean=rule)
    ii, jj, kk = model.grid_coords
    scalp = np.argwhere(labels == SCALP)
    ejs = {f: json.load(open(os.path.join(m2m, f))) for f in {v[1] for v in MONTAGES.values()}}
    centers = {}
    for tag, (pairs, ejfile) in MONTAGES.items():
        for pair in pairs:
            for lab in pair:
                if lab not in centers:
                    centers[lab] = tuple(int(x) for x in
                                         snap(scalp, mri.world_to_voxel(saff, ejs[ejfile][lab])))
    support, composition = {}, {}
    for mode in MODES:
        support[mode], composition[mode] = {}, {}
        for lab, c in centers.items():
            idx = support_nodes(mode, model, labels, c, dx)
            support[mode][lab] = idx
            lv = labels[ii[idx], jj[idx], kk[idx]]
            composition[mode][lab] = {CHARM_LABELS[int(t)]: int(n)
                                      for t, n in zip(*np.unique(lv, return_counts=True))}

    solver, ncomp = make_solver(model.G)
    print(f"  {cid}: {model.n_nodes} nodes, {len(centers)} electrodes, set up in "
          f"{time.time()-t0:.0f}s", flush=True)
    env = {mode: {} for mode in MODES}
    for mode in MODES:
        for tag, (pairs, _) in MONTAGES.items():
            F = []
            for a, b in pairs:
                V = solver(pair_injection(model, Electrode(support[mode][a], a),
                                          Electrode(support[mode][b], b), CURRENT))
                F.append(tuple(np.nan_to_num(f).astype(np.float32)
                               for f in efield_components(model, V)))
            env[mode][tag] = np.nan_to_num(envelope(F[0], F[1])).astype(np.float32)
        print(f"    {mode}: 4 montages solved ({time.time()-t0:.0f}s)", flush=True)

    per_tag = {}
    for tag in MONTAGES:
        try:
            E1, aff, sh = load_simnibs_E(fem_dir(cid, tag, 1))
            E2, _, _ = load_simnibs_E(fem_dir(cid, tag, 2))
        except Exception:
            continue
        per_tag[tag] = (np.nan_to_num(envelope(E1, E2)), aff, sh)
    if not per_tag:
        raise RuntimeError("no FEM output for this subject")
    aff, sh = per_tag[next(iter(per_tag))][1], per_tag[next(iter(per_tag))][2]
    li = nib.load(seg)
    gm = resample_to(np.isin(np.asanyarray(li.dataobj), SCORE).astype(np.float32),
                     li.affine, aff, sh) > 0.5
    rois = json.load(open(os.path.join(m2m, "roi_coords.json")))
    gidx = np.argwhere(gm)
    gw = (aff[:3, :3] @ gidx.T).T + aff[:3, 3]
    rmask = {r: roi_mask(sh, gidx, gw, rois[r], ROI_RADIUS_MM) for r in ROI_ORDER if r in rois}
    omask = {r: gm & ~roi_mask(sh, gidx, gw, rois[r], EXCLUDE_MM) for r in rmask}

    def tof(e, r):
        T = float(np.median(e[rmask[r]]))
        O = float(np.percentile(e[omask[r]], 99))
        return T, O, (T / O if O > 0 else float("nan"))

    rows = []
    for mode in MODES:
        for tag, (fem, _, _) in per_tag.items():
            spg = resample_to(env[mode][tag], saff, aff, sh)
            x, y = spg[gm].astype(float), fem[gm].astype(float)
            k = np.isfinite(x) & np.isfinite(y) & (y > 0)
            s = stats(x[k], y[k])
            roi = {}
            for r in rmask:
                Ts, Os, Fs = tof(spg, r)
                Tf, Of, Ff = tof(fem, r)
                roi[r] = dict(T_sp=Ts, O_sp=Os, F_sp=Fs, T_fem=Tf, O_fem=Of, F_fem=Ff)
            rows.append(dict(mode=mode, montage=tag, n_vox=int(k.sum()), roi=roi, **s))
    return dict(subject=cid, site=site, dx=dx, rule=rule, n_nodes=int(model.n_nodes),
                composition=composition, rows=rows, seconds=round(time.time() - t0, 1))


def stage_score(dx, only=None, rule="arithmetic"):
    pdir = os.path.join(OUTC, "per_subject")
    os.makedirs(pdir, exist_ok=True)
    subs = subjects(8)
    if only:
        subs = [s for s in subs if s[0] in only]
    for i, (cid, m2m, age, site) in enumerate(subs, 1):
        tag = "" if rule == "arithmetic" else f"_{rule}"
        p = os.path.join(pdir, f"{cid}__dx{dx:g}{tag}.json")
        if os.path.exists(p):
            print(f"[{i}/{len(subs)}] {cid} (cached)", flush=True)
            continue
        try:
            res = score_subject(cid, m2m, site, dx, rule)
        except Exception as ex:
            print(f"[{i}/{len(subs)}] {cid} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        with open(p + ".part", "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        os.replace(p + ".part", p)
        print(f"[{i}/{len(subs)}] {cid} OK ({res['seconds']/60:.1f} min)", flush=True)


def stage_collect(dx):
    """Tabulate every (interface rule, contact support) configuration against FEM.

    The published model is rule = arithmetic, support = volume. Each other configuration is
    reported on its own and as a paired change from the published one in the same subject.
    """
    pdir = os.path.join(OUTC, "per_subject")
    res = []
    for f in sorted(os.listdir(pdir)):
        if f.startswith(".") or not f.endswith(".json") or f"__dx{dx:g}" not in f:
            continue
        r = json.load(open(os.path.join(pdir, f), encoding="utf-8"))
        r.setdefault("rule", "arithmetic")
        res.append(r)
    expected = [s[0] for s in subjects(8)]
    rules = sorted({r["rule"] for r in res}, key=lambda x: x != "arithmetic")
    for rule in rules:
        got = {r["subject"] for r in res if r["rule"] == rule}
        miss = [c for c in expected if c not in got]
        print(f"rule {rule}: {len(got)}/{len(expected)} subjects" +
              (f"; MISSING {miss}" if miss else ""))

    # 1) the published configuration must reproduce the published multi-montage table
    pub = {}
    for r in csv.DictReader(open(PUBLISHED, encoding="utf-8")):
        if float(r["dx_mm"]) == dx:
            pub[(r["subject"], r["montage"])] = r
    dev = []
    for s in res:
        if s["rule"] != "arithmetic":
            continue
        for row in s["rows"]:
            p = pub.get((s["subject"], row["montage"]))
            # the published configuration's contact support: volumetric for v21, skin-surface since v22 (fixed 30 Oct
            # 2026; the v23 surface rows reproduce the v23 multi-montage table to 3e-10, the volume rows differ by 6 %)
            if row["mode"] == ("volume" if config.CONTACT == "volume" else "surface") and p:
                dev.append(max(abs(row[k] - float(p[k])) / max(abs(float(p[k])), 1e-12)
                               for k in ("slope", "r", "ccc", "mean_spicehead")))
    if dev:
        print(f"published configuration vs agreement_by_montage.csv: {len(dev)} rows, "
              f"max relative deviation {max(dev):.1e}")

    # 2) per-row tables
    os.makedirs(OUTC, exist_ok=True)
    agr, roi_rows = [], []
    for s in res:
        for row in s["rows"]:
            agr.append(dict(subject=s["subject"], site=s["site"], rule=s["rule"],
                            support=row["mode"], montage=row["montage"], dx_mm=dx,
                            n_vox=row["n_vox"], mean_spicehead=row["mean_spicehead"],
                            mean_fem=row["mean_fem"],
                            rel_bias_pct=100 * (row["mean_spicehead"] / row["mean_fem"] - 1),
                            raw_bias=row["raw_bias"], loa_lo=row["loa_lo"], loa_hi=row["loa_hi"],
                            slope=row["slope"], intercept=row["intercept"], r=row["r"],
                            ccc=row["ccc"]))
            for r, v in row["roi"].items():
                roi_rows.append(dict(subject=s["subject"], rule=s["rule"], support=row["mode"],
                                     montage=row["montage"], roi=r, dx_mm=dx, **v,
                                     T_relerr_pct=100 * (v["T_sp"] / v["T_fem"] - 1)
                                     if v["T_fem"] > 0 else float("nan")))
    for name, rows in (("contact_agreement.csv", agr), ("contact_roi.csv", roi_rows)):
        with open(os.path.join(OUTC, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    configs = [(ru, mo) for ru in rules for mo in MODES
               if any(a["rule"] == ru and a["support"] == mo for a in agr)]

    # 3) agreement by montage and configuration, and paired change from the published one
    print(f"\nagreement with FEM at {dx:g} mm (mean over subjects):")
    print(f"{'montage':<9}{'rule':<11}{'support':<9}{'n':>3}{'r':>7}{'slope':>8}{'CCC':>7}"
          f"{'rel bias %':>11}{'thal ROI %':>11}")
    summary, paired = [], []
    for tag in MONTAGES:
        for ru, mo in configs:
            sub = [a for a in agr if a["montage"] == tag and a["rule"] == ru and a["support"] == mo]
            if not sub:
                continue
            m = {k: st.mean(a[k] for a in sub) for k in ("r", "slope", "ccc", "rel_bias_pct",
                                                         "raw_bias")}
            th = [x["T_relerr_pct"] for x in roi_rows if x["montage"] == tag and x["rule"] == ru
                  and x["support"] == mo and x["roi"] == "thalamus"]
            m["thalamus_T_relerr_pct"] = st.mean(th) if th else float("nan")
            summary.append(dict(montage=tag, rule=ru, support=mo, n=len(sub), **m))
            print(f"{tag:<9}{ru:<11}{mo:<9}{len(sub):>3}{m['r']:>7.3f}{m['slope']:>8.3f}"
                  f"{m['ccc']:>7.3f}{m['rel_bias_pct']:>+11.1f}{m['thalamus_T_relerr_pct']:>+11.1f}")
    print("\npaired change from the published configuration (arithmetic, volume), "
          "mean [min, max] over subjects:")
    base = {(a["subject"], a["montage"]): a for a in agr
            if a["rule"] == "arithmetic" and a["support"] == "volume"}
    for ru, mo in configs:
        if (ru, mo) == ("arithmetic", "volume"):
            continue
        for tag in MONTAGES:
            for k in ("rel_bias_pct", "r", "slope", "ccc"):
                d = [a[k] - base[(a["subject"], tag)][k] for a in agr
                     if a["montage"] == tag and a["rule"] == ru and a["support"] == mo
                     and (a["subject"], tag) in base]
                if d:
                    paired.append(dict(rule=ru, support=mo, montage=tag, metric=k, n=len(d),
                                       mean=st.mean(d), min=min(d), max=max(d)))
                    print(f"  {ru:<11}{mo:<9}{tag:<9}{k:<13}{st.mean(d):+9.3f}  "
                          f"[{min(d):+.3f}, {max(d):+.3f}]")

    # 4) decision level: best of the four montages per subject and target, per configuration
    print("\nbest-of-four montage agreement with FEM (subject x target pairs):")
    decisions = []
    for ru, mo in configs:
        for q, label in (("T", "target intensity"), ("F", "focality")):
            agree = total = 0
            for s in sorted({r["subject"] for r in roi_rows}):
                for r in ROI_ORDER:
                    v = [x for x in roi_rows if x["subject"] == s and x["roi"] == r
                         and x["rule"] == ru and x["support"] == mo]
                    if len(v) < 2:
                        continue
                    bs = max(v, key=lambda x: x[f"{q}_sp"])["montage"]
                    bf = max(v, key=lambda x: x[f"{q}_fem"])["montage"]
                    agree += bs == bf
                    total += 1
            decisions.append(dict(rule=ru, support=mo, quantity=label, agree=agree, total=total))
            print(f"  {ru:<11}{mo:<9}{label:<17} {agree}/{total}")

    # 5) what each support actually contains (independent of the rule)
    print("\ncontact support composition (arithmetic runs, all electrodes, all subjects):")
    comp_rows = []
    for s in res:
        if s["rule"] != "arithmetic":
            continue
        for mode in MODES:
            for lab, c in s["composition"][mode].items():
                n = sum(c.values())
                comp_rows.append(dict(subject=s["subject"], support=mode, electrode=lab,
                                      n_nodes=n, non_scalp_pct=100 * (1 - c.get("scalp", 0) / n),
                                      **{f"n_{t}": c.get(t, 0) for t in
                                         ("scalp", "compact_bone", "spongy_bone", "muscle",
                                          "csf", "gray_matter", "blood", "eyeballs")}))
    if comp_rows:
        with open(os.path.join(OUTC, "contact_support.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comp_rows[0]))
            w.writeheader()
            w.writerows(comp_rows)
        for mode in MODES:
            sub = [c for c in comp_rows if c["support"] == mode]
            ns = [c["non_scalp_pct"] for c in sub]
            worst = max(sub, key=lambda c: c["non_scalp_pct"])
            print(f"  {mode:<8} nodes/electrode median {st.median(c['n_nodes'] for c in sub):.0f}; "
                  f"non-scalp share median {st.median(ns):.2f}%, max {max(ns):.2f}% "
                  f"({worst['subject']} {worst['electrode']})")
    with open(os.path.join(OUTC, "contact_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(dx=dx, rules=rules,
                       published_reproduction_max_rel_dev=max(dev) if dev else None,
                       agreement=summary, paired_change=paired, decisions=decisions), fh,
                  indent=1)
    print(f"\nwrote {OUTC}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("--score", "--collect"):
        raise SystemExit(__doc__)
    args = sys.argv[2:]
    rule = "arithmetic"
    if "--rule" in args:
        i = args.index("--rule")
        rule = args[i + 1]
        args = args[:i] + args[i + 2:]
    dx = float(args[0]) if args else 2.0
    if sys.argv[1] == "--score":
        stage_score(dx, set(args[1:]) or None, rule)
    else:
        stage_collect(dx)
