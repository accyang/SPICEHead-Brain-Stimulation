# -*- coding: utf-8 -*-
"""Direct four-pad FEM check of the montages chosen in the full-space comparison.

The FEM leadfield of v22_fem_fullsearch.py meshes all 19 pads in one geometry, which makes
superposition exact but leaves 15 unused conductive pads on the scalp for every montage. Those
pads shunt current: against the two-pad direct runs, the leadfield's grey-matter field is weaker
by a nearly uniform 2-13% depending on the pair (3 October 2026). A real montage carries only its
four pads, as does the SPICEhead model. This script therefore simulates, with SimNIBS and only the
four pads of the montage meshed, the montages that decide the comparison: for every head and
target, the optimum of the FEM leadfield, of SPICEhead 1 mm and of SPICEhead 2 mm. Each montage
needs two runs (one per channel, the other channel's pads present at zero current), scored with
the same masks, grid and envelope as the full search.

Stages (resumable; --shard k/n; --heads A,B):
  --sim     SimNIBS interpreter: two runs per montage, volumes kept off Dropbox, meshes deleted
  --score   project interpreter: F, T, O of every simulated montage; writes fem4pad_scores.csv
            and fem4pad_summary.json beside the full-search results

  "<SimNIBS>\\simnibs_env\\python.exe" pipeline/v22_fem_4pad_check.py --sim [--shard k/n]
  python pipeline/v22_fem_4pad_check.py --score
"""
import csv
import json
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)
from v22_fem_fullsearch import (heads, select, CACHE, CURRENT, ELEC_RADIUS_MM, ROI_ORDER,  # noqa: E402
                                out_dir, FEM_BASIS)

SIMS = os.environ.get("SPICEHEAD_FEM4PAD_ROOT", os.path.join(os.path.dirname(FEM_BASIS), "fem_4pad"))
MODELS = ("fem", "sp1", "sp2")


def parse(label):
    """'A+B / C+D' -> (('A','B'), ('C','D'))."""
    return tuple(tuple(p.split("+")) for p in label.split(" / "))


def montages_to_check(cid):
    """{label: [(model, roi), ...]} for the optima of the three models in this head."""
    od = out_dir()
    want = {}
    for m in MODELS:
        rois = json.load(open(os.path.join(od, f"{cid}__{m}.json"), encoding="utf-8"))["rois"]
        for r in ROI_ORDER:
            best = max(rois[r], key=lambda k: rois[r][k][0])
            want.setdefault(best, []).append((m, r))
    return want


def run_dir(cid, label, ch):
    safe = label.replace(" / ", "__").replace("+", "-")
    return os.path.join(SIMS, cid, safe, f"ch{ch}")


def stage_sim(args):
    from simnibs import sim_struct, run_simnibs
    subs = select(args, heads())
    for hi, (cid, m2m) in enumerate(subs, 1):
        ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
        want = montages_to_check(cid)
        print(f"[{hi}/{len(subs)}] {cid}: {len(want)} distinct montages", flush=True)
        for mi, label in enumerate(sorted(want), 1):
            (a1, b1), (a2, b2) = parse(label)
            pads = (a1, b1, a2, b2)
            for ch, (a, b) in enumerate(((a1, b1), (a2, b2)), 1):
                d = run_dir(cid, label, ch)
                vol = os.path.join(d, "subject_volumes")
                if os.path.isdir(vol) and any(f.endswith("_E.nii.gz") for f in os.listdir(vol)):
                    continue
                os.makedirs(d, exist_ok=True)
                S = sim_struct.SESSION()
                S.subpath, S.pathfem = m2m, d
                S.fields, S.map_to_vol, S.open_in_gmsh = "E", True, False
                t = S.add_tdcslist()
                cur = []
                for i, lab in enumerate(pads):
                    e = t.add_electrode()
                    e.channelnr = i + 1
                    e.centre = ej[lab]
                    e.shape = "ellipse"
                    e.dimensions = [2 * ELEC_RADIUS_MM, 2 * ELEC_RADIUS_MM]
                    e.thickness = [2.0]
                    cur.append(CURRENT if lab == a else (-CURRENT if lab == b else 0.0))
                t.currents = cur
                t0 = time.time()
                try:
                    run_simnibs(S)
                except Exception as ex:
                    print(f"    {label} ch{ch} FAILED {type(ex).__name__}: {ex}", flush=True)
                    continue
                for f in os.listdir(d):
                    if f.endswith((".msh", ".geo")):
                        try:
                            os.remove(os.path.join(d, f))
                        except OSError:
                            pass
                print(f"    [{mi}/{len(want)}] {label} ch{ch} OK ({(time.time()-t0)/60:.1f} min)", flush=True)


def stage_score(args):
    from spicehead.envelope_fast import envelope_max_direction_flat
    from v21_simnibs_compare import load_simnibs_E
    od = out_dir()
    rows = []
    for cid, m2m in heads():
        geo = np.load(os.path.join(CACHE, f"{cid}__grid.npz"), allow_pickle=True)
        gi = geo["gm_idx"].astype(int)
        roi_sel, out_sel = geo["roi_sel"].item(), geo["out_sel"].item()
        lf = {m: json.load(open(os.path.join(od, f"{cid}__{m}.json"), encoding="utf-8"))["rois"]
              for m in MODELS}
        want = montages_to_check(cid)
        for label, chosen_by in sorted(want.items()):
            try:
                E = [load_simnibs_E(run_dir(cid, label, ch))[0] for ch in (1, 2)]
            except Exception as ex:
                print(f"  {cid} {label}: not simulated ({type(ex).__name__})", flush=True)
                continue
            A = [c[gi[:, 0], gi[:, 1], gi[:, 2]] for c in E[0]]
            B = [c[gi[:, 0], gi[:, 1], gi[:, 2]] for c in E[1]]
            env = envelope_max_direction_flat(*A, *B)
            for r in ROI_ORDER:
                T = float(np.median(env[roi_sel[r]]))
                O = float(np.percentile(env[out_sel[r]], 99))
                rec = dict(subject=cid, roi=r, montage=label, F_4pad=T / O, T_4pad=T, O_4pad=O,
                           chosen_by=";".join(m for m, rr in chosen_by if rr == r))
                for m in MODELS:
                    v = lf[m][r].get(label)
                    rec[f"F_{m}"] = v[0] if v else float("nan")
                rows.append(rec)
        print(f"  {cid}: {len(want)} montages scored", flush=True)
    if not rows:
        raise SystemExit("nothing simulated yet")
    with open(os.path.join(od, "fem4pad_scores.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)


def summarize(rows):
    """Retention of each model's optimum among the selected candidates under four-pad FEM, and
    the leadfield-to-four-pad F ratio for explicitly labelled populations (review R4-B, R4-C).

    The denominator of "retention" is the best four-pad F among the montages that the three
    models selected for that head and target (up to three distinct candidates), NOT a four-pad
    optimum over all 11,628 montages, which this check does not evaluate."""
    od = out_dir()
    summ, per = {}, []
    for cid, _ in heads():
        for r in ROI_ORDER:
            sub = [x for x in rows if x["subject"] == cid and x["roi"] == r and x["chosen_by"]]
            if not sub:
                continue
            best = max(x["F_4pad"] for x in sub)
            rec = dict(subject=cid, roi=r, n_candidates=len(sub))
            for m in MODELS:
                pick = [x for x in sub if m in x["chosen_by"].split(";")]
                if pick:
                    rec[f"keep_{m}"] = 100 * pick[0]["F_4pad"] / best
                    rec[f"Fratio_lf_over_4pad_{m}"] = pick[0]["F_fem"] / pick[0]["F_4pad"]
            per.append(rec)
    for m in MODELS:
        k = [x[f"keep_{m}"] for x in per if f"keep_{m}" in x]
        if k:
            summ[f"keep_{m}"] = dict(n=len(k), median=float(np.median(k)), min=float(min(k)),
                                     n_best=int(sum(abs(v - 100) < 1e-9 for v in k)))
    summ["n_candidates"] = dict(total=int(sum(x["n_candidates"] for x in per)),
                                median=float(np.median([x["n_candidates"] for x in per])))

    def rng(v):
        return dict(n=len(v), median=float(np.median(v)), min=float(min(v)), max=float(max(v)))
    # leadfield F / four-pad F of the same montage and target; three populations kept apart
    fr = [x["Fratio_lf_over_4pad_fem"] for x in per if "Fratio_lf_over_4pad_fem" in x]
    if fr:
        summ["F_leadfield_over_4pad"] = rng(fr)                       # FEM-leadfield-selected winners
    sel = [x["F_fem"] / x["F_4pad"] for x in rows if x["chosen_by"]]
    summ["F_leadfield_over_4pad_selected"] = rng(sel)                 # each selected candidate once
    allr = [x["F_fem"] / x["F_4pad"] for x in rows]
    summ["F_leadfield_over_4pad_all_rows"] = rng(allr)                # incl. targets not chosen for
    flips = n_pairs = 0
    for cid, _ in heads():
        for r in ROI_ORDER:
            sub = [x for x in rows if x["subject"] == cid and x["roi"] == r]
            f = [x for x in sub if "fem" in x["chosen_by"].split(";")]
            s_ = [x for x in sub if "sp1" in x["chosen_by"].split(";")]
            if f and s_ and f[0]["montage"] != s_[0]["montage"]:
                n_pairs += 1
                flips += s_[0]["F_4pad"] > f[0]["F_4pad"]
    summ["fem_vs_sp1_disagreements"] = n_pairs
    summ["sp1_choice_better_under_4pad"] = flips
    json.dump(summ, open(os.path.join(od, "fem4pad_summary.json"), "w"), indent=1)
    with open(os.path.join(od, "fem4pad_per_target.csv"), "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for x in per for k in x}, key=lambda k: (k not in ("subject", "roi"), k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(per)
    print(json.dumps(summ, indent=1))
    return summ, per


def read_scores():
    od = out_dir()
    rows = list(csv.DictReader(open(os.path.join(od, "fem4pad_scores.csv"), encoding="utf-8")))
    for x in rows:
        for k in list(x):
            if k.startswith(("F_", "T_", "O_")):
                x[k] = float(x[k]) if x[k] not in ("", "nan") else float("nan")
    return rows


def stage_qc(args):
    """SimNIBS electrode-current diagnostics of the four-pad runs (review R4-A).

    SimNIBS solves each four-pad run as three internal electrode pairs against the first pad
    (logged as "Simulating electrode pair 2101 - 210k"), each followed by its flux-calibration
    diagnostic, and then weights the pairs by the requested currents: the weight of pair
    (2101, 210k) is the current on pad k. A warning on a zero-weight pair does not enter the
    delivered field. The diagnostic compares two estimates of the electrode current; it is not an
    error of the brain field. Writes fem4pad_qc.json with the post hoc sensitivity on the
    head-target comparisons whose selected candidates carry no warning at all (conservative)."""
    import re
    import ast
    od = out_dir()
    pat_pair = re.compile(r"Simulating electrode pair (\d+) - (\d+)")
    pat_val = re.compile(r"(?:Estimated error value: |Estimated current calibration error: )([\d.]+)%")
    logs, flagged_geom, active_geom = [], set(), set()
    for cid, _ in heads():
        for label in sorted(montages_to_check(cid)):
            for ch in (1, 2):
                d = run_dir(cid, label, ch)
                lg = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".log")] \
                    if os.path.isdir(d) else []
                if not lg:
                    logs.append(dict(subject=cid, montage=label, channel=ch, missing=True))
                    continue
                txt = open(lg[-1], encoding="utf-8", errors="replace").read()
                cur = ast.literal_eval(re.search(r"Currents \(A\): (\[[^\n]+\])", txt).group(1))
                ev, pair = [], None
                for line in txt.splitlines():
                    m = pat_pair.search(line)
                    if m:
                        pair = (int(m.group(1)), int(m.group(2)))
                        continue
                    m = pat_val.search(line)
                    if m and pair:
                        coef = cur[pair[1] - pair[0]]
                        ev.append(dict(pair=pair, pct=float(m.group(1)), warn="WARNING" in line,
                                       active=coef != 0))
                rec = dict(subject=cid, montage=label, channel=ch, log=lg[-1], n_events=len(ev),
                           n_warn=sum(e["warn"] for e in ev),
                           n_warn_active=sum(e["warn"] and e["active"] for e in ev),
                           max_pct=max((e["pct"] for e in ev), default=float("nan")),
                           completed=os.path.isdir(os.path.join(d, "subject_volumes")))
                logs.append(rec)
                if rec["n_warn"]:
                    flagged_geom.add((cid, label))
                if rec["n_warn_active"]:
                    active_geom.add((cid, label))
    ok = [x for x in logs if not x.get("missing")]
    qc = dict(n_logs=len(ok), n_missing=len(logs) - len(ok),
              n_completed=sum(x["completed"] for x in ok),
              n_events=sum(x["n_events"] for x in ok),
              n_warn_messages=sum(x["n_warn"] for x in ok),
              n_warn_active_messages=sum(x["n_warn_active"] for x in ok),
              n_logs_any_warn=sum(x["n_warn"] > 0 for x in ok),
              n_logs_active_warn=sum(x["n_warn_active"] > 0 for x in ok),
              n_geometries=len({(x["subject"], x["montage"]) for x in ok}),
              n_geometries_flagged=len(flagged_geom), n_geometries_active_flagged=len(active_geom),
              max_pct=max(x["max_pct"] for x in ok))
    rows = read_scores()
    comps, clean, flagged_comps = 0, [], []
    for cid, _ in heads():
        for r in ROI_ORDER:
            sel = [x for x in rows if x["subject"] == cid and x["roi"] == r and x["chosen_by"]]
            if not sel:
                continue
            comps += 1
            if any((cid, x["montage"]) in flagged_geom for x in sel):
                flagged_comps.append(f"{cid}:{r}")
            else:
                clean.append(sel)
    qc.update(n_comparisons=comps, n_comparisons_flagged=len(flagged_comps),
              n_comparisons_clean=len(clean), comparisons_flagged=flagged_comps)
    sens = {}
    for m in MODELS:
        k = []
        for sel in clean:
            best = max(x["F_4pad"] for x in sel)
            pick = [x for x in sel if m in x["chosen_by"].split(";")]
            if pick:
                k.append(100 * pick[0]["F_4pad"] / best)
        if k:
            sens[f"keep_{m}"] = dict(n=len(k), median=float(np.median(k)), min=float(min(k)))
    qc["sensitivity_clean_comparisons"] = sens
    json.dump(dict(qc, logs=logs), open(os.path.join(od, "fem4pad_qc.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in qc.items() if k != "comparisons_flagged"}, indent=1))


def stage_pads(args):
    """Passive-pad effect per head and pair: the leadfield (19 pads) against the two-pad direct
    runs of the four-montage comparison, on the grey-matter voxels. Writes fem_passive_pads.csv."""
    from v21_simnibs_compare import load_simnibs_E
    from v22_fem_fullsearch import DIRECT, FEM_REF, REF
    od = out_dir()
    rows = []
    for cid, m2m in heads():
        z = np.load(os.path.join(CACHE, f"{cid}__fem_basis.npz"), allow_pickle=True)
        basis, labs = z["basis"].astype(np.float64), list(z["labs"])
        gi = np.load(os.path.join(CACHE, f"{cid}__grid.npz"), allow_pickle=True)["gm_idx"].astype(int)
        idx = {lab: i for i, lab in enumerate(labs)}
        for tag, (p1, p2, dirs) in DIRECT.items():
            for (a, b), dd in zip((p1, p2), dirs):
                (Ex, Ey, Ez), _, _ = load_simnibs_E(os.path.join(FEM_REF, cid, dd))
                D = np.stack([q[gi[:, 0], gi[:, 1], gi[:, 2]] for q in (Ex, Ey, Ez)]).astype(np.float64)
                R = (basis[idx[a]] if a != REF else 0.0) - (basis[idx[b]] if b != REF else 0.0)
                if np.abs(R + D).max() < np.abs(R - D).max():
                    D = -D
                mag = np.linalg.norm(D, axis=0)
                ok = mag > 1e-6
                ratio = np.linalg.norm(R, axis=0)[ok] / mag[ok]
                rows.append(dict(subject=cid, pair=f"{a}-{b}",
                                 magnitude_ratio_median=round(float(np.median(ratio)), 5),
                                 deficit_pct=round(100 * (1 - float(np.median(ratio))), 3),
                                 voxel_rel_median=round(float(np.median(
                                     np.linalg.norm(R - D, axis=0)[ok] / mag[ok])), 5)))
        print(f"  {cid}: {sum(r['subject'] == cid for r in rows)} pairs", flush=True)
    with open(os.path.join(od, "fem_passive_pads.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    d = [r["deficit_pct"] for r in rows]
    print(f"leadfield field weaker than the two-pad direct run by {min(d):.1f}-{max(d):.1f}% "
          f"(median over GM voxels; {len(rows)} head-pairs)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "--sim":
        stage_sim(a)
    elif a[0] == "--score":
        stage_score(a)
    elif a[0] == "--pads":
        stage_pads(a)
    elif a[0] == "--summarize":
        summarize(read_scores())
    elif a[0] == "--qc":
        stage_qc(a)
    else:
        raise SystemExit(__doc__)
