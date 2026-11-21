# -*- coding: utf-8 -*-
"""Review analyses of the 300-head, 163-region v23 atlas (31 Oct 2026; review round of RESULTS_v23_2026-10-31_r2).

Re-scoring only: the per-head F and T of every montage (L:/.../nchc_fullsearch, h-codes only) are read once, as in
pipeline/v23_atlas_parameters.py, and no field is solved. The review asked for:

  F > 1 at the reported montage   heads_F_above_1_pct of the atlas summaries uses the constrained optimum
  cohort montage per head         T and F of the cohort montage relative to each head's constrained optimum; "served"
                                  = the cohort montage meets the head's T floor AND keeps >= 90 % of its constrained F
  FEM feasibility                 FEM's T at SPICEhead's reported montage relative to FEM's best T (the T floor under
                                  FEM), and retention counting a montage that fails FEM's floor as a failure
  stability                       bootstrap over heads: 95 % intervals of regional median T and F at the reported
                                  montage; how often the cohort montage is re-selected (SPICEhead)
  selection-rule sensitivity      floor {none, 0.7, 0.8, 0.9} x band {0, 3, 5 %} x tie-break {on, off}: the headline
                                  counts of the paper under each variant
  region size                     voxels per region (1 mm grid) and the F-size association
  demographics                    age, sex and site on T at the cohort montage (the fixed-protocol case); design VIFs;
                                  cohort composition by site, decade and sex (aggregate counts only)

Output: results_v23/v23_atlas_review/ (aggregates only). Below-normal priority on 4 CPUs; reads about 17 GB.
    python pipeline/v23_atlas_review.py
"""
import csv
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, "nchc")]
from collect_targets_v2 import EASY, bh                              # noqa: E402
from v23_atlas_parameters import SETS, MODELS, load, CODEMAP, COHORT  # noqa: E402
from v23_fullsearch_collect import low_footprint                     # noqa: E402

OUT = os.path.join(ROOT, "results_v23", "v23_atlas_review")
NEG = np.float32(-1e9)
FLOORS, BANDS, TIES = (0.0, 0.7, 0.8, 0.9), (0.0, 0.03, 0.05), (True, False)
DEFAULT = (0.8, 0.03, True)
B_MED, B_COH, SERVE = 1000, 100, 0.90
rng = np.random.default_rng(20261031)


def choose(F, T, easy, floor=0.8, band=0.03, tie=True):
    """The approved rule (collect_targets_v2.choose) with its settings as arguments. F, T: (n, montages)."""
    adm = T >= floor * T.max(1, keepdims=True)
    Fa = np.where(adm, F, NEG)
    wc = Fa.argmax(1)
    Fc = Fa[np.arange(len(F)), wc]
    if not tie:
        return wc, wc
    inb = Fa >= (1 - band) * Fc[:, None]
    wa = np.where(inb, easy[None, :] * 10.0 + F, NEG).argmax(1)
    return wc, wa


def demographics():
    """Age, sex and site per h-code (private maps read locally; nothing identifying leaves this function)."""
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    crow = {r["CaseID"]: r for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))}
    return cmap, crow


def ols(y, X):
    from scipy.stats import t as tdist
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / (len(y) - X.shape[1]))
    return b, 2 * tdist.sf(np.abs(b / se), len(y) - X.shape[1])


def main():
    if os.name == "nt":
        low_footprint()
    t0 = time.time()
    rows, sens, seen, codes0 = [], {}, set(), None
    a = sys.argv[1:]
    only = a[a.index("--sets") + 1].split(",") if "--sets" in a else None     # test runs: --sets thal --out <dir>
    out_dir = a[a.index("--out") + 1] if "--out" in a else OUT
    os.makedirs(out_dir, exist_ok=True)
    for sname, src in SETS:
        if only and sname not in only:
            continue
        d0, suf0, _ = src["sp1v23"]
        codes = sorted(f.split("__")[0] for f in os.listdir(d0) if f.endswith(suf0))
        assert len(codes) == 300 and (codes0 is None or codes == codes0)
        codes0 = codes
        per, monts = {}, []
        for m in MODELS:
            d, suf, name = src[m]
            F0, T0, names, tiers, _, mont = load(os.path.join(d, codes[0] + suf), name)
            monts.append(mont)
            assert monts[0] == mont
            F = np.zeros((300, len(names), len(mont)), np.float32)
            T = np.zeros_like(F)
            nroi = np.zeros((300, len(names)), np.int64)
            for hi, c in enumerate(codes):
                Fh, Th, nm, _, nr, mo = load(os.path.join(d, c + suf), name)
                assert nm == names and mo == mont
                F[hi], T[hi], nroi[hi] = Fh, Th, nr
            per[m] = (F, T, nroi, names, tiers)
            print(f"{sname} {m}: loaded ({time.time() - t0:.0f} s)", flush=True)
        easy = np.array([sum(e in EASY for p in mo.split(" / ") for e in p.split("+")) for mo in monts[0]], np.float32)
        for name in per["sp1v23"][3]:
            key = name.replace("ref6:", "")
            if key in seen and name.startswith("ref6:"):
                continue
            arr = {}
            for m in MODELS:
                F, T, nroi, nm, tiers = per[m]
                t = nm.index(name)
                v = nroi[:, t] >= 5
                if v.sum() < 10:
                    break                                  # VTA of the first set: no grey matter (scored in the brainstem set)
                assert v.all(), (name, m, int(v.sum()))    # every atlas region is valid in all 300 heads
                arr[m] = (F[:, t], T[:, t], nroi[:, t], tiers[t])
            if len(arr) < len(MODELS):
                continue
            Fs, Ts, ns, tier = arr["sp1v23"]
            Ff, Tf, _, _ = arr["fem"]
            hi = np.arange(300)
            r = dict(target=name, tier=tier, set=sname, voxels_median=int(np.median(ns)), voxels_min=int(ns.min()))
            out = {}
            for m, (F, T, _, _) in arr.items():
                wc, wa = choose(F, T, easy)
                _, cc = choose(np.median(F, 0)[None], np.median(T, 0)[None], easy)
                cc = int(cc[0])
                tmax = T.max(1)
                adm_c = T[:, cc] >= 0.8 * tmax
                keep_c = F[:, cc] / F[hi, wc]
                out[m] = (wc, wa, cc)
                r.update({f"{m}__F_above_1_reported_pct": round(100 * float(np.mean(F[hi, wa] > 1)), 1),
                          f"{m}__F_above_1_constrained_pct": round(100 * float(np.mean(F[hi, wc] > 1)), 1),
                          f"{m}__cohort_T_rel_reported_median": round(float(np.median(T[:, cc] / T[hi, wa])), 4),
                          f"{m}__cohort_F_rel_constrained_median": round(float(np.median(keep_c)), 4),
                          f"{m}__cohort_served_pct": round(100 * float(np.mean(adm_c & (keep_c >= SERVE))), 1),
                          f"{m}__cohort_admissible_pct": round(100 * float(np.mean(adm_c)), 1),
                          f"{m}__T_cohort_median": round(float(np.median(T[:, cc])), 5),
                          f"{m}__F_cohort_median": round(float(np.median(F[:, cc])), 4)})
                # bootstrap over heads: regional medians at the reported montage
                tw, fw = T[hi, wa], F[hi, wa]
                bi = rng.integers(0, 300, (B_MED, 300))
                for k, a in (("T", tw), ("F", fw)):
                    q = np.percentile(np.median(a[bi], 1), [2.5, 97.5])
                    r[f"{m}__{k}_reported_median"] = round(float(np.median(a)), 5)
                    r[f"{m}__{k}_reported_ci_lo"], r[f"{m}__{k}_reported_ci_hi"] = round(float(q[0]), 5), round(float(q[1]), 5)
            # FEM feasibility of SPICEhead's reported montage, and retention counting infeasible choices as failures
            _, wa_s, cc_s = out["sp1v23"]
            wc_f, _, _ = out["fem"]
            feas = Tf[hi, wa_s] / Tf.max(1)
            ret = 100 * Ff[hi, wa_s] / Ff[hi, wc_f]
            ok = feas >= 0.8
            r.update(fem_T_at_sp_choice_rel_median=round(float(np.median(feas)), 4),
                     sp_choice_fails_fem_floor_pct=round(100 * float(np.mean(~ok)), 1),
                     retention_median=round(float(np.median(ret)), 2), retention_p5=round(float(np.percentile(ret, 5)), 2),
                     retention_ok_and_feasible_pct=round(100 * float(np.mean(ok & (ret >= 90))), 1),
                     retention_over_100_pct=round(100 * float(np.mean(ret > 100)), 1))
            # cohort-montage stability: bootstrap re-selection (SPICEhead)
            same = 0
            for _ in range(B_COH):
                b = rng.integers(0, 300, 300)
                _, c2 = choose(np.median(Fs[b], 0)[None], np.median(Ts[b], 0)[None], easy)
                same += int(c2[0]) == cc_s
            r["sp1v23__cohort_reselected_pct"] = round(100 * same / B_COH, 1)
            # selection-rule sensitivity: per variant, the quantities behind the paper's counts
            for fl in FLOORS:
                for bd in BANDS:
                    for tie in TIES:
                        if bd == 0.0 and not tie:
                            continue                               # identical to (band 0, tie-break on)
                        res = {}
                        for m, (F, T, _, _) in arr.items():
                            wc, wa = choose(F, T, easy, fl, bd, tie)
                            _, cc = choose(np.median(F, 0)[None], np.median(T, 0)[None], easy, fl, bd, tie)
                            res[m] = (wc, wa, int(cc[0]), F, T)
                        wc_s, wa_s2, cc2, F, T = res["sp1v23"]
                        wc_f2, wa_f2, ccf, F2, T2 = res["fem"]
                        ret2 = 100 * F2[hi, wa_s2] / F2[hi, wc_f2]
                        gain = F[hi, wc_s] / F[:, cc2]
                        sens.setdefault((fl, bd, tie), []).append(dict(
                            target=name, same=100 * float(np.mean(wa_s2 == wa_f2)), p5=float(np.percentile(ret2, 5)),
                            cohort_same=cc2 == ccf, cohort_changed=cc2 != cc_s if (fl, bd, tie) != DEFAULT else False,
                            F1=100 * float(np.mean(F[hi, wa_s2] > 1)), g10=100 * float(np.mean(gain >= 1.10)),
                            adm=100 * float(np.mean(T[:, cc2] >= fl * T.max(1))), T=float(np.median(T[hi, wa_s2])),
                            F=float(np.median(F[hi, wa_s2]))))
            r["_ts"] = Ts[hi, out["sp1v23"][2]]                            # T at the cohort montage, for the regressions
            rows.append(r)
            seen.add(key)
        del per
        print(f"{sname}: done ({time.time() - t0:.0f} s)", flush=True)

    # demographics on T at the cohort montage (SPICEhead): log T ~ age per decade + sex + site; FDR across regions
    cmap, crow = demographics()
    age = np.array([float(crow[cmap[c]]["Age_Scan"]) for c in codes0])
    male = np.array([crow[cmap[c]]["Sex_1M0F"] == "1" for c in codes0], float)
    site = [crow[cmap[c]]["Site"] for c in codes0]
    sites = sorted(set(site))
    X = np.column_stack([np.ones(300), age / 10, male] + [[s == k for s in site] for k in sites[1:]]).astype(float)
    res = [ols(np.log(r["_ts"].astype(float)), X) for r in rows]
    qa, qs = bh(np.array([p[1] for _, p in res])), bh(np.array([p[2] for _, p in res]))
    for r, (b, p), a, s in zip(rows, res, qa, qs):
        r.update(cohortT_age_pct_per_decade=round(100 * (np.exp(b[1]) - 1), 2), cohortT_age_q=float(a),
                 cohortT_sex_pct_men=round(100 * (np.exp(b[2]) - 1), 2), cohortT_sex_q=float(s))
        del r["_ts"]
    vif = {}
    for j, nm in ((1, "age"), (2, "sex")):
        others = [k for k in range(X.shape[1]) if k != j]
        bb, *_ = np.linalg.lstsq(X[:, others], X[:, j], rcond=None)
        e = X[:, j] - X[:, others] @ bb
        vif[nm] = round(float(1 / (e @ e / ((X[:, j] - X[:, j].mean()) @ (X[:, j] - X[:, j].mean())))), 2)
    dec = np.minimum((age // 10).astype(int) * 10, 80)
    comp = [dict(site=s, decade=f"{d}-{d + 9}", men=int(np.sum((np.array(site) == s) & (dec == d) & (male == 1))),
                 women=int(np.sum((np.array(site) == s) & (dec == d) & (male == 0)))) for s in sites for d in range(20, 90, 10)]

    keys = list(rows[0])
    with open(os.path.join(out_dir, "region_review.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    srows = []
    for (fl, bd, tie), lst in sens.items():
        a = lambda k: np.array([x[k] for x in lst], float)
        srows.append(dict(floor=fl, band=bd, tiebreak=tie, default=(fl, bd, tie) == DEFAULT, regions=len(lst),
                          same_choice_median=round(float(np.median(a("same"))), 1),
                          retention_p5_ge90=int((a("p5") >= 90).sum()), cohort_same_as_fem=int(a("cohort_same").sum()),
                          cohort_changed_vs_default=int(a("cohort_changed").sum()),
                          F_above_1_majority=int((a("F1") > 50).sum()), F_above_1_rare=int((a("F1") < 5).sum()),
                          gain10_ge20=int((a("g10") >= 20).sum()), gain10_lt5=int((a("g10") < 5).sum()),
                          cohort_admissible_median=round(float(np.median(a("adm"))), 1),
                          T_median=round(float(np.median(a("T"))), 4), F_median=round(float(np.median(a("F"))), 3)))
    with open(os.path.join(out_dir, "rule_sensitivity.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(srows[0]))
        w.writeheader()
        w.writerows(srows)
    with open(os.path.join(out_dir, "cohort_composition.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(comp[0]))
        w.writeheader()
        w.writerows(comp)
    json.dump(dict(script="pipeline/v23_atlas_review.py", date=time.strftime("%Y-%m-%d %H:%M"), regions=len(rows),
                   heads=300, bootstrap_medians=B_MED, bootstrap_cohort=B_COH, serve_threshold=SERVE,
                   default_rule=dict(floor=0.8, band=0.03, tiebreak=True), variants=len(srows), vif=vif,
                   seed=20261031, inputs="L:/BrainProbe/SPICEhead/nchc_fullsearch (atlas_v23, targets_out, atlas_ext, "
                   "atlas_thal; h-codes only)", seconds=round(time.time() - t0)),
              open(os.path.join(out_dir, "PROVENANCE.json"), "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_dir}: {len(rows)} regions, {len(srows)} rule variants, VIF {vif} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
