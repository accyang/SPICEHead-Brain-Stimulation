# -*- coding: utf-8 -*-
"""v23 uncertainty analysis over all atlas targets, one head (24 Oct 2026; PI: "the uncertainty run should also
generalize to all brain region targets"; amendment 1 of results_v23/v23_uq/ANALYSIS_PLAN.md).

Same 75 runs per head as uq_v23.py: the perturbations come from its functions with the same seeds, and the basis is
computed with the same code, so the perturbed models are identical to the six-target run. Each run scores every
atlas target:
  - 90 AAL regions;
  - tier-2 psychiatric targets;
  - the six Paper A spheres (ref6).
Scoring and target definitions are those of score_targets_v2.py (atlas), unchanged. Per run and target it stores:
  - the unconstrained winner, the constrained optimum and the reported choice of the atlas rule (T >= 80 % of the
    head's best T, top band 3 %, most easy electrodes): their indices and F, T, O;
  - the highest achievable T;
  - F, T, O at four fixed montages: the FEM cohort choice and the FEM cohort unconstrained winner of the 300-head
    atlas, and the head's nominal unconstrained winner and nominal reported choice;
  - envelope peak and 99.9th percentile in grey matter at the FEM cohort choice.
Per-target target files (subject-derived) come from prep_uq_targets.py and are uploaded by the PI.
Output: $SPICEHEAD_UQ_ATLAS_OUT/<code>/<run>.npz, targets.json, DONE.
    SPICEHEAD_MODEL=v23 SPICEHEAD_SOLVER=gpu python pipeline/nchc/uq_v23_atlas.py --head h002 --targets <code>__uqtargets.npz [--m2m DIR] [--only nominal,cond_00]
--target-set thal (25 Oct 2026) scores the six Paper A spheres and the 16 AAL3 thalamic-nucleus and ACC_sup targets of
atlas_thal_score.py instead, from --targets <code>__exttargets.npz, with SPICEHEAD_UQ_ATLAS_FIXED pointing to
uq_atlas_fixed_montages_thal.json (their FEM cohort choices from the 300-head extension).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "pipeline")); sys.path.insert(0, HERE)
import uq_v23 as U                                                     # noqa: E402  (perturbations and constants)

OUT = os.environ.get("SPICEHEAD_UQ_ATLAS_OUT", os.path.join(ROOT, "results_v23", "v23_uq", "runs_atlas"))
FIXED = os.environ.get("SPICEHEAD_UQ_ATLAS_FIXED", os.path.join(ROOT, "results_v23", "v23_uq", "uq_atlas_fixed_montages.json"))
# atlas reporting rule, copied unchanged from collect_targets_v2.py (22 Oct 2026, approved by the PI)
TMIN_FRAC, BAND = 0.80, 0.03
EASY = {"Fp1", "Fp2", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2"}
NEG = np.float32(-1e9)


def choose(F, T, easy):
    """F, T: (nt, nm). Per-target indices and band statistics under the approved rule (collect_targets_v2.choose)."""
    tmax = T.max(1, keepdims=True)
    adm = T >= TMIN_FRAC * tmax
    Fa = np.where(adm, F, NEG)
    wc = Fa.argmax(1); Fc = Fa[np.arange(len(F)), wc]
    band = Fa >= (1 - BAND) * Fc[:, None]
    nband = band.sum(1)
    wa = np.where(band, easy[None, :] * 10.0 + F, NEG).argmax(1)
    w0 = F.argmax(1)
    return dict(w0=w0, wc=wc, wa=wa, nband=nband)


def main():
    a = sys.argv[1:]
    code = a[a.index("--head") + 1]
    tfile = a[a.index("--targets") + 1]
    only = a[a.index("--only") + 1].split(",") if "--only" in a else None
    import v22_fem_fullsearch as fs
    import score_targets_v2 as ST
    from spicehead import config, mri
    from spicehead.homogenize import homogenized_model
    from spicehead.electrodes import Electrode, contact_nodes, pair_injection
    from spicehead.field import efield_components
    from spicehead.score_gpu import _envelope
    from montage_search import snap
    from v21_head_convergence import make_solver
    import torch
    assert config.NETWORK == "homogenized"
    m2m = a[a.index("--m2m") + 1] if "--m2m" in a else os.path.join(os.environ["SPICEHEAD_CHARM_ROOT"], f"m2m_{code}")
    od = os.path.join(OUT, code); os.makedirs(od, exist_ok=True)
    t0 = time.time()
    geo = fs.grid_geometry(code, m2m)
    gm_world = np.asarray(geo["gm_world"], np.float32)        # as uq_v23.py and the validated 1 mm runs
    zt = np.load(tfile)
    if not np.array_equal(zt["gm_idx"].astype(np.int64), np.asarray(geo["gm_idx"]).astype(np.int64)):
        raise SystemExit(f"{code}: the atlas target grid differs from the scoring grid")
    # target files v2 (24 Oct 2026, PI: "approve AAL3 and add cerebellum/brainstem") add the 26 AAL3 cerebellar regions
    # (grey matter, scored like every other region) and the 12 AAL3 brainstem nuclei (brain-tissue points outside the
    # grey-matter set: T = median envelope over the nucleus points, O = 99th percentile over grey matter beyond 20 mm)
    if "--target-set" in a and a[a.index("--target-set") + 1] == "thal":
        # 25 Oct 2026 (PI: "covering these 16 targets for uncertainty analysis"): ref6 + 7 AAL3 thalamic nuclei and ACC_sup,
        # left and right, from the 300-head extension's target files (<code>__exttargets.npz), defined as in atlas_thal_score.py
        from atlas_thal_score import build_targets_thal
        targets, _ = build_targets_thal(zt, fs.EXCLUDE_MM)
        tiers = [t[1] for t in targets]
        v2 = False
    else:
        v2 = "bs_idx" in zt.files
        cereb = json.loads(str(zt["cereb_spec"])) if v2 else {}
        targets = ST.build_targets(zt, json.loads(str(zt["centres"])), json.loads(str(zt["aal_names"])), None,
                                   dict(labels=zt["t2_labels"], spec={**json.loads(str(zt["t2_spec"])), **cereb}))
        tiers = ["cerebellum" if t[0] in cereb else t[1] for t in targets]
    n_gm = len(geo["gm_idx"])
    pts = gm_world
    if v2:
        from scipy.spatial import cKDTree
        aff_t = np.asarray(zt["affine"], float)
        bs_idx = zt["bs_idx"].astype(np.int64); bs_lab = zt["bs_labels"]
        bs_world = ((aff_t[:3, :3] @ bs_idx.T).T + aff_t[:3, 3]).astype(np.float32)
        pts = np.concatenate([gm_world, bs_world])
        pad = np.zeros(len(bs_idx), bool)
        targets = [(n, tr, np.concatenate([r, pad]), np.concatenate([o, pad])) for n, tr, r, o in targets]
        for name, vals_ in json.loads(str(zt["brainstem_spec"])).items():
            sel = np.isin(bs_lab, vals_)
            if sel.sum() < ST.MIN_REGION_VOX:
                roi = np.zeros(len(pts), bool); out = np.zeros(len(pts), bool)
            else:
                dist, _ = cKDTree(bs_world[sel].astype(np.float64)).query(gm_world.astype(np.float64), k=1)
                roi = np.concatenate([np.zeros(n_gm, bool), sel]); out = np.concatenate([dist > fs.EXCLUDE_MM, pad])
            targets.append((f"brainstem:{name}", "brainstem", roi, out)); tiers.append("brainstem")   # prefix: tier 2 also has a (grey-matter) VTA
    new_t = np.array([t for t, tr in enumerate(tiers) if tr in ("cerebellum", "brainstem")], np.int64)
    tnames = [t[0] for t in targets]
    n_roi = np.array([int(t[2].sum()) for t in targets]); n_out = np.array([int(t[3].sum()) for t in targets])
    # 31 Oct 2026 (extension to all 300 heads; PI: a 35-head uncertainty analysis is "not acceptable"): --also-thal
    # <code>__exttargets.npz scores the thalamic-nucleus/ACC_sup set of --target-set thal in the same run, from the
    # same basis, and writes it to SPICEHEAD_UQ_THAL_OUT with the fixed montages of SPICEHEAD_UQ_THAL_FIXED. The
    # stored values equal those of a separate --target-set thal run (same basis, same scoring code).
    thal = None
    if "--also-thal" in a:
        from atlas_thal_score import build_targets_thal
        zth = np.load(a[a.index("--also-thal") + 1])
        if not np.array_equal(zth["gm_idx"].astype(np.int64), np.asarray(geo["gm_idx"]).astype(np.int64)):
            raise SystemExit(f"{code}: the thal target grid differs from the scoring grid")
        th_t, _ = build_targets_thal(zth, fs.EXCLUDE_MM)                  # ref6 + 16, the layout of the 35-head thal files
        th_fixed = json.load(open(os.environ["SPICEHEAD_UQ_THAL_FIXED"]))["targets"]
        th_out = os.path.join(os.environ["SPICEHEAD_UQ_THAL_OUT"], code); os.makedirs(th_out, exist_ok=True)
        thal = dict(targets=th_t, out=th_out, fixed=th_fixed)
    # 31 Oct 2026 (review D7: registration sensitivity of the target masks): mask_erode / mask_dilate score the
    # nominal model with every target region eroded or dilated by one voxel (6-connectivity) within its own point set
    # (grey matter, or the brainstem points); the exclusion masks (beyond 20 mm) are unchanged. Scoring only.
    MASK_RUNS = ("mask_erode", "mask_dilate")

    def mask_variant(tg, which, shape, idx_all, n_gm_):
        from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
        st = generate_binary_structure(3, 1)
        is_gm = np.arange(len(idx_all)) < n_gm_
        out_t, changed = [], 0
        for name, tr, roi, out in tg:
            if roi.sum() == 0:
                out_t.append((name, tr, roi, out)); continue
            cls = is_gm if roi[:n_gm_].any() else ~is_gm
            ii = idx_all[roi]
            lo = np.maximum(ii.min(0) - 2, 0); hi = np.minimum(ii.max(0) + 3, np.asarray(shape))
            sub = np.zeros(hi - lo, bool); sub[tuple((ii - lo).T)] = True
            sub = binary_erosion(sub, st) if which == "mask_erode" else binary_dilation(sub, st)
            inside = np.all((idx_all >= lo) & (idx_all < hi), 1) & cls
            new = np.zeros_like(roi)
            new[inside] = sub[tuple((idx_all[inside] - lo).T)]
            changed += int((new != roi).sum())
            out_t.append((name, tr, new, out))
        return out_t, changed
    seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
    fine, aff_f, _ = mri.load_label_volume(seg)
    ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
    labs, pairs, montages = fs.candidate_set(ej)
    others = [x for x in labs if x != fs.REF]
    keys = [fs.label_of(pa, pb) for pa, pb in montages]
    kidx = {k: i for i, k in enumerate(keys)}
    easy = np.array([sum(e in EASY for p in m.split(" / ") for e in p.split("+")) for m in keys], np.float32)
    fx = json.load(open(FIXED))["targets"]
    fem_choice = np.array([kidx[fx[n]["cohort_choice"]] if n in fx else -1 for n in tnames])
    fem_unc = np.array([kidx[fx[n]["cohort_unconstrained"]] if n in fx else -1 for n in tnames])
    json.dump(dict(targets=tnames, tiers=tiers, n_roi=n_roi.tolist(), n_out=n_out.tolist(), new_targets=new_t.tolist(),
                   n_brainstem_points=int(len(pts) - n_gm), fem_choice=[keys[i] if i >= 0 else None for i in fem_choice]),
              open(os.path.join(od, "targets.json"), "w"))
    # the target sets scored per run: the main set, and the thal set when requested (its targets padded to the point set)
    sets = [dict(name="atlas", out=od, targets=targets, tiers=tiers, new_t=new_t, n_roi=n_roi,
                 fem_choice=fem_choice, fem_unc=fem_unc, nominal_idx={})]
    if thal is not None:
        padn = len(pts) - n_gm
        tt = [(n, tr, np.concatenate([r, np.zeros(padn, bool)]), np.concatenate([o, np.zeros(padn, bool)])) for n, tr, r, o in thal["targets"]]
        tn = [t[0] for t in tt]
        tfc = np.array([kidx[thal["fixed"][n]["cohort_choice"]] if n in thal["fixed"] else -1 for n in tn])
        tfu = np.array([kidx[thal["fixed"][n]["cohort_unconstrained"]] if n in thal["fixed"] else -1 for n in tn])
        tnr = np.array([int(t[2].sum()) for t in tt])
        json.dump(dict(targets=tn, tiers=[t[1] for t in tt], n_roi=tnr.tolist(), n_out=[int(t[3].sum()) for t in tt], new_targets=[],
                       fem_choice=[keys[i] if i >= 0 else None for i in tfc], also_thal_of=od),
                  open(os.path.join(thal["out"], "targets.json"), "w"))
        sets.append(dict(name="thal", out=thal["out"], targets=tt, tiers=[t[1] for t in tt], new_t=np.array([], np.int64), n_roi=tnr,
                         fem_choice=tfc, fem_unc=tfu, nominal_idx={}))
    idx_all = np.concatenate([np.asarray(geo["gm_idx"]).astype(np.int64), zt["bs_idx"].astype(np.int64)]) if v2 else np.asarray(geo["gm_idx"]).astype(np.int64)
    grid_shape = tuple(int(x) for x in zt["shape"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{code}: {len(targets)} targets ({int((n_roi == 0).sum())} empty)" + (f" + {len(sets[1]['targets'])} thal" if thal else "") +
          f", grid checked, {len(geo['gm_idx']):,} GM voxels ({time.time()-t0:.0f} s)", flush=True)

    def build(labels, cond):
        model, aff_c = homogenized_model(labels, aff_f, U.DX, min_fill=config.HOMOG_MIN_FILL, cond=cond)
        solver, _ = make_solver(model.G)
        return model, aff_c, solver

    def run(name, model, aff_c, solver, pos, params, mask=None):
        """One perturbed model: basis once, then every target set scored and saved. mask: a MASK_RUNS name (nominal
        model, perturbed target masks)."""
        if all(os.path.exists(os.path.join(s["out"], f"{name}.npz")) for s in sets):
            return "cached"
        t1 = time.time()
        # basis: identical to uq_v23.run
        scalp_vox = np.argwhere(model.labels == fs.SCALP_LABEL)
        elecs = {}
        for lab in labs:
            v = snap(scalp_vox, mri.world_to_voxel(aff_c, pos[lab]))
            elecs[lab] = Electrode(contact_nodes(model, tuple(int(x) for x in v), radius_nodes=fs.ELEC_RADIUS_MM / U.DX), lab)
        basis = np.zeros((len(others), 3, len(pts)), np.float32)          # grey matter, then brainstem points (v2)
        for i, lab in enumerate(others):
            V = solver(pair_injection(model, elecs[lab], elecs[fs.REF], fs.CURRENT))
            basis[i] = np.stack([fs.sample_on_gm(c, aff_c, pts) for c in efield_components(model, V)])
        t_solve = time.time() - t1
        idb = {lab: i for i, lab in enumerate(others)}
        Bz = torch.from_numpy(basis).to(dev); zero = torch.zeros_like(Bz[0])
        pf = lambda pr: (Bz[idb[pr[0]]] if pr[0] != fs.REF else zero) - (Bz[idb[pr[1]]] if pr[1] != fs.REF else zero)
        msg = []
        for s in sets:
            p = os.path.join(s["out"], f"{name}.npz")
            if os.path.exists(p):
                continue
            tg, changed = s["targets"], 0
            if mask is not None:
                tg, changed = mask_variant(tg, mask, grid_shape, idx_all, n_gm)
            nt = len(tg); new_t_, n_roi_, fem_choice_, fem_unc_, nominal_idx = s["new_t"], s["n_roi"], s["fem_choice"], s["fem_unc"], s["nominal_idx"]
            F, T, O = ST.score_model(basis, others, pairs, montages, tg, dev)
            F, T, O = F.astype(np.float64), T.astype(np.float64), O.astype(np.float64)
            ch = choose(F.astype(np.float32), T.astype(np.float32), easy)
            if not nominal_idx:                                    # the nominal run defines the head's own fixed choices
                assert name == "nominal", f"{code}: {s['name']} nominal run missing before {name}"
                nominal_idx["w0"], nominal_idx["wa"] = ch["w0"].copy(), ch["wa"].copy()
            ar = np.arange(nt)
            pick = lambda w: np.stack([np.where(w >= 0, A[ar, np.maximum(w, 0)], np.nan) for A in (F, T, O)], -1)
            idx = np.stack([ch["w0"], ch["wc"], ch["wa"]], 1).astype(np.int32)
            vals = np.stack([pick(ch["w0"]), pick(ch["wc"]), pick(ch["wa"])], 1)            # (nt, 3 choices, F/T/O)
            fixed_idx = np.stack([fem_choice_, fem_unc_, nominal_idx["w0"], nominal_idx["wa"]], 1)
            fixed_vals = np.stack([pick(fixed_idx[:, j]) for j in range(4)], 1)              # (nt, 4 montages, F/T/O)
            # envelope peak and 99.9th percentile in GM at the FEM cohort choice
            peaks = np.full((nt, 2), np.nan)
            for t in range(nt):
                if fem_choice_[t] >= 0:
                    pa, pb = montages[fem_choice_[t]]
                    env = _envelope(pf(pa)[None], pf(pb)[None])[0][:n_gm]          # grey matter only
                    peaks[t] = [float(env.max()), float(torch.quantile(env, 0.999))]
            # new targets (no FEM cohort choice yet): peaks at the head's nominal reported choice
            peaks_new = np.full((len(new_t_), 2), np.nan)
            for j, t in enumerate(new_t_):
                if n_roi_[t] > 0:
                    pa, pb = montages[int(nominal_idx["wa"][t])]
                    env = _envelope(pf(pa)[None], pf(pb)[None])[0][:n_gm]
                    peaks_new[j] = [float(env.max()), float(torch.quantile(env, 0.999))]
            prm = dict(params, code=code, run=name, n_nodes=int(model.n_nodes), t_solve_s=round(t_solve, 1), t_run_s=round(time.time() - t1, 1),
                       target_set=s["name"], **({"mask_changed_points": changed, "n_roi_masked": [int(t[2].sum()) for t in tg]} if mask else {}))
            np.savez_compressed(p + ".part.npz", idx=idx, vals=vals.astype(np.float32), Tmax=T.max(1).astype(np.float32), nband=ch["nband"].astype(np.int32),
                                fixed_idx=fixed_idx.astype(np.int32), fixed_vals=fixed_vals.astype(np.float32), peaks=peaks.astype(np.float32),
                                cols=np.array(["F", "T", "O"]), choices=np.array(["unconstrained", "constrained", "reported"]),
                                fixed=np.array(["fem_cohort_choice", "fem_cohort_unconstrained", "nominal_unconstrained", "nominal_reported"]),
                                # cerebellum and brainstem: full F and T over all montages (float16), so that the fixed protocol
                                # from the 300-head extension can be evaluated after the run; peaks at the nominal reported choice
                                new_targets=new_t_.astype(np.int32), F_new=F[new_t_].astype(np.float16), T_new=T[new_t_].astype(np.float16),
                                peaks_new=peaks_new.astype(np.float32), params=json.dumps(prm))
            os.replace(p + ".part.npz", p)
            msg.append(f"{s['name']} {prm['t_run_s']:.0f} s" + (f", {changed} mask points" if mask else ""))
        del Bz
        return "ok (" + "; ".join(msg) + ")"

    want = lambda n: only is None or n in only
    model, aff_c, solver = build(fine, None)
    pos0 = {lab: np.asarray(ej[lab], float) for lab in labs}
    ii, jj, kk = model.grid_coords
    centroid = aff_c[:3, :3] @ np.array([ii.mean(), jj.mean(), kk.mean()]) + aff_c[:3, 3]
    for s in sets:
        nom = os.path.join(s["out"], "nominal.npz")
        if os.path.exists(nom):
            with np.load(nom) as z:
                s["nominal_idx"]["w0"], s["nominal_idx"]["wa"] = z["idx"][:, 0], z["idx"][:, 2]
    if any(not s["nominal_idx"] for s in sets):
        print(f"{code} nominal: {run('nominal', model, aff_c, solver, pos0, dict(scenario='nominal'))}", flush=True)
    for n in MASK_RUNS:                                        # nominal model, perturbed target masks (scoring only)
        if want(n):
            print(f"{code} {n}: {run(n, model, aff_c, solver, pos0, dict(scenario='target_mask', which=n), mask=n)}", flush=True)
    for k in range(U.N_JIT):
        n = f"jit_{k:02d}"
        if want(n):
            print(f"{code} {n}: {run(n, model, aff_c, solver, U.jitter_positions(code, k, pos0, centroid), dict(scenario='jitter', sd_mm=U.JIT_SD_MM))}", flush=True)
    for k in range(U.N_ROT):
        n = f"rot_{k:02d}"
        if want(n):
            pos, rp = U.rotate_positions(code, k, pos0, centroid)
            print(f"{code} {n}: {run(n, model, aff_c, solver, pos, dict(scenario='rotation', **rp))}", flush=True)
    del model, solver
    Z = U.cond_samples(code)
    for k in range(U.N_COND):
        n = f"cond_{k:02d}"
        if want(n) and not all(os.path.exists(os.path.join(s_["out"], f"{n}.npz")) for s_ in sets):
            m, ac, s = build(fine, U.cond_dict(Z[k]))
            print(f"{code} {n}: {run(n, m, ac, s, pos0, dict(scenario='conductivity', log_factors=dict(zip([t for t, _, _ in U.COND_TISSUES], Z[k].round(6).tolist()))))}", flush=True)
            del m, s
    for n in U.SEG:
        if want(n) and not all(os.path.exists(os.path.join(s_["out"], f"{n}.npz")) for s_ in sets):
            Lm = U.seg_labels(fine, n)
            m, ac, s = build(Lm, None)
            print(f"{code} {n}: {run(n, m, ac, s, pos0, dict(scenario='segmentation', changed_voxels=int((Lm != fine).sum())))}", flush=True)
            del m, s, Lm
    if only is None:
        for s_ in sets:
            open(os.path.join(s_["out"], "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{code}: finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
