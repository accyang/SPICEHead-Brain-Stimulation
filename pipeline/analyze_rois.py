"""
Aggregate SPICEhead TI-envelope delivery in target ROIs across the aging cohort.

For each subject and each montage (motor, thalamic), extract the envelope amplitude in each ROI
(sphere of ROI_RADIUS_MM around the subject-warped MNI center(s), intersected with brain tissue),
then relate ROI envelope to age. Writes a metrics CSV and per-montage envelope-vs-age figures.

  python analyze_rois.py <cohort.csv> <charm_outroot> <out_dir>
"""
import os
import sys
import csv
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spicehead import mri, config

DX = 3.0
ROI_RADIUS_MM = 10.0
BRAIN = (1, 2)            # WM, GM (published definition)
GM_ONLY = (2,)            # grey matter only - removes the age-varying GM:WM composition confound
MONTAGES = ["motor", "thalamic"]
ROI_ORDER = ["M1_handknob", "thalamus", "insula", "ACC", "hippocampus_L", "hippocampus_R"]


def resolve_m2m(outroot, caseid):
    if caseid.startswith("SUBJ01"):
        return os.path.join(outroot, "m2m_SUBJ01asc")
    return os.path.join(outroot, "m2m_" + caseid)


def roi_values(env, brain_idx, brain_world, centers, radius_mm):
    m = np.zeros(brain_idx.shape[0], bool)
    for c in centers:
        m |= np.linalg.norm(brain_world - np.asarray(c, float), axis=1) <= radius_mm
    vox = brain_idx[m]
    if vox.size == 0:
        return np.array([]), 0
    return env[vox[:, 0], vox[:, 1], vox[:, 2]], int(m.sum())


def main(cohort_csv, outroot, out_dir, tissue='brain'):
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(cohort_csv, encoding="utf-8-sig")))
    recs = []
    for r in rows:
        cid = r["CaseID"]
        m2m = resolve_m2m(outroot, cid)
        rj = os.path.join(m2m, "roi_coords.json")
        # Envelope volumes are namespaced by envelope definition: the rotation-invariant
        # 'maxdir' run writes to envelopes_maxdir/ so the superseded per-axis volumes in
        # envelopes/ stay reproducible. Override with SPICEHEAD_ENVDIR.
        envdir = os.path.join(m2m, os.environ.get("SPICEHEAD_ENVDIR", "envelopes"))
        seg = os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz")
        if not (os.path.exists(rj) and os.path.exists(os.path.join(envdir, "summary.json"))):
            continue
        rois = json.load(open(rj))
        # envelopes must come from the active model version (stamped in summary.json since 24 Oct 2026;
        # unstamped folders are v21/v22 runs and are refused under any other model)
        env_model = json.load(open(os.path.join(envdir, "summary.json"))).get("model")
        if env_model != config.MODEL and not (env_model is None and config.MODEL in ("v21", "v22")):
            raise SystemExit(f"{envdir}: envelopes from model {env_model or 'v21/v22'}, active model {config.MODEL}")
        labels, affine, _, _ = mri.head_grid(seg, DX)      # the grid the envelopes were saved on
        brain = np.isin(labels, BRAIN if tissue == 'brain' else GM_ONLY)
        brain_idx = np.argwhere(brain)
        brain_world = (affine[:3, :3] @ brain_idx.T).T + affine[:3, 3]

        rec = {"CaseID": cid, "Age": float(r["Age_Scan"]), "Sex": r["Sex_1M0F"],
               "Site": r["Site"], "Decade": r["Decade"]}
        for mont in MONTAGES:
            env = np.nan_to_num(np.load(os.path.join(envdir, f"{mont}_env.npy")))
            for roi in ROI_ORDER:
                vals, nvox = roi_values(env, brain_idx, brain_world, rois[roi], ROI_RADIUS_MM)
                rec[f"{mont}_{roi}_mean"] = round(float(vals.mean()), 5) if vals.size else np.nan
                rec[f"{mont}_{roi}_p95"] = round(float(np.percentile(vals, 95)), 5) if vals.size else np.nan
                rec[f"{mont}_{roi}_nvox"] = nvox
        recs.append(rec)
        print(f"  {cid}: extracted")

    # --- write CSV ---
    cols = (["CaseID", "Age", "Sex", "Site", "Decade"] +
            [f"{m}_{r}_{s}" for m in MONTAGES for r in ROI_ORDER for s in ("mean", "p95", "nvox")])
    csv_path = os.path.join(out_dir, "cohort_roi_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rec in recs:
            w.writerow(rec)
    print(f"wrote {csv_path}  ({len(recs)} subjects)")

    # --- decade summary (mean envelope per decade, motor->M1 and thalamic->thalamus as headline) ---
    ages = np.array([r["Age"] for r in recs])
    print("\nHeadline: ROI envelope mean (V/m) by decade")
    decades = sorted(set(r["Decade"] for r in recs))
    for mont, roi in [("motor", "M1_handknob"), ("thalamic", "thalamus")]:
        print(f"  {mont} montage -> {roi}:")
        for d in decades:
            v = [r[f"{mont}_{roi}_mean"] for r in recs if r["Decade"] == d and not np.isnan(r[f"{mont}_{roi}_mean"])]
            if v:
                print(f"    {d}: n={len(v):2d}  mean={np.mean(v):.3f}  (range {min(v):.3f}-{max(v):.3f})")

    # --- figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for mont in MONTAGES:
            ncol = 2
            nrow = int(np.ceil(len(ROI_ORDER) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.9 * nrow),
                                     squeeze=False)
            flat = axes.ravel()
            for ax in flat[len(ROI_ORDER):]:
                ax.set_visible(False)
            for ax, roi in zip(flat, ROI_ORDER):
                y = np.array([r[f"{mont}_{roi}_mean"] for r in recs], float)
                ok = ~np.isnan(y)
                ax.scatter(ages[ok], y[ok], s=18, alpha=0.6, edgecolor="none")
                if ok.sum() > 2:
                    b, a = np.polyfit(ages[ok], y[ok], 1)
                    xs = np.array([ages[ok].min(), ages[ok].max()])
                    from scipy import stats as _st
                    _r = _st.linregress(ages[ok], y[ok])
                    _p = ("p<0.001" if _r.pvalue < 1e-3 else f"p={_r.pvalue:.3f}")
                    ax.plot(xs, a + b * xs, "r-", lw=1.5,
                            label=f"slope={b:+.4f} V/m/yr, {_p}")
                    ax.legend(fontsize=6.5, loc="best")
                ax.set_title(f"{roi}", fontsize=10)
                ax.set_xlabel("Age (years)", fontsize=9)
                ax.set_ylabel("mean envelope (V/m)", fontsize=9)
            fig.suptitle(f"{mont.capitalize()} montage — ROI TI envelope vs age (n={len(recs)})")
            fig.tight_layout()
            p = os.path.join(out_dir, f"{mont}_rois_vs_age.png")
            fig.savefig(p, dpi=200)
            plt.close(fig)
            print("wrote", p)
    except Exception as e:
        print("plotting skipped:", e)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else 'brain')
