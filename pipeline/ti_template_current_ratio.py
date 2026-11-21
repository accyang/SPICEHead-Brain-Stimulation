# -*- coding: utf-8 -*-
"""Exploratory: unequal channel currents for insula and hippocampus targets on the template.

The template search of Figure S1 (pipeline/fig_ti_targets_mni.py) uses 1 mA in each channel. Here
every two-pair 10-20 montage (11,628) is scored at nine current splits between the channels,
I1:I2 = 1:4 ... 4:1, with the total held at 2 mA (the same total as 1 + 1 mA). F = T/O does not
change with a common scale of both currents, so only the split matters for F; T and O are
reported in V/m at the stated currents. Targets: the bilateral insula and hippocampus of Figure S1
and their one-sided halves (as in pipeline/ti_template_unilateral.py).

The paper's own current-ratio analysis (Supplement S2.2, Table S3) held each montage fixed and
used I1/I2 = 0.5-2.0; this search is joint (montage x split) and wider. Exploratory only.

Outputs, in `Paper_A_SPICEhead/Brain Stimulation/figures/TI_targets_MNI_current_ratio/`:
  TI_envelope_<target>_MNI152_1mm.nii.gz, TI_target_<target>_MNI152_1mm.nii.gz   best joint choice
  TI_targets_MNI152_summary.json / .csv                                            per target
  scores.npz                                                    F, T, O for every montage x split

  SPICEHEAD_MODEL=v22 python pipeline/ti_template_current_ratio.py
"""
import csv
import json
import os
import sys
import time
from itertools import combinations

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)
import fig_ti_targets_mni as M                                       # noqa: E402
from spicehead.envelope_fast import envelope_max_direction_flat      # noqa: E402

M.TARGETS = {
    "insula": dict(title="Insula", parcels=["Insula_L", "Insula_R"]),
    "hippocampus": dict(title="Hippocampus", parcels=["Hippocampus_L", "Hippocampus_R"]),
    "insula_L": dict(title="Left insula", parcels=["Insula_L"]),
    "insula_R": dict(title="Right insula", parcels=["Insula_R"]),
    "hippocampus_L": dict(title="Left hippocampus", parcels=["Hippocampus_L"]),
    "hippocampus_R": dict(title="Right hippocampus", parcels=["Hippocampus_R"]),
}
OUT = os.path.join(os.path.dirname(M.OUT), "TI_targets_MNI_current_ratio")
RATIOS = [0.25, 1 / 3, 0.5, 2 / 3, 1.0, 1.5, 2.0, 3.0, 4.0]          # I1 / I2
TOTAL_MA = 2.0


def currents(r):
    return TOTAL_MA * r / (1 + r), TOTAL_MA / (1 + r)


def fmt(pa, pb, i1, i2):
    return f"{'+'.join(pa)} ({i1:.1f} mA) / {'+'.join(pb)} ({i2:.1f} mA)"


def main():
    os.makedirs(OUT, exist_ok=True)
    S = M.solve_template()
    gm = S["gm"]
    keys = list(M.TARGETS)
    flat = {p: tuple(S["pair_field"](*p)[c][gm] for c in range(3)) for p in S["pairs"]}
    tsel = {k: S["tmask"][k][gm] for k in keys}
    osel = {k: S["omask"][k][gm] for k in keys}
    montages = [(pa, pb) for pa, pb in combinations(S["pairs"], 2) if len(set(pa) | set(pb)) == 4]
    nM, nR, nK = len(montages), len(RATIOS), len(keys)
    Fs, Ts, Os = (np.zeros((nM, nR, nK), np.float32) for _ in range(3))
    t0 = time.time()
    for i, (pa, pb) in enumerate(montages):
        A, B = flat[pa], flat[pb]
        for j, r in enumerate(RATIOS):
            i1, i2 = currents(r)                                     # basis fields are per 1 mA
            env = envelope_max_direction_flat(A[0] * i1, A[1] * i1, A[2] * i1, B[0] * i2, B[1] * i2, B[2] * i2)
            for k, key in enumerate(keys):
                T = float(np.median(env[tsel[key]]))
                O = float(np.percentile(env[osel[key]], 99))
                Fs[i, j, k], Ts[i, j, k], Os[i, j, k] = (T / O if O > 0 else 0.0), T, O
        if (i + 1) % 500 == 0 or i + 1 == nM:
            el = time.time() - t0
            print(f"  {i + 1}/{nM} montages x {nR} splits ({el / 60:.1f} min, ~{el / (i + 1) * (nM - i - 1) / 60:.0f} min left)",
                  flush=True)
    np.savez_compressed(os.path.join(OUT, "scores.npz"), F=Fs, T=Ts, O=Os, ratios=np.array(RATIOS),
                        targets=np.array(keys), montages=np.array([f"{'+'.join(a)}/{'+'.join(b)}" for a, b in montages]))

    t1img = nib.load(os.path.join(M.M2M, "T1.nii.gz"))
    lab05 = nib.load(os.path.join(M.M2M, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    lab1 = np.rint(np.asanyarray(resample_from_to(lab05, t1img, order=0).dataobj)).astype(int)
    brain1 = np.isin(lab1, (M.WM, M.GM))
    aal1 = np.rint(np.asanyarray(resample_from_to(nib.load(M.AAL), t1img, order=0).dataobj)).astype(int)
    value = {r["name"]: r["value"] for r in json.load(open(M.AAL_REGIONS))["regions"]}
    j1 = RATIOS.index(1.0)
    rows = []
    for k, key in enumerate(keys):
        t = M.TARGETS[key]
        ie = int(np.argmax(Fs[:, j1, k]))                            # equal-current optimum
        i, j = np.unravel_index(int(np.argmax(Fs[:, :, k])), Fs.shape[:2])
        pa, pb = montages[i]
        i1, i2 = currents(RATIOS[j])
        fixed = Fs[ie, :, k]                                         # equal-current montage, all splits
        A, B = S["pair_field"](*pa), S["pair_field"](*pb)
        env2 = envelope_max_direction_flat(*[c.ravel() * i1 for c in A], *[c.ravel() * i2 for c in B]).reshape(S["labels"].shape)
        env2 = np.where(S["brain"], np.nan_to_num(env2), 0.0).astype(np.float32)
        env1 = M.to_1mm(env2, S["aff"], t1img, order=1).astype(np.float32)
        env1[~brain1] = 0.0
        tgt1 = np.isin(aal1, [value[p] for p in t["parcels"]]) & (lab1 == M.GM)
        nib.save(nib.Nifti1Image(env1, t1img.affine), os.path.join(OUT, f"TI_envelope_{key}_MNI152_1mm.nii.gz"))
        nib.save(nib.Nifti1Image(tgt1.astype(np.uint8), t1img.affine), os.path.join(OUT, f"TI_target_{key}_MNI152_1mm.nii.gz"))
        pe, pf = montages[ie]
        rows.append(dict(
            target=key, title=t["title"], aal_parcels="+".join(t["parcels"]),
            montage=fmt(pa, pb, i1, i2), pair1="+".join(pa), pair2="+".join(pb),
            I1_mA=round(i1, 3), I2_mA=round(i2, 3), ratio_I1_I2=round(RATIOS[j], 3),
            F=round(float(Fs[i, j, k]), 3), T_V_per_m=round(float(Ts[i, j, k]), 4), O_V_per_m=round(float(Os[i, j, k]), 4),
            equal_montage=f"{'+'.join(pe)} / {'+'.join(pf)}", F_equal=round(float(Fs[ie, j1, k]), 3),
            gain_joint_pct=round(100 * (float(Fs[i, j, k]) / float(Fs[ie, j1, k]) - 1), 1),
            F_equal_montage_best_split=round(float(fixed.max()), 3),
            best_split_for_equal_montage=round(RATIOS[int(np.argmax(fixed))], 3),
            gain_fixed_montage_pct=round(100 * (float(fixed.max()) / float(fixed[j1]) - 1), 1)))
        r_ = rows[-1]
        print(f"{t['title']:<19} equal {r_['equal_montage']:<16} F={r_['F_equal']:.3f} | same montage, best split "
              f"{r_['best_split_for_equal_montage']:.2f}: F={r_['F_equal_montage_best_split']:.3f} ({r_['gain_fixed_montage_pct']:+.1f}%) | "
              f"joint best {r_['montage']} F={r_['F']:.3f} ({r_['gain_joint_pct']:+.1f}%)", flush=True)
    json.dump(dict(model=f"SPICEhead template search with unequal currents; total {TOTAL_MA} mA; ratios I1/I2 {RATIOS}",
                   grid_mm=M.DX, n_montages=nM, targets=rows),
              open(os.path.join(OUT, "TI_targets_MNI152_summary.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(OUT, "TI_targets_MNI152_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
