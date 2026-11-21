# -*- coding: utf-8 -*-
"""Check that the model-version switch (spicehead.config) changes exactly what it should.

  SPICEHEAD_MODEL=v21 python pipeline/v22_switch_check.py v21
      The production grid-convergence path must reproduce the published v21 values for SUBJ01
      at 5 and 4 mm exactly (node count and the six ROI medians).
  SPICEHEAD_MODEL=v22 python pipeline/v22_switch_check.py v22
      The production FEM-comparison path, now defaulting to skin-surface contacts and the
      harmonic rule, must reproduce the explicit (harmonic, surface) configuration of the
      contact-sensitivity analysis for SUBJ02 (thalamic montage, 2 mm).
"""
import csv
import json
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)
from spicehead import config


def check_v21():
    import v21_head_convergence as H
    assert config.MODEL == "v21" and H.CMEAN == "arithmetic", (config.MODEL, H.CMEAN)
    m2m = os.path.join(H.CH, "m2m_SUBJ01")
    pub = {float(r["dx_mm"]): r for r in csv.DictReader(open(os.path.join(
        ROOT, "results", "v21_head_convergence", "head_convergence.csv"), encoding="utf-8"))}
    worst = 0.0
    for dx in (5.0, 4.0):
        got = H.envelope_at(m2m, dx)
        assert got["n_nodes"] == int(pub[dx]["n_nodes"]), (dx, got["n_nodes"], pub[dx]["n_nodes"])
        for roi, v in got["roiT"].items():
            ref = float(pub[dx]["T_" + roi])
            worst = max(worst, abs(v - ref) / ref)
        print(f"  dx={dx:g}: nodes {got['n_nodes']} match; ROI medians max rel dev so far "
              f"{worst:.1e}", flush=True)
    print("v21 regression", "PASSED" if worst < 1e-6 else f"FAILED ({worst:.1e})")


def check_v22():
    import nibabel as nib
    import v21_simnibs_compare as C
    from v21_simnibs_validation import subjects
    from v21_multimontage_fem import fem_dir
    from v21_agreement_raw import stats
    from montage_search import SCORE
    assert config.MODEL == "v22" and config.CONTACT == "surface"
    cid, m2m = next((s[0], s[1]) for s in subjects(8) if s[0] == "SUBJ02")
    (S1, S2), saff, labels = C.spicehead_E(m2m, 2.0)
    E1, aff, sh = C.load_simnibs_E(fem_dir(cid, "THAL", 1))
    E2, _, _ = C.load_simnibs_E(fem_dir(cid, "THAL", 2))
    fem = np.nan_to_num(C.envelope(E1, E2))
    li = nib.load(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    gm = C.resample_to(np.isin(np.asanyarray(li.dataobj), SCORE).astype(np.float32),
                       li.affine, aff, sh) > 0.5
    spg = C.resample_to(np.nan_to_num(C.envelope(S1, S2)), saff, aff, sh)
    x, y = spg[gm].astype(float), fem[gm].astype(float)
    k = np.isfinite(x) & np.isfinite(y) & (y > 0)
    got = stats(x[k], y[k])
    ref = next(r for r in json.load(open(os.path.join(
        ROOT, "results", "v21_contact_sensitivity", "per_subject", f"{cid}__dx2_harmonic.json"),
        encoding="utf-8"))["rows"] if r["mode"] == "surface" and r["montage"] == "THAL")
    dev = max(abs(got[k_] - ref[k_]) / abs(ref[k_]) for k_ in ("mean_spicehead", "slope", "r", "ccc"))
    print(f"  {cid} THAL 2 mm: mean_spicehead {got['mean_spicehead']:.6f} vs "
          f"{ref['mean_spicehead']:.6f}; max relative deviation {dev:.1e}")
    print("v22 cross-check", "PASSED" if dev < 1e-6 else f"FAILED ({dev:.1e})")


if __name__ == "__main__":
    print(config.describe(), flush=True)
    {"v21": check_v21, "v22": check_v22}[sys.argv[1]]()
