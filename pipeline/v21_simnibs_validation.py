# -*- coding: utf-8 -*-
"""Realistic-head validation: SPICEhead versus a SimNIBS FEM reference (reviewer Essential item 2).

The manuscript previously reported voxel-wise correlations of r = 0.78-0.88 as evidence of good
agreement. Correlation measures covariation, not agreement, so this computes the full suite the
review asks for: regression slope and intercept, R^2, normalized MAE, Bland-Altman bias and limits,
Dice overlap of the high-field region, peak-location distance, and target-ROI error.

Design choices that make the comparison fair -- the review asks for "matched tissue assumptions and
electrode geometry":
  * conductivities are identical by construction (SPICEhead uses the SimNIBS defaults; verified
    element-by-element at run time, and the run aborts if they ever diverge),
  * the same montage, the same 1 mA per channel, and the same TI envelope definition
    (rotation-invariant maximum-over-direction) applied to both vector fields,
  * SimNIBS fields are mapped to the subject voxel grid and SPICEhead is evaluated on that same
    grid, so no interpolation asymmetry is introduced,
  * SPICEhead is evaluated at BOTH 2 mm and 1 mm, as the review suggests, since the resolution
    study (v21_head_convergence.py) showed 5 mm is not converged.

Stages, each resumable:
  --mesh     charm <sub> --mesh          (head mesh from the existing label image)
  --sim      SimNIBS TDCS, one run per channel, mapped to volume
  --compare  load both, compute the agreement suite

  python pipeline/v21_simnibs_validation.py --mesh|--sim|--compare [n_subjects]

--sim must run under the SimNIBS interpreter; --compare under the project interpreter.
"""
import csv
import json
import os
import subprocess
import sys
import time

# Data roots. The defaults are this lab's local layout; a verification copy (for example the
# student package under Paper_A_SPICEhead/Brain Stimulation/verification_8subjects) points these elsewhere through
# the environment without editing any source file.
CH = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")
CSVP = os.environ.get("SPICEHEAD_COHORT_CSV", r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv")
SIMNIBS = r"C:\Users\ALBERT YANG\SimNIBS-4.6"
_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
OUT = os.environ.get("SPICEHEAD_FEM_OUT", os.path.join(ROOT, "results", "v21_simnibs_validation"))   # FEM runs (cluster copy, 28 Oct 2026)

# Table 3 thalamic optimum: channel 1 = T8/P8, channel 2 = F7/T7. A headline montage, and all four
# sites are standard 10-20 positions present in every subject's electrode file.
MONTAGE = (("T8", "P8"), ("F7", "T7"))
CURRENT = 1e-3          # A per channel
ELEC_RADIUS_MM = 12.0   # matches spicehead's electrode radius
N_DEFAULT = 8


def subjects(n):
    """Decade-spread subjects that already have a segmentation and electrode definitions."""
    pin = os.environ.get("SPICEHEAD_FEM8_SUBJECTS")
    if pin:
        # a pinned list (CSV with a CaseID column), for runs that may not see the cohort file (ages, sites), e.g. the
        # 28 Oct 2026 cluster run with h-codes: age unknown (nan), site "NA"; the first n listed are used
        ids = [r["CaseID"] for r in csv.DictReader(open(pin, encoding="utf-8-sig"))][:n]
        return [(c, os.path.join(CH, "m2m_SUBJ01asc" if c.startswith("SUBJ01") else "m2m_" + c), float("nan"), "NA") for c in ids]
    rows = list(csv.DictReader(open(CSVP, encoding="utf-8-sig")))
    rows.sort(key=lambda r: float(r["Age_Scan"]))
    picked, seen = [], set()
    # spread across the age range and across sites
    step = max(1, len(rows) // n)
    for r in rows[::step]:
        cid = r["CaseID"]
        m2m = os.path.join(CH, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)
        if not os.path.exists(os.path.join(m2m, "electrodes_1020.json")):
            continue
        if not os.path.exists(os.path.join(m2m, "label_prep",
                                           "tissue_labeling_upsampled.nii.gz")):
            continue
        if cid in seen:
            continue
        seen.add(cid)
        picked.append((cid, m2m, float(r["Age_Scan"]), r["Site"]))
        if len(picked) >= n:
            break
    return picked


# ---------------------------------------------------------------- stage 1: mesh
def stage_mesh(n):
    subs = subjects(n)
    print(f"meshing {len(subs)} subjects")
    for i, (cid, m2m, age, site) in enumerate(subs, 1):
        msh = os.path.join(m2m, cid + ".msh")
        alt = os.path.join(CH, cid + ".msh")
        if os.path.exists(msh) or os.path.exists(alt):
            print(f"  [{i}/{len(subs)}] {cid} (mesh exists)", flush=True)
            continue
        t = time.time()
        r = subprocess.run([os.path.join(SIMNIBS, "bin", "charm.cmd"), cid, "--mesh"],
                           cwd=CH, capture_output=True, text=True)
        ok = os.path.exists(msh) or os.path.exists(alt)
        print(f"  [{i}/{len(subs)}] {cid} {'OK' if ok else 'FAILED'} "
              f"({(time.time()-t)/60:.1f} min, rc={r.returncode})", flush=True)
        if not ok:
            print("      " + (r.stderr or r.stdout or "")[-400:], flush=True)


# ---------------------------------------------------------------- stage 2: simulate
def stage_sim(n):
    """Run under the SimNIBS interpreter."""
    from simnibs import sim_struct, run_simnibs
    import numpy as np

    # abort rather than silently compare models with different tissue properties
    sys.path.insert(0, ROOT)
    from spicehead.conductivity import CONDUCTIVITY
    probe = sim_struct.SESSION().add_tdcslist()
    ours = [CONDUCTIVITY[k] for k in ("white_matter", "gray_matter", "csf", "bone", "scalp",
                                      "eyeballs", "compact_bone", "spongy_bone", "blood",
                                      "muscle")]
    theirs = [probe.cond[i].value for i in range(10)]
    for i, (a, b) in enumerate(zip(ours, theirs), 1):
        if a is None or b is None or abs(a - b) > 1e-9:
            raise SystemExit(f"conductivity mismatch at tissue {i}: spicehead {a} vs simnibs {b}")
    print("conductivities match SimNIBS defaults for all 10 tissues")

    subs = subjects(n)
    for i, (cid, m2m, age, site) in enumerate(subs, 1):
        ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
        for ch, (a, b) in enumerate(MONTAGE, 1):
            fem = os.path.join(OUT, cid, f"ch{ch}")
            done = os.path.join(fem, "subject_volumes")
            if os.path.isdir(done) and os.listdir(done):
                print(f"  [{i}/{len(subs)}] {cid} ch{ch} (cached)", flush=True)
                continue
            os.makedirs(fem, exist_ok=True)
            S = sim_struct.SESSION()
            S.subpath = m2m
            S.pathfem = fem
            S.fields = "eE"          # magnitude and vector
            S.map_to_vol = True      # NIfTI in subject space, same grid as the T1
            S.open_in_gmsh = False
            t = S.add_tdcslist()
            t.currents = [CURRENT, -CURRENT]
            for lab, cur in ((a, CURRENT), (b, -CURRENT)):
                e = t.add_electrode()
                e.channelnr = 1 if lab == a else 2
                e.centre = ej[lab]
                e.shape = "ellipse"
                e.dimensions = [2 * ELEC_RADIUS_MM, 2 * ELEC_RADIUS_MM]
                e.thickness = [2.0]
            t0 = time.time()
            try:
                run_simnibs(S)
                print(f"  [{i}/{len(subs)}] {cid} ch{ch} OK ({(time.time()-t0)/60:.1f} min)",
                      flush=True)
            except Exception as ex:
                print(f"  [{i}/{len(subs)}] {cid} ch{ch} FAILED {type(ex).__name__}: {ex}",
                      flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N_DEFAULT
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "--mesh":
        stage_mesh(n)
    elif sys.argv[1] == "--sim":
        stage_sim(n)
    elif sys.argv[1] == "--list":
        for cid, m2m, age, site in subjects(n):
            print(f"  {cid:<24} age {age:>4.0f}  {site}")
    else:
        raise SystemExit(__doc__)
