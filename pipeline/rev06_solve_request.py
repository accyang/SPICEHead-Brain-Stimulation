# -*- coding: utf-8 -*-
"""minimum_new_solve_request.csv for A2 of the Rev0.5 analysis request: the 28 SPICEHead cohort montages the archive did not keep, with cost
measured from the one-head desktop pilot (C:/spicehead_v23_logs/rev06_pilot) and the archived H200 solve times. Nothing is submitted here.
    python pipeline/rev06_solve_request.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_v23", "v23_uq_recollect_2026-11-03")
SU_PER_CORE_H, CPUS_PER_GPU, OCC = 2.5, 2, 0.77                        # pipeline/nchc/estimate_su.py
BUILD_S, MODEL_REBUILDS, SETUP_S = 28.0, 44, 40.0                      # pilot: 19-28 s per rebuilt model; 40 conductivity + 4 segmentation rebuilds


def main():
    p = json.load(open(os.path.join(OUT, "uq_sp_protocols_28.json"), encoding="utf-8"))["targets"]
    inv = pd.read_csv(os.path.join(OUT, "archive_inventory.csv"))
    a = inv[(inv["set"] == "runs_atlas") & (inv["run"] != "mask_erode") & (inv["run"] != "mask_dilate")]
    ext = a[~a["head"].isin(open(os.path.join(ROOT, "results_v23", "v23_uq", "uq_heads.txt")).read().split())]
    per_head = ext.groupby("head").t_solve_s.sum()
    h50 = (per_head.median() + MODEL_REBUILDS * BUILD_S + SETUP_S) / 3600
    h95 = (per_head.quantile(0.95) + MODEL_REBUILDS * BUILD_S + SETUP_S) / 3600
    su = lambda h: 300 * h / OCC * CPUS_PER_GPU * SU_PER_CORE_H
    rows = [dict(target=t, tier=v["tier"], spicehead_montage=v["spicehead"], fem_control_montage=v["fem"], heads=300, physical_runs_per_head=74, nominal_solves_per_head=1, mask_rescorings_per_head=2,
                 basis_fields_per_model=18, note="same seeds, perturbations, solver tolerance and masks as the uncertainty run; all 28 share one set of solves per head") for t, v in p.items()]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "minimum_new_solve_request.csv"), index=False)
    est = dict(status="NOT SUBMITTED: awaits the PI's approval of this cost", targets=len(rows), heads=300, models_solved_per_head=75, reason_not_recoverable="perturbed basis fields were never written; fixed scores were saved only for the FEM cohort choice, FEM cohort unconstrained and each head's own nominal choices",
               pilot=dict(head="h002", hardware="desktop RTX 3080, 8 logical CPUs at below-normal priority", seconds_per_model=[109, 113], seconds_per_rebuild=[19, 28], control="FEM montage F, T, O reproduced the archive exactly (0 relative difference) in 28 targets and 7 evaluations", storage_kb_per_run=9),
               nano4_h200=dict(hours_per_head_median=round(h50, 2), hours_per_head_p95=round(h95, 2), cpus_per_gpu=CPUS_PER_GPU, occupancy=OCC, su_at_median=round(su(h50)), su_if_every_head_at_p95=round(su(h95)), elapsed_hours_16_nodes=round(300 * h50 / OCC / (16 * 8), 1), source="archived t_solve_s of the 265 extension heads, H200 GPUs shared within a node, plus pilot rebuild times"),
               storage_gb_total=round(300 * 77 * 9e3 / 1e9, 2),
               options=dict(nano4="Code, tarballs and target files were deployed for the uncertainty run; needs the PI to reopen the ssh session and a check that the target files are still there. No upload expected.",
                            nano5="No SPICEHead install or subject data; needs environment install, a PI-run upload of 300 heads and a measured pilot; its billing was not measured.",
                            forerunner="CPU only for this code: about 105 min per model, about 131 h per head; not practical.",
                            desktop=f"About {(75 * 111 + MODEL_REBUILDS * 25 + SETUP_S) / 3600:.1f} h per head, about {300 * (75 * 111 + MODEL_REBUILDS * 25 + SETUP_S) / 3600 / 24:.0f} days continuous on the GPU; not practical."))
    json.dump(est, open(os.path.join(OUT, "minimum_new_solve_estimate.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(est["nano4_h200"], indent=1)); print(est["options"]["desktop"])


if __name__ == "__main__":
    main()
