"""Compare GPU scoring with an existing CPU score file for one head and model.

  SPICEHEAD_SCORE=gpu python pipeline/gpu_score_check.py <code> <model: sp2|sp1|fem>

Needs the head's basis and grid caches (SPICEHEAD_FULLSEARCH_CACHE) and the CPU result
<out>/<code>__<model>.json. Reports max relative differences of F, T and O over all montages and
targets, winner agreement per target, and the two timings. Writes <out>/<code>__<model>__gpucheck.json.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("SPICEHEAD_MODEL", "v22")
import v22_fem_fullsearch as FS

code, model = sys.argv[1], sys.argv[2]
m2m = dict(FS.heads())[code]
od = FS.out_dir()
cpu = json.load(open(os.path.join(od, f"{code}__{model}.json"), encoding="utf-8"))
os.environ["SPICEHEAD_SCORE"] = "gpu"
t0 = time.time()
gpu = FS.score_head(code, m2m, model)
t_gpu = time.time() - t0
rep = {"subject": code, "model": model, "t_cpu_s": cpu.get("t_score_s"), "t_gpu_s": round(t_gpu, 1), "targets": {}}
worst = {"F": 0.0, "T": 0.0, "O": 0.0}
for r in FS.ROI_ORDER:
    a, b = cpu["rois"][r], gpu["rois"][r]
    keys = sorted(set(a) & set(b))
    diffs = {q: 0.0 for q in worst}
    for k in keys:
        for qi, q in enumerate(("F", "T", "O")):
            den = abs(a[k][qi]) or 1.0
            diffs[q] = max(diffs[q], abs(a[k][qi] - b[k][qi]) / den)
    wa = max(keys, key=lambda k: a[k][0]); wb = max(keys, key=lambda k: b[k][0])
    rep["targets"][r] = {"n": len(keys), "missing": len(set(a) ^ set(b)), "max_rel_diff": diffs,
                         "winner_cpu": wa, "winner_gpu": wb, "same_winner": wa == wb}
    for q in worst:
        worst[q] = max(worst[q], diffs[q])
    print(f"{r:<15} n={len(keys)} maxrel F {diffs['F']:.1e} T {diffs['T']:.1e} O {diffs['O']:.1e} | winner same={wa == wb} ({wa})")
rep["worst"] = worst
print(f"CPU {rep['t_cpu_s']} s vs GPU {t_gpu:.1f} s | worst relative differences {worst}")
json.dump(rep, open(os.path.join(od, f"{code}__{model}__gpucheck.json"), "w"), indent=1)
