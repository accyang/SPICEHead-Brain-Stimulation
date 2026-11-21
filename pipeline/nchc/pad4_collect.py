# -*- coding: utf-8 -*-
r"""Collect the four-pad FEM check (review decision D4, 31 Oct / 1 Nov 2026) for the ref6 cohort montages in every head.

Three FEM values per head x target (same grid, masks, envelope, currents and conductivities):
  4pad   direct SimNIBS runs with only the four active pads, on the mesh rebuilt on Nano 4        (pad4 JSON "targets")
  19s    19-pad leadfield superposition on the SAME rebuilt mesh                                  (pad4 JSON "targets19")
  19a    19-pad leadfield superposition on the archived mesh = Paper A's FEM reference            (<code>__fem.json)
Ratios: pad effect = 4pad/19s (passive pads, one mesh); re-mesh effect = 19s/19a (charm meshing is not bit-reproducible);
total = 4pad/19a. Inputs: <runs>/<code>__pad4.json (fetched by wsl_pad4_nano4.sh) and
L:\...\nchc_fullsearch\heads\<code>\<code>__fem.json (rois[target][montage] = [F, T, O]).
Output: <out>/{per_head.csv, summary.csv, SUMMARY.md, PROVENANCE.json}; h-codes only.
    python pipeline/nchc/pad4_collect.py [--runs DIR] [--out DIR]
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "results_v23", "v23_pad4_check")
RUNS = os.path.join(OUT, "nano4", "results")
HEADS = r"L:\BrainProbe\SPICEhead\nchc_fullsearch\heads"
MESH_SHA_LOCAL = r"C:\spicehead_v22_logs\pad4_nano4\mesh_sha_local.txt"
MONTAGES = os.path.join(ROOT, "results_v23", "v23_uq", "uq_fixed_montages.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pct(x, q):
    return round(float(np.percentile(x, q)), 4)


def main():
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    runs, out = arg("--runs", RUNS), arg("--out", OUT)
    os.makedirs(out, exist_ok=True)
    order = list(json.load(open(MONTAGES))["montages"])
    local_sha = {}
    if os.path.exists(MESH_SHA_LOCAL):
        for line in open(MESH_SHA_LOCAL):
            parts = line.split()
            if len(parts) == 2:
                local_sha[os.path.basename(parts[1]).replace(".msh", "")] = parts[0]
    files = sorted(f for f in os.listdir(runs) if f.endswith("__pad4.json"))
    rows, n_lf, n_same_mesh, minutes = [], 0, 0, []
    for f in files:
        j = json.load(open(os.path.join(runs, f)))
        code = j["code"]
        fem = json.load(open(os.path.join(HEADS, code, f"{code}__fem.json")))["rois"]
        t19 = j.get("targets19")
        n_lf += t19 is not None
        same = local_sha.get(code) is not None and j.get("mesh_sha256") == local_sha.get(code)
        n_same_mesh += same
        minutes.append(j.get("minutes", np.nan))
        for tgt, r in j["targets"].items():
            m = r["montage"]
            ref = fem[tgt].get(m) or fem[tgt].get(" / ".join(m.split(" / ")[::-1]))
            if ref is None:
                continue
            F19a, T19a, O19a = ref
            row = dict(code=code, target=tgt, montage=m, n_roi=r["n_roi"], mesh_identical_to_archive=int(same),
                       T_4pad=r["T"], O_4pad=r["O"], F_4pad=r["F"], T_19pad_archived=T19a, O_19pad_archived=O19a, F_19pad_archived=F19a,
                       total_T_ratio=r["T"] / T19a, total_O_ratio=r["O"] / O19a, total_F_ratio=r["F"] / F19a)
            if t19 is not None and tgt in t19:
                s = t19[tgt]
                row.update(T_19pad_samemesh=s["T"], O_19pad_samemesh=s["O"], F_19pad_samemesh=s["F"],
                           pad_T_ratio=r["T"] / s["T"], pad_O_ratio=r["O"] / s["O"], pad_F_ratio=r["F"] / s["F"],
                           remesh_T_ratio=s["T"] / T19a, remesh_O_ratio=s["O"] / O19a, remesh_F_ratio=s["F"] / F19a)
            rows.append(row)
    if not rows:
        raise SystemExit(f"no pad4 JSON with matching fem.json under {runs}")
    cols = ["code", "target", "montage", "n_roi", "mesh_identical_to_archive", "T_4pad", "O_4pad", "F_4pad",
            "T_19pad_samemesh", "O_19pad_samemesh", "F_19pad_samemesh", "T_19pad_archived", "O_19pad_archived", "F_19pad_archived",
            "pad_T_ratio", "pad_O_ratio", "pad_F_ratio", "remesh_T_ratio", "remesh_O_ratio", "remesh_F_ratio",
            "total_T_ratio", "total_O_ratio", "total_F_ratio"]
    with open(os.path.join(out, "per_head.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    def block(kind, title):
        """kind in pad / remesh / total -> summary rows + markdown table lines"""
        srows, lines = [], []
        for tgt in sorted({r["target"] for r in rows}, key=lambda t: order.index(t) if t in order else 99):
            R = [r for r in rows if r["target"] == tgt and f"{kind}_T_ratio" in r]
            if not R:
                continue
            tr, orr, fr = (np.array([r[f"{kind}_{k}_ratio"] for r in R]) for k in ("T", "O", "F"))
            s = dict(effect=kind, target=tgt, montage=R[0]["montage"], heads=len(R),
                     T_ratio_median=round(float(np.median(tr)), 4), T_ratio_p5=pct(tr, 5), T_ratio_p95=pct(tr, 95),
                     O_ratio_median=round(float(np.median(orr)), 4), F_ratio_median=round(float(np.median(fr)), 4),
                     F_ratio_p5=pct(fr, 5), F_ratio_p95=pct(fr, 95),
                     F_within_3pct_share=round(100 * float(np.mean(np.abs(fr - 1) <= 0.03)), 1),
                     T_within_5pct_share=round(100 * float(np.mean(np.abs(tr - 1) <= 0.05)), 1))
            srows.append(s)
            lines.append(f"| {tgt} | {s['montage']} | {s['heads']} | {s['T_ratio_median']:.3f} ({s['T_ratio_p5']:.3f}-{s['T_ratio_p95']:.3f}) | "
                         f"{s['O_ratio_median']:.3f} | {s['F_ratio_median']:.3f} ({s['F_ratio_p5']:.3f}-{s['F_ratio_p95']:.3f}) | {s['F_within_3pct_share']:.0f}% |")
        if not srows:
            return srows, []
        allT = np.array([r[f"{kind}_T_ratio"] for r in rows if f"{kind}_T_ratio" in r])
        allF = np.array([r[f"{kind}_F_ratio"] for r in rows if f"{kind}_F_ratio" in r])
        head = [f"## {title}", "",
                f"All head-targets (n = {len(allT)}): T ratio median {np.median(allT):.3f} (5th-95th {np.percentile(allT, 5):.3f}-{np.percentile(allT, 95):.3f}); "
                f"F ratio median {np.median(allF):.3f} ({np.percentile(allF, 5):.3f}-{np.percentile(allF, 95):.3f}); "
                f"F within 3% in {100 * np.mean(np.abs(allF - 1) <= 0.03):.0f}% and T within 5% in {100 * np.mean(np.abs(allT - 1) <= 0.05):.0f}% of head-targets.", "",
                "| Target | Cohort montage | Heads | T ratio, median (5th-95th) | O ratio, median | F ratio, median (5th-95th) | F within 3% |",
                "|---|---|---|---|---|---|---|", *lines, ""]
        return srows, head

    s_pad, md_pad = block("pad", "Passive-pad effect: four active pads / 19-pad leadfield, same rebuilt mesh")
    s_rm, md_rm = block("remesh", "Re-mesh effect: 19-pad leadfield on the rebuilt mesh / 19-pad leadfield on the archived mesh (Paper A reference)")
    s_tot, md_tot = block("total", "Total: four active pads on the rebuilt mesh / archived 19-pad reference")
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        allrows = s_pad + s_rm + s_tot
        w = csv.DictWriter(fh, fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
    md = [f"# Four-pad FEM check of the ref6 cohort montages: {len(files)} heads", "",
          f"Collected {time.strftime('%Y-%m-%d %H:%M')} by `pipeline/nchc/pad4_collect.py` from `{runs}`. Per head: the mesh was rebuilt on Nano 4 "
          f"(`charm --mesh`, SimNIBS 4.6; identical to the archived mesh in {n_same_mesh} of {len(files)} heads), then direct SimNIBS runs with only "
          "the four active pads (two per montage, 24 mm pads, 2 mm, default conductivities) and, in "
          f"{n_lf} of {len(files)} heads, a 19-pad leadfield on the same mesh; both scored on the grid, masks and envelope of the 300-head comparison. "
          f"Median wall time per head {np.nanmedian(minutes):.0f} min (8 threads).", "", *md_pad, *md_rm, *md_tot]
    open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(dict(collected=time.strftime("%Y-%m-%d %H:%M"), runs=runs, n_files=len(files), n_with_leadfield=n_lf, n_mesh_identical=n_same_mesh,
                   fem_json_sha={f[:4]: sha(os.path.join(HEADS, f[:4], f"{f[:4]}__fem.json")) for f in files[:3]},
                   note="first three fem.json SHAs recorded; the script reads every head's file"),
              open(os.path.join(out, "PROVENANCE.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
