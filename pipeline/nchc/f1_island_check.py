# -*- coding: utf-8 -*-
"""Which v23 networks contained islands linked only through zero-conductance edges (27 Oct 2026)?

Before the fix of 27 Oct, spicehead.homogenize stored a zero conductance as an edge between neighbouring blocks, so a
block joined to the network only through non-conducting tubes passed the giant-component filter (h062 at 5 mm: a
two-node island, singular matrix). Run with the PRE-FIX package to see which earlier v23 results could be affected: per
head and grid, the explicit zeros in G and the nodes outside the main component once those zeros are removed.
Output: one CSV line per head and grid (h-codes only).
    SPICEHEAD_MODEL=v23 python f1_island_check.py <heads.csv> <charm root> <out.csv> <dx,dx,...> <shard> <nshards>
"""
import csv
import os
import sys
import time

import numpy as np
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline")]
from spicehead import mri                                                    # noqa: E402


def main():
    heads_csv, charm, out, dxs, shard, nsh = sys.argv[1:7]
    codes = [r["CaseID"] for r in csv.DictReader(open(heads_csv, encoding="utf-8-sig"))][int(shard)::int(nsh)]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["code", "dx_mm", "nodes", "explicit_zeros", "components_without_zeros", "nodes_outside_main", "seconds"])
        for c in codes:
            for dx in (float(x) for x in dxs.split(",")):
                t0 = time.time()
                model, _ = mri.head_model(os.path.join(charm, f"m2m_{c}", "label_prep", "tissue_labeling_upsampled.nii.gz"), dx)
                G = model.G.tocsr()
                zeros = int((G.data == 0).sum())
                G.eliminate_zeros()
                n, comp = connected_components(G, directed=False)
                outside = int(G.shape[0] - np.bincount(comp).max())
                w.writerow([c, dx, G.shape[0], zeros, n, outside, round(time.time() - t0, 1)]); fh.flush()
                del model, G


if __name__ == "__main__":
    main()
