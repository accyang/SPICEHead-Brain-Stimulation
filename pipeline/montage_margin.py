"""
How well separated is each region's "optimal" montage?

Changing the focality metric (grey-matter-only, 20 mm halo) reshuffled the winning montage in ~63%
of AAL regions. That is either alarming (the optimiser is unstable) or benign (many montages are
near-tied and the winner is arbitrary among them). This distinguishes the two:

  1. rank-1 vs rank-2 margin per region;
  2. how many of the retained top-10 sit within 1 / 2 / 5 % of the winner;
  3. electrode overlap between rank 1 and rank 2 (a "different" montage sharing 3 of 4 electrodes
     is not really a different recommendation);
  4. THE DECISIVE ONE - for every region whose winner changed between metrics, was the new winner
     already in the OLD metric's top-10, and at what margin? If yes, the reshuffle is reordering
     within a near-tied set, not the optimiser picking something genuinely different.

  python montage_margin.py [new_topk.json] [old_topk.json]
"""
import os
import sys
import json

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABL = os.path.join(ROOT, "results", "montage_optimization", "ablation")


def key(pairs):
    """Canonical montage id: pair order and within-pair order are both irrelevant."""
    return frozenset(frozenset(p) for p in pairs)


def elecs(pairs):
    return set(pairs[0]) | set(pairs[1])


def main(new_path, old_path):
    new = {r["name"]: r for r in json.load(open(new_path))["regions"]}
    old = {r["name"]: r for r in json.load(open(old_path))["regions"]}
    common = sorted(set(new) & set(old))
    print(f"regions: new={len(new)} old={len(old)} common={len(common)}\n")

    # ---- 1-3: separation within the new metric ----
    marg, w1, w2, w5, ov = [], 0, 0, 0, []
    for n in common:
        top = new[n]["top"]
        if len(top) < 2:
            continue
        f1, f2 = top[0]["focality"], top[1]["focality"]
        marg.append(100 * (f1 - f2) / f1)
        w1 += sum(1 for t in top if t["focality"] >= f1 * 0.99)
        w2 += sum(1 for t in top if t["focality"] >= f1 * 0.98)
        w5 += sum(1 for t in top if t["focality"] >= f1 * 0.95)
        ov.append(len(elecs(top[0]["pairs"]) & elecs(top[1]["pairs"])))
    marg = np.array(marg); ov = np.array(ov)
    k = len(new[common[0]]["top"])
    print(f"rank-1 vs rank-2 margin (n={len(marg)} regions):")
    print(f"   median {np.median(marg):.2f}%   IQR [{np.percentile(marg,25):.2f}, "
          f"{np.percentile(marg,75):.2f}]   max {marg.max():.2f}%")
    print(f"   regions with margin <1%: {int((marg<1).sum())}/{len(marg)}"
          f"   <2%: {int((marg<2).sum())}   <5%: {int((marg<5).sum())}")
    print(f"\nnear-ties per region (top-{k} retained, so these are LOWER BOUNDS):")
    print(f"   within 1% of winner: {w1/len(marg):.1f} montages   "
          f"2%: {w2/len(marg):.1f}   5%: {w5/len(marg):.1f}")
    print(f"\nelectrodes shared between rank 1 and rank 2 (of 4): "
          f"median {int(np.median(ov))}, mean {ov.mean():.2f}")
    print(f"   share >=3 of 4: {int((ov>=3).sum())}/{len(ov)} regions   "
          f"share 0: {int((ov==0).sum())}")

    # ---- 4: is a reshuffled winner just a near-tie in the other metric? ----
    changed = [n for n in common if key(new[n]["pairs"]) != key(old[n]["pairs"])]
    print(f"\n{'='*66}\nwinner changed between metrics: {len(changed)}/{len(common)} regions")
    ranks, lost = [], 0
    for n in changed:
        oldtop = {key(t["pairs"]): (i, t["focality"]) for i, t in enumerate(old[n]["top"])}
        hit = oldtop.get(key(new[n]["pairs"]))
        if hit is None:
            lost += 1
        else:
            i, f = hit
            f1 = old[n]["top"][0]["focality"]
            ranks.append((i + 1, 100 * (f1 - f) / f1))
    print(f"   new winner WAS in the old top-{k}: {len(ranks)}/{len(changed)}")
    print(f"   new winner NOT in old top-{k}:     {lost}/{len(changed)}")
    if ranks:
        rk = np.array([r for r, _ in ranks]); df = np.array([d for _, d in ranks])
        print(f"   its old rank: median {int(np.median(rk))}, "
              f"{int((rk<=3).sum())}/{len(rk)} were in the old top 3")
        print(f"   how far below the old winner it scored: median {np.median(df):.2f}%, "
              f"{int((df<2).sum())}/{len(df)} within 2%")
    verdict = ("near-tie reordering - the winner is arbitrary among a cluster of "
               "near-equivalent montages") if ranks and np.median(
        [d for _, d in ranks]) < 3 and lost < 0.3 * max(len(changed), 1) else \
        "genuine divergence - the two metrics prefer materially different montages"
    print(f"\n   VERDICT: {verdict}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ABL, "topk_gm_on.json"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(ABL, "topk_gmwm_off.json"))
