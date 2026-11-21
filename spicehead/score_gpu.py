"""GPU scoring of every two-pair montage from a basis of single-electrode fields (PyTorch, CUDA).

Reproduces ``v22_fem_fullsearch.score_head`` on the GPU: pair fields E(a,b) = E(a,ref) - E(b,ref) from
the 18 basis fields, the rotation-invariant envelope of ``spicehead.envelope_fast`` (same closed
form, same float32 arithmetic), then for every target T = median of the envelope over the target
gray matter and O = 99th percentile over the off-target gray matter, with numpy's conventions
(median averages the two middle values; percentile uses linear interpolation at 0.99 (n - 1)).

    from spicehead.score_gpu import score_montages_gpu
    res = score_montages_gpu(basis, labs, ref, pairs, montages, roi_sel, out_sel, roi_order, label_of)
    # res[roi][label] = [F, T, O] rounded to 7 significant digits, as the CPU path writes

Memory: the 171 pair fields are held on the device in float32 (about 1.2 GB for 600k GM voxels);
montages are processed in batches. Selected by SPICEHEAD_SCORE=gpu in v22_fem_fullsearch.
"""
from __future__ import annotations

import math
import time

import numpy as np


def _g7(x):
    return float(f"{x:.7g}")


def _envelope(A, B):
    """A, B: (k, 3, n) float32 on device -> (k, n) envelope, same formula as envelope_fast."""
    import torch
    na2 = (A * A).sum(1)
    nb2 = (B * B).sum(1)
    swap = (nb2 > na2)[:, None, :]
    A, B = torch.where(swap, B, A), torch.where(swap, A, B)
    na2, nb2 = torch.where(swap[:, 0], nb2, na2), torch.where(swap[:, 0], na2, nb2)
    dot = (A * B).sum(1)
    B = torch.where((dot < 0)[:, None, :], -B, B)
    dot = dot.abs()
    D = A - B
    C = torch.cross(B, D, dim=1)
    ndiff = torch.sqrt((D * D).sum(1))
    oblique = 2.0 * torch.sqrt((C * C).sum(1)) / ndiff
    res = torch.where(nb2 <= dot, 2.0 * torch.sqrt(nb2), oblique)
    return torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


def _median_sorted(vals):
    """numpy median along dim 1 for a (k, n) tensor (average of the two middle values when n is even)."""
    import torch
    n = vals.shape[1]
    s, _ = torch.sort(vals, dim=1)
    if n % 2:
        return s[:, n // 2]
    return 0.5 * (s[:, n // 2 - 1] + s[:, n // 2])


def _percentile99(vals):
    """numpy percentile(…, 99) along dim 1 (linear interpolation) for a (k, n) tensor."""
    import torch
    n = vals.shape[1]
    pos = 0.99 * (n - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    v_lo = torch.kthvalue(vals, lo + 1, dim=1).values
    if lo + 1 >= n or frac == 0.0:
        return v_lo
    v_hi = torch.kthvalue(vals, lo + 2, dim=1).values
    return v_lo + frac * (v_hi - v_lo)


def score_montages_gpu(basis, labs, ref, pairs, montages, roi_sel, out_sel, roi_order, label_of,
                       batch=48, device=None, verbose=True):
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    idx = {lab: i for i, lab in enumerate(labs)}
    Bz = torch.from_numpy(np.ascontiguousarray(basis, dtype=np.float32)).to(device)      # (18, 3, n)
    zero = torch.zeros_like(Bz[0])
    pair_index = {p: j for j, p in enumerate(pairs)}
    P = torch.stack([(Bz[idx[a]] if a != ref else zero) - (Bz[idx[b]] if b != ref else zero)
                     for a, b in pairs])                                                    # (171, 3, n)
    del Bz
    roi_idx = {r: torch.from_numpy(np.flatnonzero(roi_sel[r])).to(device) for r in roi_order}
    out_idx = {r: torch.from_numpy(np.flatnonzero(out_sel[r])).to(device) for r in roi_order}
    res = {r: {} for r in roi_order}
    labels = [label_of(pa, pb) for pa, pb in montages]
    ia = torch.tensor([pair_index[pa] for pa, _ in montages], device=device)
    ib = torch.tensor([pair_index[pb] for _, pb in montages], device=device)
    nm = len(montages)
    for s in range(0, nm, batch):
        e = min(s + batch, nm)
        env = _envelope(P[ia[s:e]], P[ib[s:e]])                                             # (k, n)
        for r in roi_order:
            if roi_idx[r].numel() == 0:
                continue
            T = _median_sorted(env[:, roi_idx[r]]).double().cpu().numpy()
            O = _percentile99(env[:, out_idx[r]]).double().cpu().numpy()
            for j in range(e - s):
                t, o = float(T[j]), float(O[j])
                res[r][labels[s + j]] = [_g7(t / o if o > 0 else 0.0), _g7(t), _g7(o)]
        if verbose and ((e // batch) % 50 == 0 or e == nm):
            print(f"      GPU scored {e}/{nm} ({time.time()-t0:.0f} s)", flush=True)
    return res
