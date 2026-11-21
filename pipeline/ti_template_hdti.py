# -*- coding: utf-8 -*-
"""Exploratory: multi-electrode current steering for TI ("HD-TI") on the template, 10-10 cap.

Two-pair TI drives one electrode pair per channel. Here each of the two channels may drive up to
MAX_E of the 76 10-10 positions with signed currents, so each channel's field can be shaped. The
model is linear, so a channel's field is a weighted sum of the 75 single-electrode basis fields
already cached by ti_template_1010.py (relative to Cz); no new head solve is needed.

Constraints (same current budget as two-pair TI at 1 mA per channel):
  each channel's currents sum to zero and inject 1 mA (sum |c| = 2 mA); at most MAX_E electrodes
  per channel; an electrode belongs to one channel only; active pads at least 25 mm apart (12 mm
  pads do not overlap); within a channel, pads of opposite polarity at least 40 mm apart (the pair
  rule of the two-pair searches); every active electrode carries at least MIN_I = 0.05 mA

Objective: F = median target GM envelope / p99 of GM more than 20 mm from the target, exactly as in
the two-pair searches (numpy-compatible median and linear-interpolated percentile, on the GPU).

Search: greedy, on the exact objective. From a start montage, every move that shifts delta mA from
an active electrode to any allowed electrode within the same channel (then rescales the channel to
1 mA) is scored in one batch; the best improving move is taken; when none improves, delta is halved
(0.25 -> 0.01 mA). Starts: the eight best two-pair montages of the 10-10 search for that target and
variant. The result can only be at least as good as its start. Final F is recomputed on the CPU
with the project's numpy envelope.

Targets: left insula, left anterior and posterior insula (approximate geometric split, see
ti_template_1010.SUBREGIONS), left hippocampus. Variants: all 76 positions, and without the
inferior row (positions near ears, face and neck). Exploratory only.

Outputs in `Paper_A_SPICEhead/Brain Stimulation/figures/TI_targets_MNI_hdti/`.

  SPICEHEAD_MODEL=v22 python pipeline/ti_template_hdti.py [targets...]
"""
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)
import ti_template_1010 as X                                         # noqa: E402
import fig_ti_targets_mni as M                                       # noqa: E402
from spicehead.envelope_fast import envelope_max_direction_flat      # noqa: E402

TARGETS = {"insula_L": X.TARGETS["insula_L"], "insula_ant_L": X.SUBREGIONS["insula_ant_L"],
           "insula_post_L": X.SUBREGIONS["insula_post_L"], "hippocampus_L": X.TARGETS["hippocampus_L"]}
SCREEN = {"insula_L": "", "hippocampus_L": "", "insula_ant_L": "_sub", "insula_post_L": "_sub"}
OUT = os.path.join(os.path.dirname(M.OUT), "TI_targets_MNI_hdti")
I_MA = 1.0
MAX_E = 6
MIN_PAD_MM, MIN_OPP_MM = 25.0, 40.0
DELTAS = [0.25, 0.12, 0.06, 0.03, 0.015]
N_STARTS, BATCH = 8, 192
MIN_I = 0.05                    # mA: smallest current an active electrode may carry
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def envelope_t(a, b, eps=1e-20):
    """Torch version of envelope_fast.envelope_max_direction_flat; components on dim -2."""
    na2, nb2 = (a * a).sum(-2), (b * b).sum(-2)
    swap = (nb2 > na2).unsqueeze(-2)
    a, b = torch.where(swap, b, a), torch.where(swap, a, b)
    na2, nb2 = torch.maximum(na2, nb2), torch.minimum(na2, nb2)
    dot = (a * b).sum(-2)
    b = torch.where((dot < 0).unsqueeze(-2), -b, b)
    dot = dot.abs()
    d = a - b
    c = torch.linalg.cross(b, d, dim=-2)
    oblique = 2 * torch.sqrt((c * c).sum(-2)) / torch.sqrt((d * d).sum(-2) + eps)
    return torch.where(nb2 <= dot, 2 * torch.sqrt(nb2), oblique)


def quantile_np(x, q):
    """numpy-compatible (linear) quantile along the last dim."""
    n = x.shape[-1]
    pos = q * (n - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    vlo = torch.kthvalue(x, lo + 1, dim=-1).values
    vhi = vlo if hi == lo else torch.kthvalue(x, hi + 1, dim=-1).values
    return vlo + (pos - lo) * (vhi - vlo)


class Scorer:
    def __init__(self, B, tsel, osel):
        self.B = torch.as_tensor(B, device=DEV)                        # (E, 3, N) per mA
        self.t = torch.as_tensor(np.flatnonzero(tsel), device=DEV)
        self.o = torch.as_tensor(np.flatnonzero(osel), device=DEV)

    def fields(self, C):                                               # C (2, E) -> (2, 3, N)
        return torch.einsum("ke,ecn->kcn", torch.as_tensor(C, device=DEV, dtype=torch.float32), self.B)

    def F(self, E1, E2):                                               # batched (b, 3, N)
        env = envelope_t(E1, E2)
        T = quantile_np(env[..., self.t], 0.5)
        O = quantile_np(env[..., self.o], 0.99)
        return T / O


def legal(c_new, c_other, D, allowed):
    """Rules for one channel's new currents given the other channel's."""
    act = np.flatnonzero(np.abs(c_new) > 1e-9)
    oth = np.flatnonzero(np.abs(c_other) > 1e-9)
    if len(act) < 2 or len(act) > MAX_E or not allowed[act].all():
        return False
    if (np.abs(c_new[act]) < MIN_I - 1e-9).any():
        return False
    s = np.sign(c_new[act])
    if not ((s > 0).any() and (s < 0).any()):
        return False
    if np.intersect1d(act, oth).size or (D[np.ix_(act, oth)] < MIN_PAD_MM).any():
        return False
    Da = D[np.ix_(act, act)] + np.eye(len(act)) * 1e9
    if (Da < MIN_PAD_MM).any():
        return False
    opp = s[:, None] != s[None]
    return not ((Da < MIN_OPP_MM) & opp).any()


def greedy(C0, sc, D, allowed, log=None):
    """Greedy current shifting on the exact F; returns currents and F."""
    C = C0.copy()
    E = sc.fields(C)
    F = float(sc.F(E[0:1], E[1:2])[0])
    nE = C.shape[1]
    for delta in DELTAS:
        while True:
            cands = []                                                 # (channel, from, to, scale, new currents)
            for k in (0, 1):
                for i in np.flatnonzero(np.abs(C[k]) > 1e-9):
                    c = C[k].copy()                            # drop electrode i
                    c[i] = 0.0
                    c -= c.sum() / max(1, np.count_nonzero(c)) * (c != 0)
                    if np.abs(c).sum() > 1e-9:
                        c *= 2 * I_MA / np.abs(c).sum()
                        if legal(c, C[1 - k], D, allowed):
                            cands.append((k, c))
                    for j in range(nE):
                        if j == i or not allowed[j]:
                            continue
                        for step in (delta, -delta):                   # shift current from i to j
                            c = C[k].copy()
                            c[i] -= step
                            c[j] += step                               # the sum stays zero
                            c[np.abs(c) < 1e-6] = 0.0
                            c -= c.sum() / max(1, np.count_nonzero(c)) * (c != 0)
                            l1 = np.abs(c).sum()
                            if l1 < 1e-9:
                                continue
                            c *= 2 * I_MA / l1
                            if legal(c, C[1 - k], D, allowed):
                                cands.append((k, c))
            if not cands:
                break
            best_F, best = F, None
            Bm = sc.B
            for s in range(0, len(cands), BATCH):
                chunk = cands[s:s + BATCH]
                Cb = torch.as_tensor(np.stack([c for _, c in chunk]), device=DEV, dtype=torch.float32)
                newE = torch.einsum("be,ecn->bcn", Cb, Bm)
                ks = torch.as_tensor([k for k, _ in chunk], device=DEV)
                E1 = torch.where((ks == 0)[:, None, None], newE, E[0].expand_as(newE))
                E2 = torch.where((ks == 1)[:, None, None], newE, E[1].expand_as(newE))
                Fb = sc.F(E1, E2)
                m = int(torch.argmax(Fb))
                if float(Fb[m]) > best_F + 1e-6:
                    best_F, best = float(Fb[m]), chunk[m]
            if best is None:
                break
            k, c = best
            C[k] = c
            E = sc.fields(C)
            F = best_F
            if log:
                log(F, delta, C)
    return C, F


def exact_np(C, B_np, tsel, osel):
    E = [np.einsum("e,ecn->cn", c, B_np) for c in C]
    env = envelope_max_direction_flat(*E[0], *E[1])
    T, O = float(np.median(env[tsel])), float(np.percentile(env[osel], 99))
    return T / O, T, O


def main():
    which = [a for a in sys.argv[1:] if a in TARGETS] or list(TARGETS)
    X.TARGETS = TARGETS
    S = X.setup()
    labs = S["labs"]
    pairs, D = X.pairs_and_dist(S)
    gm = S["gm"]
    bidx_gm = np.flatnonzero(S["gm"].ravel()[S["brain"].ravel()])
    B_np = X.load_basis(S, bidx_gm)                                    # (76, 3, N_gm); Cz row = 0
    inferior = np.array([l in X.INFERIOR for l in labs])

    t1img = nib.load(os.path.join(M.M2M, "T1.nii.gz"))
    lab05 = nib.load(os.path.join(M.M2M, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    lab1 = np.rint(np.asanyarray(resample_from_to(lab05, t1img, order=0).dataobj)).astype(int)
    brain1 = np.isin(lab1, (M.WM, M.GM))
    aal1 = np.rint(np.asanyarray(resample_from_to(nib.load(M.AAL), t1img, order=0).dataobj)).astype(int)
    value = {r["name"]: r["value"] for r in json.load(open(M.AAL_REGIONS))["regions"]}
    Bfull = None
    os.makedirs(OUT, exist_ok=True)
    summary = os.path.join(OUT, "TI_targets_MNI152_summary.json")
    rows = json.load(open(summary, encoding="utf-8"))["targets"] if os.path.exists(summary) else []
    for key in which:
        t = TARGETS[key]
        tsel, osel = S["tmask"][key][gm], S["omask"][key][gm]
        sc = Scorer(B_np, tsel, osel)
        z = np.load(os.path.join(X.CACHE, f"screen_scores{SCREEN[key]}.npz"))
        kk = list(z["keys"]).index(key)
        I, J, Fsc = z["I"], z["J"], z["F"][:, kk]
        inf_m = inferior[[a for a, _ in pairs]] | inferior[[b for _, b in pairs]]
        for variant, allowed in (("all", np.ones(len(labs), bool)), ("no_inferior", ~inferior)):
            ok = np.ones(len(I), bool) if variant == "all" else ~(inf_m[I] | inf_m[J])
            cand = np.flatnonzero(ok)
            top = cand[np.argsort(-Fsc[cand])[:60]]                    # rescore the screening top 60
            exact2 = []
            for m in top:
                C = np.zeros((2, len(labs)), np.float32)
                (a, b), (c, d) = pairs[I[m]], pairs[J[m]]
                C[0, a], C[0, b], C[1, c], C[1, d] = 1, -1, 1, -1
                E = sc.fields(C)
                exact2.append((float(sc.F(E[0:1], E[1:2])[0]), C))
            exact2.sort(key=lambda e: -e[0])
            F2, _ = exact2[0]
            t0 = time.time()
            best = None
            for s_i, (Fs, C0) in enumerate(exact2[:N_STARTS]):
                C, F = greedy(C0, sc, D, allowed)
                if best is None or F > best[1]:
                    best = (C, F, s_i + 1, Fs)
            C, Fg, start, Fstart = best
            F, T, O = exact_np(C, B_np, tsel, osel)
            ch = [[(labs[e], float(C[k_, e])) for e in np.flatnonzero(np.abs(C[k_]) > 1e-9)] for k_ in (0, 1)]
            order = lambda lst: sorted(lst, key=lambda x: -abs(x[1]))
            fmt = lambda lst: ", ".join(f"{l} {c:+.2f}" for l, c in order(lst))
            rid = key if variant == "all" else f"{key}_noinf"
            row = dict(target=rid, title=t["title"] + ("" if variant == "all" else " (no inferior row)"),
                       variant=variant,
                       montage=" / ".join(", ".join(l for l, _ in order(c_)) for c_ in ch),
                       pair1="+".join(l for l, _ in ch[0]), pair2="+".join(l for l, _ in ch[1]),
                       currents_ch1_mA=fmt(ch[0]), currents_ch2_mA=fmt(ch[1]),
                       currents_ch1=[[l, round(c, 6)] for l, c in order(ch[0])],     # full precision
                       currents_ch2=[[l, round(c, 6)] for l, c in order(ch[1])],
                       n_electrodes=[len(ch[0]), len(ch[1])], F=round(F, 3), T_V_per_m=round(T, 4),
                       O_V_per_m=round(O, 4), F_two_pair=round(F2, 3), gain_pct=round(100 * (F / F2 - 1), 1),
                       start_rank=start, F_gpu=round(Fg, 4))
            rows = [r for r in rows if r["target"] != rid] + [row]
            print(f"{row['title']:<44} two-pair F={F2:.3f} -> HD-TI F={F:.3f} ({row['gain_pct']:+.1f}%), T={T:.3f} V/m, "
                  f"{len(ch[0])}+{len(ch[1])} electrodes, start {start} ({(time.time() - t0) / 60:.1f} min; GPU F {Fg:.4f})\n"
                  f"    ch1: {row['currents_ch1_mA']}\n    ch2: {row['currents_ch2_mA']}", flush=True)
            if Bfull is None:
                Bfull = X.load_basis(S, np.arange(int(S["brain"].sum())))
            E = [np.einsum("e,ecn->cn", c, Bfull) for c in C]
            env2 = np.zeros(S["labels"].shape, np.float32)
            env2[S["brain"]] = envelope_max_direction_flat(*E[0], *E[1])
            env1 = M.to_1mm(env2, S["aff"], t1img, order=1).astype(np.float32)
            env1[~brain1] = 0.0
            tgt1 = X.region(aal1, value, t, t1img.affine) & (lab1 == M.GM)
            nib.save(nib.Nifti1Image(env1, t1img.affine), os.path.join(OUT, f"TI_envelope_{rid}_MNI152_1mm.nii.gz"))
            nib.save(nib.Nifti1Image(tgt1.astype(np.uint8), t1img.affine), os.path.join(OUT, f"TI_target_{rid}_MNI152_1mm.nii.gz"))
            json.dump(dict(model="SPICEhead template, multi-electrode TI current steering over the 10-10 cap (greedy, exact F)",
                           I_mA_per_channel=I_MA, max_electrodes_per_channel=MAX_E, min_pad_mm=MIN_PAD_MM,
                           min_opposite_mm=MIN_OPP_MM, deltas_mA=DELTAS, targets=rows),
                      open(summary, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
