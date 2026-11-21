# -*- coding: utf-8 -*-
"""3-D see-through views of the optimal 10-20 TI montages for four template targets.

For the insula, hippocampus, anterior cingulate cortex and orbitofrontal cortex, the saved
template results of fig_ti_targets_mni.py (MNI152 template, SPICEhead v22, 2 mm solve resampled to
1 mm, 1 mA per channel, montage chosen to maximize F over all 11,628 two-pair 10-20 montages) are
drawn as perspective renderings of the head:

  scalp and brain    translucent surfaces of the model's own tissue labels (brighter at grazing
                     angles, so the outlines read like an x-ray)
  TI field           the envelope inside the brain (gray and white matter) drawn as a glowing
                     volume: color and opacity rise with field strength, so the focus (the
                     strongest region) glows dark red and weak field is transparent. The scale is
                     each montage's own: 100% = the 99.9th percentile of the brain envelope
                     (single hot voxels would otherwise set it); values below GLOW_S0 of that are
                     not drawn
  target             the bilateral atlas region scored as the target, as a cyan see-through
                     shell (outline plus a faint fill), so the field inside it stays visible
  electrodes         12 mm-radius pads following the scalp; channel 1 dark, channel 2 light;
                     the other 10-20 positions as small gray marks

No field is re-solved: every value comes from the saved NIfTI maps and summary JSON. The renderer
is a small z-buffer rasterizer in numpy (no VTK is installed here): each surface contributes its
front-most fragment; the field is ray-cast (emission-absorption) and split at the target's front
surface; all layers are composited back to front with their opacities.

  SPICEHEAD_MODEL=v22 python pipeline/fig_ti_targets_3d.py [insula hippocampus acc ofc]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from matplotlib.colors import ListedColormap, Normalize
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from skimage.measure import marching_cubes

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)
import fig_ti_targets_mni as M                                       # noqa: E402

OUT = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "figures", "TI_targets_3D")
MM = 1 / 25.4
PAD_R = M.ELEC_RADIUS_MM                         # 12 mm pads, as in the solve
SITE_R = 3.5                                     # marks for the other 10-20 positions
ALL_1010 = False                                 # True: pads and marks at all 76 10-10 positions
FIELD_PCT = 99.9                                 # 100% of the field scale: brain 99.9th percentile
GLOW_S0, GLOW_GAMMA, GLOW_K = 0.3, 3.0, 0.4      # drawn from 30% of the scale; opacity per mm at 100%
GLOW_CMAP = ListedColormap(matplotlib.colormaps["YlOrRd"](np.linspace(0.2, 1.0, 256)))
SKIN, BRAIN, TARGET = "#e9b99c", "#c9c9c9", "#00a6d6"
CH1, CH2, SITE = "#2b2b2b", "#d9d9d9", "#8c8c8c"

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.facecolor": "white"})


def rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# ------------------------------------------------------------------ geometry
def surface(mask, affine, sigma=1.0, step=1):
    """Smoothed isosurface of a binary mask in world (MNI) mm, faces wound outward."""
    vol = gaussian_filter(mask.astype(np.float32), sigma) if sigma else mask.astype(np.float32)
    v, f, _, _ = marching_cubes(vol, 0.5, step_size=step, allow_degenerate=False)
    V = v @ affine[:3, :3].T + affine[:3, 3]
    N = vnormals(V, f)
    if np.sum(N * (V - V.mean(0))) < 0:                               # make normals point outward
        f = f[:, ::-1]
        N = -N
    return V, f, N


def vnormals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def patch(V, F, N, centre, radius, lift=1.2):
    """The scalp faces within `radius` of `centre`, lifted along the normals: a pad that follows
    the head's curvature."""
    inside = np.linalg.norm(V - centre, axis=1) <= radius
    keep = inside[F].all(1)
    used = np.unique(F[keep])
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    return V[used] + lift * N[used], remap[F[keep]], N[used]


# ------------------------------------------------------------------ camera and rasterizer
class View:
    """Perspective camera looking at `centre` from azimuth `az` (0 = anterior, -90 = left) and
    elevation `el` (degrees), `size` pixels square."""

    def __init__(self, centre, az, el, size, dist=640.0, fov=25.0):
        a, e = np.radians(az), np.radians(el)
        self.eye = centre + dist * np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
        f = centre - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        self.R = np.stack([r, np.cross(r, f), f])
        self.size = size
        self.focal = size / 2 / np.tan(np.radians(fov / 2))
        self.light = -0.45 * r + 0.55 * self.R[1] - 0.70 * f        # key light, upper left of camera
        self.light /= np.linalg.norm(self.light)

    def project(self, P):
        c = (P - self.eye) @ self.R.T
        return np.c_[self.size / 2 + self.focal * c[:, 0] / c[:, 2],
                     self.size / 2 - self.focal * c[:, 1] / c[:, 2]], c[:, 2]


def shade(V, N, view, base, a0, a1):
    """Per-vertex RGBA: two-sided Lambert + ambient + a little specular; opacity rises from a0
    (facing the camera) to a1 (grazing), which outlines translucent surfaces."""
    vd = view.eye - V
    vd /= np.linalg.norm(vd, axis=1, keepdims=True)
    ndv = np.abs((N * vd).sum(1))
    ndl = np.abs(N @ view.light)
    h = vd + view.light
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    spec = np.abs((N * h).sum(1)) ** 40
    col = np.clip(rgb(base)[None] * (0.4 + 0.6 * ndl)[:, None] + 0.15 * spec[:, None], 0, 1)
    alpha = a0 + (a1 - a0) * (1 - ndv) ** 2.2
    return np.c_[col, alpha]


def rasterize(xy, z, F, C, W, H, chunk=4_000_000):
    """Front-most fragment of one mesh: RGBA image (H, W, 4) and depth (H, W), inf where empty.
    Pixel centres are tested against every triangle's bounding box in groups of equal box size."""
    P, Z, Cf = xy[F], z[F], C[F]
    lo = np.floor(P.min(1)).astype(int)
    hi = np.floor(P.max(1)).astype(int)
    x0, y0, x1, y1, x2, y2 = P[:, 0, 0], P[:, 0, 1], P[:, 1, 0], P[:, 1, 1], P[:, 2, 0], P[:, 2, 1]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    ok = (Z.min(1) > 0) & (np.abs(area) > 1e-9) & (hi[:, 0] >= 0) & (lo[:, 0] < W) & (hi[:, 1] >= 0) & (lo[:, 1] < H)
    bw, bh = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
    pix, dep, col = [], [], []
    for w_, h_ in {(a, b) for a, b in zip(bw[ok], bh[ok])}:
        idx = np.nonzero(ok & (bw == w_) & (bh == h_))[0]
        oy, ox = np.divmod(np.arange(w_ * h_), w_)
        for s in range(0, len(idx), max(1, chunk // (w_ * h_))):
            t = idx[s:s + max(1, chunk // (w_ * h_))]
            gx, gy = lo[t, 0, None] + ox[None], lo[t, 1, None] + oy[None]
            cx, cy = gx + 0.5, gy + 0.5
            a = area[t, None]
            l0 = ((x1[t, None] - cx) * (y2[t, None] - cy) - (x2[t, None] - cx) * (y1[t, None] - cy)) / a
            l1 = ((x2[t, None] - cx) * (y0[t, None] - cy) - (x0[t, None] - cx) * (y2[t, None] - cy)) / a
            l2 = 1 - l0 - l1
            ins = (l0 >= 0) & (l1 >= 0) & (l2 >= 0) & (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
            if not ins.any():
                continue
            r_, c_ = np.nonzero(ins)
            L = np.stack([l0[r_, c_], l1[r_, c_], l2[r_, c_]], 1)
            tt = t[r_]
            pix.append(gy[r_, c_] * W + gx[r_, c_])
            dep.append((L * Z[tt]).sum(1))
            col.append(np.einsum("nk,nkc->nc", L, Cf[tt]))
    img, depth = np.zeros((H * W, 4)), np.full(H * W, np.inf)
    if pix:
        pix, dep, col = np.concatenate(pix), np.concatenate(dep), np.concatenate(col)
        o = np.lexsort((dep, pix))
        first = o[np.r_[True, pix[o][1:] != pix[o][:-1]]]
        img[pix[first]], depth[pix[first]] = col[first], dep[first]
    return img.reshape(H, W, 4), depth.reshape(H, W)


def composite(layers, H, W):
    """Back-to-front alpha compositing of the front-most fragment of every layer, on white."""
    D = np.stack([d for _, d in layers])
    Cs = np.stack([c for c, _ in layers])
    order = np.argsort(-D, axis=0)                                   # far (and empty) first
    out = np.ones((H, W, 3))
    for k in range(len(layers)):
        c = np.take_along_axis(Cs, order[k][None, ..., None], 0)[0]
        out = out * (1 - c[..., 3:]) + c[..., :3] * c[..., 3:]
    return out


# ------------------------------------------------------------------ scene
_SCENE = {}


def scene():
    """Template surfaces and electrode sites, built once."""
    if _SCENE:
        return _SCENE
    t1 = nib.load(os.path.join(M.M2M, "T1.nii.gz"))
    lab05 = nib.load(os.path.join(M.M2M, "label_prep", "tissue_labeling_upsampled.nii.gz"))
    lab = np.rint(np.asanyarray(resample_from_to(lab05, t1, order=0).dataobj)).astype(int)
    A = t1.affine
    print("surfaces ...")
    scalp = surface(lab > 0, A, sigma=1.5)
    brain = surface(np.isin(lab, (M.WM, M.GM)), A, sigma=0.8)
    sV, sF, sN = scalp
    if ALL_1010:                                                    # every position of the 10-10 cap
        cap = {}
        for line in open(M.CAP):
            p = line.strip().split(",")
            if len(p) >= 5 and p[0].endswith("Electrode"):
                cap[p[4]] = np.array([float(p[1]), float(p[2]), float(p[3])])
    else:
        cap = M.read_cap()
    sites = {}
    for l in (cap if ALL_1010 else M.LABELS_1020):
        c = sV[np.argmin(np.linalg.norm(sV - cap[l], axis=1))]
        sites[l] = c
    top = sV[sV[:, 2] > -45]                                       # head above the neck
    _SCENE.update(A=A, lab=lab, gm=lab == M.GM, scalp=scalp, brain=brain, sites=sites,
                  centre=np.r_[(top.min(0) + top.max(0))[:2] / 2, 8.0])        # z: include face and neck
    return _SCENE


def target_layers(key):
    """Target surface and the montage's envelope, normalized to its own field scale."""
    S = scene()
    d = M.OUT
    env = np.asanyarray(nib.load(os.path.join(d, f"TI_envelope_{key}_MNI152_1mm.nii.gz")).dataobj).astype(np.float32)
    tgt = np.asanyarray(nib.load(os.path.join(d, f"TI_target_{key}_MNI152_1mm.nii.gz")).dataobj) > 0
    brain = np.isin(S["lab"], (M.WM, M.GM))
    vmax = float(np.percentile(env[brain], FIELD_PCT))
    s = np.where(brain, np.clip(env / vmax, 0, 1), 0).astype(np.float32)
    core = brain & (env >= 0.7 * vmax)                               # the focus, for the printed summary
    return dict(target=surface(tgt, S["A"], sigma=0.8), s=s, vmax=vmax,
                tgt_median=float(np.median(env[tgt & S["gm"]])),
                core_in_target=float((core & tgt).sum() / max(core.sum(), 1)),
                target_in_core=float((core & tgt).sum() / tgt.sum()))


def glow(view, s, inv_aff, bounds, tdepth, div=4, step=1.0):
    """Ray-cast the normalized field (emission-absorption, front to back) on a grid `div` times
    coarser than the image, then upsample. Samples in front of and behind the target's front
    surface (`tdepth`, full-resolution depth image) are kept as two layers so the target shell
    composites between them."""
    S = view.size
    n = S // div
    pc = (np.arange(n) + 0.5) * div
    dirs = np.stack(np.meshgrid((pc - S / 2) / view.focal, -(pc - S / 2) / view.focal), -1).reshape(-1, 2)
    D = np.c_[dirs, np.ones(len(dirs))] @ view.R                       # world direction per unit depth
    c, r = bounds
    dc = (c - view.eye) @ view.R[2]
    zs = np.arange(dc - r, dc + r, step)
    td = tdepth.reshape(n, div, n, div).min((1, 3)).reshape(-1)        # nearest target depth per ray
    acc = {k: (np.zeros((len(D), 3)), np.zeros(len(D)), np.zeros(len(D))) for k in ("front", "back")}
    L, t = inv_aff[:3, :3], inv_aff[:3, 3]
    for z in zs:
        vox = (view.eye + z * D) @ L.T + t
        v = map_coordinates(s, vox.T, order=1, cval=0.0, prefilter=False)
        tf = np.clip((v - GLOW_S0) / (1 - GLOW_S0), 0, 1)
        live = tf > 0
        if not live.any():
            continue
        a = np.zeros(len(D))
        a[live] = 1 - np.exp(-GLOW_K * tf[live] ** GLOW_GAMMA * step)
        col = GLOW_CMAP(tf)[:, :3]
        for part, sel in (("front", z < td), ("back", z >= td)):
            C, A, Z = acc[part]
            w = np.where(sel, (1 - A) * a, 0.0)
            C += w[:, None] * col
            Z += w * z
            A += w
    out = []
    for part in ("back", "front"):
        C, A, Z = acc[part]
        rgba = np.c_[C / np.maximum(A, 1e-9)[:, None], A].reshape(n, n, 4)
        dep = np.where(A > 1e-4, Z / np.maximum(A, 1e-9), 1e9).reshape(n, n)
        rgba = np.stack([zoom(rgba[..., k], div, order=1) for k in range(4)], -1)
        dep = zoom(dep, div, order=0)
        has_t = np.isfinite(tdepth)                                     # keep each part on its side
        dep = np.where(has_t, np.maximum(dep, tdepth + 0.01) if part == "back" else np.minimum(dep, tdepth - 0.01), dep)
        rgba[..., 3] = np.where(rgba[..., 3] > 1e-4, rgba[..., 3], 0.0)
        out.append((np.clip(rgba, 0, 1), np.where(rgba[..., 3] > 0, dep, np.inf)))
    return out


def render(key, info, az, el, size=1400, ss=2):
    """One perspective view: image (size x size x 3) and the electrode label positions."""
    S, T = scene(), target_layers_cached(key)
    pa, pb = info["pair1"].split("+"), info["pair2"].split("+")
    view = View(S["centre"], az, el, size * ss)
    H = W = size * ss
    meshes = [(S["scalp"], SKIN, 0.06, 0.5), (S["brain"], BRAIN, 0.05, 0.45),
              (T["target"], TARGET, 0.12, 0.95)]                        # target: see-through shell
    sV, sF, sN = S["scalp"]
    layers = []
    for (V, F, N), col, a0, a1 in meshes:
        xy, z = view.project(V)
        layers.append(rasterize(xy, z, F, shade(V, N, view, col, a0, a1), W, H))
    pads, off = [], 0                                                   # all pads and marks: one layer
    for l, c in S["sites"].items():
        col = CH1 if l in pa else CH2 if l in pb else SITE
        V, F, N = patch(sV, sF, sN, c, PAD_R if col != SITE else SITE_R, 1.6)
        pads.append((V, F + off, shade(V, N, view, col, 1.0, 1.0)))
        off += len(V)
    V = np.concatenate([p[0] for p in pads])
    xy, z = view.project(V)
    layers.append(rasterize(xy, z, np.concatenate([p[1] for p in pads]), np.concatenate([p[2] for p in pads]), W, H))
    bV = S["brain"][0]
    bc = (bV.min(0) + bV.max(0)) / 2
    layers += glow(view, T["s"], np.linalg.inv(S["A"]), (bc, np.linalg.norm(bV - bc, axis=1).max() + 2),
                   layers[2][1])
    img = composite(layers, H, W).reshape(size, ss, size, ss, 3).mean((1, 3))
    labels = []
    for l in pa + pb:
        c = S["sites"][l]
        n = sN[np.argmin(np.linalg.norm(sV - c, axis=1))]
        xy, _ = view.project((c + 2.0 * n)[None])
        front = n @ (view.eye - c) > 0
        labels.append((l, xy[0] / ss, l in pa, front))
    tmask = np.isfinite(layers[2][1]).reshape(size, ss, size, ss).mean((1, 3))   # target footprint
    return img, labels, tmask


_T = {}


def target_layers_cached(key):
    if key not in _T:
        _T[key] = target_layers(key)
    return _T[key]


def draw(ax, img, labels, tmask, title=None):
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    # target footprint, drawn on top: the field inside the outline is the field in (or in line
    # with) the target
    ax.contour(tmask, levels=[0.5], colors=[TARGET], linewidths=1.1,
               path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    pts = np.array([xy for _, xy, _, _ in labels], float)
    gap = 0.075 * img.shape[1]                       # labels closer than this are pushed apart
    for _ in range(30):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = pts[j] - pts[i]
                if np.hypot(*d) < gap:
                    s = 1.0 if d[1] >= 0 else -1.0
                    step = (gap - np.hypot(*d)) / 2 + 1
                    pts[i, 1] -= s * step
                    pts[j, 1] += s * step
                    moved = True
        if not moved:
            break
    for (l, _, ch1, front), (x, y) in zip(labels, pts):
        fg, halo = ("white", "black") if ch1 else ("black", "white")   # readable on pad, skin or paper
        ax.text(x, y, l, ha="center", va="center", fontsize=8, fontweight="bold", color=fg,
                alpha=1 if front else 0.75, path_effects=[pe.withStroke(linewidth=1.8, foreground=halo)])
    if title:
        ax.set_title(title, fontsize=8.5, pad=2)


def field_bar(fig, rect, key, info):
    """Color scale of the field glow in V/m, from where it starts to be drawn to 100%; the
    target's median exposure T is marked."""
    T = target_layers_cached(key)
    lo, hi = GLOW_S0 * T["vmax"], T["vmax"]
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(lo, hi), GLOW_CMAP), cax=cax)
    cb.outline.set_linewidth(0.5)
    t = info["T_V_per_m"]
    cb.set_ticks([lo, hi])
    cb.set_ticklabels([f"{lo:.2f}", f"{hi:.2f}"])
    cax.tick_params(labelsize=7, length=2, width=0.5)
    if lo <= t <= hi:
        cax.plot([0, 1], [t, t], color=TARGET, lw=1.6, transform=cax.get_yaxis_transform(), clip_on=False)
        cax.text(1.5, t, f"T {t:.2f}", transform=cax.get_yaxis_transform(), ha="left", va="center",
                 fontsize=7, color="#006d8f", fontweight="bold")
    cax.set_title("V/m", fontsize=7, pad=3)


def legend(fig, y=0.0, ncol=5):
    h = [Line2D([], [], color=TARGET, lw=1.6, label="target outline (atlas region)"),
         Patch(facecolor=GLOW_CMAP(0.85), edgecolor=GLOW_CMAP(0.35), lw=2,
               label="TI envelope (darker red = stronger)"),
         Line2D([], [], marker="o", ls="", ms=8, mfc=CH1, mec="0.2", label="channel 1 pads"),
         Line2D([], [], marker="o", ls="", ms=8, mfc=CH2, mec="0.2", label="channel 2 pads"),
         Line2D([], [], marker="o", ls="", ms=4, mfc=SITE, mec=SITE, label=f"other {'10-10' if ALL_1010 else '10-20'} sites")]
    fig.legend(handles=h, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y),
               handletextpad=0.4, columnspacing=1.2)


# (azimuth, elevation); the top view looks down from slightly behind, so the nose points up and
# the subject's left is on the left, as in the scalp diagrams of Figure S1
VIEWS = {"insula": [(-55, 20), (125, 20), (180, 82)], "hippocampus": [(-70, 12), (110, 12), (180, 82)],
         "acc": [(-30, 22), (30, 22), (180, 82)], "ofc": [(-35, 8), (35, 8), (180, 82)],
         # one-sided targets (pipeline/ti_template_unilateral.py): the target's own side first
         "insula_L": [(-75, 15), (105, 15), (180, 82)], "insula_R": [(75, 15), (-105, 15), (180, 82)],
         "hippocampus_L": [(-75, 10), (105, 10), (180, 82)], "hippocampus_R": [(75, 10), (-105, 10), (180, 82)]}
VIEW_NAMES = {"insula": ("left anterior", "right posterior", "top"),
              "hippocampus": ("left", "right posterior", "top"),
              "acc": ("left anterior", "right anterior", "top"),
              "ofc": ("left anterior", "right anterior", "top"),
              "insula_L": ("left", "right", "top"), "insula_R": ("right", "left", "top"),
              "hippocampus_L": ("left", "right", "top"), "hippocampus_R": ("right", "left", "top")}


def crop(renders, margin=0.015):
    """Crop every view of a figure to the same box around the union of their content, so the
    panels share one scale; label positions are shifted to match."""
    m = np.zeros(renders[0][0].shape[:2], bool)
    for img, _, _ in renders:
        m |= img.min(2) < 0.985
    ys, xs = np.nonzero(m)
    H, W = m.shape
    p = int(margin * W)
    y0, y1, x0, x1 = max(ys.min() - p, 0), min(ys.max() + p + 1, H), max(xs.min() - p, 0), min(xs.max() + p + 1, W)
    return [(img[y0:y1, x0:x1], [(l, xy - [x0, y0], c, f) for l, xy, c, f in lab], tm[y0:y1, x0:x1])
            for img, lab, tm in renders]


def summary_line(key, info):
    return f"F = {info['F']:.2f};  target median T = {info['T_V_per_m']:.3f} V/m"


def per_target(key, info, size=1100):
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 76 * MM))
    fig.subplots_adjust(left=0.0, right=0.93, top=0.8, bottom=0.13, wspace=0.02)
    renders = crop([render(key, info, az, el, size=size) for az, el in VIEWS[vkey(key)]])
    for a, (img, lab, tm), vn in zip(axs, renders, VIEW_NAMES[vkey(key)]):
        draw(a, img, lab, tm, f"{vn} view")
    fig.suptitle(f"{info['title']}: {info['montage']}", fontsize=9 if len(info["montage"]) < 50 else 8,
                 fontweight="bold", y=0.985)
    fig.text(0.5, 0.905, summary_line(key, info), ha="center", fontsize=8)
    field_bar(fig, [0.955, 0.25, 0.012, 0.45], key, info)
    legend(fig)
    save(fig, f"{PREFIX}{key}")


PREFIX = "TI_3D_"                                  # file-name prefix of the result set being drawn
SETS = {"unilateral": ("TI_targets_MNI_unilateral", "TI_3D_", "TI_3D_unilateral_four_targets"),
        "ratio": ("TI_targets_MNI_current_ratio", "TI_3D_ratio_", "TI_3D_ratio_targets"),
        "1010": ("TI_targets_MNI_1010", "TI_3D_1010_", "TI_3D_1010_targets"),
        "hdti": ("TI_targets_MNI_hdti", "TI_3D_hdti_", "TI_3D_hdti_targets"),
        "1010sub": ("TI_targets_MNI_1010_subregions", "TI_3D_1010_", "TI_3D_1010_subregion_targets")}


def vkey(key):
    """View set of a target: variants (_noinf) and insular subregions use their parent's."""
    k = key.replace("_noinf", "")
    return {"insula_ant_L": "insula_L", "insula_post_L": "insula_L"}.get(k, k)


def head(info):
    """Title and montage; long multi-electrode montages go on their own line."""
    m = info["montage"]
    return f"{info['title']}\n{m}" if len(m) > 34 else f"{info['title']}: {m}"


def four_targets(infos, size=1500, name="TI_3D_four_targets"):
    """All targets of the set, two per row, one shared crop."""
    keys = list(infos)
    rows = (len(keys) + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(180 * MM, 100 * rows * MM), squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.94, top=1 - 0.055 * 2 / rows, bottom=0.07 * 2 / rows,
                        wspace=0.14, hspace=0.16)
    renders = crop([render(k, infos[k], *VIEWS[vkey(k)][0], size=size) for k in keys])
    for a in axs.flat[len(keys):]:
        a.set_axis_off()
    for a, k, (img, lab, tm), letter in zip(axs.flat, keys, renders, "ABCDEFGH"):
        info = infos[k]
        draw(a, img, lab, tm)
        a.set_title(f"{head(info)}\n{summary_line(k, info)}", fontsize=8, pad=2)
        a.text(-0.02, 1.0, letter, transform=a.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")
        p = a.get_position()
        field_bar(fig, [p.x1 + 0.012, p.y0 + 0.2 * p.height, 0.011, 0.55 * p.height], k, info)
    legend(fig, y=0.005 * 2 / rows, ncol=3)
    save(fig, name)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext, dpi in (("png", 600), ("pdf", 600)):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".png/.pdf"))


if __name__ == "__main__":
    # exploratory result sets: --set unilateral (one-sided targets) or --set ratio (unequal currents)
    which_set = "unilateral" if "--unilateral" in sys.argv else (
        sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else None)
    overview = "TI_3D_four_targets"
    if which_set:
        folder, PREFIX, overview = SETS[which_set]
        ALL_1010 = which_set in ("1010", "1010sub", "hdti")
        SITE_R = 2.5 if ALL_1010 else SITE_R                         # 76 marks: keep them small
        M.OUT = os.path.join(os.path.dirname(M.OUT), folder)
    summ = json.load(open(os.path.join(M.OUT, "TI_targets_MNI152_summary.json"), encoding="utf-8"))
    infos = {t["target"]: t for t in summ["targets"]}
    which = [a for a in sys.argv[1:] if a in infos] or list(infos)
    for k in which:
        per_target(k, infos[k])
        T = target_layers_cached(k)
        print(f"  {k}: field scale {T['vmax']:.3f} V/m (brain p{FIELD_PCT:g}); target median "
              f"{infos[k]['T_V_per_m']:.3f} V/m; focus (>= 70% of scale) inside target "
              f"{100 * T['core_in_target']:.1f}%, target inside focus {100 * T['target_in_core']:.1f}%")
    if len(which) == len(infos):
        if which_set == "hdti":                                      # one overview per variant
            order = ["insula_L", "insula_ant_L", "insula_post_L", "hippocampus_L"]
            four_targets({k: infos[k] for k in order if k in infos}, name=overview)
            four_targets({k + "_noinf": infos[k + "_noinf"] for k in order if k + "_noinf" in infos},
                         name=overview + "_no_inferior")
        else:
            four_targets(infos, name=overview)
