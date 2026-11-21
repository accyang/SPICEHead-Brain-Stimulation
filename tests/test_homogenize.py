"""
Tests for the v23 homogenized network (spicehead.homogenize) and its package wiring.

Runnable directly (``python tests/test_homogenize.py``) or under pytest.

1. test_uniform_blocks_equal_harmonic  -- block-uniform tissue: the homogenized network equals the
                                          v22 harmonic network on the coarse grid, edge for edge.
2. test_thin_layer_series              -- a one-voxel skull layer inside a block: the edge conductance
                                          equals the analytic series/parallel value; nearest-label
                                          resampling would have lost the layer.
3. test_node_rule_and_islands          -- blocks under half full are not nodes; detached islands are
                                          dropped and counted.
4. test_world_axis_field               -- on a permuted and flipped (non-RAS) grid, a linear potential
                                          gives the correct world-axis field after rotation.
5. test_config_and_builder             -- SPICEHEAD_MODEL=v23 selects the homogenized builder and its own
                                          results folder; model_from_labels and interface-rule overrides are
                                          refused; head_grid returns the model's grid; v22 head_model and
                                          head_grid are the old calls, unchanged (subprocesses, because the
                                          model version is read at import).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spicehead.conductivity import CONDUCTIVITY
from spicehead.field import efield_components
from spicehead.homogenize import homogenized_model
from spicehead.model import SPICEHeadModel

GM, CSF, BONE, SCALP = 2, 3, 4, 5
SIG = {GM: CONDUCTIVITY["gray_matter"], CSF: CONDUCTIVITY["csf"], BONE: CONDUCTIVITY["bone"], SCALP: CONDUCTIVITY["scalp"]}
FINE = 0.5                                   # mm
AFF = np.diag([FINE, FINE, FINE, 1.0])


def _edge(G, model, a, b):
    return -G[model.node_id[a], model.node_id[b]]


def test_uniform_blocks_equal_harmonic():
    rng = np.random.default_rng(0)
    coarse = rng.choice([GM, CSF, BONE, SCALP], size=(5, 4, 6))
    for f in (2, 4):
        fine = np.kron(coarse, np.ones((f, f, f), int))              # every block uniform
        hom, _ = homogenized_model(fine, np.diag([FINE, FINE, FINE, 1.0]), dx_mm=f * FINE)
        sig = np.vectorize(SIG.get)(coarse).astype(float)
        std = SPICEHeadModel(sig, dx=f * FINE * 1e-3, conductance_mean="harmonic")
        assert hom.n_nodes == std.n_nodes and hom.dropped_nodes == 0
        d = abs(hom.G - std.G).max()
        print(f"[uniform blocks] f={f}: max |G_hom - G_v22| = {d:.1e} S (max entry {abs(std.G).max():.1e})")
        assert d < 1e-12 * abs(std.G).max()


def test_thin_layer_series():
    f = 4
    dxf = FINE * 1e-3
    fine = np.full((2 * f, f, f), SCALP, int)
    fine[f - 1, :, :] = BONE                                           # one fine voxel of bone, last slice of block A
    hom, _ = homogenized_model(fine, AFF, dx_mm=f * FINE)
    # each of the f*f tubes: (f/2 - 1) scalp + 1 bone in A's half, f/2 scalp in B's half
    r_tube = (f - 1) / (SIG[SCALP] * dxf) + 1 / (SIG[BONE] * dxf)
    g_expected = f * f / r_tube
    g = _edge(hom.G, hom, (0, 0, 0), (1, 0, 0))
    # nearest-label resampling at 2 mm keeps the majority label (scalp) and loses the bone layer
    g_nearest = SIG[SCALP] * f * dxf
    print(f"[thin layer] edge {g:.4e} S, analytic {g_expected:.4e} S, nearest-label network {g_nearest:.4e} S")
    assert abs(g - g_expected) < 1e-12 * g_expected
    assert g < 0.2 * g_nearest
    assert hom.labels[0, 0, 0] == SCALP                                # majority label for the contacts


def test_node_rule_and_islands():
    f = 2
    fine = np.zeros((6 * f, 2 * f, 2 * f), int)
    fine[: 4 * f] = GM                                                 # main conductor: 4 x 2 x 2 blocks
    fine[5 * f:, :f, :f] = GM                                          # detached island: one block
    fine[4 * f, 0, 0] = GM                                             # 1 of 8 fine voxels: not a node
    hom, _ = homogenized_model(fine, AFF, dx_mm=f * FINE)
    print(f"[node rule] nodes {hom.n_nodes}, dropped {hom.dropped_nodes}")
    assert hom.n_nodes == 16 and hom.dropped_nodes == 1
    assert hom.labels[5, 0, 0] == 0 and hom.labels[4, 0, 0] == 0


def test_zero_conductance_island():
    # a block joined to the network only through tubes that cross air is not connected (fixed 27 Oct 2026: such a block
    # passed the giant-component filter through stored zero conductances and made the matrix singular)
    f = 2
    fine = np.zeros((5 * f, f, f), int)
    fine[: 4 * f] = GM                                                 # main conductor: 4 x 1 x 1 blocks
    fine[4 * f + 1:] = GM                                              # half-filled block (a node) whose contact face is air
    hom, _ = homogenized_model(fine, AFF, dx_mm=f * FINE)
    G = hom.G.tocsr()
    print(f"[zero-conductance island] nodes {hom.n_nodes}, dropped {hom.dropped_nodes}, explicit zeros {int((G.data == 0).sum())}")
    assert hom.n_nodes == 4 and hom.dropped_nodes == 1 and not (G.data == 0).any()


def test_world_axis_field():
    f = 2
    fine = np.full((8 * f, 6 * f, 10 * f), GM, int)
    # voxel axes: i -> -z (flipped), j -> x, k -> y
    A = np.array([[0, FINE, 0, 10.0], [0, 0, FINE, -20.0], [-FINE, 0, 0, 30.0], [0, 0, 0, 1.0]])
    hom, aff_c = homogenized_model(fine, A, dx_mm=f * FINE)
    E0 = np.array([3.0, -1.5, 0.5])                                    # V/m in world axes
    ii, jj, kk = hom.grid_coords
    xyz = (aff_c[:3, :3] @ np.vstack([ii, jj, kk]) + aff_c[:3, 3:4]) * 1e-3   # node positions in metres
    V = -(E0 @ xyz)
    E = efield_components(hom, V)
    inner = (slice(1, -1),) * 3
    err = max(np.nanmax(np.abs(E[w][inner] - E0[w])) for w in range(3))
    print(f"[world axes] max |E - E0| in the interior = {err:.1e} V/m")
    assert err < 1e-9


def _run(model, code):
    env = dict(os.environ, SPICEHEAD_MODEL=model, PYTHONPATH=ROOT)
    env.pop("SPICEHEAD_RESULTS_DIR", None)          # the test checks the default results folders
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_config_and_builder():
    import nibabel as nib
    rng = np.random.default_rng(1)
    vol = np.zeros((24, 20, 28), np.int16)
    vol[2:-2, 2:-2, 2:-2] = SCALP
    vol[4:-4, 4:-4, 4:-4] = BONE
    vol[5:-5, 5:-5, 5:-5] = rng.choice([GM, CSF], size=(14, 10, 18))
    tmp = tempfile.mkdtemp()
    seg = os.path.join(tmp, "seg.nii.gz")
    nib.save(nib.Nifti1Image(vol, AFF), seg)
    code = f"""
import numpy as np
from spicehead import config, mri
model, aff = mri.head_model(r"{seg}", 1.0)
print(config.MODEL, config.NETWORK, config.RESULTS.split(chr(92))[-1].split('/')[-1], getattr(model, 'network', 'nearest'), model.n_nodes)
if config.MODEL == 'v22':
    labels, a2, z = mri.load_label_volume(r"{seg}", dx_mm=1.0)
    old = mri.model_from_labels(labels, None, dx_mm=1.0)
    l3, a3, z3, m3 = mri.head_grid(r"{seg}", 1.0)
    print('same_as_old', abs(old.G - model.G).max() == 0 and np.array_equal(a2, aff) and np.array_equal(old.labels, model.labels)
          and np.array_equal(l3, labels) and np.array_equal(a3, a2) and m3 is None)
else:
    l3, a3, z3, m3 = mri.head_grid(r"{seg}", 1.0)
    ok = m3 is not None and np.array_equal(l3, model.labels) and np.array_equal(a3, aff) and z3 == (1.0, 1.0, 1.0)
    for bad in (lambda: mri.model_from_labels(np.ones((3, 3, 3), int), None, dx_mm=1.0),
                lambda: mri.head_model(r"{seg}", 1.0, conductance_mean='arithmetic')):
        try:
            bad(); ok = False
        except ValueError:
            pass
    print('refused', ok)
"""
    v22 = _run("v22", code).split()
    v23 = _run("v23", code).split()
    print(f"[config] v22: {' '.join(v22)} | v23: {' '.join(v23)}")
    assert v22[:4] == ["v22", "nearest", "results_v22", "nearest"] and v22[-1] == "True"
    assert v23[:4] == ["v23", "homogenized", "results_v23", "homogenized"] and v23[-1] == "True"


if __name__ == "__main__":
    print("Running v23 homogenized-network tests\n" + "-" * 44)
    test_uniform_blocks_equal_harmonic()
    test_thin_layer_series()
    test_node_rule_and_islands()
    test_zero_conductance_island()
    test_world_axis_field()
    test_config_and_builder()
    print("-" * 44 + "\nAll v23 checks passed.")
