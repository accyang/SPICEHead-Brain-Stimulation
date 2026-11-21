"""Verify the GPU conjugate-gradient solver against the CPU solver on a real head.

  SPICEHEAD_MODEL=v22 python pipeline/gpu_solver_check.py <CaseID> [dx_mm] [n_cpu_rhs]

Builds the head network at dx (default 2 mm), the 18 single-electrode injections against Cz used by
the basis sweeps, solves all of them on the GPU in one block, solves the first n_cpu_rhs (default 3)
with the CPU Jacobi-CG of v21_head_convergence.make_solver, and reports the largest relative
difference of the potentials and of the gray-matter field components, plus timings. Writes
results_v22/gpu_solver_check/<CaseID>_dx<dx>.json.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("SPICEHEAD_MODEL", "v22")
from spicehead import config, mri
import v21_dx2_cohort_check as D          # reuses its electrode/injection helpers and constants
from v21_head_convergence import make_solver
from spicehead.solve_gpu import make_gpu_solver

cid = sys.argv[1] if len(sys.argv) > 1 else "SUBJ01"
dx = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
n_cpu = int(sys.argv[3]) if len(sys.argv) > 3 else 3

m2m = D.resolve_m2m(D.CH, cid)
model, affine = mri.head_model(os.path.join(m2m, "label_prep", "tissue_labeling_upsampled.nii.gz"), dx)   # v21/v22: nearest-label network; v23: homogenized
labels = model.labels
gm = np.isin(labels, D.SCORE)
scalp_vox = np.argwhere(labels == D.SCALP)
ej = json.load(open(os.path.join(m2m, "electrodes_1020.json")))
labs = sorted(ej)
elecs = {}
for lab in labs:
    v = D.snap(scalp_vox, mri.world_to_voxel(affine, ej[lab]))
    elecs[lab] = D.Electrode(D.contact_nodes(model, tuple(int(x) for x in v), radius_nodes=D.ELEC_RADIUS_MM / dx), lab)
others = [l for l in labs if l != D.REF]
inj = np.stack([D.pair_injection(model, elecs[l], elecs[D.REF], 1e-3) for l in others], axis=1)   # (N, 18)
print(f"{cid} dx={dx:g}: {model.n_nodes} nodes, nnz {model.G.nnz}, {len(others)} injections", flush=True)

t0 = time.time()
gsolve, ncomp_g = make_gpu_solver(model.G)
t_setup = time.time() - t0
t0 = time.time()
Vg = gsolve.many(inj)                                   # all 18 at once
t_gpu_all = time.time() - t0
t0 = time.time()
Vg1 = gsolve(inj[:, 0])                                 # single-rhs path
t_gpu_one = time.time() - t0
print(f"GPU: setup {t_setup:.1f}s, 18 rhs in {t_gpu_all:.1f}s, 1 rhs in {t_gpu_one:.1f}s, device {gsolve.device}", flush=True)

os.environ.pop("SPICEHEAD_SOLVER", None)
t0 = time.time()
csolve, ncomp_c = make_solver(model.G)
t_csetup = time.time() - t0
rel_v, rel_e, t_cpu = [], [], []
for j in range(min(n_cpu, inj.shape[1])):
    t0 = time.time()
    Vc = csolve(inj[:, j])
    t_cpu.append(time.time() - t0)
    rel_v.append(float(np.abs(Vg[:, j] - Vc).max() / np.abs(Vc).max()))
    Ec = np.stack([np.nan_to_num(c[gm]) for c in D.efield_components(model, Vc)])
    Eg = np.stack([np.nan_to_num(c[gm]) for c in D.efield_components(model, Vg[:, j])])
    rel_e.append(float(np.abs(Eg - Ec).max() / np.abs(Ec).max()))
    print(f"  rhs {j} ({others[j]}): CPU {t_cpu[-1]:.0f}s | max rel diff V {rel_v[-1]:.2e}, GM field {rel_e[-1]:.2e}", flush=True)
same = float(np.abs(Vg1 - Vg[:, 0]).max() / np.abs(Vg[:, 0]).max())
print(f"single-rhs vs block GPU solution: {same:.2e}; components GPU {ncomp_g} / CPU {ncomp_c}")

out = os.path.join(config.RESULTS, "gpu_solver_check"); os.makedirs(out, exist_ok=True)
rec = dict(subject=cid, dx_mm=dx, n_nodes=int(model.n_nodes), nnz=int(model.G.nnz), n_rhs=int(inj.shape[1]),
           gpu_device=gsolve.device, t_gpu_setup_s=t_setup, t_gpu_18rhs_s=t_gpu_all, t_gpu_1rhs_s=t_gpu_one,
           t_cpu_setup_s=t_csetup, t_cpu_per_rhs_s=t_cpu, max_rel_diff_V=rel_v, max_rel_diff_gm_field=rel_e,
           single_vs_block=same, ncomp=int(ncomp_g), rtol=1e-10, date=time.strftime("%Y-%m-%d %H:%M"))
json.dump(rec, open(os.path.join(out, f"{cid}_dx{dx:g}.json"), "w"), indent=1)
print("wrote", os.path.join(out, f"{cid}_dx{dx:g}.json"))
