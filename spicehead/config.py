"""Which physical model every SPICEhead analysis uses, and where its results go.

Three model versions exist:

  v23 (added 2026-10-23, approved by the PI after the 300-head FEM comparison; the model of Paper A's
      final results since 2026-10-26 and the package default since 2026-11-02): v22 contacts and
      interface rule, but the coarse network is built from the fine segmentation instead of
      nearest-label resampling: each edge conductance is the parallel sum of series conduction
      tubes through the 0.5 mm CHARM voxels between two block centres (spicehead.homogenize).
      The coarse grid follows the CHARM voxel axes; field components are rotated to world axes.
      Build heads with spicehead.mri.head_model(); mri.model_from_labels refuses to run under v23.
  v22 (default 2026-10-01 to 2026-11-01; the model of the v22 records r1-r7): each electrode injects through the skin
      under its pad only, i.e. scalp nodes within the pad radius that touch the non-conductor, and
      the two voxel conductivities on an edge are combined harmonically (resistors in series),
      which is the physically correct rule for thin layers such as the skull.
  v21: the configuration of every result produced before 2026-10-01: volumetric contacts (all
      conducting nodes within the pad radius) and arithmetic averaging.

Select with the environment variable SPICEHEAD_MODEL=v21|v22|v23 (unset = v23). The v22 record and
figure scripts (values_v22, render_v22, fig_v22_results, fig_v22_pipeline, make_submission_figures)
default to v22 themselves. Model-dependent results go to
results/ for v21 and results_<model>/ otherwise, and caches carry the model tag, so results from
different versions can never be mixed. The FEM reference (SimNIBS output) does not depend on this
choice and stays in results/v21_simnibs_validation for all of them.
"""
import os

MODELS = ("v21", "v22", "v23")
MODEL = os.environ.get("SPICEHEAD_MODEL", "v23")
if MODEL not in MODELS:
    raise ValueError(f"SPICEHEAD_MODEL must be one of {', '.join(MODELS)}, not {MODEL!r}")

CONTACT = {"v21": "volume", "v22": "surface", "v23": "surface"}[MODEL]
CONDUCTANCE_MEAN = {"v21": "arithmetic", "v22": "harmonic", "v23": "harmonic"}[MODEL]
NETWORK = {"v21": "nearest", "v22": "nearest", "v23": "homogenized"}[MODEL]
HOMOG_MIN_FILL = 0.5          # v23 node rule: share of a block's fine voxels that must conduct

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SPICEHEAD_RESULTS_DIR redirects all model-dependent results (smoke tests, scratch runs); unset = the project folders
RESULTS = os.environ.get("SPICEHEAD_RESULTS_DIR") or os.path.join(PROJECT_ROOT, "results" if MODEL == "v21" else f"results_{MODEL}")
FEM_DIR = os.path.join(PROJECT_ROOT, "results", "v21_simnibs_validation")


def tagged(name):
    """A cache or file name for this model: unchanged for v21, suffixed otherwise."""
    return name if MODEL == "v21" else f"{name}_{MODEL}"


def describe():
    net = ("network from the fine segmentation (homogenized conductances)" if NETWORK == "homogenized"
           else "network from nearest-label resampling")
    try:
        where = os.path.relpath(RESULTS, PROJECT_ROOT)
    except ValueError:                       # SPICEHEAD_RESULTS_DIR on another drive (Windows)
        where = RESULTS
    return (f"SPICEhead model {MODEL}: {CONTACT} contacts, {CONDUCTANCE_MEAN} interface rule, {net}; "
            f"results in {where}")
