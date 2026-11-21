"""
SPICEhead: a circuit-integrated, netlist-compatible head-modeling framework for
simulating temporal interference stimulation (TIS).

Reconstruction of the method described in Chen et al., "SPICEhead: A Circuit-
Integrated, Netlist-Compatible Head-Modeling Framework for Simulating Temporal
Interference Stimulation."

Pipeline:  MRI/phantom -> conductivity volume -> resistor network (model.py)
           -> solve G V = I (solve.py / SPICE netlist netlist.py)
           -> E = -grad V, TI envelope A = 2 min(|a|,|b|) (field.py)
"""

from .model import SPICEHeadModel
from .solve import solve_potentials, make_solver
from .field import (
    efield_components,
    field_magnitude,
    field_magnitude_inplane,
    ti_envelope_components,
    ti_envelope_magnitude,
    ti_envelope_max_direction,
    calibrate_k,
)
from .electrodes import Electrode, nodes_in_sphere, nearest_node, pair_injection
from .geometry import cylinder_model, place_rim_electrode, rim_point, cylinder_center
from .pipeline import simulate_ti, TIResult
from . import netlist
from . import conductivity
from . import mri
from . import homogenize
from . import config

__all__ = [
    "SPICEHeadModel",
    "solve_potentials", "make_solver",
    "efield_components", "field_magnitude", "field_magnitude_inplane",
    "ti_envelope_components", "ti_envelope_magnitude", "ti_envelope_max_direction", "calibrate_k",
    "Electrode", "nodes_in_sphere", "nearest_node", "pair_injection",
    "cylinder_model", "place_rim_electrode", "rim_point", "cylinder_center",
    "simulate_ti", "TIResult",
    "netlist", "conductivity", "mri", "homogenize", "config",
]

__version__ = "0.1.0"
