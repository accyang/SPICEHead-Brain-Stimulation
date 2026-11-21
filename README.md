# SPICEHead

SPICEHead is an MRI-derived, circuit-compatible resistor-network model for temporal interference stimulation. This deposit accompanies the manuscript *SPICEHead: Whole-brain temporal interference targeting across healthy aging*.

- `spicehead/` contains the Python implementation.
- `pipeline/` contains analysis scripts and examples of cluster jobs. Full cohort analyses require separately authorized inputs and a configured execution environment.
- `examples/` contains an analytic validation example and a worked MNI152 template example. The template provenance and licence are described in [the example README](examples/mni152_head/README.md).
- `tests/` contains numerical validation tests.
- `example_outputs/` contains selected aggregate benchmarks.
- [Supplementary Data S1](data/Supplementary_Data_S1.xlsx) contains the reported targeting parameters for 163 regions.

## Worked example

Install the packages in `requirements.txt` in a separate Python environment and run from the repository root.

```sh
python examples/mni152_full_search.py --dx 3
```

The example uses the template segmentation supplied in this deposit. SimNIBS is a separate dependency for analyses that use its segmentation or FEM facilities.

## Data access

The workbook and aggregate benchmarks can be accessed by readers with repository access. Participant MRI, derived head models and head-level outputs are held separately. Access is subject to the conditions of each contributing cohort and the applicable approvals. Use the corresponding-author contacts in the manuscript for inquiries.

## Software licence

The project licence is pending author selection. This deposit does not grant a software licence. Third-party template terms apply independently.
