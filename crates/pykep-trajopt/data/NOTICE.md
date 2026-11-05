# Embedded reference data

The four TOPS JSON files are verbatim data from ESA PyKEP at
`c01e19debad5ea22b7079189f786f473046a79d5`. The launcher table is an extraction
of the numeric values in `pykep/trajopt/_launchers.py` at the same revision.
The upstream MPL-2.0 license and copyright notices apply to adapted source
and embedded data. The repository license and upstream notice give the
project attribution. [PyKEP source](https://github.com/esa/pykep/tree/c01e19debad5ea22b7079189f786f473046a79d5)

`provenance.json` records source IDs and SHA-256 hashes. Regenerate using
`tools/upstream-comparison/export_data.py` with the pinned source checkout.

The `solutions/` vectors were generated here with the Rust evaluator and
SciPy's bounded least-squares solver. Each record contains the solver
configuration, feasibility and independent forward residual.
