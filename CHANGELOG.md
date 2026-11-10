# Changelog

All notable changes to this project will be documented in this file. The
format follows Keep a Changelog and releases use semantic versioning. During
the pre-1.0 series, minor releases may deliberately evolve the public API.

## [Unreleased]

### Added

- Fixed-duration Sims–Flanagan multiple shooting: `pykep_trajopt::SimsFlanaganChain`
  and `pykep_rust.trajopt.SimsFlanaganChain` (with a type stub). Each arc has
  seven matching rows and its squared throttle norms; the total duration,
  impulse model and global forward/backward cut equal the single leg's. The
  constraint Jacobian is analytic in a fixed CSR pattern, `elements` gives
  the arc row groups for a partitioned SQP Hessian, and `lift` builds node
  states from a single-leg decision. SQP-craft's SQP5 scaling study uses it
  up to 4,994 variables.
- `SimsFlanaganLeg::matchpoint_states` returns the two endpoints propagated
  independently to the cut.
- Added an optional heyoka example, `python/examples/heyoka/`. pykep-rust
  and fcmaes find a Cassini1 route in patched conics. heyoka re-propagates
  each leg with VSOP2013 ephemerides and planetary perturbations, locates
  the closest approaches with events, and retargets the legs with
  variational equations. heyoka is not a dependency of the crate or the
  wheel, and the example is not run in CI.

### Changed

- README: links the [overview of the 18 related repositories](https://github.com/dietmarwo/fcmaes-rust/blob/main/docs/repositories.md).
- PERFORMANCE_GUIDE.md now uses the 0.1.5 measurements against PyKEP
  3.1.0: all eleven Taylor fixtures with their heyoka setup times, the
  seven upstream analytical benchmark programs, complete ZOH legs, Python
  gym evaluation and the Sims–Flanagan optimizer comparison. The 0.1.4
  tables are replaced by a short note. `docs/performance.md`,
  `ai-context.md`, ADR 0005, the Taylor guide and the documentation map
  now point to the 0.1.5 results instead of describing them as
  development work.

### Fixed

- `ai-context.md`'s API table said only Rust nominal mismatch/history can select Taylor and that Python stays DOP853. Rust and Python select Taylor or DOP853 for mismatch, history and the full Jacobian, as the same guide and the README say elsewhere; the table now agrees and names which model Jacobians are numerical.
- The error panel of `docs/images/taylor-results.svg` contradicted the table beside it. Its Taylor bars were taller than the DOP853 bars under "lower is better", although Taylor's error is smaller at every tolerance. It also had no axis values, and bars covered its title and legend. The new script `tools/taylor_results_svg.py` redraws the figure from `docs/data/taylor-kepler-benchmark.csv`, with the errors on a labelled log axis. The speed-up panel is unchanged.
- The documentation site loads MathJax (`mathjax-support` in `book.toml`), so the
  Taylor-series formula in the Taylor integration guide renders instead of
  showing its TeX source.

## [0.1.5] - 2026-11-05

### Added

- Added the optimizer-independent trajectory gym: 14 classical PyKEP 3.1.0
  mission problems, 34 fixed and 34 moving TOPS low-thrust problems with
  analytic Jacobians and uniform or softmax time grids, and six launcher
  tables. It lives in the unpublished `pykep-trajopt` workspace crate and is
  included in the `pykep-rust` wheels as `pykep_rust.trajopt.gym`, a typed,
  PyGMO-compatible problem protocol with ordered float64 batches.
- Added `TopsOptions::maximum_steps` and the Python `maximum_steps` option.
  Fixed-boundary problems keep the upstream limit of 1,000 steps; moving
  problems, whose upstream factories impose no limit, default to 1,000,000.
- Added the optional `simd` feature (dependency `wide`) for four-lane Taylor
  batches with independent adaptive steps, `Taylor::propagate_batch`.
- Added analytic Taylor plane, sphere and radial-velocity events for the
  built-in Cartesian models: `Taylor::propagate_until_event`,
  `Taylor::propagate_with_events`, `TaylorEvent`, `TaylorEventHit` and
  `TaylorEventHistory`. `TaylorEvent` is `#[non_exhaustive]`.
- Added `ZohLeg::mismatch_jacobian_with_method` for complete Taylor or DOP853
  ZOH-leg Jacobians, and a keyword-only `method` on the Python
  `ZohLeg.mismatch_jacobian` and `ZohLeg.mismatch_jacobian_batch`.
- Added explicit Python backend selection for nominal dynamics, ZOH
  schedules, and ZOH-leg mismatch and history calls.
- Added matched PyKEP 3.1.0 benchmarks with shared fixtures, full-Jacobian
  comparisons, bounded restartable studies, optimizer comparisons and
  trajectory replay examples (`tools/upstream-benchmark`,
  `tools/gym-benchmark`, `tools/upstream-comparison`,
  `tools/taylor-comparison`, `tools/taylor-codegen`). Raw measurements are
  attached to the GitHub release.
- Added native regression tests for the gym: finite-difference Jacobian
  checks for every TOPS family, boundary type and time encoding, catalog
  midpoint evaluations, launcher values from the pinned source, ordered
  batches, and the two moving-CR3BP close-lunar-pass vectors.
- Added "Astrodynamics for programmers", an introduction from orbits,
  Lambert arcs and gravity assists to low thrust and trajectory optimization
  with fcmaes-rust, optimization-jobs and experiment-craft, with a runnable
  tour (`python/examples/astrodynamics_tour.py`), a Python fcmaes example,
  the Rust `cassini1_retry` example and the figure script.
- Added a measurement-backed guide for choosing between the original
  Python/C++ fcmaes/pykep stack and the native Rust crates, plus guides for
  the gym, TOPS, impulsive formulations and matched benchmarks.

### Changed

- DOP853 sensitivity propagation for `ZohKeplerDynamics` and
  `ZohCr3bpDynamics` now uses analytic model Jacobians instead of fixed-step
  centered differences. The old differences lost accuracy near a primary: on
  one backward segment of a moving CR3BP TOPS vector with a close lunar pass,
  the DOP853 transition matrix was 43% off. Equinoctial and solar-sail ZOH
  Jacobians still use centered differences.
- Taylor sensitivities propagate tangent coefficient recurrences instead of
  `2W + 1` centered propagations, share nominal coefficients across practical
  seed widths and avoid a heap allocation for single-jet propagation.
- Store coefficient histories only for generated Taylor recurrence operands,
  and generate fixed-model Rust coefficient kernels ahead of time for
  selected Taylor paths. CI checks that the generated kernels are current.
- Sample Taylor dense output from accepted-step polynomials; on baseline x86
  builds without FMA the polynomial evaluation is unfused.
- The nominal Taylor step safety factor is 0.85, previously 0.75, for
  `Cr3bpDynamics`, `ZohSolarSailDynamics`, `EquinoctialMassOptimal` and
  `EquinoctialTimeOptimal`. Default step counts and results change within
  tolerance.
- With the `simd` feature, Taylor batch results may differ from scalar
  results at rounding level.
- The release smoke test now fails if a wheel or source distribution lacks
  the trajectory gym, and the wheels ship `UPSTREAM_NOTICE.md` and the gym
  dataset notice alongside the licence.
- Updated the security policy, implementation status, validation history,
  release instructions, package READMEs and AI operational context for the
  current integration defaults, ZOH method selection, parallel batch
  semantics and pure-Rust performance boundary. The performance guide's
  tables are framed as the 0.1.4 baseline.
- Replaced the stale Phase 18 release-candidate status probe with a value
  derived automatically from the Cargo package version.

### Fixed

- Restarting `Taylor::propagate_until_event` from a returned event state
  could return the same event again, or fail with "event time does not
  advance", depending on the sign of the rounding residual. Roots within
  rounding of the initial time are now excluded as documented.
- Safeguarded Lagrange anomaly iteration with a bracketed fallback for
  difficult elliptic and hyperbolic cases found by the expanded benchmarks.
- `pykep-gym-benchmark validate` now restores the recorded backend and
  settings and exits with an error when a replay disagrees with its record.
- `pykep-gym-benchmark` now uses fcmaes-core 0.2.1. Its MODE fixes the SBX,
  polynomial-mutation and front-sorting defects of fcmaes-core 0.1.x; MODE
  runs recorded with 0.1.3, including the JUICE front in the measurement
  archive, used the older MODE.
- Forced Python CI to install the wheel built from the current checkout rather
  than allowing an already-published wheel with the same workspace version to
  win pip's candidate selection.

### Known limitations

- Two moving-CR3BP stress vectors (P1, P13) differ from the PyKEP gradient
  by a scaled `1.07e-6` and `4.23e-6`, above the strict `1e-7` gate. Both
  pass close to the Moon; the Rust Taylor gradient varies by up to `2.4e-5`
  across tolerances on the same vectors.
- Moving MEE retains the documented source chain-rule discrepancies, and the
  moving CR3BP softmax variant is a Rust-only formulation.
- The original SPICE batch wrapper cell remains a reference-only failure in
  the pinned source.

## [0.1.4] - 2026-07-28

### Added

- Added a contributor guide covering the complete evaluated-dynamics,
  Jacobian, Taylor, ZOH, batch, Python, validation, benchmark, documentation,
  and quality-gate workflow for a new ODE system.
- Added method-selecting ZOH-leg mismatch and state-history APIs. Existing
  methods retain their DOP853 behavior, while built-in legs can explicitly
  select the accelerated Taylor backend.

### Changed

- Replaced repeated full-series evaluation across all eleven Taylor models.
  Kepler and ZOH Kepler use specialized incremental recurrences; the other
  nine models use cached fixed-expression coefficient tapes with structural
  common-subexpression elimination. Pontryagin costates are generated by
  reverse symbolic differentiation with explicit minimized-control
  stop-gradients. The eight models migrated after the initial Kepler, ZOH
  Kepler, and Cartesian mass work measured 6.45× to 28.97× faster in the
  representative warmed propagation protocol. Coefficient-level reference
  tests cover Taylor order 24 and both stable smoothed-throttle branches.

## [0.1.3] - 2026-07-28

### Added

- Added a pure-Rust adaptive Taylor-series backend for all eleven built-in
  dynamics types, with analytic, DOP853, upstream-fixture, and heyoka 7.10.1
  cross-validation plus reproducible matched-accuracy benchmarks.
- Added `Taylor`, `IntegrationMethod`, `AdaptiveIntegrator`, and explicit
  method-selecting propagation APIs for the built-in dynamics and ZOH
  schedules.

### Changed

- Changed `AdaptiveIntegrator::default()` and nominal
  `KeplerDynamics`/`Cr3bpDynamics`/`BcpDynamics` propagation to Taylor. This
  aligns the TA-derived Rust models with original pykep's heyoka
  Taylor-adaptive families. Direct `Dop853` calls remain available for
  arbitrary user dynamics and events; STM and ZOH sensitivity convenience
  methods retain DOP853's direct variational propagation until native Taylor
  variational recurrences are implemented.

## [0.1.2] - 2026-07-26

### Added

- Added deterministic serial/parallel batches for the existing propagation,
  Lambert, ephemeris, anomaly, vector, transfer, encoding, flyby, MIMA,
  evaluated-dynamics, ZOH, Pontryagin, and trajectory-leg computations.
- Exposed the complete batch surface through the typed Python API, with
  scalar-parity and shape tests for every top-level batch export.
- Documented the SpOC 4 and `pykep-lambert` motivation, worker selection,
  nested-parallelism guidance, and the boundary that batching does not add new
  physical models.

## [0.1.1] - 2026-07-26

### Changed

- Bumped the synchronized Rust and Python distribution version for the first
  fully automated Trusted Publishing release.
- Made Python wheel and source-distribution CI checks derive their expected
  version from the Cargo workspace instead of hard-coding `0.1.0`.

## [0.1.0] - 2026-07-26

### Added

- Independent Rust workspace and private PyO3 extension scaffold.
- Pinned pykep/kep3 3.0.1 provenance, full MPL-2.0 license, and deterministic
  C++ foundation oracle data.
- Cross-platform formatting, lint, test, documentation, coverage, packaging,
  dependency-license, and vulnerability CI gates.
- Physical constants, Julian-date conversions, numerically stable Stumpff
  functions, Kepler-equation residuals/derivatives, and fixed small linear
  algebra in Rust and Python.
- Explicit numerical error taxonomy, fixed-shape type decision, golden C++
  parity tests, independent derivative/geometry tests, and foundation
  benchmarks.
- Immutable microsecond-resolution epochs with MJD2000/MJD/JD and cropped ISO
  conversion, checked duration arithmetic, rich Python comparison, and
  explicit time-scale limitations.
- Elliptic and hyperbolic anomaly conversions, bounded solvers, scalar and
  batch Python APIs, seeded C++ parity vectors, round-trip tests, and
  benchmarks.
- Cartesian, classical Keplerian, and prograde/retrograde modified
  equinoctial conversions with named Rust element values, analytic
  output-by-input Jacobians, NumPy batch APIs, and explicit singularity
  handling.
- Forward/backward two-body propagation using Lagrange coefficients and
  universal variables, analytic Lagrangian and Reynolds STMs, time-grid and
  GIL-releasing NumPy batch APIs, golden parity vectors, invariant and
  finite-difference validation, and Criterion benchmarks.
- Hohmann and bi-elliptic transfers, reversible alpha/eta encodings, flyby
  constraints and analytic Jacobian, unpowered flyby mapping, deterministic
  single/multi-revolution Lambert branches, and both MIMA approximations in
  Rust and Python.
- Object-safe, thread-safe ephemeris providers with explicit optional
  capabilities and metadata, derived period/elements behavior, a Keplerian
  provider, shared Python `Planet`, and GIL-releasing epoch batches.
- JPL low-precision heliocentric ephemerides for Mercury through Neptune,
  traceable coefficient data, explicit 1800–2050 validity checks, all element
  representations, configurable safe radii, and scalar/batch Python access.
- Feature-gated pure-Rust VSOP2013 ephemerides for all nine upstream bodies,
  embedded MPL-2.0 coefficient data down to `1e-9`, direct ICRF state
  evaluation without JIT or network access, and explicit Python availability
  and threshold reporting.
- A pykep-owned adaptive DOP853 integration facade with fixed-size evaluated
  model contracts, exact final-time and backward propagation, dense output,
  terminal events, step diagnostics, and allocation-free first-order
  sensitivity states, plus a reproducible candidate decision benchmark.
- Stateless evaluated Kepler, CR3BP, and bicircular dynamics with explicit
  rotating-frame conventions, CR3BP effective potential and Jacobi constant,
  analytic state/parameter Jacobians, adaptive propagation and STMs, Python
  wrappers, C++ Taylor-reference trajectories, and Criterion benchmarks.
- A validated zero-order-hold control schedule plus Kepler, CR3BP,
  modified-equinoctial, and ideal solar-sail models with deterministic
  switches, backward propagation, fixed-width segment sensitivities, Python
  wrappers, Taylor-reference variations, and scaling benchmarks.
- Cartesian and modified-equinoctial Pontryagin state/costate dynamics for
  mass- and time-optimal control, including controls, Hamiltonians,
  propagation, sensitivities, Python APIs, parity vectors, and benchmarks.
- Fixed- and variable-duration Sims–Flanagan legs with immutable validation,
  mismatch/throttle constraints, analytic gradients, Python classes, and
  C++/independent derivative validation.
- A generic ZOH leg over all four controlled dynamics families, with
  endpoint/control/time-grid sensitivities, state histories, batch evaluation,
  contextual integration failures, and Python bindings.
- A complete typed Python surface audit, including all-export Pytest coverage,
  adversarial NumPy/lifetime/thread tests, explicit GIL-releasing batches,
  migration matrix, clean-wheel CI, Mypy example, and wrapper benchmarks.
- Paired deterministic Rust/Python examples for every major public module,
  plus quick starts and a batch-throughput example.
- Release-candidate packaging, performance regression tooling, bounded fuzz
  targets, memory/cache profiling procedure, and local empty-project smoke
  tests for the crate and wheel.
- A Reynolds-STM overflow fuzz target, reusable Markdown link checker, and
  executable Rustdoc examples on every major module landing page.
- An mdBook documentation site and GitHub Pages deployment workflow.
- Tag-gated crates.io and PyPI release workflows using protected environments,
  short-lived OIDC credentials, installed-artifact tests, and attestations.
- Registry bootstrap instructions and installed-package smoke utilities for
  synchronized `pykep-core` and `pykep-rust` releases.

### Changed

- Enabled independent packaging of `pykep-core`; the internal PyO3
  implementation crate remains `publish = false` and the Python distribution
  remains a separate wheel.
- ZOH state histories use one dense-output solve per segment. Backward
  histories use an equivalent increasing-time coordinate to avoid the
  selected backend's decreasing-time interpolation boundary defect.
- Python stub audits now compare runtime parameter names, order, defaults,
  return-annotation presence, and documented class members.

### Fixed

- Reynolds state-transition matrices now return a typed overflow error instead
  of panicking when finite inputs overflow intermediate products.
- `alpha_to_direct` rejects both zero and one with the documented invalid-input
  error class.
- Extended signed-year ISO strings emitted by `Epoch` now parse back to the
  same epoch.
- Lambert solution properties now expose runtime docstrings, and anomaly
  wrapper parameter names match the shipped stub.
