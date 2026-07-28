# Adding an ODE system

This guide describes the complete path for adding a first-order ordinary
differential equation to `pykep-core`, including the evaluated Rust model,
sensitivities, the optional fixed-system Taylor backend, batches, Python
bindings, tests, benchmarks, and documentation.

The central contract is

```text
dy/dt = f(t, y, p)
```

where the state has a compile-time dimension `N` and the constant parameter
vector has a compile-time dimension `P`.

## Decide the scope first

There are two materially different extension levels.

### External or DOP853-only model

Any downstream crate can implement
`DynamicsModel<N, P>` and pass the model to `Dop853`. This is the stable public
extension point. It supports nominal propagation, dense output, events, and,
after implementing `DifferentiableDynamicsModel`, direct variational
sensitivities.

An external model cannot implement Taylor support. `TaylorCoefficientModel`
is private and `TaylorDynamicsModel` is deliberately sealed. This prevents the
crate from promising a symbolic/coefficient API before it has a stable
third-party contract.

### Built-in pykep model

A model shipped by this repository normally needs the complete product
surface:

- evaluated Rust right-hand side;
- domain validation and typed errors;
- state and parameter Jacobians when sensitivities are meaningful;
- Taylor coefficients and the `TaylorDynamicsModel` marker;
- convenience propagation methods;
- ordered scalar and parallel-batch APIs where users will evaluate many
  independent cases;
- Python scalar and batch bindings if the model is part of the Python
  product;
- independent validation, performance evidence, Rustdoc, mdBook
  documentation, source-map/status updates, and a changelog entry.

Do not add a model to the default `AdaptiveIntegrator` surface until its
Taylor implementation exists and has been validated. A deliberately
DOP853-only built-in model must use `Dop853` directly and document that
limitation.

## 1. Write down the numerical contract

Before coding, record:

1. The model name and source equation or upstream reference.
2. `N`, the state dimension, and the exact order of every state component.
3. `P`, the parameter dimension, and the exact order of every parameter.
4. Units, normalization, coordinate frame, epoch/time convention, and sign
   conventions.
5. The valid domain of every state and parameter.
6. Singular geometries and non-analytic switching surfaces.
7. Whether the equations depend explicitly on time.
8. Conserved quantities, symmetries, closed-form cases, equilibria, or
   reversible cases that can serve as independent checks.
9. Whether analytic state and parameter Jacobians are available.
10. Whether users need dense output, events, sensitivities, ZOH control,
    batches, or Python access.

Treat state and parameter ordering as API. Changing it later is a breaking
change even if the Rust type remains `[f64; N]`.

## 2. Add the evaluated Rust model

Place a substantial new family in
`crates/pykep-core/src/dynamics/<system>.rs` and export the module from
`crates/pykep-core/src/dynamics.rs`. A small closely related model can live
beside its family.

Prefer a zero-sized, copyable model:

```rust
use pykep_core::integration::{DynamicsModel, Dop853, InitialValueProblem};
use pykep_core::integration::IntegratorOptions;
use pykep_core::{PykepError, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OscillatorDynamics;

impl DynamicsModel<2, 1> for OscillatorDynamics {
    const NAME: &'static str = "harmonic oscillator";

    fn validate(
        &self,
        time: f64,
        state: &[f64; 2],
        parameters: &[f64; 1],
    ) -> Result<()> {
        if !time.is_finite() {
            return Err(PykepError::NonFiniteInput { parameter: "time" });
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err(PykepError::NonFiniteInput { parameter: "state" });
        }
        if !parameters[0].is_finite() {
            return Err(PykepError::NonFiniteInput {
                parameter: "angular_frequency",
            });
        }
        if parameters[0] <= 0.0 {
            return Err(PykepError::InvalidInput {
                parameter: "angular_frequency",
                reason: "must be greater than zero".into(),
            });
        }
        Ok(())
    }

    fn rhs(
        &self,
        time: f64,
        state: &[f64; 2],
        parameters: &[f64; 1],
        derivative: &mut [f64; 2],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let omega = parameters[0];
        *derivative = [state[1], -omega * omega * state[0]];
        if derivative.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err(PykepError::NumericalOverflow {
                operation: Self::NAME,
            })
        }
    }
}
```

Inside `pykep-core`, reuse the crate-private finite-input and finite-output
helpers instead of duplicating them. External crates should construct the
public `PykepError` variants as above.

The implementation rules are:

- `rhs` writes every derivative component into caller-owned storage.
- `rhs` must not allocate.
- Validate before divisions, roots, logarithms, normalizations, or other
  domain-sensitive operations.
- Do not silently clamp, normalize, or cross a physical singularity.
- Use `InvalidInput` for finite values outside the declared domain,
  `NonFiniteInput` for NaN/infinity, `SingularGeometry` for mathematically
  undefined geometry, and `NumericalOverflow` for a non-finite result from
  otherwise finite inputs.
- Keep model-domain errors distinct from `IntegrationFailure`.
- Return a stable, descriptive `NAME`; integration errors include it.
- Do not keep mutable propagation state in the model object. Constant model
  configuration belongs in the parameter array or in an immutable model
  value.

An external model is immediately usable with DOP853:

```rust
let result = Dop853.propagate(
    &OscillatorDynamics,
    InitialValueProblem::new(0.0, [1.0, 0.0], 10.0, [2.0]),
    IntegratorOptions::default(),
)?;
```

Taylor event location is not implemented. Event-driven propagation must use
`Dop853`, even for a built-in Taylor-capable model.

## 3. Add Jacobians and sensitivity support

Implement `DifferentiableDynamicsModel<N, P>` when state-transition matrices,
parameter sensitivities, or gradients are part of the model's use case:

```rust
use pykep_core::integration::DifferentiableDynamicsModel;

impl DifferentiableDynamicsModel<2, 1> for OscillatorDynamics {
    fn jacobians(
        &self,
        time: f64,
        state: &[f64; 2],
        parameters: &[f64; 1],
        state_jacobian: &mut [[f64; 2]; 2],
        parameter_jacobian: &mut [[f64; 1]; 2],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let omega = parameters[0];
        *state_jacobian = [[0.0, 1.0], [-omega * omega, 0.0]];
        *parameter_jacobian = [[0.0], [-2.0 * omega * state[0]]];
        Ok(())
    }
}
```

Matrix layout is output-by-input:

```text
state_jacobian[i][j]     = d f_i / d state_j
parameter_jacobian[i][k] = d f_i / d parameter_k
```

Always overwrite both complete output matrices. Never assume the caller
provided zeros.

Prefer analytic Jacobians. If a family uses numerical Jacobians, reuse the
existing scale-aware helper and list every strictly positive state or
parameter index. Numerical differentiation must not step across a mass,
radius, barrier, or other one-sided domain.

DOP853 integrates these Jacobians directly. The current Taylor sensitivity
path uses centered differences of complete Taylor propagations and scales as
`2W + 1` propagations for `W` seed directions. Keep DOP853 as the default for
wide sensitivity matrices unless measurement supports a different choice.

## 4. Add convenience APIs

For a public built-in model, mirror the shape of the existing dynamics APIs:

- `evaluate`;
- `propagate`;
- `propagate_with_method`;
- `propagate_with_stm` or a general sensitivity method;
- `propagate_with_stm_method`;
- model-specific invariants or controls, if they are already part of the
  underlying physics.

The no-suffix `propagate` method may use `AdaptiveIntegrator::default()` only
after Taylor support is complete. The method-selecting variant should accept
`IntegrationMethod`, and documentation must say which method is the default.

Avoid inventing derived physics merely to make an API appear symmetrical.
Expose only quantities actually defined by the implemented system.

## 5. Add the fixed-system Taylor evaluator

This section applies only to models shipped inside `pykep-core`.

Add the model to
`crates/pykep-core/src/integration/taylor/systems.rs` by implementing the
private coefficient contract and the public sealed marker:

```rust
impl TaylorCoefficientModel<2, 1> for OscillatorDynamics {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 2],
        parameters: &[f64; 1],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 2],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        oscillator_tape().coefficients(time, state, parameters, order, jet);
        Ok(())
    }
}

impl TaylorDynamicsModel<2, 1> for OscillatorDynamics {}
```

There are two implementation strategies.

### Specialized recurrence

Use a hand-written recurrence for a small equation where doing so clearly
reduces work. The Kepler and ZOH Kepler implementations are the templates.

The recurrence must:

1. Clear `jet`.
2. Copy the initial state into coefficient zero.
3. Advance only coefficient `n + 1` from already available coefficients
   `0..=n`.
4. Divide the right-hand-side coefficient by `n + 1`.
5. Stop at the requested `order`, never beyond `MAX_ORDER`.

Provide an independent coefficient reference in tests. Hand-written
recurrences without coefficient-level tests are not acceptable.

### Cached expression tape

Use `TapeBuilder` for a larger fixed expression:

```rust
use std::sync::OnceLock;

fn oscillator_tape() -> &'static IncrementalTape<2> {
    static TAPE: OnceLock<IncrementalTape<2>> = OnceLock::new();
    TAPE.get_or_init(build_oscillator_tape)
}

fn build_oscillator_tape() -> IncrementalTape<2> {
    let builder = TapeBuilder::new();
    let outputs = {
        let position = builder.state(0);
        let velocity = builder.state(1);
        let omega = builder.parameter(0);
        [velocity, -omega * omega * position].map(Expression::index)
    };
    builder.finish(outputs)
}
```

The lexical block is important: `finish` consumes the builder after all
borrowed expressions have been reduced to output indices.

The tape currently supports:

- constants, time, state components, and parameters;
- addition, subtraction, multiplication, division, and negation;
- real powers and square roots;
- exponentials;
- sine, cosine, and paired `sin_cos`;
- `stop_gradient`;
- reverse symbolic gradients.

Use `builder.time()` for explicit time dependence. Treating the initial time
as a constant produces incorrect higher-order coefficients.

`TapeBuilder` performs structural common-subexpression elimination. Reuse
the same algebraic form where practical, but do not obscure the equations
solely to reduce the operation count.

If the system needs a new elementary operation, adding only an evaluated
operation is insufficient. Extend all of these together:

1. the `Operation` enum;
2. the builder and `Expression` API;
3. constant folding;
4. the incremental Taylor recurrence;
5. reverse differentiation, when gradients can traverse the node;
6. companion/workspace metadata, if required;
7. direct elementary-function and derivative tests in `tape.rs`.

### Piecewise expressions and optimal control

A Taylor series is valid only on an analytic branch. For algebraically
equivalent stable formulas, keep one `OnceLock` tape per branch and select the
branch from coefficient-zero state/parameter values at the start of each
step. Test every branch and its boundary policy.

For minimized Pontryagin Hamiltonians, controls may evolve as Taylor series
while remaining excluded from the Hamiltonian partial derivative. Wrap those
direction or throttle expressions with `stop_gradient`, matching the envelope
convention. Omitting this changes the costate equations.

Do not smooth a real discontinuity merely to make Taylor integration possible.
Document the boundary and use segmented propagation or DOP853 when the
physical model requires it.

## 6. Add ZOH support when applicable

If controls are constant over user-supplied segments, implement the internal
`ZeroOrderHoldModel<N, C, K, P>` mapping:

```text
parameters = parameters(control[C], constants[K])
```

Also map control and constant sensitivity seeds into parameter seeds. Define
which side owns an exact switch time, validate the complete schedule once,
and test forward and backward segment traversal.

Add the model to every relevant ZOH dispatch point:

- scalar segment propagation;
- dense/history output;
- sensitivities;
- leg support, if the state/control semantics match the leg;
- Python `ZohModel` selection, if exposed there.

Do not hide a control lookup inside `rhs`; a validated schedule should split
the integration into constant-parameter segments.

## 7. Add ordered batch APIs

Provide a batch when callers will naturally evaluate many independent
states, controls, epochs, or final times. A batch is an explicit API
extension, not an implicit change to scalar behavior.

Use `pykep_core::batch::try_map` so that:

- `workers = 0` uses Rayon's global pool;
- `workers = 1` is deterministic serial execution;
- larger values use a cached fixed-size pool;
- outputs preserve input order;
- if multiple items fail, the earliest failing input is reported.

Test empty input, one item, multiple worker counts, mismatched input lengths,
deterministic output order, and deterministic error order. Every batch result
must match the scalar API for the same item.

Avoid nested parallelism. If the objective is parallelized at a higher level,
call the scalar or `workers = 1` model API inside each objective evaluation.

## 8. Add Python bindings

Python exposure is optional for a low-level internal model and expected for a
user-facing pykep model.

Implement bindings in `crates/pykep-py/src/dynamics.rs`:

1. Parse dynamically sized Python inputs into fixed Rust arrays before
   releasing the GIL.
2. Call the same `pykep-core` scalar implementation; do not duplicate the
   equations in the binding.
3. Convert `PykepError` with the shared `to_python` mapping.
4. Release the GIL with `Python::detach` for propagation and batch work.
5. Use NumPy `N × state_dimension` arrays for batch states.
6. Accept `workers` on parallel batches and preserve input order.
7. Register every function in `dynamics::register`.

Then update:

- `python/pykep_rust/__init__.py` imports and `__all__`;
- `python/pykep_rust/_pykep_rust.pyi` signatures and docstrings;
- `python/tests/test_smoke.py` for scalar values and error mapping;
- `python/tests/test_parallel_batch.py` for scalar/batch parity, shapes,
  worker counts, and deterministic failures;
- a Python example when the usage is not obvious.

Use consistent names:

```text
<system>_rhs
<system>_rhs_batch
propagate_<system>
propagate_<system>_batch
propagate_<system>_with_stm
propagate_<system>_with_stm_batch
```

If the Rust model supports both DOP853 and Taylor but Python intentionally
exposes only the default, state that explicitly. Do not expose internal tape
objects.

## 9. Build the test pyramid

No single reference is sufficient. Add the following layers in order.

### Evaluated right-hand side

- One or more hand-computed or authoritative reference states.
- Exact zero/control-free/equilibrium reductions where available.
- Explicit-time cases at more than one time.
- Forward-frame and sign-convention checks.
- Every invalid parameter range.
- Every singular geometry.
- NaN and infinity for time, state, and parameters.
- Finite inputs that would overflow the result.

Put small local tests beside the implementation. Put source-parity and
cross-family tests in a dedicated
`crates/pykep-core/tests/phase*_*.rs` integration test.

### Jacobians

Compare every state and parameter column with independent, scale-adjusted
central differences over several non-singular states. Use perturbations based
on each variable's scale, not one absolute epsilon for all columns.

Also test:

- the documented output-by-input layout;
- analytically zero columns;
- one-sided positive domains;
- consistency between the Jacobian and propagated sensitivities.

### Propagation

Use at least two independent checks:

- a closed-form solution, invariant, upstream trajectory, or independently
  generated fixture;
- a same-problem DOP853 comparison at a tighter reference tolerance.

Cover forward and backward time, zero duration, dense sampling, step limits,
rejected steps when relevant, and a duration long enough to expose drift or
branch errors. Validation tolerances must be justified by the state scale and
reference accuracy; they are not general user guarantees.

### Taylor coefficients

Keep a test-only full-series form of the right-hand side and compare its jet
with the optimized recurrence/tape at representative orders such as
`[8, 15, MAX_ORDER]`.

The test should verify:

- every state component and every coefficient;
- explicit-time coefficients;
- trigonometric/exponential/power paths used by the model;
- every stable algebraic branch;
- an operation-count ceiling for expression tapes.

Choose a scaled relative comparison:

```text
abs(incremental - reference) <= tolerance * max(abs(reference), 1)
```

High-order coefficients of poorly scaled systems can be ill-conditioned.
Use a realistic state, record why a looser coefficient tolerance is needed,
and retain the independent propagated-state comparison.

### Cross-backend and regression tests

Add the model to the central Taylor-versus-DOP853 tests. If an upstream
heyoka/C++ implementation exists, generate committed fixtures with a pinned
upstream version and keep the generator outside the normal runtime
dependency graph.

Compare physical outputs, not only internal work counters. Taylor coefficient
sweeps and DOP853 RHS evaluations are not equivalent units.

### Batch and Python tests

For every scalar Python function, test the corresponding batch when one
exists. Verify dtype/shape, empty batches, mismatched lengths, exception
classes, worker behavior, and equality to scalar calls.

## 10. Add benchmarks and apply a performance gate

Add Criterion entries in `crates/pykep-core/benches/dynamics.rs` for:

- one RHS evaluation;
- one representative DOP853 propagation;
- one Taylor propagation for a Taylor-capable built-in;
- one state/parameter sensitivity propagation when it is important;
- representative scalar and batch throughput.

Benchmark release builds. Warm lazy `OnceLock` tapes before recording steady
state, but report first-call setup separately if it is material.

For an optimization, declare the benchmark, correctness tolerance, and
minimum meaningful gain before changing the implementation. Pin:

- initial time, state, parameters, and final time;
- integration tolerances and maximum step;
- release profile and dependency versions;
- iteration/sample counts;
- CPU affinity when comparing small kernels;
- final-state checksum or independent accuracy measure.

Keep an optimization only when the gain is larger than run-to-run noise and
all accuracy checks remain satisfied. Record development-host timings as
evidence, not portable latency promises.

## 11. Document the model

Every public type and method needs Rustdoc describing:

- state and parameter order;
- units and reference frame;
- default integration backend;
- singularities and domain restrictions;
- output layout;
- `# Errors`;
- a short runnable example for a non-obvious API.

Update the relevant mdBook pages:

- `docs/dynamics.md`, `docs/zero-order-hold.md`, or
  `docs/pontryagin.md`;
- `docs/taylor-integration.md` when Taylor support changes;
- `docs/python-api.md` and `docs/python-migration.md` for Python;
- `docs/batch-processing.md` for a new batch family;
- `docs/validation.md` with the independent evidence and tolerances;
- `docs/performance.md` with the benchmark protocol and interpretation;
- `docs/status.md` and `docs/source-map.md`;
- `docs/SUMMARY.md` for any new page;
- `CHANGELOG.md` under `Unreleased`.

Update model counts such as “eleven built-in models” wherever the new model
changes them. Search for the old count rather than fixing only one page:

```bash
rg -n "eleven|11 built-in|supported models|TaylorDynamicsModel" \
  README.md CHANGELOG.md docs crates
```

If the model ports an upstream equation, add the exact source file, upstream
version/commit, and Rust destination to `docs/source-map.md`.

## 12. Run the complete quality gate

From the `public` repository:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features --locked
cargo test --workspace --all-features --locked --release
cargo test -p pykep-core --no-default-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo +1.88.0 check --workspace --locked
python tools/check_markdown_links.py
mdbook build
```

For Python changes, build the extension in an isolated environment and run
the Python suite:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "maturin[patchelf]>=1.7,<2" pytest
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" \
  PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/maturin develop --release \
  --manifest-path crates/pykep-py/Cargo.toml
.venv/bin/python -m pytest
```

Also run the relevant Criterion benchmark and coverage report:

```bash
cargo bench -p pykep-core --bench dynamics
cargo llvm-cov -p pykep-core --all-features --all-targets --summary-only
```

Use Miri for new unsafe-sensitive structure or recurrence code if applicable.
The crate forbids unsafe code, but Miri can still catch invalid assumptions in
dependency-free state manipulation. Fuzz new parsers or dynamically shaped
input boundaries; do not fuzz a fixed smooth RHS merely to increase a metric.

## Definition of done

A built-in ODE system is complete only when:

- [ ] State, parameter, unit, frame, and domain contracts are written down.
- [ ] `DynamicsModel` is allocation-free, validated, and fully documented.
- [ ] Errors use the stable public taxonomy.
- [ ] Jacobians are implemented and independently checked, or their absence
      is explicitly justified.
- [ ] Taylor support is implemented and coefficient-tested, or the model is
      explicitly documented as DOP853-only.
- [ ] Propagation matches at least two independent references or invariants.
- [ ] Forward, backward, zero-duration, singular, and non-finite cases pass.
- [ ] Every analytic branch and control switch policy is tested.
- [ ] Scalar and batch APIs agree for all worker modes.
- [ ] Python functions, stubs, exports, error mapping, and tests are complete
      when Python exposure is in scope.
- [ ] RHS, propagation, sensitivity, and batch benchmarks are recorded where
      relevant.
- [ ] Rustdoc, mdBook, status, source map, validation, performance, and
      changelog are updated.
- [ ] Formatting, Clippy, debug/release/no-default tests, Rustdoc, MSRV,
      Markdown links, mdBook, Python tests, and relevant coverage checks pass.
- [ ] The `public` worktree contains only intended changes and the work is
      committed.

