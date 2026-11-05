# Trajectory optimization gym

Since 0.1.5 the `pykep-rust` wheels include the current PyKEP mission
definitions and TOPS nonlinear programs from the Rust crate
`pykep-trajopt`. Constructing a problem reads embedded data and creates
native Rust state. The benchmark tools supply optional optimizers.

`pykep-trajopt` is not published on crates.io; Rust users depend on it
through the repository workspace or a path. Maturin uses the feature list
in `pyproject.toml` by default. With an explicit Maturin feature override,
include `trajopt` alongside any other requested features.

| Catalog | Entries |
|---|---|
| Classical trajectories | Cassini1 in three encodings, Cassini2, three EVE variants, Rosetta, Messenger, Earth–Mars with 3/5/7 impulses, JUICE and multiobjective JUICE |
| TOPS fixed boundaries | 5 two-body Cartesian, 9 MEE, 14 CR3BP, 6 solar sail |
| TOPS moving boundaries | The same family-qualified case keys with a moving endpoint formulation |
| Launcher tables | Ariane 5, Ariane 64, Atlas 501, Atlas 551, Soyuz-Fregat, Falcon 9 |

The stable catalog has 82 instance IDs. Each TOPS ID also accepts segment,
cut, time encoding, constraint and integration options, including
`maximum_steps`. These settings belong to the problem specification and are
recorded with experiment results. Classical `gym.*` IDs have fixed
definitions and ignore the TOPS options.

Fixed-boundary TOPS problems default to the upstream limit of 1,000
integration steps per propagation. The upstream moving-boundary factories
impose no limit, so moving problems default to 1,000,000.

```bash
cargo run --release --locked -p pykep-gym-benchmark -- list
cargo run --release --locked -p pykep-gym-benchmark -- evaluate --problem gym.eve_mga1dsm
cargo run --release --locked -p pykep-gym-benchmark -- replay --problem tops.twobody.P4 --nseg 16 --count 64
```

```python
from pykep_rust.trajopt import gym

problem = gym.tops_twobody("P4", nseg=16, time_encoding="softmax")
lower, upper = problem.get_bounds()
x = [(a + b) * 0.5 for a, b in zip(lower, upper)]
result = problem.evaluate(x)
jacobian = problem.gradient(x)
```

`fitness(x)` returns minimized objectives, equalities, then inequalities.
Raw equalities target zero. Raw inequalities are feasible at nonpositive
values. The full gradient is flattened in row-major order for the PyGMO
protocol. `fitness_batch` takes a two-dimensional float64 NumPy array and
returns the same rows in input order. Classical mission functions expose
`has_gradient() == False`.

A decision inside the bounds can still be numerically unusable, for
example a TOPS schedule that exhausts the propellant before the cut. Such
calls raise the mapped integration or numerical exception rather than
returning a penalty, and `fitness_batch` raises for the first failing row
in input order. Wrap the call when an optimizer needs a penalty value; the
native `tools/gym-benchmark` adapters map failures to a large merit.

The Rust API also exposes configurable `ImpulsiveProblem` constructors,
`TopsProblem`, `ProblemSpec`, `Launcher` and the `Problem` trait. The
optimizer adapters live in `tools/gym-benchmark`. Native execution can use
DE, DE followed by CMA-ES, or MODE. Frozen coordinates are removed inside
the solver and restored exactly for evaluation. Replayed results retain
raw physical objectives and residuals.

```bash
cargo run --release --locked -p pykep-gym-benchmark -- optimize \
  --problem gym.eve_mga1dsm --optimizer de-cma --evaluations 4096 \
  --seed 42 --workers 1 --output artifacts/gym/eve
cargo run --release --locked -p pykep-gym-benchmark -- validate \
  --run artifacts/gym/eve/run.json
```

`validate` restores the recorded settings and backend, replays the best
decision with a tighter nominal tolerance and exits with an error if the
objectives differ from the record by more than `1e-6`, a recorded feasible
point becomes infeasible, or a TOPS forward residual exceeds `1e-6`.

The run directory contains a schema-v1 manifest, the complete problem
specification, convergence data and a decoded trajectory when available.
The optional wall deadline is checked between populations. Actual budget
and elapsed time are recorded. PyGMO can wrap the Python problem directly
when installed. The native library has no PyGMO dependency.

See the [TOPS walkthrough](tops.md), [impulsive formulations](trajectory-formulations.md)
and [benchmark results](matched-benchmarks.md).
