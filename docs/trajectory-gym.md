# Trajectory optimization gym

The development checkout includes the current PyKEP mission definitions and
TOPS nonlinear programs in the separate Rust crate `pykep-trajopt`. The
wheel built by this checkout enables its `trajopt` feature. Constructing a
problem reads embedded data and creates native Rust state. The benchmark
tools supply optional optimizers.

`pykep-trajopt` is currently a workspace or path dependency. Maturin uses
the feature list in `pyproject.toml` by default. With an explicit Maturin
feature override, include `trajopt` alongside any other requested features.

| Catalog | Entries |
|---|---|
| Classical trajectories | Cassini1 in three encodings, Cassini2, three EVE variants, Rosetta, Messenger, Earth–Mars with 3/5/7 impulses, JUICE and multiobjective JUICE |
| TOPS fixed boundaries | 5 two-body Cartesian, 9 MEE, 14 CR3BP, 6 solar sail |
| TOPS moving boundaries | The same family-qualified case keys with a moving endpoint formulation |
| Launcher tables | Ariane 5, Ariane 64, Atlas 501, Atlas 551, Soyuz-Fregat, Falcon 9 |

The stable catalog has 82 instance IDs. Each TOPS ID also accepts segment,
cut, time encoding, constraint and integration options. These settings
belong to the problem specification and are recorded with experiment results.

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

The run directory contains a schema-v1 manifest, the complete problem
specification, convergence data and a decoded trajectory when available.
The optional wall deadline is checked between populations. Actual budget
and elapsed time are recorded. PyGMO can wrap the Python problem directly
when installed. The native library has no PyGMO dependency.

See the [TOPS walkthrough](tops.md), [impulsive formulations](trajectory-formulations.md)
and [benchmark results](matched-benchmarks.md).
