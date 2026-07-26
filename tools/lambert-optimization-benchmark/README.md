# Lambert optimization benchmark

This native Rust benchmark ports the fixed-leg optimization workload from
[`pykep-lambert`](https://github.com/dietmarwo/pykep-lambert) at commit
`e1e4bb36a9e26470e0f8268180cd3c3c77a48443`. It uses the first two bodies from
`problems/easy.kttsp` and preserves the source defaults:

- decision variables: wait time and time of flight, in days;
- bounds: `[0, 12]` days for wait and `[0.001, 12]` days for flight;
- objective: arrival day plus linear and quadratic penalties above `600 m/s`;
- reporting: delta-v excess through `1e-7 m/s` is considered feasible;
- Lambert search: both directions and up to two complete revolutions;
- optimizer budget: 4,096 evaluations, with a 32-member CMA-ES population.

`pykep-core` supplies mean-anomaly Kepler propagation and all Lambert solution
branches. The crates.io `fcmaes-core` crate supplies native CMA-ES and BiteOpt;
the benchmark does not call Python, Numba, C, or C++.

Run the complete benchmark in release mode:

```bash
cargo run --release -p pykep-lambert-optimization-benchmark
```

Choose one optimizer or shorten a diagnostic run:

```bash
cargo run --release -p pykep-lambert-optimization-benchmark -- \
  --optimizer cma --objective-evaluations 1024 --optimizer-evaluations 1024
```

Use `--help` for all options. Output is line-oriented and reports the fixed
configuration, raw objective evaluations per second, and each optimizer's
best wait, flight time, arrival, delta-v, and penalized objective.

The source project's stored timings remain separate reference evidence. Do
not compare them directly with this executable unless both runs use the same
machine, build mode, evaluation budget, worker policy, and measurement
protocol.
