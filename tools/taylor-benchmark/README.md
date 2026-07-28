# Taylor integration benchmark

Run the fixed matched-accuracy protocol from the repository root:

```bash
cargo run --release -p pykep-taylor-benchmark \
  > docs/data/taylor-kepler-benchmark.csv
```

The output records medians, final-state error against analytical Lagrange
propagation, energy drift, and backend-specific work counters. A Taylor
coefficient sweep is not an ordinary RHS call, so `work_units` must not be
compared across methods; wall time at matched accuracy is the primary metric.
