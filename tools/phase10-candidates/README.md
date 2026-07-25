# Phase 10 solver-candidate benchmark

This standalone, non-published crate reproduces the decision-gate comparison
between:

- the pykep facade over `differential-equations` 0.6.1 DOP853; and
- `ode_solvers` 0.6.2 DOP853.

Both solve the same six-state Kepler problem from `t = 0` to
`t = 5.785665678258923`, with `rtol = atol = 1e-12`. The release executable
warms both paths, then reports 30 elapsed-time sample means of 200 complete
propagations. The checksum prevents dead-code elimination.

Run:

```bash
cargo run --release --manifest-path tools/phase10-candidates/Cargo.toml
```

This orientation tool is intentionally separate from the production
workspace. The maintained Criterion benchmark for the selected facade is
`crates/pykep-core/benches/integration.rs`.
