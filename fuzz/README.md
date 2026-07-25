# Bounded fuzz targets

These targets exercise parser and numerical validation boundaries without
shipping a fuzzing runtime in `pykep-core`:

```bash
cargo +nightly fuzz run epoch_parser -- -max_total_time=30
cargo +nightly fuzz run element_conversions -- -max_total_time=30
cargo +nightly fuzz run lambert_inputs -- -max_total_time=30
```

`element_conversions` combines arbitrary IEEE-754 inputs with a bounded
physically meaningful mapping. `lambert_inputs` maps bytes to finite positions,
positive time/`mu`, both directions, and up to four revolutions. Any persisted
crash must be reduced and added as a deterministic regression test before a
fix is accepted.

Cargo-fuzz/libFuzzer is development-only and may invoke a native compiler; it
is excluded from the release workspace and packaged core dependency graph.
