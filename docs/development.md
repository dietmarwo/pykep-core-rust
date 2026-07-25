# Development

The workspace MSRV is Rust 1.88.0. The normal local quality gate is:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo +1.88.0 check --workspace --locked
```

Run benchmarks separately so timing work is never hidden in the test suite:

```bash
cargo bench -p pykep-core
cargo run --release -p pykep-release-benchmark -- --quick --check
python python/benchmarks/wrapper_overhead.py --scaling
```

The standalone Phase 10 candidate comparison is reproducible with:

```bash
cargo run --release --manifest-path tools/phase10-candidates/Cargo.toml
```

For Python integration:

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

Coverage:

```bash
cargo llvm-cov -p pykep-core --all-targets --summary-only
cargo llvm-cov --workspace --all-targets --summary-only
```

Release-candidate dynamic checks:

```bash
cargo +nightly miri test -p pykep-core --lib --no-default-features fixed_matrix_operations
cargo +nightly fuzz run epoch_parser -- -max_total_time=30
cargo +nightly fuzz run element_conversions -- -max_total_time=30
cargo +nightly fuzz run lambert_inputs -- -max_total_time=30
valgrind --error-exitcode=1 --leak-check=full \
  target/release/pykep-release-benchmark --quick
```

The full protocol, profiler commands/results, and Miri floating-point scope are
documented in [stabilization.md](stabilization.md). Packaging and
clean-artifact consumption are documented in `RELEASE.md`.

Build state belongs in `target/` or `.venv/` and is ignored. Development-only
C++ oracle tools and internal planning notes are not part of this standalone
repository.
