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

Build state belongs in `target/` or `.venv/` and is ignored. Development-only
C++ oracle tools and internal planning notes are not part of this standalone
repository.
