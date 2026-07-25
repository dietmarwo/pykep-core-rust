# Release process

`pykep-core` and the `pykep-rust` Python distribution are separate artifacts.
The `pykep-py` binding implementation crate remains `publish = false`.

## Reproducible candidate

From a clean checkout:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features --locked
cargo test --workspace --all-features --locked --release
cargo test -p pykep-core --no-default-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo +1.88.0 check --workspace --locked
cargo audit
cargo deny check
cargo package -p pykep-core --locked
maturin build --release --locked --manifest-path crates/pykep-py/Cargo.toml
maturin sdist --manifest-path crates/pykep-py/Cargo.toml
```

Inspect the crate list, `.crate`, wheel, sdist, licenses, SBOM, type stub, and
native wheel dependencies. Install the `.crate` through an empty path-only
Cargo project and the wheel into a fresh CPython environment with
`LD_LIBRARY_PATH`/`PYTHONPATH` unset. The CI matrix repeats wheel builds on
CPython 3.11–3.13 for Linux, macOS, and Windows.

## External release boundary

Before publication, a release owner must:

1. assign the permanent repository URL, owners, security contact, and trusted
   publisher environments;
2. conduct external API review and explicitly approve the public-name freeze;
3. publish `pykep-core` to crates.io and `pykep-rust` to PyPI using trusted
   OIDC publishing from a green protected environment;
4. download both registry artifacts into empty projects and repeat the smoke
   tests, rather than substituting local archives;
5. verify registry metadata and docs.rs, then create the tag and hosted
   release from the exact green reproducible commit.

No registry token belongs in the repository. Never reuse a published version
or tag a build that did not pass clean-artifact tests. Local packaging is not
evidence that registry publication or cross-platform CI has completed.
