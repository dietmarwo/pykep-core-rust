# Contributing

Numerical changes must preserve the conventions and evidence requirements in
`docs/validation.md` and update `docs/source-map.md`.

Before opening a change:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

Run the MSRV check separately:

```bash
cargo +1.88.0 check --workspace --locked
```

Python-facing changes must build the extension in a clean virtual
environment, run Pytest, update runtime docstrings and `.pyi` declarations,
and demonstrate that wrappers call `pykep-core` rather than duplicate
numerical formulas.

Translated or adapted code must retain `SPDX-License-Identifier: MPL-2.0`,
identify its pinned upstream source, and document material algorithm changes.
Never commit generated build directories, virtual environments, extension
modules, or credentials.
