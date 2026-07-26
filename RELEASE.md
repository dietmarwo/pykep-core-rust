# Release process

The public repository produces two synchronized artifacts:

| Registry | Package | Version source |
|---|---|---|
| crates.io / docs.rs | `pykep-core` | `[workspace.package].version` |
| PyPI | `pykep-rust` | derived by Maturin from `pykep-py` |

The `pykep-py` binding implementation crate remains `publish = false`.
Publication is irreversible: never reuse a version after uploading it to
either registry.

## One-time GitHub and registry setup

1. In `dietmarwo/pykep-core-rust`, create a protected GitHub environment named
   `release`. Require a reviewer and restrict deployment tags to `v*`.
2. In the repository's Pages settings, choose **GitHub Actions** as the Pages
   source. `.github/workflows/docs.yml` will publish the mdBook site from
   `main`; pull requests only build it.
3. In PyPI, create a pending Trusted Publisher for project `pykep-rust` with
   owner `dietmarwo`, repository `pykep-core-rust`, workflow
   `python-release.yml`, and environment `release`. PyPI supports this before
   the first project upload.
4. crates.io requires the first `pykep-core` version to be published manually.
   After that upload, configure its Trusted Publisher with owner `dietmarwo`,
   repository `pykep-core-rust`, workflow `publish-crates.yml`, and
   environment `release`.
5. Only after crates.io accepts that publisher, set the GitHub repository
   variable `CRATES_IO_TRUSTED_PUBLISHING` to `true`:

   ```bash
   gh variable set CRATES_IO_TRUSTED_PUBLISHING \
     --repo dietmarwo/pykep-core-rust \
     --body true
   ```

With the variable absent, the crates.io workflow performs all validation and
intentionally skips the upload job. If the tag names the manually published
bootstrap version, the enabled upload job verifies that the exact version
already exists and skips the duplicate; later versions publish normally.
Neither workflow needs a long-lived registry secret.

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
cargo publish -p pykep-core --dry-run --locked
maturin build --release --locked --manifest-path crates/pykep-py/Cargo.toml
maturin sdist --manifest-path crates/pykep-py/Cargo.toml
```

Inspect the crate list, `.crate`, wheel, sdist, licenses, type stub, README
rendering, and native wheel dependencies. Install the `.crate` through an
empty path-only Cargo project and the wheel and sdist into fresh CPython
environments with `LD_LIBRARY_PATH` and `PYTHONPATH` unset.

The release commit must have the intended version in `Cargo.toml` and
`Cargo.lock`, a dated changelog entry, a clean working tree, passing hosted
CI, and no uncommitted generated artifacts.

## First crates.io release

Log in locally without putting a token in shell history, publish the exact
validated commit, and remove the local credential afterward if it is not
stored by an operating-system credential provider:

```bash
cargo login
cargo publish -p pykep-core --locked
cargo logout
```

Verify the crate page, compile `cargo add pykep-core` in a clean project, and
check the docs.rs build. Then configure crates.io Trusted Publishing and the
repository variable described above. Do this before creating the synchronized
release tag, so the tag can publish both registries.

## Optional TestPyPI rehearsal

Build artifacts from the intended release commit and upload to TestPyPI with a
short-lived test credential or a dedicated trusted workflow. Install with
production PyPI available only for dependencies:

```bash
version="$(python scripts/package_version.py)"
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "pykep-rust==$version"
```

If testing requires changing an artifact, increment the version. Registry
files cannot be replaced.

## Tag and publish

After both trusted publishers are configured, the first crates.io bootstrap
is complete, external API review approves the public-name freeze, and all
pre-release checks pass:

```bash
version="$(python scripts/package_version.py)"
git tag -a "v$version" -m "Release pykep-rust $version"
git push origin "v$version"
```

The tag must exactly equal `v` plus the Cargo package version. The Python
workflow builds and installs CPython 3.11–3.13 wheels for Linux, Windows, and
Intel/Apple-silicon macOS, tests the source distribution, attests every
artifact, and publishes through PyPI Trusted Publishing. The crates workflow
revalidates and publishes through a short-lived crates.io OIDC credential.

Create the GitHub release only after both registry pages and docs.rs have been
verified.

## Post-release verification

Use empty directories rather than local archives:

```bash
cargo new pykep-registry-test
cd pykep-registry-test
cargo add pykep-core
cargo check

python -m venv pykep-wheel-test
pykep-wheel-test/bin/python -m pip install pykep-rust
pykep-wheel-test/bin/python -c \
  'import pykep_rust as pk; print(pk.port_status())'
```

Confirm the README, license, project links, wheel matrix, provenance, and
release notes on crates.io, docs.rs, PyPI, GitHub Pages, and GitHub Releases.
No registry token belongs in this repository.
