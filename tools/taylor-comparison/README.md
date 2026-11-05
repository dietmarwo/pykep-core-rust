# Taylor comparisons

Run the release benchmark with the same compiler flags and features on each
checkout. It reports native warmed medians and scaled endpoint errors against
tight DOP853 reference solves.

```sh
cargo run --release --locked -p pykep-taylor-benchmark --bin model_comparison
cargo run --release --locked -p pykep-taylor-benchmark --features pykep-core/simd --bin model_comparison
```

For official upstream comparisons, install PyKEP 3.0.1 and heyoka 7.10.1 in
a development Python environment alongside the wheel from this checkout.
Export the same native fixtures and pass them to the comparison script.

```sh
cargo run --release --locked -p pykep-taylor-benchmark --bin model_comparison -- --fixtures > fixtures.jsonl
python tools/taylor-comparison/upstream.py fixtures.jsonl
```

Both sides of the upstream comparison use Python calls. The upstream path
reuses an integrator and resets its state. The Rust path includes its Python
wrapper and uses the selected backend. The upstream setup column may include
JIT compilation or a cache hit. Compare timing rows at a common achieved
error threshold. Use the native benchmark for measurements of Rust kernels.

The Linux job runner records status and bounded output, applies an inherited
per-process address-space limit, and samples process-group RSS where procfs
exposes it. It terminates a process group at the deadline. Run heavyweight
jobs in sequence and retain the system or container memory limit.

```sh
python tools/taylor-comparison/run.py --output result.json --timeout 120 -- \
  python tools/taylor-comparison/upstream.py fixtures.jsonl
```

Use distinct output paths when retaining several experiment records.

Use `--reference-only --references output.json` to regenerate the
all-model endpoint reference without importing the Rust extension. The
PyKEP 3.0.1 wheel tested here omitted its TOPS JSON resources. They can be
restored in a development environment from upstream commit
`53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e` under
`pykep/trajopt/gym/tops`. With heyoka.py 7.10.1, the tested wheel pair
also crashed when destroying a PyKEP integrator. In a bounded standalone
comparison process, `--isolated-exit` keeps the integrators alive through
measurement and flushes results before exiting the process.

## Portable and host-tuned builds

The release profile already enables thin LTO and one codegen unit. Use
separate target directories and record `rustc -Vv` with each experiment.
Build both revisions with the same flags for an algorithm comparison.

```bash
RUSTFLAGS='' CARGO_TARGET_DIR=target/taylor-portable \
  cargo build --release --locked -p pykep-taylor-benchmark \
  --features pykep-core/simd --bin model_comparison
RUSTFLAGS='-C target-cpu=native' CARGO_TARGET_DIR=target/taylor-native \
  cargo build --release --locked -p pykep-taylor-benchmark \
  --features pykep-core/simd --bin model_comparison
```

Use the host-tuned binary on the CPU it was built for. Keep distribution
wheels and general release builds on their portable target.

Run three paired rounds with an idle CPU set available on your host. The
comparator preserves all seven samples per pass and computes intervals from
paired rounds. A small number of rounds makes those intervals exploratory.

```bash
python tools/taylor-comparison/compare.py \
  --baseline target/taylor-portable/release/model_comparison \
  --candidate target/taylor-native/release/model_comparison \
  --cpus 2-5 --rounds 3 --output results/cpu-builds.json -- \
  --only=kepler_grid,bcp_population
```

Pass `--scalar-suite --taylor-only --run-tolerance=1e-12` for the original
nominal fixtures at fixed tolerance. `--accuracy-corpus` runs the broader
accuracy checks and reports reference convergence and pass/fail for each
case. Its `--tolerance=` override supports work-precision experiments.
Use `--grid-sweep`, `--batch-sweep`, `--seed-sweep`, and `--leg-sweep` for
complete-call measurements at several grid sizes, population sizes, seed
widths, and segment counts. The first-error batch contract is covered by
`crates/pykep-core/tests/taylor_integration.rs`.

Run `python tools/taylor-comparison/history_sweep.py` with the installed
wheel to measure complete Python ZOH history calls. The script checks each
sample against a separate DOP853 endpoint solve before timing the call.

```bash
cargo test --release --locked -p pykep-core --lib \
  profile_taylor_step_stages -- --ignored --nocapture
```

This developer-only profile separates one step's coefficient generation,
step bound, and fused Horner cost. Complete propagation timings include
validation and other overhead. SIMD utilization is computed from the
accepted-step counts in each static four-lane group.

`run.py --resume` reuses successful results with the same experiment
identity. Pass `--input path/to/fixture` for indirect inputs referenced
inside a script or configuration. Command-line executables and files,
source differences, and the recorded build environment are fingerprinted.

## Native upstream SDK benchmark

The developer-only C++ executable calls an installed PyKEP SDK and
its heyoka integrators. It covers the same eleven nominal fixtures and
reports setup, first call, reset-only, and warmed reset-plus-propagation
measurements. Build with an SDK providing `kep3-config.cmake` and the matching
heyoka headers and libraries.

```bash
python tools/taylor-comparison/native_input.py \
  crates/pykep-core/tests/data/taylor-pykep-v1.json > fixtures.txt
cmake -S tools/taylor-comparison/native -B target/taylor-upstream \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/sdk
cmake --build target/taylor-upstream --parallel 2
python tools/taylor-comparison/run.py --output results/upstream-native.json \
  --timeout 300 -- target/taylor-upstream/pykep-native-comparison fixtures.txt
```

The Python wheel comparison has its own timing boundary. Compare the C++
and Rust endpoint errors before drawing conclusions about native speed.

## fcmaes objective replay

Copy `fcmaes_replay.rs` to `src/main.rs` in a standalone Cargo project. Add
local dependencies on `pykep-core` with feature `simd` and on `fcmaes-core`
from the pinned checkout `8b48919d7e0776c20e60fadcd01b1bae8dc086cd`. Use the
same release profile above and lock the dependencies. Build the baseline and
candidate in separate target directories.

```toml
[dependencies]
pykep-core = { path = "/path/to/pykep-core-rust/crates/pykep-core", features = ["simd"] }
fcmaes-core = { path = "/path/to/fcmaes-rust/crates/fcmaes-core" }

[profile.release]
codegen-units = 1
lto = "thin"
```

Run the executable with `--values --workers=1` or `--workers=4`. The replay
uses the same 1,024 points for both revisions and emits objective and
constraint values for each point. Each DE run uses 4,096 evaluations at
seeds 11, 19, and 42. The optional `--replay-only` mode skips the adaptive
optimizer. Compare objective replay throughput separately from optimizer
outcomes and time to the declared target.

See the [PyKEP 3.1.0 studies](../upstream-comparison/README.md) for the current paired native measurements.
