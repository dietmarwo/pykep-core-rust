# Matched PyKEP studies

The reference is [PyKEP 3.1.0 at c01e19de](https://github.com/esa/pykep/tree/c01e19debad5ea22b7079189f786f473046a79d5).
The C++ SDK, Python oracle and optional reference optimizers belong to the
developer comparison environment.

## Reference environment

Build the pinned source as a release SDK with a C++23 compiler, heyoka 7.10.0,
Boost 1.88, fmt 12.1, xtensor 0.27.1, xtensor-blas 0.23.0 and OpenBLAS.
The Python studies used CPython 3.12, heyoka.py 7.10.1 and PyGMO 2.20.0.
The SDK and Python extension must use a compatible heyoka ABI. The
[reference SDK notes](reference-sdk/README.md) record the exact compiler,
dependency revisions and header adaptation used for the 0.1.5 measurements.

```bash
git clone https://github.com/esa/pykep.git upstream-pykep
git -C upstream-pykep checkout --detach c01e19debad5ea22b7079189f786f473046a79d5
cmake -S upstream-pykep -B target/kep3-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$DEPS_PREFIX" \
  -DCMAKE_INSTALL_PREFIX="$SDK_PREFIX" -DPYKEP_INSTALL_PATH="$SDK_PREFIX/python" \
  -Dkep3_BUILD_PYTHON_BINDINGS=ON
cmake --build target/kep3-build --parallel 2
cmake --install target/kep3-build
cmake -S tools/upstream-comparison/native -B target/upstream-native \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$SDK_PREFIX;$DEPS_PREFIX"
cmake --build target/upstream-native --parallel 2
cmake -S tools/taylor-comparison/native -B target/taylor-native \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$SDK_PREFIX;$DEPS_PREFIX"
cmake --build target/taylor-native --parallel 2
cargo build --release --locked -p pykep-upstream-benchmark \
  -p pykep-taylor-benchmark --features pykep-core/simd
```

Set `PYTHONPATH` to the reference Python installation for the Python studies.
Configure the loader for the SDK libraries and verify the imported module path.

## Native benchmarks

Choose a logical CPU available to the process. Start with the smoke profile.

```bash
python tools/upstream-comparison/matched_study.py \
  --exporter target/upstream-native/pykep-export-fixtures \
  --native target/upstream-native/pykep-native-benchmark \
  --rust target/release/pykep-upstream-benchmark --output results/native-smoke --cpu 2
python tools/upstream-comparison/taylor_study.py \
  --native target/taylor-native/pykep-native-comparison \
  --rust target/release/model_comparison --output results/taylor --cpu 2
```

`--profile full` uses the counts in [specs.json](specs.json). Repeat
`--only` to select suites. The ZOH set uses five passes per arm and
200 cases per segment count. Run the source derivative tolerance of
`1e-8` and a separate `--derivative-tol 1e-13` arm. The Rust executable
also accepts `--zoh-backend=dop853` on the same fixture.

Each fixture begins with `PYKEP_BENCH_V1 SUITE ROWS WIDTH` and a newline.
The payload contains `ROWS * WIDTH` little-endian binary64 values. The
runner records the fixture and executable hashes. Retain the fixture
bytes when repeating a comparison on another platform. The SF population
uses a shared deterministic stream.

## Python and optimizer studies

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p results
python tools/upstream-comparison/gym_oracle.py --nseg 16 --samples 2 \
  --reference-derivative-tol 1e-13 --output results/gym.json
python tools/upstream-comparison/notebooks.py upstream-pykep symbolic
python tools/upstream-comparison/notebooks.py upstream-pykep interface
target/upstream-native/pykep-export-fixtures sf-conv5 100 results/sf5.bin
python tools/upstream-comparison/sf_study.py results/sf5.bin \
  --output results/sf5-convergence.json
cargo build --release --locked -p pykep-gym-benchmark
python tools/gym-benchmark/study.py --binary target/release/pykep-gym-benchmark \
  --cpu 2 --output results/gym-optimizer-pilot --pilot
```

Use separate output directories for changed binaries, datasets or budgets.
`study.py` in this directory executes an ordered job manifest through the
Linux bounded runner. It checkpoints status and preserves failures. The
default address-space limit is 4 GiB per process.
[Matched benchmark report](../../docs/matched-benchmarks.md)
