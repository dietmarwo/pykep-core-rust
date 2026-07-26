# Examples and quick starts

The examples are deliberately small and deterministic. Runtime statements are
orientation for a release build, not performance guarantees; use the
maintained benchmark harnesses for measurements.

## Rust quick start

Add the native crate and propagate a normalized circular orbit:

```rust
use pykep_core::astro::propagation::propagate_lagrangian;

fn main() -> pykep_core::Result<()> {
    let initial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let final_state =
        propagate_lagrangian(&initial, core::f64::consts::FRAC_PI_2, 1.0)?;
    assert!((final_state[1] - 1.0).abs() < 1e-12);
    Ok(())
}
```

From this workspace, run `cargo run --release -p pykep-examples --bin
propagation`. The default feature set includes the embedded VSOP2013
coefficients; use `pykep-core` with `default-features = false` when they are
not needed.

## Python quick start

Install a wheel in a virtual environment and use the same Rust core:

```bash
python -m venv .venv
.venv/bin/python -m pip install target/wheels/pykep_rust-*.whl
.venv/bin/python python/examples/elements_propagation.py
```

```python
import numpy as np
import pykep_rust as pk

states = np.tile([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (4096, 1))
times = np.linspace(0.0, 1.0, len(states))
result = pk.propagate_lagrangian_batch(states, times, 1.0, workers=8)
assert result.shape == (4096, 6)
```

The Python package ships type information and accepts NumPy batch arrays only
at `float64`; see [python-api.md](python-api.md) and
[batch-processing.md](batch-processing.md).

## Runnable matrix

| Capability | Rust binary | Python script | Units and expected output | Runtime / features |
|---|---|---|---|---|
| Epoch/anomaly | `epoch-anomalies` | `epoch_anomalies.py` | MJD2000 days/radians; 180-day offset and exact round trips | Constant, normally <1 ms; default |
| Elements | `elements` | `elements_propagation.py` | SI/radians; finite state and stable element round trip | Constant, normally <1 ms; default |
| Propagation/STM | `propagation` | `elements_propagation.py` | Normalized or consistent SI; quarter orbit / 60-second state | Constant, normally <1 ms; default |
| Lambert | `lambert` | `lambert.py` | Normalized; seven ordered zero/multi-revolution branches | Bounded, normally <1 ms; default |
| Ephemerides | `ephemeris-comparison` | `ephemeris_comparison.py` | MJD2000, metres, m/s; two finite frame-labelled states | First call normally <1 ms; default `vsop2013` |
| Gravity assist | `gravity-assist` | `gravity_assist.py` | SI/radians; constraints, positive delta-v, outgoing velocity | Constant, normally <1 ms; default |
| Sims–Flanagan | `low-thrust-legs` | `low_thrust_legs.py` | Consistent units; 7 mismatch values and 7 × 13 Jacobian | Four segments, normally <1 ms; default |
| CR3BP/ZOH | `dynamics` | `dynamics.py` | Normalized; finite six-/seven-state propagation | Two short adaptive solves, normally <1 ms; default |
| Batch throughput | core Criterion benches | `batch.py` | Normalized; newly owned 4096 × 6 output | One native O(N) call; NumPy |

Every Rust binary is compiled by the workspace test and clippy gates. Every
Python script is executed by `python/tests/test_examples.py`, including from
the clean-wheel CI matrix.

## Documentation map

- [conventions.md](conventions.md) defines units, epochs, frames, layout, and
  accuracy interpretation.
- Dynamics, Pontryagin, zero-order-hold, ephemeris, Sims–Flanagan, and generic
  ZOH-leg behavior have dedicated module guides in this directory.
- [python-migration.md](python-migration.md) maps the C++/Python upstream
  surface and records deliberate differences and unavailable ecosystem areas.
- The `decisions/` records explain fixed-size types, errors, VSOP data, and
  adaptive-integration dependencies.
- [performance.md](performance.md) gives benchmark methodology and results.
- [status.md](status.md), [source-map.md](source-map.md), and
  [validation.md](validation.md) state implemented scope, limitations, and
  evidence without implying support for deferred modules.
