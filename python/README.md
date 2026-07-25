# Python API

The `pykep_rust` package is a thin, typed Python interface to `pykep-core`.
It currently exposes physical constants, Julian-date arithmetic, stable
Stumpff functions, Kepler-equation residuals and derivatives, and small
three-vector operations. It also provides an immutable, microsecond-resolution
`Epoch` class and elliptic/hyperbolic anomaly conversions.
Cartesian, classical, and modified-equinoctial conversions are available as
scalar functions and NumPy `N × 6` batches; analytic Jacobians use
output-by-input row-major order.
`Planet` owns thread-safe Keplerian or JPL low-precision ephemerides and
evaluates scalar or NumPy epoch batches. Feature-gated VSOP2013 providers
expose their availability and minimum embedded coefficient threshold.

```python
import pykep_rust as pk

assert pk.jd_to_mjd2000(2_451_544.5) == 0.0
assert pk.cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == [0.0, 0.0, 1.0]
epoch = pk.Epoch.from_iso("2030-01")
assert epoch.to_iso() == "2030-01-01T00:00:00.000000"
eccentric = pk.mean_to_eccentric_anomaly(0.1, 0.5)
assert abs(pk.eccentric_to_mean_anomaly(eccentric, 0.5) - 0.1) < 1e-15
state = pk.classical_to_cartesian([7e6, 0.01, 0.4, 1.0, 0.5, 0.2], 3.986004418e14)
earth = pk.Planet.jpl_low_precision("earth")
earth_state = earth.state(0.0)
emb = pk.Planet.vsop2013("earth_moon")
emb_state = emb.state(0.5)
```

Angles are radians and Julian conversions are arithmetic day counts without
an implied UTC, TT, or TDB time scale. Public numerical functions reject NaN
and infinity. Sequence batch functions preserve order and do not create a
thread pool. Numeric `Epoch` arithmetic uses days; `add_seconds()` and
`seconds_since()` make second-based duration arithmetic explicit.

The existing upstream Python wrapper is deliberately out of scope for the
first porting pass. Compatibility names and migration helpers will be designed
only after the Rust core has stable numerical behavior.
