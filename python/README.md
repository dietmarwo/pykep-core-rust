# Python API

The `pykep_rust` package is a thin, typed Python interface to `pykep-core`.
It currently exposes physical constants, Julian-date arithmetic, stable
Stumpff functions, Kepler-equation residuals and derivatives, and small
three-vector operations.

```python
import pykep_rust as pk

assert pk.jd_to_mjd2000(2_451_544.5) == 0.0
assert pk.cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == [0.0, 0.0, 1.0]
```

Angles are radians and Julian conversions are arithmetic day counts without
an implied UTC, TT, or TDB time scale. Public numerical functions reject NaN
and infinity. Sequence batch functions preserve order and do not create a
thread pool.

The existing upstream Python wrapper is deliberately out of scope for the
first porting pass. Compatibility names and migration helpers will be designed
only after the Rust core has stable numerical behavior.
