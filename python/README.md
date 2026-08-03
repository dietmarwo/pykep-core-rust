# Python API

`pykep_rust` is the typed, collision-safe Python interface to the native
`pykep-core` implementation. It covers foundations, epochs and anomalies,
element conversion, two-body propagation and STMs, mission-design utilities,
Lambert problems, ephemerides, evaluated dynamics, and low-thrust legs.

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
```

Key conventions:

- Angles are radians.
- Julian conversions are arithmetic day counts without an implied UTC, TT, or
  TDB time scale.
- Numeric `Epoch` arithmetic uses days; `add_seconds()` and `seconds_since()`
  make second-based arithmetic explicit.
- Public numerical functions reject NaN and infinity.
- Ordered batches preserve scalar semantics and release the GIL around native
  work.
- `workers=0` uses Rayon's shared pool, `workers=1` runs serially, and
  `workers=N` uses a cached pool of exactly `N` threads. Avoid nested full-size
  pools.
- Constructors copy their data into immutable native objects.

The wheel ships a `py.typed` marker and complete extension stub. The full
units, shape, default, ownership, and error contract is in
[`docs/python-api.md`](../docs/python-api.md). Users moving from `kep3` should
read [`docs/python-migration.md`](../docs/python-migration.md); no partial
legacy-name facade is installed.
