# C++ golden data

These files contain portable numerical reference values generated from
pykep/kep3 3.0.1 commit
`53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e`.

Finite `f64` values are stored as C99 hexadecimal strings. Metadata in each
file records the schema, units, generator, compiler, source revision, and
random seed. The development-only exporter lives outside the standalone
public repository; no C++ code or runtime is part of `pykep-core`.

Regenerate the foundation and phase-specific data from the staging source
root:

```bash
cmake -S oracle -B .oracle-build -G Ninja
cmake --build .oracle-build
.oracle-build/foundations_oracle
.oracle-build/phase3_oracle
.oracle-build/phase4_oracle
.oracle-build/phase5_oracle
.oracle-build/phase6_oracle
.oracle-build/phase7_oracle
.oracle-build/phase8_oracle
.oracle-build/phase9_oracle
.oracle-build/phase13_oracle
```

Compare the output byte-for-byte with `foundations-v1.json` and
`phase3-v1.json`, `phase4-v1.json`, `phase5-v1.json`, `phase6-v1.json`, and
`phase7-v1.json`, `phase8-v1.json`, and `phase9-v1.json`. Generation must be
deterministic.

`phase13-v1.json` covers Cartesian and modified-equinoctial Pontryagin
trajectories in mass- and time-optimal modes, the upstream dimensional
100-day trajectory, and first-order costate/`lambda0` variations.
