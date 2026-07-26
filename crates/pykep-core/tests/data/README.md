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
.oracle-build/phase11_oracle
.oracle-build/phase12_oracle
.oracle-build/phase13_oracle
.oracle-build/phase14_oracle
.oracle-build/phase15_oracle
```

Compare every command's output byte-for-byte with its corresponding
`foundations-v1.json` or `phaseN-v1.json` file. Generation must be
deterministic.

`phase11-v1.json` covers sampled Kepler, CR3BP, and BCP trajectories and
their final state-transition matrices. It has SHA-256
`1c2b67eb203da62baf921db9f811b0ad0f763cbaa03f0ca80d6319870b0da807`.

`phase12-v1.json` covers final states and state/control sensitivities for all
four built-in ZOH dynamics models. It has SHA-256
`294bab93355628cd22991b83563c6f93deefc19dd93911dfc4b654a101764f22`.

`phase13-v1.json` covers Cartesian and modified-equinoctial Pontryagin
trajectories in mass- and time-optimal modes, the upstream dimensional
100-day trajectory, and first-order costate/`lambda0` variations.

`phase14-v1.json` covers fixed-duration Sims–Flanagan constraints and
analytic gradients at representative cut boundaries, plus equal- and
irregular-duration alpha legs. It has SHA-256
`0bd6adddc72d850f1de5e4ab95a436128d9b8181f8de2b3bbbc67300e004d542`.

`phase15-v1.json` covers generic ZOH-leg mismatches and endpoint, control, and
time-grid gradients for all four built-in ZOH dynamics. It has SHA-256
`9d8d424c2af10ceee497f74f62acb30c53924f78f7be8a528dbe44bf14767935`.
