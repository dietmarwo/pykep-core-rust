# Source map

The port baseline is pykep/kep3 3.0.1 at commit
`53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e`. A checked box means the Rust
module, C++ golden parity, independent validation, Python binding, and
documentation required by the definition of done are complete.

## Header-only numerical sources

- [x] `include/kep3/core_astro/constants.hpp` → `constants` (Phase 2)
- [x] `include/kep3/core_astro/convert_julian_dates.hpp` → `time::julian`
      (Phase 2)
- [x] `include/kep3/core_astro/kepler_equations.hpp` →
      `math::kepler_equations` (Phase 2)
- [x] `include/kep3/core_astro/special_functions.hpp` → `math::stumpff`
      (Phase 2)
- [x] `include/kep3/core_astro/convert_anomalies.hpp` → `astro::anomalies`
      (Phase 3)

## Translation units

- [x] `src/linalg.cpp` → `math::linalg` (Phase 2)
- [x] `src/epoch.cpp` → `time::epoch` (Phase 3)
- [x] `src/core_astro/ic2par2ic.cpp` → `astro::elements::classical` (Phase 4)
- [x] `src/core_astro/mee2par2mee.cpp` →
      `astro::elements::equinoctial` (Phase 4)
- [x] `src/core_astro/ic2mee2ic.cpp` →
      `astro::elements::equinoctial` (Phase 4)
- [x] `src/core_astro/propagate_lagrangian.cpp` →
      `astro::propagation::lagrangian` (Phase 5)
- [x] `src/core_astro/stm.cpp` → `astro::propagation::stm` (Phase 5)
- [x] `src/core_astro/basic_transfers.cpp` → `astro::transfers::basic`
      (Phase 6)
- [x] `src/core_astro/encodings.cpp` → `astro::encodings` (Phase 6)
- [x] `src/core_astro/flyby.cpp` → `astro::flyby` (Phase 6)
- [x] `src/lambert_problem.cpp` → `astro::lambert` (Phase 6)
- [x] `src/core_astro/mima.cpp` → `astro::mima` (Phase 6)
- [x] `src/planet.cpp` → `ephemeris` (Phase 7)
- [x] `src/udpla/keplerian.cpp` → `ephemeris::keplerian` (Phase 7)
- [x] `src/udpla/jpl_lp.cpp` → `ephemeris::jpl_lp` (Phase 8)
- [x] `src/udpla/vsop2013.cpp` → `ephemeris::vsop2013` (Phase 9)
- [x] heyoka integration requirements → `integration` facade and ADR 0004
      (Phase 10; infrastructure rather than a source translation)
- [x] `src/ta/kep.cpp` → `dynamics::KeplerDynamics` (Phase 11)
- [x] `src/ta/cr3bp.cpp` → `dynamics::Cr3bpDynamics` (Phase 11)
- [x] `src/ta/bcp.cpp` → `dynamics::BcpDynamics` (Phase 11)
- [x] `src/ta/zoh_kep.cpp` → `dynamics::zoh::ZohKeplerDynamics` (Phase 12)
- [x] `src/ta/zoh_cr3bp.cpp` → `dynamics::zoh::ZohCr3bpDynamics` (Phase 12)
- [x] `src/ta/zoh_eq.cpp` → `dynamics::zoh::ZohEquinoctialDynamics` (Phase 12)
- [x] `src/ta/zoh_ss.cpp` → `dynamics::zoh::ZohSolarSailDynamics` (Phase 12)
- [x] `src/ta/pontryagin_cartesian.cpp` →
      `dynamics::pontryagin::{CartesianMassOptimal, CartesianTimeOptimal}`
      (Phase 13)
- [x] `src/ta/pontryagin_equinoctial.cpp` →
      `dynamics::pontryagin::{EquinoctialMassOptimal, EquinoctialTimeOptimal}`
      (Phase 13)
- [x] `src/leg/sf_checks.cpp` → `leg::sims_flanagan` validation (Phase 14)
- [x] `src/leg/sims_flanagan.cpp` → `leg::sims_flanagan` (Phase 14)
- [x] `src/leg/sims_flanagan_alpha.cpp` →
      `leg::sims_flanagan_alpha` (Phase 14)
- [x] `src/leg/zoh.cpp` → `leg::zoh` (Phase 15)

The C++-specific visibility, serialization, and type-erasure support headers
are reviewed for semantics but are not port targets.

## Explicitly unavailable upstream ecosystem areas

These are not unchecked source-map rows. They are technically outside the
native numerical-core product and have stable internal tracking identifiers:

- `PY-EXT-001`: SPICE kernels, TLE parsing, and Python-defined UDPLA providers
  require external data/runtime and dynamic Python callback contracts that are
  absent from the C/C++-free core.
- `TA-SYMBOLIC-001`: arbitrary heyoka expression graphs and Taylor-integrator
  objects cannot be reproduced without exposing a symbolic/JIT runtime; native
  evaluated dynamics and DOP853 propagation are the supported contract.
- `PY-ECOSYSTEM-001`: plotting, trajectory-optimization UDPs, gym utilities,
  and third-party integrations belong to their Python ecosystems rather than
  the numerical core.
- `MIMA-HOP-001`: `mima_from_hop` and `mima2_from_hop` depend on an upstream
  higher-order-propagation object that the native API intentionally does not
  expose.

The Python migration matrix gives user-visible alternatives. A future change
must resolve the corresponding tracking item and add a source-map row before
claiming support.
