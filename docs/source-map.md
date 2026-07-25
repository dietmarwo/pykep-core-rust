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
- [ ] `src/udpla/jpl_lp.cpp` → `ephemeris::jpl_lp` (Phase 8)
- [ ] `src/udpla/vsop2013.cpp` → `ephemeris::vsop2013` (Phase 9)
- [ ] `src/ta/kep.cpp` → `dynamics::kepler` (Phase 11)
- [ ] `src/ta/cr3bp.cpp` → `dynamics::cr3bp` (Phase 11)
- [ ] `src/ta/bcp.cpp` → `dynamics::bcp` (Phase 11)
- [ ] `src/ta/zoh_kep.cpp` → `dynamics::zoh::kepler` (Phase 12)
- [ ] `src/ta/zoh_cr3bp.cpp` → `dynamics::zoh::cr3bp` (Phase 12)
- [ ] `src/ta/zoh_eq.cpp` → `dynamics::zoh::equinoctial` (Phase 12)
- [ ] `src/ta/zoh_ss.cpp` → `dynamics::zoh::spacecraft` (Phase 12)
- [ ] `src/ta/pontryagin_cartesian.cpp` →
      `dynamics::pontryagin::cartesian` (Phase 13)
- [ ] `src/ta/pontryagin_equinoctial.cpp` →
      `dynamics::pontryagin::equinoctial` (Phase 13)
- [ ] `src/leg/sf_checks.cpp` → `leg::validation` (Phase 14)
- [ ] `src/leg/sims_flanagan.cpp` → `leg::sims_flanagan` (Phase 14)
- [ ] `src/leg/sims_flanagan_alpha.cpp` →
      `leg::sims_flanagan_alpha` (Phase 14)
- [ ] `src/leg/zoh.cpp` → `leg::zoh` (Phase 15)

The C++-specific visibility, serialization, and type-erasure support headers
are reviewed for semantics but are not port targets.
