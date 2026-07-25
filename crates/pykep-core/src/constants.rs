// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from include/kep3/core_astro/constants.hpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Physical constants and normalized model parameters.

/// Ratio of a circle's circumference to its diameter.
pub const PI: f64 = core::f64::consts::PI;
/// Half of [`PI`].
pub const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;
/// IAU 2012 astronomical unit in metres.
pub const ASTRONOMICAL_UNIT: f64 = 149_597_870_700.0;
/// Newtonian gravitational constant in m³·kg⁻¹·s⁻².
pub const CAVENDISH_CONSTANT: f64 = 6.67430e-11;
/// DE440 solar gravitational parameter in m³·s⁻².
pub const MU_SUN: f64 = 1.327_124_400_412_794_2e20;
/// DE440 terrestrial gravitational parameter in m³·s⁻².
pub const MU_EARTH: f64 = 398_600_435_507_000.0;
/// DE440 lunar gravitational parameter in m³·s⁻².
pub const MU_MOON: f64 = 4_902_800_118_000.0;
/// Reference mean Earth orbital velocity in m·s⁻¹.
pub const EARTH_ORBITAL_VELOCITY: f64 = 29_784.691_834_309_11;
/// Earth's dimensionless second zonal harmonic.
pub const EARTH_J2: f64 = 1.082_626_68e-3;
/// Earth's reference equatorial radius in metres.
pub const EARTH_RADIUS: f64 = 6_378_137.0;
/// Multiplicative conversion from degrees to radians.
pub const DEGREES_TO_RADIANS: f64 = PI / 180.0;
/// Multiplicative conversion from radians to degrees.
pub const RADIANS_TO_DEGREES: f64 = 180.0 / PI;
/// Seconds in one mean solar day.
pub const DAY_TO_SECONDS: f64 = 86_400.0;
/// Mean solar days in one second.
pub const SECONDS_TO_DAY: f64 = 1.0 / DAY_TO_SECONDS;
/// Mean solar days in one Julian year.
pub const JULIAN_YEAR_DAYS: f64 = 365.25;
/// Julian years in one mean solar day.
pub const DAYS_TO_JULIAN_YEAR: f64 = 1.0 / JULIAN_YEAR_DAYS;
/// Conventional standard gravity in m·s⁻².
pub const STANDARD_GRAVITY: f64 = 9.806_65;
/// Earth–Moon CR3BP lunar mass parameter.
pub const CR3BP_MU_EARTH_MOON: f64 = MU_MOON / (MU_MOON + MU_EARTH);
/// Earth–Moon bicircular-problem lunar mass parameter.
pub const BCP_MU_EARTH_MOON: f64 = MU_MOON / (MU_MOON + MU_EARTH);
/// Bicircular-problem normalized solar mass parameter.
pub const BCP_MU_SUN: f64 = MU_SUN / (MU_MOON + MU_EARTH);
/// Normalized Sun to Earth–Moon barycentre distance used by the BCP model.
pub const BCP_SUN_DISTANCE: f64 = 3.888_111_43e2;
/// Normalized solar angular velocity used by the BCP model.
pub const BCP_SUN_ANGULAR_VELOCITY: f64 = -9.251_959_85e-1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_constants_match_the_cpp_binary64_values() {
        assert_eq!(PI.to_bits(), 0x4009_21fb_5444_2d18);
        assert_eq!(ASTRONOMICAL_UNIT, 149_597_870_700.0);
        assert_eq!(MU_EARTH, 398_600_435_507_000.0);
        assert_eq!(DAY_TO_SECONDS, 86_400.0);
        assert_eq!(SECONDS_TO_DAY * DAY_TO_SECONDS, 1.0);
        assert_eq!(DEGREES_TO_RADIANS * RADIANS_TO_DEGREES, 1.0);
    }
}
