// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::Elements6;

/// Classical Keplerian elements `[a, e, i, Ω, ω, ν]`.
///
/// `a` is the semi-major axis in the caller's length unit, `e` is
/// dimensionless, and all four angles are radians. Ellipses use `a > 0` and
/// `0 <= e < 1`; hyperbolae use `a < 0` and `e > 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassicalElements {
    /// Semi-major axis.
    pub semi_major_axis: f64,
    /// Eccentricity.
    pub eccentricity: f64,
    /// Inclination in `[0, π]`.
    pub inclination: f64,
    /// Longitude of the ascending node.
    pub longitude_ascending_node: f64,
    /// Argument of periapsis.
    pub argument_periapsis: f64,
    /// True anomaly.
    pub true_anomaly: f64,
}

impl ClassicalElements {
    /// Constructs a classical-element value in `[a, e, i, Ω, ω, ν]` order.
    #[must_use]
    pub const fn new(
        semi_major_axis: f64,
        eccentricity: f64,
        inclination: f64,
        longitude_ascending_node: f64,
        argument_periapsis: f64,
        true_anomaly: f64,
    ) -> Self {
        Self {
            semi_major_axis,
            eccentricity,
            inclination,
            longitude_ascending_node,
            argument_periapsis,
            true_anomaly,
        }
    }

    /// Returns `[a, e, i, Ω, ω, ν]`.
    #[must_use]
    pub const fn to_array(self) -> Elements6 {
        [
            self.semi_major_axis,
            self.eccentricity,
            self.inclination,
            self.longitude_ascending_node,
            self.argument_periapsis,
            self.true_anomaly,
        ]
    }
}

impl From<Elements6> for ClassicalElements {
    fn from(values: Elements6) -> Self {
        Self::new(
            values[0], values[1], values[2], values[3], values[4], values[5],
        )
    }
}

impl From<ClassicalElements> for Elements6 {
    fn from(elements: ClassicalElements) -> Self {
        elements.to_array()
    }
}

/// Modified equinoctial elements `[p, f, g, h, k, L]`.
///
/// `p` is the positive semilatus rectum in the caller's length unit; `f`,
/// `g`, `h`, and `k` are dimensionless; and true longitude `L` is radians.
/// The prograde convention is singular at inclination `π`, while the
/// retrograde convention is singular at inclination zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModifiedEquinoctialElements {
    /// Semilatus rectum.
    pub semilatus_rectum: f64,
    /// First eccentricity component.
    pub f: f64,
    /// Second eccentricity component.
    pub g: f64,
    /// First inclination component.
    pub h: f64,
    /// Second inclination component.
    pub k: f64,
    /// True longitude.
    pub true_longitude: f64,
}

impl ModifiedEquinoctialElements {
    /// Constructs a modified-equinoctial value in `[p, f, g, h, k, L]` order.
    #[must_use]
    pub const fn new(
        semilatus_rectum: f64,
        f: f64,
        g: f64,
        h: f64,
        k: f64,
        true_longitude: f64,
    ) -> Self {
        Self {
            semilatus_rectum,
            f,
            g,
            h,
            k,
            true_longitude,
        }
    }

    /// Returns `[p, f, g, h, k, L]`.
    #[must_use]
    pub const fn to_array(self) -> Elements6 {
        [
            self.semilatus_rectum,
            self.f,
            self.g,
            self.h,
            self.k,
            self.true_longitude,
        ]
    }
}

impl From<Elements6> for ModifiedEquinoctialElements {
    fn from(values: Elements6) -> Self {
        Self::new(
            values[0], values[1], values[2], values[3], values[4], values[5],
        )
    }
}

impl From<ModifiedEquinoctialElements> for Elements6 {
    fn from(elements: ModifiedEquinoctialElements) -> Self {
        elements.to_array()
    }
}
