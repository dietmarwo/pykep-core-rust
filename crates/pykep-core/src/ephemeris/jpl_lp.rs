// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/udpla/jpl_lp.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.
//
// Coefficients: JPL Solar System Dynamics, "Approximate Positions of the
// Planets", https://ssd.jpl.nasa.gov/planets/approx_pos.html.

//! JPL low-precision heliocentric planetary ephemerides for 1800–2050.

use super::{ElementRepresentation, Ephemeris, EphemerisMetadata};
use crate::astro::anomalies::{mean_to_true_anomaly, true_to_mean_anomaly};
use crate::astro::elements::{
    ClassicalElements, classical_to_cartesian_unbounded_inclination,
    classical_to_modified_equinoctial,
};
use crate::constants::{ASTRONOMICAL_UNIT, DEGREES_TO_RADIANS, MU_SUN};
use crate::error::ensure_finite;
use crate::{CartesianState, Elements6, PykepError, Result};

const MINIMUM_MJD2000: f64 = -73_048.0;
const MAXIMUM_MJD2000: f64 = 18_263.0;

#[derive(Clone, Copy)]
struct Coefficients {
    elements: [f64; 6],
    rates: [f64; 6],
    body_mu: f64,
    radius: f64,
    safe_scale: f64,
}

const MERCURY: Coefficients = Coefficients {
    elements: [
        0.38709927,
        0.20563593,
        7.00497902,
        252.25032350,
        77.45779628,
        48.33076593,
    ],
    rates: [
        0.00000037,
        0.00001906,
        -0.00594749,
        149472.67411175,
        0.16047689,
        -0.12534081,
    ],
    body_mu: 22_032e9,
    radius: 2_440_000.0,
    safe_scale: 1.1,
};
const VENUS: Coefficients = Coefficients {
    elements: [
        0.72333566,
        0.00677672,
        3.39467605,
        181.97909950,
        131.60246718,
        76.67984255,
    ],
    rates: [
        0.00000390,
        -0.00004107,
        -0.00078890,
        58517.81538729,
        0.00268329,
        -0.27769418,
    ],
    body_mu: 324_859e9,
    radius: 6_052_000.0,
    safe_scale: 1.1,
};
const EARTH: Coefficients = Coefficients {
    elements: [
        1.00000261,
        0.01671123,
        -0.00001531,
        100.46457166,
        102.93768193,
        0.0,
    ],
    rates: [
        0.00000562,
        -0.00004392,
        -0.01294668,
        35999.37244981,
        0.32327364,
        0.0,
    ],
    body_mu: 398_600.441_8e9,
    radius: 6_378_000.0,
    safe_scale: 1.1,
};
const MARS: Coefficients = Coefficients {
    elements: [
        1.52371034,
        0.09339410,
        1.84969142,
        -4.55343205,
        -23.94362959,
        49.55953891,
    ],
    rates: [
        0.00001847,
        0.00007882,
        -0.00813131,
        19140.30268499,
        0.44441088,
        -0.29257343,
    ],
    body_mu: 42_828e9,
    radius: 3_397_000.0,
    safe_scale: 1.1,
};
const JUPITER: Coefficients = Coefficients {
    elements: [
        5.20288700,
        0.04838624,
        1.30439695,
        34.39644051,
        14.72847983,
        100.47390909,
    ],
    rates: [
        -0.00011607,
        -0.00013253,
        -0.00183714,
        3034.74612775,
        0.21252668,
        0.20469106,
    ],
    body_mu: 126_686_534e9,
    radius: 71_492_000.0,
    safe_scale: 9.0,
};
const SATURN: Coefficients = Coefficients {
    elements: [
        9.53667594,
        0.05386179,
        2.48599187,
        49.95424423,
        92.59887831,
        113.66242448,
    ],
    rates: [
        -0.00125060,
        -0.00050991,
        0.00193609,
        1222.49362201,
        -0.41897216,
        -0.28867794,
    ],
    body_mu: 37_931_187e9,
    radius: 60_330_000.0,
    safe_scale: 1.1,
};
const URANUS: Coefficients = Coefficients {
    elements: [
        19.18916464,
        0.04725744,
        0.77263783,
        313.23810451,
        170.95427630,
        74.01692503,
    ],
    rates: [
        -0.00196176,
        -0.00004397,
        -0.00242939,
        428.48202785,
        0.40805281,
        0.04240589,
    ],
    body_mu: 5_793_939e9,
    radius: 25_362_000.0,
    safe_scale: 1.1,
};
const NEPTUNE: Coefficients = Coefficients {
    elements: [
        30.06992276,
        0.00859048,
        1.77004347,
        -55.12002969,
        44.96476227,
        131.78422574,
    ],
    rates: [
        0.00026291,
        0.00005105,
        0.00035372,
        218.45945325,
        -0.32241464,
        -0.00508664,
    ],
    body_mu: 6_836_529e9,
    radius: 24_622_000.0,
    safe_scale: 1.1,
};

/// JPL approximate heliocentric ephemeris, valid from 1800 through 2050.
#[derive(Clone, Debug, PartialEq)]
pub struct JplLowPrecision {
    name: String,
    elements: [f64; 6],
    rates: [f64; 6],
    metadata: EphemerisMetadata,
}

impl JplLowPrecision {
    /// Constructs one of Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus,
    /// or Neptune. Lookup is ASCII case-insensitive.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported name.
    pub fn new(name: &str) -> Result<Self> {
        let canonical = name.to_ascii_lowercase();
        let coefficients = match canonical.as_str() {
            "mercury" => MERCURY,
            "venus" => VENUS,
            "earth" => EARTH,
            "mars" => MARS,
            "jupiter" => JUPITER,
            "saturn" => SATURN,
            "uranus" => URANUS,
            "neptune" => NEPTUNE,
            _ => {
                return Err(PykepError::InvalidInput {
                    parameter: "name",
                    reason: "supported names are mercury, venus, earth, mars, jupiter, saturn, uranus, and neptune".into(),
                });
            }
        };
        Ok(Self {
            name: format!("{canonical}(jpl_lp)"),
            elements: coefficients.elements,
            rates: coefficients.rates,
            metadata: EphemerisMetadata {
                central_mu: Some(MU_SUN),
                body_mu: Some(coefficients.body_mu),
                radius: Some(coefficients.radius),
                safe_radius: Some(coefficients.safe_scale * coefficients.radius),
            },
        })
    }

    /// Returns all supported canonical body names.
    #[must_use]
    pub const fn supported_bodies() -> [&'static str; 8] {
        [
            "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune",
        ]
    }

    /// Changes the safe encounter radius.
    ///
    /// # Errors
    ///
    /// Returns an error unless the new value is finite and at least the
    /// physical radius.
    pub fn set_safe_radius(&mut self, safe_radius: f64) -> Result<()> {
        ensure_finite("safe_radius", safe_radius)?;
        let radius = self
            .metadata
            .radius
            .ok_or_else(|| PykepError::UnsupportedCapability {
                provider: self.name.clone(),
                capability: "radius",
            })?;
        if safe_radius < radius {
            return Err(PykepError::InvalidInput {
                parameter: "safe_radius",
                reason: "must be at least the physical radius".into(),
            });
        }
        self.metadata.safe_radius = Some(safe_radius);
        Ok(())
    }

    fn classical_elements(&self, epoch_mjd2000: f64) -> Result<ClassicalElements> {
        ensure_finite("epoch_mjd2000", epoch_mjd2000)?;
        if epoch_mjd2000 <= MINIMUM_MJD2000 || epoch_mjd2000 >= MAXIMUM_MJD2000 {
            return Err(PykepError::InvalidInput {
                parameter: "epoch_mjd2000",
                reason: "JPL low-precision ephemerides require -73048 < epoch < 18263 (approximately 1800-2050)".into(),
            });
        }
        let centuries = (epoch_mjd2000 - 0.5) / 36_525.0;
        let updated = core::array::from_fn::<_, 6, _>(|index| {
            self.elements[index] + self.rates[index] * centuries
        });
        let eccentricity = updated[1];
        let mean_anomaly = (updated[3] - updated[4]) * DEGREES_TO_RADIANS;
        Ok(ClassicalElements::new(
            updated[0] * ASTRONOMICAL_UNIT,
            eccentricity,
            updated[2] * DEGREES_TO_RADIANS,
            updated[5] * DEGREES_TO_RADIANS,
            (updated[4] - updated[5]) * DEGREES_TO_RADIANS,
            mean_to_true_anomaly(mean_anomaly, eccentricity)?,
        ))
    }
}

impl Ephemeris for JplLowPrecision {
    fn state(&self, epoch_mjd2000: f64) -> Result<CartesianState> {
        classical_to_cartesian_unbounded_inclination(
            self.classical_elements(epoch_mjd2000)?,
            MU_SUN,
        )
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn metadata(&self) -> EphemerisMetadata {
        self.metadata.clone()
    }

    fn elements(
        &self,
        epoch_mjd2000: f64,
        representation: ElementRepresentation,
    ) -> Result<Elements6> {
        let mut elements = self.classical_elements(epoch_mjd2000)?;
        match representation {
            ElementRepresentation::ClassicalTrue => Ok(elements.to_array()),
            ElementRepresentation::ClassicalMean => {
                elements.true_anomaly =
                    true_to_mean_anomaly(elements.true_anomaly, elements.eccentricity)?;
                Ok(elements.to_array())
            }
            ElementRepresentation::ModifiedEquinoctial => {
                classical_to_modified_equinoctial(elements, false).map(|value| value.to_array())
            }
            ElementRepresentation::ModifiedEquinoctialRetrograde => {
                classical_to_modified_equinoctial(elements, true).map(|value| value.to_array())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_metadata_and_validity_boundaries_are_explicit() {
        let mut neptune = JplLowPrecision::new("nePTUne").unwrap();
        assert_eq!(neptune.name(), "neptune(jpl_lp)");
        assert_eq!(neptune.metadata().body_mu, Some(6_836_529e9));
        assert!(neptune.state(MINIMUM_MJD2000).is_err());
        assert!(neptune.state(MAXIMUM_MJD2000).is_err());
        assert!(JplLowPrecision::new("pluto").is_err());
        assert!(neptune.set_safe_radius(1.0).is_err());
        neptune.set_safe_radius(30_000_000.0).unwrap();
        assert_eq!(neptune.metadata().safe_radius, Some(30_000_000.0));
    }
}
