// Copyright (c) 2020-2026 Francesco Biscani (bluescarni@gmail.com),
//                         Dario Izzo (dario.izzo@gmail.com)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from heyoka 7.10.0 commit
// 15bd82547b5b2d701727335c3b3dbcbeafa20018 and src/udpla/vsop2013.cpp
// at pykep commit 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Pure-Rust VSOP2013 analytical planetary ephemerides.

use super::{Ephemeris, EphemerisMetadata};
#[cfg(feature = "vsop2013")]
use crate::constants::{ASTRONOMICAL_UNIT, DAY_TO_SECONDS};
use crate::error::ensure_finite;
use crate::{CartesianState, PykepError, Result};
#[cfg(feature = "vsop2013")]
use std::sync::Arc;

/// Smallest coefficient threshold included in the embedded data.
pub const VSOP2013_MINIMUM_THRESHOLD: f64 = 1e-9;

#[cfg(feature = "vsop2013")]
const LAMBDA_LINEAR: [[f64; 2]; 17] = [
    [4.402_608_631_669, 26_087.903_140_685_55],
    [3.176_134_461_576, 10_213.285_547_434_45],
    [1.753_470_369_433, 6_283.075_850_353_215],
    [6.203_500_014_141, 3_340.612_434_145_457],
    [4.091_360_003_050, 1_731.170_452_721_855],
    [1.713_740_719_173, 1_704.450_855_027_201],
    [5.598_641_292_287, 1_428.948_917_844_273],
    [2.805_136_360_408, 1_364.756_513_629_99],
    [2.326_989_734_620, 1_361.923_207_632_842],
    [0.599_546_107_035, 529.690_961_562_325],
    [0.874_018_510_107, 213.299_086_108_488],
    [5.481_225_395_663, 74.781_659_030_778],
    [5.311_897_933_164, 38.132_972_226_125],
    [0.0, 0.359_536_228_504_930_9],
    [5.198_466_400_630, 77_713.771_448_180_4],
    [1.627_905_136_020, 84_334.661_571_783_7],
    [2.355_555_638_750, 83_286.914_247_714_7],
];

#[cfg(feature = "vsop2013")]
const PLANET_MUS: [f64; 9] = [
    4.912_547_451_450_812e-11,
    7.243_452_486_162_703e-10,
    8.997_011_603_631_609e-10,
    9.549_535_105_779_258e-11,
    2.825_345_842_083_778e-7,
    8.459_715_185_680_659e-8,
    1.292_024_916_781_969_4e-8,
    1.524_358_900_784_276_2e-8,
    2.188_699_765_425_969_6e-12,
];
#[cfg(feature = "vsop2013")]
const SUN_MU: f64 = 2.959_122_083_684_144e-4;

#[cfg(feature = "vsop2013")]
const COEFFICIENTS: &[u8] = include_bytes!("vsop2013.bin");
#[cfg(feature = "vsop2013")]
const HEADER_SIZE: usize = 24;
#[cfg(feature = "vsop2013")]
const GROUP_HEADER_SIZE: usize = 8;
#[cfg(feature = "vsop2013")]
const ENCODED_TERM_SIZE: usize = 40;

#[cfg(feature = "vsop2013")]
#[derive(Clone, Debug)]
struct Term {
    multipliers: [i32; 17],
    sine: f64,
    cosine: f64,
}

#[cfg(feature = "vsop2013")]
#[derive(Clone, Debug)]
struct PlanetSeries {
    variables: [Vec<Vec<Term>>; 6],
}

/// Analytical heliocentric ephemeris for a VSOP2013 body.
///
/// The embedded dataset supports coefficient thresholds greater than or equal
/// to [`VSOP2013_MINIMUM_THRESHOLD`]. The optional `vsop2013` Cargo feature is
/// enabled by default and can be disabled to remove the 4.3 MiB coefficient
/// asset while retaining the Keplerian and JPL low-precision providers.
#[derive(Clone, Debug)]
pub struct Vsop2013 {
    name: String,
    threshold: f64,
    #[cfg(feature = "vsop2013")]
    planet_index: u8,
    #[cfg(feature = "vsop2013")]
    series: Arc<PlanetSeries>,
}

impl Vsop2013 {
    /// Constructs a VSOP2013 provider with the upstream default threshold
    /// `1e-5`.
    ///
    /// Lookup is ASCII case-insensitive.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported body or unavailable feature.
    pub fn new(name: &str) -> Result<Self> {
        Self::with_threshold(name, 1e-5)
    }

    /// Constructs a VSOP2013 provider at a coefficient threshold.
    ///
    /// Supported names are Mercury, Venus, `earth_moon`, Mars, Jupiter,
    /// Saturn, Uranus, Neptune, and Pluto. The threshold is applied to
    /// `hypot(S, C)` independently for every series term, matching heyoka.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite or too-small threshold, unsupported
    /// body, unavailable feature, or corrupt embedded data.
    pub fn with_threshold(name: &str, threshold: f64) -> Result<Self> {
        ensure_finite("threshold", threshold)?;
        if threshold < VSOP2013_MINIMUM_THRESHOLD {
            return Err(PykepError::InvalidInput {
                parameter: "threshold",
                reason: format!(
                    "must be at least {VSOP2013_MINIMUM_THRESHOLD:e}; lower-threshold coefficients are not embedded"
                ),
            });
        }
        let canonical = name.to_ascii_lowercase();
        let planet_index = match canonical.as_str() {
            "mercury" => 1,
            "venus" => 2,
            "earth_moon" => 3,
            "mars" => 4,
            "jupiter" => 5,
            "saturn" => 6,
            "uranus" => 7,
            "neptune" => 8,
            "pluto" => 9,
            _ => {
                return Err(PykepError::InvalidInput {
                    parameter: "name",
                    reason: "supported names are mercury, venus, earth_moon, mars, jupiter, saturn, uranus, neptune, and pluto".into(),
                });
            }
        };

        #[cfg(not(feature = "vsop2013"))]
        {
            let _ = planet_index;
            let _ = threshold;
            Err(PykepError::UnsupportedCapability {
                provider: canonical,
                capability: "vsop2013 Cargo feature",
            })
        }
        #[cfg(feature = "vsop2013")]
        {
            let series = decode_planet(planet_index, threshold)?;
            Ok(Self {
                name: format!("vsop2013 {canonical}, threshold={threshold}"),
                threshold,
                planet_index,
                series: Arc::new(series),
            })
        }
    }

    /// Returns the canonical supported body names.
    #[must_use]
    pub const fn supported_bodies() -> [&'static str; 9] {
        [
            "mercury",
            "venus",
            "earth_moon",
            "mars",
            "jupiter",
            "saturn",
            "uranus",
            "neptune",
            "pluto",
        ]
    }

    /// Returns whether the coefficient feature was compiled in.
    #[must_use]
    pub const fn available() -> bool {
        cfg!(feature = "vsop2013")
    }

    /// Returns this provider's coefficient threshold.
    #[must_use]
    pub const fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl Ephemeris for Vsop2013 {
    fn state(&self, epoch_mjd2000: f64) -> Result<CartesianState> {
        ensure_finite("epoch_mjd2000", epoch_mjd2000)?;
        #[cfg(not(feature = "vsop2013"))]
        {
            let _ = epoch_mjd2000;
            Err(PykepError::UnsupportedCapability {
                provider: self.name.clone(),
                capability: "vsop2013 Cargo feature",
            })
        }
        #[cfg(feature = "vsop2013")]
        {
            let time = (epoch_mjd2000 - 0.5) / 365_250.0;
            let elements = self.series.evaluate(time);
            cartesian_icrf(elements, self.planet_index)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn metadata(&self) -> EphemerisMetadata {
        EphemerisMetadata::default()
    }
}

#[cfg(feature = "vsop2013")]
impl PlanetSeries {
    fn evaluate(&self, time: f64) -> [f64; 6] {
        core::array::from_fn(|variable| {
            let chunks = &self.variables[variable];
            chunks.iter().rev().fold(0.0, |accumulator, terms| {
                let part = terms.iter().fold(0.0, |sum, term| {
                    let argument = term.multipliers.iter().zip(LAMBDA_LINEAR).fold(
                        0.0,
                        |angle, (&multiplier, lambda)| {
                            angle + f64::from(multiplier) * (lambda[0] + time * lambda[1])
                        },
                    );
                    sum + term.sine * argument.sin() + term.cosine * argument.cos()
                });
                accumulator * time + part
            })
        })
    }
}

#[cfg(feature = "vsop2013")]
fn decode_planet(planet_index: u8, threshold: f64) -> Result<PlanetSeries> {
    if COEFFICIENTS.len() < HEADER_SIZE
        || &COEFFICIENTS[..8] != b"PKVSOP13"
        || read_u32(COEFFICIENTS, 8)? != 1
        || read_u32(COEFFICIENTS, 20)? != 54
    {
        return Err(PykepError::DataUnavailable {
            dataset: "VSOP2013 embedded coefficient header",
        });
    }
    let mut variables: [Vec<Vec<Term>>; 6] = core::array::from_fn(|_| Vec::new());
    let mut cursor = HEADER_SIZE;
    while cursor < COEFFICIENTS.len() {
        let header = slice(COEFFICIENTS, cursor, GROUP_HEADER_SIZE)?;
        cursor += GROUP_HEADER_SIZE;
        let planet = header[0];
        let variable = header[1];
        let alpha = header[2];
        let count = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        if !(1..=9).contains(&planet) || !(1..=6).contains(&variable) {
            return Err(PykepError::DataUnavailable {
                dataset: "VSOP2013 embedded coefficient group",
            });
        }
        if planet == planet_index {
            let chunks = &mut variables[usize::from(variable - 1)];
            while chunks.len() <= usize::from(alpha) {
                chunks.push(Vec::new());
            }
            let output = &mut chunks[usize::from(alpha)];
            for _ in 0..count {
                let encoded = slice(COEFFICIENTS, cursor, ENCODED_TERM_SIZE)?;
                cursor += ENCODED_TERM_SIZE;
                let term = decode_term(encoded);
                if term.sine.hypot(term.cosine) >= threshold {
                    output.push(term);
                }
            }
        } else {
            cursor = cursor.checked_add(count * ENCODED_TERM_SIZE).ok_or(
                PykepError::DataUnavailable {
                    dataset: "VSOP2013 embedded coefficient range",
                },
            )?;
            if cursor > COEFFICIENTS.len() {
                return Err(PykepError::DataUnavailable {
                    dataset: "VSOP2013 embedded coefficient range",
                });
            }
        }
    }
    if variables.iter().any(Vec::is_empty) || cursor != COEFFICIENTS.len() {
        return Err(PykepError::DataUnavailable {
            dataset: "VSOP2013 embedded coefficients",
        });
    }
    Ok(PlanetSeries { variables })
}

#[cfg(feature = "vsop2013")]
fn decode_term(encoded: &[u8]) -> Term {
    let mut multipliers = [0_i32; 17];
    for index in 0..9 {
        multipliers[index] = i32::from(i8::from_le_bytes([encoded[index]]));
    }
    for index in 0..3 {
        multipliers[14 + index] = i32::from(i8::from_le_bytes([encoded[9 + index]]));
    }
    for index in 0..4 {
        let offset = 12 + 2 * index;
        multipliers[9 + index] = i32::from(i16::from_le_bytes(
            encoded[offset..offset + 2].try_into().unwrap(),
        ));
    }
    multipliers[13] = i32::from_le_bytes(encoded[20..24].try_into().unwrap());
    Term {
        multipliers,
        sine: f64::from_le_bytes(encoded[24..32].try_into().unwrap()),
        cosine: f64::from_le_bytes(encoded[32..40].try_into().unwrap()),
    }
}

#[cfg(feature = "vsop2013")]
fn slice(data: &[u8], start: usize, length: usize) -> Result<&[u8]> {
    data.get(start..start + length)
        .ok_or(PykepError::DataUnavailable {
            dataset: "VSOP2013 embedded coefficient range",
        })
}

#[cfg(feature = "vsop2013")]
fn read_u32(data: &[u8], start: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        slice(data, start, 4)?.try_into().unwrap(),
    ))
}

#[cfg(feature = "vsop2013")]
fn cartesian_icrf(elements: [f64; 6], planet_index: u8) -> Result<CartesianState> {
    let [a, lambda, k, h, q_source, p_source] = elements;
    let sine_half_squared = q_source * q_source + p_source * p_source;
    let cosine_half = (1.0 - sine_half_squared).sqrt();
    let q = q_source / cosine_half;
    let p = p_source / cosine_half;
    let eccentricity_squared = h * h + k * k;
    let eccentricity_quotient = 1.0 + (1.0 - eccentricity_squared).sqrt();
    let eccentric_longitude = solve_eccentric_longitude(h, k, lambda)?;
    let (sine_f, cosine_f) = eccentric_longitude.sin_cos();
    let lambda_f = h * cosine_f - k * sine_f;
    let lambda_f_quotient = lambda_f / eccentricity_quotient;
    let x1 = a * (cosine_f - k - h * lambda_f_quotient);
    let y1 = a * (sine_f - h + k * lambda_f_quotient);
    let p_squared = p * p;
    let q_squared = q * q;
    let p2_minus_q2 = p_squared - q_squared;
    let p2_plus_q2 = p_squared + q_squared;
    let two_p = p + p;
    let two_pq = two_p * q;
    let two_q = q + q;
    let x = (1.0 - p2_minus_q2) * x1 + two_pq * y1;
    let y = two_pq * x1 + (1.0 + p2_minus_q2) * y1;
    let z = two_q * y1 - two_p * x1;

    let mu = SUN_MU + PLANET_MUS[usize::from(planet_index - 1)];
    let mean_motion = (mu / a.powi(3)).sqrt();
    let f_prime = mean_motion / (1.0 - h * sine_f - k * cosine_f);
    let mean_minus_prime = mean_motion - f_prime;
    let derivative_quotient = mean_minus_prime / eccentricity_quotient;
    let vx1 = a * (-sine_f * f_prime - h * derivative_quotient);
    let vy1 = a * (cosine_f * f_prime + k * derivative_quotient);
    let vx = (1.0 - p2_minus_q2) * vx1 + two_pq * vy1;
    let vy = two_pq * vx1 + (1.0 + p2_minus_q2) * vy1;
    let vz = two_q * vy1 - two_p * vx1;
    let divisor = 1.0 + p2_plus_q2;

    let ecliptic = [
        x / divisor,
        y / divisor,
        z / divisor,
        vx / divisor,
        vy / divisor,
        vz / divisor,
    ];
    let epsilon: f64 = 0.409_092_626_586_596_2;
    let phi: f64 = -2.515_213_377_596_228_5e-7;
    let (sine_epsilon, cosine_epsilon) = epsilon.sin_cos();
    let (sine_phi, cosine_phi) = phi.sin_cos();
    let rotate = |x_value: f64, y_value: f64, z_value: f64| {
        [
            cosine_phi * x_value - sine_phi * cosine_epsilon * y_value
                + sine_phi * sine_epsilon * z_value,
            sine_phi * x_value + cosine_phi * cosine_epsilon * y_value
                - cosine_phi * sine_epsilon * z_value,
            sine_epsilon * y_value + cosine_epsilon * z_value,
        ]
    };
    let position = rotate(ecliptic[0], ecliptic[1], ecliptic[2]);
    let velocity = rotate(ecliptic[3], ecliptic[4], ecliptic[5]);
    let position_scale = ASTRONOMICAL_UNIT;
    let velocity_scale = ASTRONOMICAL_UNIT / DAY_TO_SECONDS;
    let state = [
        position[0] * position_scale,
        position[1] * position_scale,
        position[2] * position_scale,
        velocity[0] * velocity_scale,
        velocity[1] * velocity_scale,
        velocity[2] * velocity_scale,
    ];
    if state.iter().all(|value| value.is_finite()) {
        Ok(state)
    } else {
        Err(PykepError::NumericalOverflow {
            operation: "VSOP2013 evaluation",
        })
    }
}

#[cfg(feature = "vsop2013")]
fn solve_eccentric_longitude(h: f64, k: f64, lambda: f64) -> Result<f64> {
    if h * h + k * k >= 1.0 {
        return Err(PykepError::InvalidInput {
            parameter: "VSOP2013 elements",
            reason: "eccentricity must be below one".into(),
        });
    }
    let reduced = lambda.rem_euclid(core::f64::consts::TAU);
    let (sine_lambda, cosine_lambda) = reduced.sin_cos();
    let k_sine = k * sine_lambda;
    let h_cosine = h * cosine_lambda;
    let k_cosine_plus_h_sine = k * cosine_lambda + h * sine_lambda;
    let k_sine_minus_h_cosine = k_sine - h_cosine;
    let mut value = reduced
        + k_sine_minus_h_cosine
        + (k * k - h * h) * cosine_lambda * sine_lambda
        + h * k * (sine_lambda * sine_lambda - cosine_lambda * cosine_lambda)
        + 0.5
            * k_sine_minus_h_cosine
            * (2.0 * k_cosine_plus_h_sine.powi(2) - k_sine_minus_h_cosine.powi(2));
    let mut lower = -1.0;
    let mut upper = core::f64::consts::TAU + 1.0;
    value = value.clamp(lower, upper);
    let tolerance = 4.0 * f64::EPSILON;
    for _ in 0..20 {
        let (sine, cosine) = value.sin_cos();
        let residual = value - reduced + h * cosine - k * sine;
        if residual >= 0.0 {
            upper = value;
        }
        if residual <= 0.0 {
            lower = value;
        }
        if residual.abs() <= tolerance || upper - lower <= tolerance {
            return Ok(value.rem_euclid(core::f64::consts::TAU));
        }
        let derivative = 1.0 - h * sine - k * cosine;
        let mut next = value - residual / derivative;
        if next > upper {
            next = 0.5 * (value + upper);
        }
        if next < lower {
            next = 0.5 * (value + lower);
        }
        value = next;
    }
    Err(PykepError::ConvergenceFailure {
        operation: "VSOP2013 eccentric longitude",
        iterations: 20,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_thresholds_and_feature_reporting_are_explicit() {
        assert_eq!(Vsop2013::supported_bodies().len(), 9);
        assert!(Vsop2013::with_threshold("venus", f64::NAN).is_err());
        assert!(Vsop2013::with_threshold("venus", 1e-10).is_err());
        assert!(Vsop2013::new("goofy").is_err());
        assert_eq!(Vsop2013::available(), cfg!(feature = "vsop2013"));
    }

    #[cfg(feature = "vsop2013")]
    #[test]
    fn provider_is_cloneable_and_matches_upstream_venus_reference() {
        let provider = Vsop2013::with_threshold("vEnUs", 1e-9).unwrap();
        assert_eq!(provider.threshold(), 1e-9);
        let clone = provider.clone();
        assert_eq!(provider.state(123.0).unwrap(), clone.state(123.0).unwrap());
        let state = provider.state(123.0).unwrap();
        let expected = [
            103_304_986_899.798_1,
            32_220_404_104.119_9,
            7_957_719_449.515_38,
            -10_696.505_905_435_035,
            30_061.035_989_651_813,
            14_201.000_904_921_95,
        ];
        for (actual, expected) in state.iter().zip(expected) {
            assert!((actual - expected).abs() <= 5e-13 * expected.abs().max(1.0));
        }
    }

    #[cfg(feature = "vsop2013")]
    #[test]
    fn shared_provider_is_thread_safe_and_deterministic() {
        let provider = Arc::new(Vsop2013::new("earth_moon").unwrap());
        let expected = provider.state(42.0).unwrap();
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let provider = Arc::clone(&provider);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        assert_eq!(provider.state(42.0).unwrap(), expected);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }
}
