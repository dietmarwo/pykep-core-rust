// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from include/kep3/core_astro/convert_anomalies.hpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Conversions among mean, eccentric/hyperbolic, true, and Gudermannian
//! anomalies.
//!
//! The hyperbolic mean-anomaly solver deliberately uses an `asinh(M/e)`
//! initial estimate and an unrestricted Newton domain. This converges for
//! finite cases outside the pinned C++ implementation's fixed `1 ± 20π`
//! bracket; iteration exhaustion remains an explicit error.

use crate::constants::PI;
use crate::error::{ensure_finite_output, ensure_finite_values};
use crate::{PykepError, Result};

const MAX_ITERATIONS: usize = 100;

fn validate_elliptic(angle_name: &'static str, angle: f64, eccentricity: f64) -> Result<()> {
    ensure_finite_values(&[(angle_name, angle), ("eccentricity", eccentricity)])?;
    if (0.0..1.0).contains(&eccentricity) {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter: "eccentricity",
            reason: "elliptic eccentricity must satisfy 0 <= e < 1".into(),
        })
    }
}

fn validate_hyperbolic(angle_name: &'static str, angle: f64, eccentricity: f64) -> Result<()> {
    ensure_finite_values(&[(angle_name, angle), ("eccentricity", eccentricity)])?;
    if eccentricity > 1.0 {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter: "eccentricity",
            reason: "hyperbolic eccentricity must satisfy e > 1".into(),
        })
    }
}

/// Converts elliptic mean anomaly to principal eccentric anomaly.
///
/// Inputs and output are radians. Mean anomalies outside one revolution are
/// reduced through sine/cosine, matching the pinned upstream principal branch
/// in `[-π, π]`.
///
/// # Errors
///
/// Returns an error for non-finite input, `eccentricity` outside `0 <= e < 1`,
/// numerical overflow, or failure to converge in 100 iterations.
pub fn mean_to_eccentric_anomaly(mean_anomaly: f64, eccentricity: f64) -> Result<f64> {
    validate_elliptic("mean_anomaly", mean_anomaly, eccentricity)?;
    let sine = mean_anomaly.sin();
    let cosine = mean_anomaly.cos();
    let reduced_mean = sine.atan2(cosine);
    let mut anomaly = reduced_mean
        + eccentricity * sine
        + eccentricity.powi(2) * sine * cosine
        + eccentricity.powi(3) * sine * (1.5 * cosine.powi(2) - 0.5);
    let mut lower = -PI;
    let mut upper = PI;

    for _ in 0..MAX_ITERATIONS {
        let residual = anomaly - eccentricity * anomaly.sin() - reduced_mean;
        if residual.abs() <= 4.0 * f64::EPSILON * (1.0 + reduced_mean.abs()) {
            return ensure_finite_output("mean_to_eccentric_anomaly", anomaly);
        }
        if residual > 0.0 {
            upper = anomaly;
        } else {
            lower = anomaly;
        }
        let derivative = 1.0 - eccentricity * anomaly.cos();
        let newton = anomaly - residual / derivative;
        let next = if newton > lower && newton < upper && newton.is_finite() {
            newton
        } else {
            0.5 * (lower + upper)
        };
        if (next - anomaly).abs() <= 2.0 * f64::EPSILON * next.abs().max(1.0) {
            return ensure_finite_output("mean_to_eccentric_anomaly", next);
        }
        anomaly = next;
    }
    Err(PykepError::ConvergenceFailure {
        operation: "mean_to_eccentric_anomaly",
        iterations: MAX_ITERATIONS,
    })
}

/// Converts eccentric anomaly to elliptic mean anomaly.
///
/// Inputs and output are radians; the input revolution count is preserved.
///
/// # Errors
///
/// Returns an error for non-finite input, invalid eccentricity, or overflow.
pub fn eccentric_to_mean_anomaly(eccentric_anomaly: f64, eccentricity: f64) -> Result<f64> {
    validate_elliptic("eccentric_anomaly", eccentric_anomaly, eccentricity)?;
    ensure_finite_output(
        "eccentric_to_mean_anomaly",
        eccentric_anomaly - eccentricity * eccentric_anomaly.sin(),
    )
}

/// Converts eccentric anomaly to principal true anomaly in `[-π, π]`.
///
/// # Errors
///
/// Returns an error for non-finite input or invalid eccentricity.
pub fn eccentric_to_true_anomaly(eccentric_anomaly: f64, eccentricity: f64) -> Result<f64> {
    validate_elliptic("eccentric_anomaly", eccentric_anomaly, eccentricity)?;
    ensure_finite_output(
        "eccentric_to_true_anomaly",
        2.0 * (((1.0 + eccentricity) / (1.0 - eccentricity)).sqrt()
            * (eccentric_anomaly / 2.0).tan())
        .atan(),
    )
}

/// Converts true anomaly to principal eccentric anomaly in `[-π, π]`.
///
/// # Errors
///
/// Returns an error for non-finite input or invalid eccentricity.
pub fn true_to_eccentric_anomaly(true_anomaly: f64, eccentricity: f64) -> Result<f64> {
    validate_elliptic("true_anomaly", true_anomaly, eccentricity)?;
    ensure_finite_output(
        "true_to_eccentric_anomaly",
        2.0 * (((1.0 - eccentricity) / (1.0 + eccentricity)).sqrt() * (true_anomaly / 2.0).tan())
            .atan(),
    )
}

/// Converts elliptic mean anomaly to principal true anomaly.
///
/// # Errors
///
/// Returns the validation or convergence errors from
/// [`mean_to_eccentric_anomaly`] and [`eccentric_to_true_anomaly`].
pub fn mean_to_true_anomaly(mean_anomaly: f64, eccentricity: f64) -> Result<f64> {
    eccentric_to_true_anomaly(
        mean_to_eccentric_anomaly(mean_anomaly, eccentricity)?,
        eccentricity,
    )
}

/// Converts true anomaly to elliptic mean anomaly on the principal branch.
///
/// # Errors
///
/// Returns the validation errors from [`true_to_eccentric_anomaly`] and
/// [`eccentric_to_mean_anomaly`].
pub fn true_to_mean_anomaly(true_anomaly: f64, eccentricity: f64) -> Result<f64> {
    eccentric_to_mean_anomaly(
        true_to_eccentric_anomaly(true_anomaly, eccentricity)?,
        eccentricity,
    )
}

/// Converts Gudermannian anomaly to hyperbolic true anomaly.
///
/// Both angles use the principal `[-π, π]` branch.
///
/// # Errors
///
/// Returns an error for non-finite input or `eccentricity <= 1`.
pub fn gudermannian_to_true_anomaly(gudermannian_anomaly: f64, eccentricity: f64) -> Result<f64> {
    validate_hyperbolic("gudermannian_anomaly", gudermannian_anomaly, eccentricity)?;
    ensure_finite_output(
        "gudermannian_to_true_anomaly",
        2.0 * (((1.0 + eccentricity) / (eccentricity - 1.0)).sqrt()
            * (gudermannian_anomaly / 2.0).tan())
        .atan(),
    )
}

/// Converts hyperbolic true anomaly to Gudermannian anomaly.
///
/// # Errors
///
/// Returns an error for non-finite input or `eccentricity <= 1`.
pub fn true_to_gudermannian_anomaly(true_anomaly: f64, eccentricity: f64) -> Result<f64> {
    validate_hyperbolic("true_anomaly", true_anomaly, eccentricity)?;
    ensure_finite_output(
        "true_to_gudermannian_anomaly",
        2.0 * (((eccentricity - 1.0) / (1.0 + eccentricity)).sqrt() * (true_anomaly / 2.0).tan())
            .atan(),
    )
}

/// Converts hyperbolic mean anomaly to hyperbolic anomaly.
///
/// Unlike the pinned C++ solver, this uses an `asinh(M/e)` initial estimate
/// without a fixed anomaly bracket, so some large finite anomalies accepted
/// here are rejected upstream.
///
/// # Errors
///
/// Returns an error for non-finite input, `eccentricity <= 1`, overflow, or
/// failure to converge in 100 Newton iterations.
pub fn hyperbolic_mean_to_anomaly(mean_anomaly: f64, eccentricity: f64) -> Result<f64> {
    validate_hyperbolic("mean_anomaly", mean_anomaly, eccentricity)?;
    let mut anomaly = (mean_anomaly / eccentricity).asinh();
    for _ in 0..MAX_ITERATIONS {
        let residual = eccentricity * anomaly.sinh() - anomaly - mean_anomaly;
        if residual.abs() <= 4.0 * f64::EPSILON * (1.0 + mean_anomaly.abs()) {
            return ensure_finite_output("hyperbolic_mean_to_anomaly", anomaly);
        }
        let derivative = eccentricity * anomaly.cosh() - 1.0;
        let step = residual / derivative;
        anomaly -= step;
        if !anomaly.is_finite() {
            return Err(PykepError::NumericalOverflow {
                operation: "hyperbolic_mean_to_anomaly",
            });
        }
        if step.abs() <= 2.0 * f64::EPSILON * anomaly.abs().max(1.0) {
            return Ok(anomaly);
        }
    }
    Err(PykepError::ConvergenceFailure {
        operation: "hyperbolic_mean_to_anomaly",
        iterations: MAX_ITERATIONS,
    })
}

/// Converts hyperbolic anomaly to hyperbolic mean anomaly.
///
/// # Errors
///
/// Returns an error for non-finite input, `eccentricity <= 1`, or overflow.
pub fn hyperbolic_anomaly_to_mean(hyperbolic_anomaly: f64, eccentricity: f64) -> Result<f64> {
    validate_hyperbolic("hyperbolic_anomaly", hyperbolic_anomaly, eccentricity)?;
    ensure_finite_output(
        "hyperbolic_anomaly_to_mean",
        eccentricity * hyperbolic_anomaly.sinh() - hyperbolic_anomaly,
    )
}

/// Converts hyperbolic anomaly to principal true anomaly.
///
/// # Errors
///
/// Returns an error for non-finite input or `eccentricity <= 1`.
pub fn hyperbolic_anomaly_to_true(hyperbolic_anomaly: f64, eccentricity: f64) -> Result<f64> {
    validate_hyperbolic("hyperbolic_anomaly", hyperbolic_anomaly, eccentricity)?;
    ensure_finite_output(
        "hyperbolic_anomaly_to_true",
        2.0 * (((1.0 + eccentricity) / (eccentricity - 1.0)).sqrt()
            * (hyperbolic_anomaly / 2.0).tanh())
        .atan(),
    )
}

/// Converts true anomaly to hyperbolic anomaly.
///
/// The true anomaly must lie inside the physical hyperbolic asymptote.
///
/// # Errors
///
/// Returns an error for non-finite input, `eccentricity <= 1`, or an anomaly
/// outside the `atanh` domain.
pub fn true_to_hyperbolic_anomaly(true_anomaly: f64, eccentricity: f64) -> Result<f64> {
    validate_hyperbolic("true_anomaly", true_anomaly, eccentricity)?;
    let argument =
        ((eccentricity - 1.0) / (1.0 + eccentricity)).sqrt() * (true_anomaly / 2.0).tan();
    if argument.abs() >= 1.0 {
        return Err(PykepError::InvalidInput {
            parameter: "true_anomaly",
            reason: "lies outside the physical hyperbolic asymptote".into(),
        });
    }
    ensure_finite_output("true_to_hyperbolic_anomaly", 2.0 * argument.atanh())
}

/// Converts hyperbolic mean anomaly to principal true anomaly.
///
/// # Errors
///
/// Returns validation, convergence, or overflow errors from the component
/// conversions.
pub fn hyperbolic_mean_to_true(mean_anomaly: f64, eccentricity: f64) -> Result<f64> {
    hyperbolic_anomaly_to_true(
        hyperbolic_mean_to_anomaly(mean_anomaly, eccentricity)?,
        eccentricity,
    )
}

/// Converts true anomaly to hyperbolic mean anomaly.
///
/// # Errors
///
/// Returns validation or overflow errors from the component conversions.
pub fn true_to_hyperbolic_mean(true_anomaly: f64, eccentricity: f64) -> Result<f64> {
    hyperbolic_anomaly_to_mean(
        true_to_hyperbolic_anomaly(true_anomaly, eccentricity)?,
        eccentricity,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn angle_close(left: f64, right: f64, tolerance: f64) {
        assert!((left.sin() - right.sin()).abs() <= tolerance);
        assert!((left.cos() - right.cos()).abs() <= tolerance);
    }

    #[test]
    fn elliptic_conversions_round_trip_across_regimes() {
        for &(mean, eccentricity) in &[(-1e6, 0.0), (-4.0, 0.5), (0.1, 0.9), (100.0, 0.999_999)] {
            let eccentric = mean_to_eccentric_anomaly(mean, eccentricity).unwrap();
            let reconstructed = eccentric_to_mean_anomaly(eccentric, eccentricity).unwrap();
            angle_close(reconstructed, mean, 2e-14);
            let true_anomaly = eccentric_to_true_anomaly(eccentric, eccentricity).unwrap();
            angle_close(
                true_to_eccentric_anomaly(true_anomaly, eccentricity).unwrap(),
                eccentric,
                2e-13,
            );
        }
    }

    #[test]
    fn hyperbolic_conversions_round_trip_across_regimes() {
        for &(mean, eccentricity) in &[(-100.0, 10.0), (-4.0, 1.5), (0.0, 1.000_001), (20.0, 100.0)]
        {
            let anomaly = hyperbolic_mean_to_anomaly(mean, eccentricity).unwrap();
            let reconstructed = hyperbolic_anomaly_to_mean(anomaly, eccentricity).unwrap();
            assert!((reconstructed - mean).abs() <= 5e-13 * mean.abs().max(1.0));
            let true_anomaly = hyperbolic_anomaly_to_true(anomaly, eccentricity).unwrap();
            let round_trip = true_to_hyperbolic_anomaly(true_anomaly, eccentricity).unwrap();
            assert!((round_trip - anomaly).abs() < 2e-13);
        }
    }

    #[test]
    fn invalid_domains_and_non_finite_values_are_errors() {
        assert!(mean_to_eccentric_anomaly(0.3, 1.0).is_err());
        assert!(hyperbolic_mean_to_anomaly(0.3, 1.0).is_err());
        assert!(mean_to_eccentric_anomaly(f64::NAN, 0.5).is_err());
        assert!(true_to_hyperbolic_anomaly(PI, 1.5).is_err());
    }
}
