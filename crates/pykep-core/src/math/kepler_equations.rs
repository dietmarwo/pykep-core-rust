// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from include/kep3/core_astro/kepler_equations.hpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Residuals and derivatives for elliptic, hyperbolic, and universal Kepler
//! equations.

use super::stumpff::{stumpff_c, stumpff_s};
use crate::error::{ensure_finite_output, ensure_finite_values};
use crate::{PykepError, Result};

fn validate_elliptic_eccentricity(eccentricity: f64) -> Result<()> {
    ensure_finite_values(&[("eccentricity", eccentricity)])?;
    if (0.0..1.0).contains(&eccentricity) {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter: "eccentricity",
            reason: "elliptic eccentricity must satisfy 0 <= e < 1".into(),
        })
    }
}

fn validate_hyperbolic_eccentricity(eccentricity: f64) -> Result<()> {
    ensure_finite_values(&[("eccentricity", eccentricity)])?;
    if eccentricity > 1.0 {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter: "eccentricity",
            reason: "hyperbolic eccentricity must satisfy e > 1".into(),
        })
    }
}

fn validate_difference_parameters(
    delta_anomaly: f64,
    sigma0: f64,
    sqrt_abs_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> Result<()> {
    ensure_finite_values(&[
        ("delta_anomaly", delta_anomaly),
        ("sigma0", sigma0),
        ("sqrt_abs_semi_major_axis", sqrt_abs_semi_major_axis),
        ("semi_major_axis", semi_major_axis),
        ("initial_radius", initial_radius),
    ])?;
    if sqrt_abs_semi_major_axis <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "sqrt_abs_semi_major_axis",
            reason: "must be positive".into(),
        });
    }
    if semi_major_axis == 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "semi_major_axis",
            reason: "must be non-zero".into(),
        });
    }
    if initial_radius <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "initial_radius",
            reason: "must be positive".into(),
        });
    }
    Ok(())
}

fn validate_universal_parameters(
    delta_s: f64,
    initial_radius: f64,
    initial_radial_velocity: f64,
    alpha: f64,
    mu: f64,
) -> Result<()> {
    ensure_finite_values(&[
        ("delta_s", delta_s),
        ("initial_radius", initial_radius),
        ("initial_radial_velocity", initial_radial_velocity),
        ("alpha", alpha),
        ("mu", mu),
    ])?;
    if initial_radius <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "initial_radius",
            reason: "must be positive".into(),
        });
    }
    if mu <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "mu",
            reason: "must be positive".into(),
        });
    }
    Ok(())
}

/// Evaluates the elliptic Kepler residual `E - e sin(E) - M`.
///
/// All anomalies are in radians and `0 <= eccentricity < 1`.
///
/// # Errors
///
/// Returns an error for non-finite inputs, an invalid eccentricity, or
/// binary64 overflow.
pub fn elliptic_kepler_residual(
    eccentric_anomaly: f64,
    mean_anomaly: f64,
    eccentricity: f64,
) -> Result<f64> {
    ensure_finite_values(&[
        ("eccentric_anomaly", eccentric_anomaly),
        ("mean_anomaly", mean_anomaly),
    ])?;
    validate_elliptic_eccentricity(eccentricity)?;
    ensure_finite_output(
        "elliptic_kepler_residual",
        eccentric_anomaly - eccentricity * eccentric_anomaly.sin() - mean_anomaly,
    )
}

/// Evaluates the first derivative `1 - e cos(E)` of the elliptic residual.
///
/// # Errors
///
/// Returns an error for non-finite input or an eccentricity outside
/// `0 <= e < 1`.
pub fn elliptic_kepler_derivative(eccentric_anomaly: f64, eccentricity: f64) -> Result<f64> {
    ensure_finite_values(&[("eccentric_anomaly", eccentric_anomaly)])?;
    validate_elliptic_eccentricity(eccentricity)?;
    ensure_finite_output(
        "elliptic_kepler_derivative",
        1.0 - eccentricity * eccentric_anomaly.cos(),
    )
}

/// Evaluates the second derivative `e sin(E)` of the elliptic residual.
///
/// # Errors
///
/// Returns an error for non-finite input or an eccentricity outside
/// `0 <= e < 1`.
pub fn elliptic_kepler_second_derivative(eccentric_anomaly: f64, eccentricity: f64) -> Result<f64> {
    ensure_finite_values(&[("eccentric_anomaly", eccentric_anomaly)])?;
    validate_elliptic_eccentricity(eccentricity)?;
    ensure_finite_output(
        "elliptic_kepler_second_derivative",
        eccentricity * eccentric_anomaly.sin(),
    )
}

/// Evaluates the hyperbolic Kepler residual `e sinh(H) - H - M`.
///
/// All anomalies are dimensionless hyperbolic angles and `eccentricity > 1`.
///
/// # Errors
///
/// Returns an error for non-finite inputs, an invalid eccentricity, or
/// binary64 overflow.
pub fn hyperbolic_kepler_residual(
    hyperbolic_anomaly: f64,
    mean_anomaly: f64,
    eccentricity: f64,
) -> Result<f64> {
    ensure_finite_values(&[
        ("hyperbolic_anomaly", hyperbolic_anomaly),
        ("mean_anomaly", mean_anomaly),
    ])?;
    validate_hyperbolic_eccentricity(eccentricity)?;
    ensure_finite_output(
        "hyperbolic_kepler_residual",
        eccentricity * hyperbolic_anomaly.sinh() - hyperbolic_anomaly - mean_anomaly,
    )
}

/// Evaluates the first derivative `e cosh(H) - 1` of the hyperbolic residual.
///
/// # Errors
///
/// Returns an error for non-finite input, `eccentricity <= 1`, or binary64
/// overflow.
pub fn hyperbolic_kepler_derivative(hyperbolic_anomaly: f64, eccentricity: f64) -> Result<f64> {
    ensure_finite_values(&[("hyperbolic_anomaly", hyperbolic_anomaly)])?;
    validate_hyperbolic_eccentricity(eccentricity)?;
    ensure_finite_output(
        "hyperbolic_kepler_derivative",
        eccentricity * hyperbolic_anomaly.cosh() - 1.0,
    )
}

/// Evaluates the second derivative `e sinh(H)` of the hyperbolic residual.
///
/// # Errors
///
/// Returns an error for non-finite input, `eccentricity <= 1`, or binary64
/// overflow.
pub fn hyperbolic_kepler_second_derivative(
    hyperbolic_anomaly: f64,
    eccentricity: f64,
) -> Result<f64> {
    ensure_finite_values(&[("hyperbolic_anomaly", hyperbolic_anomaly)])?;
    validate_hyperbolic_eccentricity(eccentricity)?;
    ensure_finite_output(
        "hyperbolic_kepler_second_derivative",
        eccentricity * hyperbolic_anomaly.sinh(),
    )
}

/// Evaluates Kepler's equation in elliptic anomaly difference `ΔE`.
///
/// `delta_mean_anomaly` and `delta_eccentric_anomaly` are radians,
/// `sqrt_semi_major_axis` is the positive square root of the semi-major-axis
/// length, and `initial_radius` uses the same length unit.
///
/// # Errors
///
/// Returns an error for non-finite inputs, non-positive square-root/radius,
/// zero semi-major axis, or binary64 overflow.
pub fn elliptic_difference_residual(
    delta_eccentric_anomaly: f64,
    delta_mean_anomaly: f64,
    sigma0: f64,
    sqrt_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> Result<f64> {
    ensure_finite_values(&[("delta_mean_anomaly", delta_mean_anomaly)])?;
    validate_difference_parameters(
        delta_eccentric_anomaly,
        sigma0,
        sqrt_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )?;
    ensure_finite_output(
        "elliptic_difference_residual",
        -delta_mean_anomaly
            + delta_eccentric_anomaly
            + sigma0 / sqrt_semi_major_axis * (1.0 - delta_eccentric_anomaly.cos())
            - (1.0 - initial_radius / semi_major_axis) * delta_eccentric_anomaly.sin(),
    )
}

/// Evaluates the first derivative of [`elliptic_difference_residual`].
///
/// # Errors
///
/// Returns the same validation and overflow errors as the residual.
pub fn elliptic_difference_derivative(
    delta_eccentric_anomaly: f64,
    sigma0: f64,
    sqrt_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> Result<f64> {
    validate_difference_parameters(
        delta_eccentric_anomaly,
        sigma0,
        sqrt_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )?;
    ensure_finite_output(
        "elliptic_difference_derivative",
        1.0 + sigma0 / sqrt_semi_major_axis * delta_eccentric_anomaly.sin()
            - (1.0 - initial_radius / semi_major_axis) * delta_eccentric_anomaly.cos(),
    )
}

/// Evaluates the second derivative of [`elliptic_difference_residual`].
///
/// # Errors
///
/// Returns the same validation and overflow errors as the residual.
pub fn elliptic_difference_second_derivative(
    delta_eccentric_anomaly: f64,
    sigma0: f64,
    sqrt_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> Result<f64> {
    validate_difference_parameters(
        delta_eccentric_anomaly,
        sigma0,
        sqrt_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )?;
    ensure_finite_output(
        "elliptic_difference_second_derivative",
        sigma0 / sqrt_semi_major_axis * delta_eccentric_anomaly.cos()
            + (1.0 - initial_radius / semi_major_axis) * delta_eccentric_anomaly.sin(),
    )
}

/// Evaluates Kepler's equation in hyperbolic anomaly difference `ΔH`.
///
/// The hyperbolic semi-major axis follows the upstream negative convention;
/// `sqrt_abs_semi_major_axis` is positive.
///
/// # Errors
///
/// Returns an error for non-finite inputs, non-positive square-root/radius,
/// zero semi-major axis, or binary64 overflow.
pub fn hyperbolic_difference_residual(
    delta_hyperbolic_anomaly: f64,
    delta_mean_anomaly: f64,
    sigma0: f64,
    sqrt_abs_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> Result<f64> {
    ensure_finite_values(&[("delta_mean_anomaly", delta_mean_anomaly)])?;
    validate_difference_parameters(
        delta_hyperbolic_anomaly,
        sigma0,
        sqrt_abs_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )?;
    ensure_finite_output(
        "hyperbolic_difference_residual",
        -delta_mean_anomaly - delta_hyperbolic_anomaly
            + sigma0 / sqrt_abs_semi_major_axis * (delta_hyperbolic_anomaly.cosh() - 1.0)
            + (1.0 - initial_radius / semi_major_axis) * delta_hyperbolic_anomaly.sinh(),
    )
}

/// Evaluates the first derivative of [`hyperbolic_difference_residual`].
///
/// # Errors
///
/// Returns the same validation and overflow errors as the residual.
pub fn hyperbolic_difference_derivative(
    delta_hyperbolic_anomaly: f64,
    sigma0: f64,
    sqrt_abs_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> Result<f64> {
    validate_difference_parameters(
        delta_hyperbolic_anomaly,
        sigma0,
        sqrt_abs_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )?;
    ensure_finite_output(
        "hyperbolic_difference_derivative",
        -1.0 + sigma0 / sqrt_abs_semi_major_axis * delta_hyperbolic_anomaly.sinh()
            + (1.0 - initial_radius / semi_major_axis) * delta_hyperbolic_anomaly.cosh(),
    )
}

/// Evaluates the second derivative of [`hyperbolic_difference_residual`].
///
/// # Errors
///
/// Returns the same validation and overflow errors as the residual.
pub fn hyperbolic_difference_second_derivative(
    delta_hyperbolic_anomaly: f64,
    sigma0: f64,
    sqrt_abs_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> Result<f64> {
    validate_difference_parameters(
        delta_hyperbolic_anomaly,
        sigma0,
        sqrt_abs_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )?;
    ensure_finite_output(
        "hyperbolic_difference_second_derivative",
        sigma0 / sqrt_abs_semi_major_axis * delta_hyperbolic_anomaly.cosh()
            + (1.0 - initial_radius / semi_major_axis) * delta_hyperbolic_anomaly.sinh(),
    )
}

/// Evaluates universal-variable Kepler's equation in anomaly difference `Δs`.
///
/// `delta_time` is in seconds, `initial_radius` in length units,
/// `initial_radial_velocity` in length per second, `alpha` in inverse length,
/// and `mu` in length³ per second². A consistent unit system is required.
///
/// # Errors
///
/// Returns an error for non-finite inputs, non-positive radius or `mu`,
/// Stumpff overflow, or residual overflow.
pub fn universal_kepler_residual(
    delta_s: f64,
    delta_time: f64,
    initial_radius: f64,
    initial_radial_velocity: f64,
    alpha: f64,
    mu: f64,
) -> Result<f64> {
    ensure_finite_values(&[("delta_time", delta_time)])?;
    validate_universal_parameters(delta_s, initial_radius, initial_radial_velocity, alpha, mu)?;
    let argument = alpha * delta_s * delta_s;
    let s = stumpff_s(argument)?;
    let c = stumpff_c(argument)?;
    let sqrt_mu = mu.sqrt();
    ensure_finite_output(
        "universal_kepler_residual",
        -sqrt_mu * delta_time
            + initial_radius * initial_radial_velocity * delta_s.powi(2) * c / sqrt_mu
            + (1.0 - alpha * initial_radius) * delta_s.powi(3) * s
            + initial_radius * delta_s,
    )
}

/// Evaluates the first derivative of [`universal_kepler_residual`].
///
/// # Errors
///
/// Returns the same validation and overflow errors as the residual.
pub fn universal_kepler_derivative(
    delta_s: f64,
    initial_radius: f64,
    initial_radial_velocity: f64,
    alpha: f64,
    mu: f64,
) -> Result<f64> {
    validate_universal_parameters(delta_s, initial_radius, initial_radial_velocity, alpha, mu)?;
    let argument = alpha * delta_s * delta_s;
    let s = stumpff_s(argument)?;
    let c = stumpff_c(argument)?;
    ensure_finite_output(
        "universal_kepler_derivative",
        initial_radius * initial_radial_velocity / mu.sqrt()
            * delta_s
            * (1.0 - alpha * delta_s.powi(2) * s)
            + (1.0 - alpha * initial_radius) * delta_s.powi(2) * c
            + initial_radius,
    )
}

/// Evaluates the second derivative of [`universal_kepler_residual`].
///
/// # Errors
///
/// Returns the same validation and overflow errors as the residual.
pub fn universal_kepler_second_derivative(
    delta_s: f64,
    initial_radius: f64,
    initial_radial_velocity: f64,
    alpha: f64,
    mu: f64,
) -> Result<f64> {
    validate_universal_parameters(delta_s, initial_radius, initial_radial_velocity, alpha, mu)?;
    let argument = alpha * delta_s * delta_s;
    let s = stumpff_s(argument)?;
    let c = stumpff_c(argument)?;
    ensure_finite_output(
        "universal_kepler_second_derivative",
        initial_radius * initial_radial_velocity / mu.sqrt() * (1.0 - alpha * delta_s.powi(2) * c)
            + (1.0 - alpha * initial_radius) * (1.0 - delta_s.powi(2) * s),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn central_first_derivative(function: impl Fn(f64) -> Result<f64>, x: f64, step: f64) -> f64 {
        (function(x + step).unwrap() - function(x - step).unwrap()) / (2.0 * step)
    }

    #[test]
    fn elliptic_reference_values_and_derivatives_match() {
        let e = 0.9;
        let anomaly = 0.1;
        let residual = elliptic_kepler_residual(anomaly, -2.0, e).unwrap();
        assert!((residual - 2.010_149_925_017_854).abs() < 5e-16);
        let numerical = central_first_derivative(
            |value| elliptic_kepler_residual(value, -2.0, e),
            anomaly,
            1e-5,
        );
        assert!((numerical - elliptic_kepler_derivative(anomaly, e).unwrap()).abs() < 3e-11);
    }

    #[test]
    fn hyperbolic_reference_values_and_derivatives_match() {
        let e = 1.5;
        let anomaly = -4.0;
        let residual = hyperbolic_kepler_residual(anomaly, 3.0, e).unwrap();
        assert!((residual + 39.934_875_795_691_624).abs() < 1e-13);
        let numerical = central_first_derivative(
            |value| hyperbolic_kepler_residual(value, 3.0, e),
            anomaly,
            1e-5,
        );
        assert!((numerical - hyperbolic_kepler_derivative(anomaly, e).unwrap()).abs() < 1e-8);
    }

    #[test]
    fn invalid_domains_and_non_finite_values_are_rejected() {
        assert!(elliptic_kepler_residual(0.1, 0.2, 1.0).is_err());
        assert!(hyperbolic_kepler_residual(0.1, 0.2, 1.0).is_err());
        assert!(elliptic_kepler_residual(f64::NAN, 0.2, 0.5).is_err());
        assert!(universal_kepler_residual(1.0, 2.0, 0.0, 0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn difference_equations_have_consistent_derivatives() {
        // Central differences subtract nearby residuals. This scale-aware
        // bound covers final-bit variation in native and Miri transcendental
        // evaluation while still detecting an incorrect analytic derivative.
        const RELATIVE_TOLERANCE: f64 = 1e-10;

        let numerical = central_first_derivative(
            |value| elliptic_difference_residual(value, 0.3, 0.2, 2.0, 4.0, 3.0),
            0.4,
            1e-5,
        );
        let analytic = elliptic_difference_derivative(0.4, 0.2, 2.0, 4.0, 3.0).unwrap();
        let error = (numerical - analytic).abs();
        let tolerance = RELATIVE_TOLERANCE * analytic.abs().max(1.0);
        assert!(
            error < tolerance,
            "elliptic finite-difference error {error:e} exceeds {tolerance:e}: numerical={numerical:e}, analytic={analytic:e}"
        );

        let numerical = central_first_derivative(
            |value| hyperbolic_difference_residual(value, 0.3, 0.2, 2.0, -4.0, 3.0),
            0.4,
            1e-5,
        );
        let analytic = hyperbolic_difference_derivative(0.4, 0.2, 2.0, -4.0, 3.0).unwrap();
        let error = (numerical - analytic).abs();
        let tolerance = RELATIVE_TOLERANCE * analytic.abs().max(1.0);
        assert!(
            error < tolerance,
            "hyperbolic finite-difference error {error:e} exceeds {tolerance:e}: numerical={numerical:e}, analytic={analytic:e}"
        );
    }

    #[test]
    fn universal_equation_derivative_matches_finite_difference() {
        let numerical = central_first_derivative(
            |value| universal_kepler_residual(value, 20.0, 7.0, 0.2, 0.01, 3.0),
            0.4,
            1e-5,
        );
        let analytic = universal_kepler_derivative(0.4, 7.0, 0.2, 0.01, 3.0).unwrap();
        assert!((numerical - analytic).abs() < 2e-9);
    }
}
