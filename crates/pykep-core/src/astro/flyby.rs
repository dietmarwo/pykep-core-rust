// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/core_astro/flyby.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Unpowered gravity-assist constraints and outgoing velocity.

use crate::error::{ensure_finite, ensure_finite_output};
use crate::math::linalg::{cross, dot, norm, normalize};
use crate::{PykepError, Result, Vector3};

fn validate(mu: f64, radius: f64) -> Result<()> {
    ensure_finite("mu", mu)?;
    ensure_finite("radius", radius)?;
    if mu <= 0.0 || radius <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "mu/radius",
            reason: "must be greater than zero".into(),
        });
    }
    Ok(())
}

/// Returns the speed-equality and minimum-turn-angle flyby constraints.
///
/// Feasibility requires equality constraint zero and inequality constraint at
/// most zero.
///
/// # Errors
///
/// Returns an error for invalid body parameters, non-finite values, or a zero
/// relative velocity.
pub fn flyby_constraints(
    incoming: &Vector3,
    outgoing: &Vector3,
    mu: f64,
    safe_radius: f64,
) -> Result<[f64; 2]> {
    validate(mu, safe_radius)?;
    let incoming_squared = dot(incoming, incoming)?;
    let outgoing_squared = dot(outgoing, outgoing)?;
    if incoming_squared == 0.0 || outgoing_squared == 0.0 {
        return Err(PykepError::SingularGeometry {
            operation: "flyby_constraints",
        });
    }
    let minimum_eccentricity = 1.0 + safe_radius / mu * incoming_squared;
    let cosine =
        (dot(incoming, outgoing)? / (incoming_squared * outgoing_squared).sqrt()).clamp(-1.0, 1.0);
    Ok([
        incoming_squared - outgoing_squared,
        1.0 - 2.0 / minimum_eccentricity.powi(2) - cosine,
    ])
}

/// Returns the analytic `2 × 6` output-by-input flyby constraint Jacobian.
///
/// # Errors
///
/// Returns the same errors as [`flyby_constraints`].
pub fn flyby_constraints_jacobian(
    incoming: &Vector3,
    outgoing: &Vector3,
    mu: f64,
    safe_radius: f64,
) -> Result<[[f64; 6]; 2]> {
    flyby_constraints(incoming, outgoing, mu, safe_radius)?;
    let incoming_norm = norm(incoming)?;
    let outgoing_norm = norm(outgoing)?;
    let cosine = dot(incoming, outgoing)? / (incoming_norm * outgoing_norm);
    let eccentricity = 1.0 + safe_radius / mu * incoming_norm.powi(2);
    let scale = 8.0 * safe_radius / mu / eccentricity.powi(3);
    let mut jacobian = [[0.0; 6]; 2];
    for index in 0..3 {
        jacobian[0][index] = 2.0 * incoming[index];
        jacobian[0][index + 3] = -2.0 * outgoing[index];
        let cosine_in = outgoing[index] / (incoming_norm * outgoing_norm)
            - cosine * incoming[index] / incoming_norm.powi(2);
        let cosine_out = incoming[index] / (incoming_norm * outgoing_norm)
            - cosine * outgoing[index] / outgoing_norm.powi(2);
        jacobian[1][index] = scale * incoming[index] - cosine_in;
        jacobian[1][index + 3] = -cosine_out;
    }
    Ok(jacobian)
}

/// Computes the minimum powered-flyby delta-v needed to connect two excess
/// velocity vectors at or above `safe_radius`.
///
/// # Errors
///
/// Returns the same errors as [`flyby_constraints`].
pub fn flyby_delta_v(
    incoming: &Vector3,
    outgoing: &Vector3,
    mu: f64,
    safe_radius: f64,
) -> Result<f64> {
    validate(mu, safe_radius)?;
    let incoming_speed = norm(incoming)?;
    let outgoing_speed = norm(outgoing)?;
    if incoming_speed == 0.0 || outgoing_speed == 0.0 {
        return Err(PykepError::SingularGeometry {
            operation: "flyby_delta_v",
        });
    }
    let eccentricity = 1.0 + safe_radius / mu * incoming_speed.powi(2);
    let angle = (dot(incoming, outgoing)? / (incoming_speed * outgoing_speed))
        .clamp(-1.0, 1.0)
        .acos();
    let excess_turn = angle - 2.0 * (1.0 / eccentricity).asin();
    let delta_v = if excess_turn > 0.0 {
        (outgoing_speed.powi(2) + incoming_speed.powi(2)
            - 2.0 * outgoing_speed * incoming_speed * excess_turn.cos())
        .sqrt()
    } else {
        (outgoing_speed - incoming_speed).abs()
    };
    ensure_finite_output("flyby_delta_v", delta_v)
}

/// Maps an inertial incoming velocity through an unpowered flyby.
///
/// `beta` rotates the flyby plane about the incoming excess-velocity axis.
///
/// # Errors
///
/// Returns an error for invalid body parameters, zero excess velocity, or a
/// planet velocity parallel to the incoming excess velocity.
pub fn flyby_outgoing_velocity(
    incoming: &Vector3,
    planet_velocity: &Vector3,
    periapsis_radius: f64,
    beta: f64,
    mu: f64,
) -> Result<Vector3> {
    validate(mu, periapsis_radius)?;
    ensure_finite("beta", beta)?;
    let relative = [
        incoming[0] - planet_velocity[0],
        incoming[1] - planet_velocity[1],
        incoming[2] - planet_velocity[2],
    ];
    let speed = norm(&relative)?;
    if speed == 0.0 {
        return Err(PykepError::SingularGeometry {
            operation: "flyby_outgoing_velocity",
        });
    }
    let eccentricity = 1.0 + periapsis_radius / mu * speed.powi(2);
    let turn = 2.0 * (1.0 / eccentricity).asin();
    let i_hat = normalize(&relative)?;
    let j_hat = normalize(&cross(&i_hat, planet_velocity)?)?;
    let k_hat = cross(&i_hat, &j_hat)?;
    let output = core::array::from_fn(|index| {
        planet_velocity[index]
            + speed * turn.cos() * i_hat[index]
            + speed * beta.cos() * turn.sin() * j_hat[index]
            + speed * beta.sin() * turn.sin() * k_hat[index]
    });
    for &value in &output {
        ensure_finite_output("flyby_outgoing_velocity", value)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_reference_cases_match() {
        let mu = 398_600_441_800_000.0;
        let radius = 7_015_800.000_000_001;
        let incoming = [7200.0, -4567.7655, 1234.4233];
        let outgoing = [7100.0, 220.123, -144.432];
        let constraints = flyby_constraints(&incoming, &outgoing, mu, radius).unwrap();
        assert!((constraints[0] - 23_748_967.808_820_15).abs() < 1e-7);
        assert!((constraints[1] + 0.191_727_031_480_289_8).abs() < 2e-15);
        assert!(
            (flyby_delta_v(&incoming, &outgoing, mu, radius).unwrap() - 1_510.704_060_449_003)
                .abs()
                < 2e-11
        );
    }

    #[test]
    fn constraint_jacobian_matches_central_differences() {
        let incoming = [7200.0, -4567.7655, 1234.4233];
        let outgoing = [7100.0, 220.123, -144.432];
        let analytic = flyby_constraints_jacobian(&incoming, &outgoing, 3.986e14, 7.0e6).unwrap();
        for column in 0..6 {
            let mut plus_in = incoming;
            let mut minus_in = incoming;
            let mut plus_out = outgoing;
            let mut minus_out = outgoing;
            let step = 1e-3;
            if column < 3 {
                plus_in[column] += step;
                minus_in[column] -= step;
            } else {
                plus_out[column - 3] += step;
                minus_out[column - 3] -= step;
            }
            let plus = flyby_constraints(&plus_in, &plus_out, 3.986e14, 7.0e6).unwrap();
            let minus = flyby_constraints(&minus_in, &minus_out, 3.986e14, 7.0e6).unwrap();
            for row in 0..2 {
                let numerical = (plus[row] - minus[row]) / (2.0 * step);
                assert!((analytic[row][column] - numerical).abs() < 2e-5);
            }
        }
    }
}
