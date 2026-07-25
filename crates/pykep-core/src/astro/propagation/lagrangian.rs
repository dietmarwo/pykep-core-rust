// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/core_astro/propagate_lagrangian.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

use core::f64::consts::PI;

use super::{
    LagrangeSolution, MAX_ITERATIONS, combine, dot, linear_combination, norm, split, validate,
    validate_state_output,
};
use crate::astro::anomalies::{
    hyperbolic_mean_to_true, mean_to_true_anomaly, true_to_hyperbolic_mean, true_to_mean_anomaly,
};
use crate::astro::elements::{cartesian_to_classical, classical_to_cartesian};
use crate::math::kepler_equations::{
    elliptic_difference_derivative, elliptic_difference_residual, hyperbolic_difference_derivative,
    hyperbolic_difference_residual, universal_kepler_derivative, universal_kepler_residual,
};
use crate::math::stumpff::{stumpff_c, stumpff_s};
use crate::{CartesianState, PykepError, Result};

fn converged(step: f64, value: f64) -> bool {
    step.abs() <= 4.0 * f64::EPSILON * value.abs().max(1.0)
}

fn solve_elliptic_difference(
    initial: f64,
    mean_difference: f64,
    sigma0: f64,
    sqrt_a: f64,
    a: f64,
    radius: f64,
) -> Result<f64> {
    let mut anomaly = initial;
    for _ in 0..MAX_ITERATIONS {
        let residual =
            elliptic_difference_residual(anomaly, mean_difference, sigma0, sqrt_a, a, radius)?;
        let derivative = elliptic_difference_derivative(anomaly, sigma0, sqrt_a, a, radius)?;
        let step = residual / derivative;
        anomaly -= step;
        if !anomaly.is_finite() {
            return Err(PykepError::NumericalOverflow {
                operation: "propagate_lagrangian",
            });
        }
        if converged(step, anomaly) {
            return Ok(anomaly);
        }
    }
    Err(PykepError::ConvergenceFailure {
        operation: "propagate_lagrangian_elliptic",
        iterations: MAX_ITERATIONS,
    })
}

fn solve_hyperbolic_difference(
    initial: f64,
    mean_difference: f64,
    sigma0: f64,
    sqrt_abs_a: f64,
    a: f64,
    radius: f64,
) -> Result<f64> {
    let mut anomaly = initial;
    for _ in 0..MAX_ITERATIONS {
        let residual = hyperbolic_difference_residual(
            anomaly,
            mean_difference,
            sigma0,
            sqrt_abs_a,
            a,
            radius,
        )?;
        let derivative = hyperbolic_difference_derivative(anomaly, sigma0, sqrt_abs_a, a, radius)?;
        let step = residual / derivative;
        let next = anomaly - step;
        if !next.is_finite() || next.abs() > 50.0 {
            anomaly *= 0.5;
            continue;
        }
        anomaly = next;
        if converged(step, anomaly) {
            return Ok(anomaly);
        }
    }
    Err(PykepError::ConvergenceFailure {
        operation: "propagate_lagrangian_hyperbolic",
        iterations: MAX_ITERATIONS,
    })
}

pub(super) fn solve_lagrangian(
    state: &CartesianState,
    time: f64,
    mu: f64,
) -> Result<LagrangeSolution> {
    validate(state, time, mu)?;
    let (initial_position, initial_velocity) = split(state);
    let initial_radius = norm(&initial_position);
    let velocity_squared = dot(&initial_velocity, &initial_velocity);
    let energy = velocity_squared / 2.0 - mu / initial_radius;

    if time == 0.0 {
        return Ok(LagrangeSolution {
            state: *state,
            initial_radius,
            final_radius: initial_radius,
            energy,
            sigma0: dot(&initial_position, &initial_velocity) / mu.sqrt(),
            semi_major_axis: -mu / (2.0 * energy),
            s0: 0.0,
            c0: 0.0,
            anomaly_difference: 0.0,
            f: 1.0,
            g: 0.0,
            f_dot: 0.0,
            g_dot: 1.0,
        });
    }
    if energy == 0.0 {
        let propagated = propagate_universal(state, time, mu)?;
        return Ok(LagrangeSolution {
            state: propagated,
            initial_radius,
            final_radius: norm(&[propagated[0], propagated[1], propagated[2]]),
            energy,
            sigma0: dot(&initial_position, &initial_velocity) / mu.sqrt(),
            semi_major_axis: f64::INFINITY,
            s0: 0.0,
            c0: 0.0,
            anomaly_difference: 0.0,
            f: 0.0,
            g: 0.0,
            f_dot: 0.0,
            g_dot: 0.0,
        });
    }

    let semi_major_axis = -mu / (2.0 * energy);
    let sigma0 = dot(&initial_position, &initial_velocity) / mu.sqrt();
    let (final_radius, s0, c0, anomaly_difference, f, g, f_dot, g_dot) = if semi_major_axis > 0.0 {
        let sqrt_a = semi_major_axis.sqrt();
        let mean_difference = (mu / semi_major_axis.powi(3)).sqrt() * time;
        let sine_mean = mean_difference.sin();
        let cosine_mean = mean_difference.cos();
        let mut cropped_mean = sine_mean.atan2(cosine_mean);
        if cropped_mean < 0.0 {
            cropped_mean += 2.0 * PI;
        }
        let s0 = sigma0 / sqrt_a;
        let c0 = 1.0 - initial_radius / semi_major_axis;
        let auxiliary = c0 * sine_mean + s0 * cosine_mean - s0;
        let derivative_auxiliary = c0 * cosine_mean - s0 * sine_mean;
        let initial = cropped_mean + c0 * sine_mean - s0 * (1.0 - cosine_mean)
            + derivative_auxiliary * auxiliary
            + 0.5
                * auxiliary
                * (2.0 * derivative_auxiliary.powi(2)
                    - auxiliary * (c0 * sine_mean + s0 * cosine_mean));
        let anomaly = solve_elliptic_difference(
            initial,
            cropped_mean,
            sigma0,
            sqrt_a,
            semi_major_axis,
            initial_radius,
        )?;
        let (sine, cosine) = anomaly.sin_cos();
        let final_radius =
            semi_major_axis + (initial_radius - semi_major_axis) * cosine + sigma0 * sqrt_a * sine;
        let f = 1.0 - semi_major_axis / initial_radius * (1.0 - cosine);
        let g = semi_major_axis * sigma0 / mu.sqrt() * (1.0 - cosine)
            + initial_radius * (semi_major_axis / mu).sqrt() * sine;
        let f_dot = -(mu * semi_major_axis).sqrt() / (final_radius * initial_radius) * sine;
        let g_dot = 1.0 - semi_major_axis / final_radius * (1.0 - cosine);
        (final_radius, s0, c0, anomaly, f, g, f_dot, g_dot)
    } else {
        let sqrt_abs_a = (-semi_major_axis).sqrt();
        let mean_difference = (-mu / semi_major_axis.powi(3)).sqrt() * time;
        let s0 = sigma0 / sqrt_abs_a;
        let c0 = 1.0 - initial_radius / semi_major_axis;
        let initial = if time > 0.0 { 1.0 } else { -1.0 };
        let anomaly = solve_hyperbolic_difference(
            initial,
            mean_difference,
            sigma0,
            sqrt_abs_a,
            semi_major_axis,
            initial_radius,
        )?;
        let sine = anomaly.sinh();
        let cosine = anomaly.cosh();
        let final_radius = semi_major_axis
            + (initial_radius - semi_major_axis) * cosine
            + sigma0 * sqrt_abs_a * sine;
        let f = 1.0 - semi_major_axis / initial_radius * (1.0 - cosine);
        let g = semi_major_axis * sigma0 / mu.sqrt() * (1.0 - cosine)
            + initial_radius * (-semi_major_axis / mu).sqrt() * sine;
        let f_dot = -(-mu * semi_major_axis).sqrt() / (final_radius * initial_radius) * sine;
        let g_dot = 1.0 - semi_major_axis / final_radius * (1.0 - cosine);
        (final_radius, s0, c0, anomaly, f, g, f_dot, g_dot)
    };

    if final_radius <= 0.0 || !final_radius.is_finite() {
        return Err(PykepError::NumericalOverflow {
            operation: "propagate_lagrangian",
        });
    }
    let final_position = linear_combination(f, &initial_position, g, &initial_velocity);
    let final_velocity = linear_combination(f_dot, &initial_position, g_dot, &initial_velocity);
    let state = validate_state_output(
        "propagate_lagrangian",
        combine(final_position, final_velocity),
    )?;
    Ok(LagrangeSolution {
        state,
        initial_radius,
        final_radius,
        energy,
        sigma0,
        semi_major_axis,
        s0,
        c0,
        anomaly_difference,
        f,
        g,
        f_dot,
        g_dot,
    })
}

/// Propagates a Cartesian state with elliptic/hyperbolic Lagrange
/// coefficients.
///
/// `state`, `time`, and `mu` must use one consistent unit system.
///
/// # Errors
///
/// Returns an error for non-finite input, non-positive `mu`, zero radius,
/// numerical overflow, or failure to converge in 100 Newton iterations.
pub fn propagate_lagrangian(state: &CartesianState, time: f64, mu: f64) -> Result<CartesianState> {
    solve_lagrangian(state, time, mu).map(|solution| solution.state)
}

/// Propagates a Cartesian state with universal variables and Stumpff
/// functions.
///
/// This formulation also covers the parabolic limit.
///
/// # Errors
///
/// Returns an error for invalid input, Stumpff overflow, or failure to
/// converge in 100 Newton iterations.
pub fn propagate_universal(state: &CartesianState, time: f64, mu: f64) -> Result<CartesianState> {
    validate(state, time, mu)?;
    if time == 0.0 {
        return Ok(*state);
    }
    let (initial_position, mut initial_velocity) = split(state);
    let backward = time < 0.0;
    let propagation_time = time.abs();
    if backward {
        initial_velocity = initial_velocity.map(|value| -value);
    }
    let initial_radius = norm(&initial_position);
    let velocity_squared = dot(&initial_velocity, &initial_velocity);
    let alpha = 2.0 / initial_radius - velocity_squared / mu;
    let radial_velocity = dot(&initial_position, &initial_velocity) / initial_radius;
    let sqrt_mu = mu.sqrt();
    let mut anomaly = if alpha > 0.0 {
        sqrt_mu * propagation_time * alpha.abs()
    } else {
        3.0
    };
    let mut lower = anomaly - 2.0 * PI;
    let mut upper = anomaly + 2.0 * PI;
    for _ in 0..MAX_ITERATIONS {
        let residual = universal_kepler_residual(
            anomaly,
            propagation_time,
            initial_radius,
            radial_velocity,
            alpha,
            mu,
        )?;
        let derivative =
            universal_kepler_derivative(anomaly, initial_radius, radial_velocity, alpha, mu)?;
        if residual < 0.0 {
            lower = anomaly;
        } else {
            upper = anomaly;
        }
        let step = residual / derivative;
        let newton = anomaly - step;
        let next = if newton > lower && newton < upper && newton.is_finite() {
            newton
        } else {
            0.5 * (lower + upper)
        };
        let actual_step = next - anomaly;
        anomaly = next;
        if converged(actual_step, anomaly) {
            let argument = alpha * anomaly * anomaly;
            let s = stumpff_s(argument)?;
            let c = stumpff_c(argument)?;
            let f = 1.0 - anomaly * anomaly / initial_radius * c;
            let g = propagation_time - anomaly.powi(3) / sqrt_mu * s;
            let final_position = linear_combination(f, &initial_position, g, &initial_velocity);
            let final_radius = norm(&final_position);
            if final_radius == 0.0 {
                return Err(PykepError::SingularGeometry {
                    operation: "propagate_universal",
                });
            }
            let f_dot = sqrt_mu / (final_radius * initial_radius) * (argument * s - 1.0) * anomaly;
            let g_dot = 1.0 - anomaly * anomaly / final_radius * c;
            let mut final_velocity =
                linear_combination(f_dot, &initial_position, g_dot, &initial_velocity);
            if backward {
                final_velocity = final_velocity.map(|value| -value);
            }
            return validate_state_output(
                "propagate_universal",
                combine(final_position, final_velocity),
            );
        }
    }
    Err(PykepError::ConvergenceFailure {
        operation: "propagate_universal",
        iterations: MAX_ITERATIONS,
    })
}

/// Propagates by converting to classical elements and advancing mean anomaly.
///
/// This slower reference path is undefined for circular or equatorial states
/// because their classical angles are singular.
///
/// # Errors
///
/// Returns conversion, anomaly-solver, or validation errors.
pub fn propagate_keplerian(state: &CartesianState, time: f64, mu: f64) -> Result<CartesianState> {
    validate(state, time, mu)?;
    if time == 0.0 {
        return Ok(*state);
    }
    let mut elements = cartesian_to_classical(state, mu)?;
    if elements.semi_major_axis > 0.0 {
        let mean_motion = (mu / elements.semi_major_axis.powi(3)).sqrt();
        let initial_mean = true_to_mean_anomaly(elements.true_anomaly, elements.eccentricity)?;
        elements.true_anomaly =
            mean_to_true_anomaly(initial_mean + mean_motion * time, elements.eccentricity)?;
    } else {
        let mean_motion = (-mu / elements.semi_major_axis.powi(3)).sqrt();
        let initial_mean = true_to_hyperbolic_mean(elements.true_anomaly, elements.eccentricity)?;
        elements.true_anomaly =
            hyperbolic_mean_to_true(initial_mean + mean_motion * time, elements.eccentricity)?;
    }
    classical_to_cartesian(elements, mu)
}

/// Propagates one initial state over a time grid relative to its first value.
///
/// An empty grid returns an empty vector. Output order matches input order.
///
/// # Errors
///
/// Returns the first validation or propagation error.
pub fn propagate_lagrangian_grid(
    state: &CartesianState,
    time_grid: &[f64],
    mu: f64,
) -> Result<Vec<CartesianState>> {
    if time_grid.is_empty() {
        return Ok(Vec::new());
    }
    let origin = time_grid[0];
    time_grid
        .iter()
        .map(|&time| propagate_lagrangian(state, time - origin, mu))
        .collect()
}
