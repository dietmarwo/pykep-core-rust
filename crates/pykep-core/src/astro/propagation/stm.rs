// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/core_astro/stm.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

use super::lagrangian::solve_lagrangian;
use super::{LagrangeSolution, cross, norm, split, validate};
use crate::math::linalg::{identity, matrix_multiply};
use crate::{CartesianState, Matrix3, Matrix6, PykepError, Result};

type Gradient = [f64; 6];

fn gradient(operation: impl Fn(usize) -> f64) -> Gradient {
    core::array::from_fn(operation)
}

fn lagrangian_stm(
    initial_state: &CartesianState,
    time: f64,
    mu: f64,
    solution: LagrangeSolution,
) -> Result<Matrix6> {
    if time == 0.0 {
        return Ok(identity());
    }
    if !solution.semi_major_axis.is_finite() {
        return Err(PykepError::InvalidInput {
            parameter: "state",
            reason: "the Lagrangian STM is undefined at the exact parabolic limit".into(),
        });
    }
    let (position, velocity) = split(initial_state);
    let radius0 = solution.initial_radius;
    let radius_final = solution.final_radius;
    let energy = solution.energy;
    let sigma0 = solution.sigma0;
    let semi_major_axis = solution.semi_major_axis;
    let s0 = solution.s0;
    let c0 = solution.c0;
    let anomaly = solution.anomaly_difference;
    let sqrt_mu = mu.sqrt();

    let d_velocity_squared = gradient(|column| {
        if column >= 3 {
            2.0 * velocity[column - 3]
        } else {
            0.0
        }
    });
    let d_radius0 = gradient(|column| {
        if column < 3 {
            position[column] / radius0
        } else {
            0.0
        }
    });
    let d_energy = gradient(|column| {
        0.5 * d_velocity_squared[column] + mu / radius0.powi(2) * d_radius0[column]
    });
    let d_sigma0 = gradient(|column| {
        if column < 3 {
            velocity[column] / sqrt_mu
        } else {
            position[column - 3] / sqrt_mu
        }
    });
    let d_a = gradient(|column| mu / (2.0 * energy.powi(2)) * d_energy[column]);

    let (d_f, d_g, d_f_dot, d_g_dot) = if semi_major_axis > 0.0 {
        let sqrt_a = semi_major_axis.sqrt();
        let sine = anomaly.sin();
        let cosine = anomaly.cos();
        let d_s0 = gradient(|column| {
            d_sigma0[column] / sqrt_a - 0.5 * sigma0 / sqrt_a.powi(3) * d_a[column]
        });
        let d_c0 = gradient(|column| {
            -d_radius0[column] / semi_major_axis + radius0 / semi_major_axis.powi(2) * d_a[column]
        });
        let d_mean = gradient(|column| -1.5 * sqrt_mu * time / sqrt_a.powi(5) * d_a[column]);
        let denominator = 1.0 + s0 * sine - c0 * cosine;
        let d_anomaly = gradient(|column| {
            (d_mean[column] - (1.0 - cosine) * d_s0[column] + sine * d_c0[column]) / denominator
        });
        let d_radius_final = gradient(|column| {
            (1.0 - cosine + 0.5 / sqrt_a * sigma0 * sine) * d_a[column]
                + cosine * d_radius0[column]
                + (sigma0 * sqrt_a * cosine - (radius0 - semi_major_axis) * sine)
                    * d_anomaly[column]
                + sqrt_a * sine * d_sigma0[column]
        });
        let d_f = gradient(|column| {
            -(1.0 - cosine) / radius0 * d_a[column]
                + semi_major_axis / radius0.powi(2) * (1.0 - cosine) * d_radius0[column]
                - semi_major_axis / radius0 * sine * d_anomaly[column]
        });
        let d_g = gradient(|column| {
            ((1.0 - solution.f) * (radius0 * d_sigma0[column] + sigma0 * d_radius0[column])
                - sigma0 * radius0 * d_f[column]
                + sqrt_a * radius0 * cosine * d_anomaly[column]
                + sqrt_a * sine * d_radius0[column]
                + 0.5 * radius0 * sine / sqrt_a * d_a[column])
                / sqrt_mu
        });
        let d_f_dot = gradient(|column| {
            sqrt_mu
                * (-sqrt_a / radius0 / radius_final * cosine * d_anomaly[column]
                    - 0.5 / sqrt_a / radius0 / radius_final * sine * d_a[column]
                    + sqrt_a / radius_final / radius0.powi(2) * sine * d_radius0[column]
                    + sqrt_a / radius_final.powi(2) / radius0 * sine * d_radius_final[column])
        });
        let d_g_dot = gradient(|column| {
            -(1.0 - cosine) / radius_final * d_a[column]
                + semi_major_axis / radius_final.powi(2) * (1.0 - cosine) * d_radius_final[column]
                - semi_major_axis / radius_final * sine * d_anomaly[column]
        });
        (d_f, d_g, d_f_dot, d_g_dot)
    } else {
        let sqrt_abs_a = (-semi_major_axis).sqrt();
        let sine = anomaly.sinh();
        let cosine = anomaly.cosh();
        let d_s0 = gradient(|column| {
            d_sigma0[column] / sqrt_abs_a + 0.5 * sigma0 / sqrt_abs_a.powi(3) * d_a[column]
        });
        let d_c0 = gradient(|column| {
            -d_radius0[column] / semi_major_axis + radius0 / semi_major_axis.powi(2) * d_a[column]
        });
        let d_mean = gradient(|column| 1.5 * sqrt_mu * time / sqrt_abs_a.powi(5) * d_a[column]);
        let denominator = s0 * sine + c0 * cosine - 1.0;
        let d_anomaly = gradient(|column| {
            (d_mean[column] - (cosine - 1.0) * d_s0[column] - sine * d_c0[column]) / denominator
        });
        let d_radius_final = gradient(|column| {
            (1.0 - cosine - 0.5 / sqrt_abs_a * sigma0 * sine) * d_a[column]
                + cosine * d_radius0[column]
                + (sigma0 * sqrt_abs_a * cosine + (radius0 - semi_major_axis) * sine)
                    * d_anomaly[column]
                + sqrt_abs_a * sine * d_sigma0[column]
        });
        let d_f = gradient(|column| {
            -(1.0 - cosine) / radius0 * d_a[column]
                + semi_major_axis / radius0.powi(2) * (1.0 - cosine) * d_radius0[column]
                + semi_major_axis / radius0 * sine * d_anomaly[column]
        });
        let d_g = gradient(|column| {
            ((1.0 - solution.f) * (radius0 * d_sigma0[column] + sigma0 * d_radius0[column])
                - sigma0 * radius0 * d_f[column]
                + sqrt_abs_a * radius0 * cosine * d_anomaly[column]
                + sqrt_abs_a * sine * d_radius0[column]
                - 0.5 * radius0 * sine / sqrt_abs_a * d_a[column])
                / sqrt_mu
        });
        let d_f_dot = gradient(|column| {
            sqrt_mu
                * (-sqrt_abs_a / radius0 / radius_final * cosine * d_anomaly[column]
                    + 0.5 / sqrt_abs_a / radius0 / radius_final * sine * d_a[column]
                    + sqrt_abs_a / radius_final / radius0.powi(2) * sine * d_radius0[column]
                    + sqrt_abs_a / radius_final.powi(2) / radius0 * sine * d_radius_final[column])
        });
        let d_g_dot = gradient(|column| {
            -(1.0 - cosine) / radius_final * d_a[column]
                + semi_major_axis / radius_final.powi(2) * (1.0 - cosine) * d_radius_final[column]
                + semi_major_axis / radius_final * sine * d_anomaly[column]
        });
        (d_f, d_g, d_f_dot, d_g_dot)
    };

    let mut matrix = [[0.0; 6]; 6];
    for row in 0..3 {
        for column in 0..6 {
            matrix[row][column] = solution.f * f64::from(row == column)
                + position[row] * d_f[column]
                + solution.g * f64::from(column == row + 3)
                + velocity[row] * d_g[column];
            matrix[row + 3][column] = solution.f_dot * f64::from(row == column)
                + position[row] * d_f_dot[column]
                + solution.g_dot * f64::from(column == row + 3)
                + velocity[row] * d_g_dot[column];
            if !matrix[row][column].is_finite() || !matrix[row + 3][column].is_finite() {
                return Err(PykepError::NumericalOverflow {
                    operation: "state_transition_matrix_lagrangian",
                });
            }
        }
    }
    Ok(matrix)
}

/// Propagates a state and returns the Lagrangian analytic state-transition
/// matrix.
///
/// The matrix is row-major `∂state_final/∂state_initial`.
///
/// # Errors
///
/// Returns propagation, validation, exact-parabolic, or overflow errors.
pub fn propagate_lagrangian_with_stm(
    initial_state: &CartesianState,
    time: f64,
    mu: f64,
) -> Result<(CartesianState, Matrix6)> {
    let solution = solve_lagrangian(initial_state, time, mu)?;
    let matrix = lagrangian_stm(initial_state, time, mu, solution)?;
    Ok((solution.state, matrix))
}

/// Returns the Lagrangian analytic state-transition matrix.
///
/// # Errors
///
/// Returns propagation, validation, exact-parabolic, or overflow errors.
pub fn state_transition_matrix_lagrangian(
    initial_state: &CartesianState,
    time: f64,
    mu: f64,
) -> Result<Matrix6> {
    propagate_lagrangian_with_stm(initial_state, time, mu).map(|(_, matrix)| matrix)
}

fn skew(vector: &[f64; 3]) -> Matrix3 {
    [
        [0.0, -vector[2], vector[1]],
        [vector[2], 0.0, -vector[0]],
        [-vector[1], vector[0], 0.0],
    ]
}

fn add3(left: &Matrix3, right: &Matrix3) -> Matrix3 {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| left[row][column] + right[row][column])
    })
}

fn subtract3(left: &Matrix3, right: &Matrix3) -> Matrix3 {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| left[row][column] - right[row][column])
    })
}

fn scale3(matrix: &Matrix3, scale: f64) -> Matrix3 {
    matrix.map(|row| row.map(|value| value * scale))
}

fn multiply_3x3_3x2(left: &Matrix3, right: &[[f64; 2]; 3]) -> [[f64; 2]; 3] {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| {
            (0..3)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

fn compute_y(
    initial_position: &[f64; 3],
    initial_velocity: &[f64; 3],
    position: &[f64; 3],
    velocity: &[f64; 3],
    time: f64,
    mu: f64,
) -> Matrix6 {
    let angular_momentum = cross(position, velocity);
    let initial_radius = norm(initial_position);
    let radius = norm(position);
    let radius_cubed = radius.powi(3);
    let mut b = [[0.0; 2]; 3];
    for row in 0..3 {
        b[row][0] = initial_position[row] / (mu * initial_radius).sqrt();
        b[row][1] = initial_velocity[row] * initial_radius / mu;
    }
    let skew_position = skew(position);
    let skew_velocity = skew(velocity);
    let skew_angular = skew(&angular_momentum);
    let product = matrix_multiply(&skew_position, &skew_velocity).expect("finite inputs");
    let top = scale3(&add3(&product, &skew_angular), -1.0);
    let position_squared = matrix_multiply(&skew_position, &skew_position).expect("finite inputs");
    let velocity_squared = matrix_multiply(&skew_velocity, &skew_velocity).expect("finite inputs");
    let bottom = subtract3(
        &scale3(&position_squared, mu / radius_cubed),
        &velocity_squared,
    );
    let top = multiply_3x3_3x2(&top, &b);
    let bottom = multiply_3x3_3x2(&bottom, &b);

    let mut result = [[0.0; 6]; 6];
    for row in 0..3 {
        for column in 0..3 {
            result[row][column] = skew_position[row][column];
            result[row + 3][column] = skew_velocity[row][column];
        }
        for column in 0..2 {
            result[row][column + 3] = top[row][column];
            result[row + 3][column + 3] = bottom[row][column];
        }
        result[row][5] = -position[row] + 1.5 * velocity[row] * time;
        result[row + 3][5] = velocity[row] / 2.0 - 1.5 * mu / radius_cubed * position[row] * time;
    }
    result
}

pub(crate) fn invert6(mut matrix: Matrix6) -> Result<Matrix6> {
    let mut inverse = identity::<6>();
    for pivot in 0..6 {
        let best = (pivot..6)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .abs()
                    .partial_cmp(&matrix[right][pivot].abs())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap_or(pivot);
        if matrix[best][pivot].abs() <= f64::EPSILON {
            return Err(PykepError::SingularGeometry {
                operation: "state_transition_matrix_reynolds",
            });
        }
        matrix.swap(pivot, best);
        inverse.swap(pivot, best);
        let scale = matrix[pivot][pivot];
        for column in 0..6 {
            matrix[pivot][column] /= scale;
            inverse[pivot][column] /= scale;
        }
        for row in 0..6 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in 0..6 {
                matrix[row][column] -= factor * matrix[pivot][column];
                inverse[row][column] -= factor * inverse[pivot][column];
            }
        }
    }
    Ok(inverse)
}

/// Computes the Reynolds Cartesian state-transition matrix between two
/// states on the same two-body trajectory.
///
/// The matrix is row-major `∂state_final/∂state_initial`.
///
/// # Errors
///
/// Returns an error for invalid input or a singular Reynolds basis.
pub fn state_transition_matrix_reynolds(
    initial_state: &CartesianState,
    final_state: &CartesianState,
    time: f64,
    mu: f64,
) -> Result<Matrix6> {
    validate(initial_state, time, mu)?;
    validate(final_state, time, mu)?;
    let (initial_position, initial_velocity) = split(initial_state);
    let (final_position, final_velocity) = split(final_state);
    let final_basis = compute_y(
        &initial_position,
        &initial_velocity,
        &final_position,
        &final_velocity,
        time,
        mu,
    );
    let initial_basis = compute_y(
        &initial_position,
        &initial_velocity,
        &initial_position,
        &initial_velocity,
        0.0,
        mu,
    );
    matrix_multiply(&final_basis, &invert6(initial_basis)?)
}
