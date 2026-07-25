// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/core_astro/mima.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Maximum-initial-mass approximations for low-thrust transfers.

use crate::astro::propagation::propagate_lagrangian_with_stm;
use crate::astro::propagation::stm::invert6;
use crate::error::{ensure_finite, ensure_finite_output};
use crate::math::linalg::{identity, matrix_multiply, matrix_vector_multiply};
use crate::{CartesianState, Matrix6, PykepError, Result, Vector3};

/// Maximum-mass approximation and its characteristic acceleration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MimaResult {
    /// Maximum initial spacecraft mass.
    pub mass: f64,
    /// Characteristic acceleration from the approximation.
    pub acceleration: f64,
}

fn validate(time: f64, maximum_thrust: f64, effective_exhaust_velocity: f64) -> Result<()> {
    for (name, value) in [
        ("time", time),
        ("maximum_thrust", maximum_thrust),
        ("effective_exhaust_velocity", effective_exhaust_velocity),
    ] {
        ensure_finite(name, value)?;
        if value <= 0.0 {
            return Err(PykepError::InvalidInput {
                parameter: name,
                reason: "must be greater than zero".into(),
            });
        }
    }
    Ok(())
}

/// Computes the Hennes-Izzo-Landau maximum-initial-mass approximation.
///
/// # Errors
///
/// Returns an error for non-finite inputs, non-positive physical parameters,
/// or binary64 overflow.
pub fn mima(
    departure_delta_v: &Vector3,
    arrival_delta_v: &Vector3,
    time: f64,
    maximum_thrust: f64,
    effective_exhaust_velocity: f64,
) -> Result<MimaResult> {
    validate(time, maximum_thrust, effective_exhaust_velocity)?;
    for &value in departure_delta_v.iter().chain(arrival_delta_v.iter()) {
        ensure_finite("delta_v", value)?;
    }
    let sum =
        core::array::from_fn::<_, 3, _>(|index| departure_delta_v[index] + arrival_delta_v[index]);
    let difference =
        core::array::from_fn::<_, 3, _>(|index| -departure_delta_v[index] + arrival_delta_v[index]);
    let ab: f64 = sum.iter().zip(difference).map(|(a, b)| a * b).sum();
    let aa: f64 = sum.iter().map(|value| value * value).sum();
    let bb: f64 = difference.iter().map(|value| value * value).sum();
    let acceleration = (aa + 2.0 * bb + 2.0 * (ab * ab + bb * bb).sqrt()).sqrt() / time;
    if acceleration == 0.0 {
        return Err(PykepError::SingularGeometry { operation: "mima" });
    }
    let mass = 2.0 * maximum_thrust
        / acceleration
        / (1.0 + (-acceleration * time / effective_exhaust_velocity).exp());
    Ok(MimaResult {
        mass: ensure_finite_output("mima", mass)?,
        acceleration: ensure_finite_output("mima", acceleration)?,
    })
}

fn add_scaled(left: &Matrix6, right: &Matrix6, right_scale: f64) -> Matrix6 {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| left[row][column] + right_scale * right[row][column])
    })
}

fn scale(matrix: &Matrix6, factor: f64) -> Matrix6 {
    matrix.map(|row| row.map(|value| value * factor))
}

fn transfer_error(
    x: f64,
    initial_state: &CartesianState,
    time: f64,
    departure_delta_v: &Vector3,
    arrival_delta_v: &Vector3,
    mu: f64,
) -> Result<(f64, f64)> {
    let tau = (x / (x * x + 1.0).sqrt() + 1.0) / 2.0;
    let first_time = time * tau;
    let second_time = time * (1.0 - tau);
    let (_, matrix12) = propagate_lagrangian_with_stm(initial_state, first_time / 2.0, mu)?;
    let (_, matrix22) = propagate_lagrangian_with_stm(initial_state, time - second_time / 2.0, mu)?;
    let (_, matrix11) = propagate_lagrangian_with_stm(initial_state, first_time, mu)?;
    let (_, matrix21) = propagate_lagrangian_with_stm(initial_state, time - second_time, mu)?;
    let (_, matrix3) = propagate_lagrangian_with_stm(initial_state, time, mu)?;

    let first_sum = add_scaled(&invert6(matrix11)?, &invert6(matrix12)?, 4.0);
    let first = scale(
        &add_scaled(&matrix_multiply(&matrix3, &first_sum)?, &matrix3, 1.0),
        1.0 / 6.0,
    );
    let second_sum = add_scaled(&invert6(matrix21)?, &invert6(matrix22)?, 4.0);
    let second = scale(
        &add_scaled(&matrix_multiply(&matrix3, &second_sum)?, &identity(), 1.0),
        1.0 / 6.0,
    );

    let mut right_hand_side = [0.0; 6];
    for row in 0..3 {
        right_hand_side[row] = (0..3)
            .map(|column| matrix3[row][column + 3] * departure_delta_v[column])
            .sum();
        right_hand_side[row + 3] = arrival_delta_v[row]
            + (0..3)
                .map(|column| matrix3[row + 3][column + 3] * departure_delta_v[column])
                .sum::<f64>();
    }
    let mut matrix = [[0.0; 6]; 6];
    for row in 0..6 {
        for column in 0..3 {
            matrix[row][column] = first[row][column + 3];
            matrix[row][column + 3] = second[row][column + 3];
        }
    }
    let impulses = matrix_vector_multiply(&invert6(matrix)?, &right_hand_side)?;
    let first_acceleration = [
        impulses[0] / tau / time,
        impulses[1] / tau / time,
        impulses[2] / tau / time,
    ];
    let second_acceleration = [
        impulses[3] / (1.0 - tau) / time,
        impulses[4] / (1.0 - tau) / time,
        impulses[5] / (1.0 - tau) / time,
    ];
    let first_squared: f64 = first_acceleration.iter().map(|value| value * value).sum();
    let second_squared: f64 = second_acceleration.iter().map(|value| value * value).sum();
    Ok((
        tau.powi(2) * (1.0 - tau).powi(2) * (second_squared - first_squared),
        first_squared.sqrt(),
    ))
}

/// Computes the STM-based second maximum-initial-mass approximation.
///
/// # Errors
///
/// Returns an error for invalid inputs, propagation/STM failures, a singular
/// transfer system, or failure to bracket/converge on the split time.
pub fn mima2(
    initial_state: &CartesianState,
    departure_delta_v: &Vector3,
    arrival_delta_v: &Vector3,
    time: f64,
    maximum_thrust: f64,
    effective_exhaust_velocity: f64,
    mu: f64,
) -> Result<MimaResult> {
    validate(time, maximum_thrust, effective_exhaust_velocity)?;
    ensure_finite("mu", mu)?;
    if mu <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "mu",
            reason: "must be greater than zero".into(),
        });
    }
    let center = transfer_error(
        0.0,
        initial_state,
        time,
        departure_delta_v,
        arrival_delta_v,
        mu,
    )?
    .0;
    let guess = if center > 0.0 { -0.5 } else { 0.5 };
    let guess_value = transfer_error(
        guess,
        initial_state,
        time,
        departure_delta_v,
        arrival_delta_v,
        mu,
    )?
    .0;
    let (mut left, mut left_value, mut right, mut right_value) = if guess < 0.0 {
        (guess, guess_value, 0.0, center)
    } else {
        (0.0, center, guess, guess_value)
    };
    if left_value.signum() == right_value.signum() && guess < 0.0 {
        for _ in 0..31 {
            right = left;
            right_value = left_value;
            left *= 2.0;
            left_value = transfer_error(
                left,
                initial_state,
                time,
                departure_delta_v,
                arrival_delta_v,
                mu,
            )?
            .0;
            if left_value.signum() != right_value.signum() {
                break;
            }
        }
    } else if left_value.signum() == right_value.signum() {
        for _ in 0..31 {
            left = right;
            left_value = right_value;
            right *= 2.0;
            right_value = transfer_error(
                right,
                initial_state,
                time,
                departure_delta_v,
                arrival_delta_v,
                mu,
            )?
            .0;
            if left_value.signum() != right_value.signum() {
                break;
            }
        }
    }
    if left_value.signum() == right_value.signum() {
        return Err(PykepError::ConvergenceFailure {
            operation: "mima2_bracket",
            iterations: 32,
        });
    }
    for _ in 0..100 {
        let root = 0.5 * (left + right);
        let value = transfer_error(
            root,
            initial_state,
            time,
            departure_delta_v,
            arrival_delta_v,
            mu,
        )?
        .0;
        if value == 0.0 || (right - left).abs() <= 8.0 * f64::EPSILON * root.abs().max(1.0) {
            let acceleration = transfer_error(
                root,
                initial_state,
                time,
                departure_delta_v,
                arrival_delta_v,
                mu,
            )?
            .1;
            let mass = 2.0 * maximum_thrust
                / acceleration
                / (1.0 + (-acceleration * time / effective_exhaust_velocity).exp());
            return Ok(MimaResult { mass, acceleration });
        }
        if value.signum() == left_value.signum() {
            left = root;
            left_value = value;
        } else {
            right = root;
        }
    }
    Err(PykepError::ConvergenceFailure {
        operation: "mima2",
        iterations: 100,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::lambert::LambertProblem;
    use crate::constants::{DAY_TO_SECONDS, MU_SUN};

    #[test]
    fn zero_impulse_is_explicitly_singular() {
        assert!(mima(&[0.0; 3], &[0.0; 3], 1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn simple_result_is_positive() {
        let result = mima(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 10.0, 0.6, 4000.0).unwrap();
        assert!(result.mass > 0.0);
        assert!(result.acceleration > 0.0);
    }

    #[test]
    fn stm_variant_matches_the_published_upstream_case() {
        let position = [
            3.574_644_002_632_926e10,
            -5.688_222_150_272_903e10,
            -1.304_897_435_568_400_6e10,
        ];
        let velocity = [
            4.666_425_901_145_393_4e4,
            2.375_697_019_573_154_5e4,
            1.165_422_004_315_22e4,
        ];
        let final_position = [
            7.672_399_994_418_636e10,
            -1.093_562_401_274_179_4e11,
            4.796_635_567_684_053e9,
        ];
        let final_velocity = [
            2.725_105_661_271_001e4,
            1.599_599_495_457_483_5e4,
            6.818_440_757_625_088e3,
        ];
        let time = 3.311_380_772_794_45e2 * DAY_TO_SECONDS;
        let lambert =
            LambertProblem::new(position, final_position, time, MU_SUN, false, 0).unwrap();
        let transfer = &lambert.solutions()[0];
        let departure_delta_v =
            core::array::from_fn(|index| transfer.departure_velocity[index] - velocity[index]);
        let arrival_delta_v =
            core::array::from_fn(|index| final_velocity[index] - transfer.arrival_velocity[index]);
        let initial_state = [
            position[0],
            position[1],
            position[2],
            transfer.departure_velocity[0],
            transfer.departure_velocity[1],
            transfer.departure_velocity[2],
        ];
        let result = mima2(
            &initial_state,
            &departure_delta_v,
            &arrival_delta_v,
            time,
            0.6,
            4000.0 * crate::constants::STANDARD_GRAVITY,
            MU_SUN,
        )
        .unwrap();
        assert!(
            (result.mass - 139.785_164_191_226_5).abs() < 2e-5,
            "{result:?}"
        );
    }
}
