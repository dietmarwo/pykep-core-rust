// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Two-body Cartesian propagation and state-transition matrices.

mod lagrangian;
mod stm;

pub use lagrangian::{
    propagate_keplerian, propagate_lagrangian, propagate_lagrangian_grid, propagate_universal,
};
pub use stm::{
    propagate_lagrangian_with_stm, state_transition_matrix_lagrangian,
    state_transition_matrix_reynolds,
};

use crate::error::{ensure_finite, ensure_finite_output};
use crate::{CartesianState, PykepError, Result};

const MAX_ITERATIONS: usize = 100;

fn validate(state: &CartesianState, time: f64, mu: f64) -> Result<()> {
    for &value in state {
        ensure_finite("state", value)?;
    }
    ensure_finite("time", time)?;
    ensure_finite("mu", mu)?;
    if mu <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "mu",
            reason: "must be greater than zero".into(),
        });
    }
    if norm(&[state[0], state[1], state[2]]) == 0.0 {
        return Err(PykepError::SingularGeometry {
            operation: "two_body_propagation",
        });
    }
    Ok(())
}

fn validate_state_output(operation: &'static str, state: CartesianState) -> Result<CartesianState> {
    for &value in &state {
        ensure_finite_output(operation, value)?;
    }
    Ok(state)
}

fn dot(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: &[f64; 3], right: &[f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(vector: &[f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn split(state: &CartesianState) -> ([f64; 3], [f64; 3]) {
    (
        [state[0], state[1], state[2]],
        [state[3], state[4], state[5]],
    )
}

fn combine(position: [f64; 3], velocity: [f64; 3]) -> CartesianState {
    [
        position[0],
        position[1],
        position[2],
        velocity[0],
        velocity[1],
        velocity[2],
    ]
}

fn linear_combination(
    left_scale: f64,
    left: &[f64; 3],
    right_scale: f64,
    right: &[f64; 3],
) -> [f64; 3] {
    [
        left_scale * left[0] + right_scale * right[0],
        left_scale * left[1] + right_scale * right[1],
        left_scale * left[2] + right_scale * right[2],
    ]
}

#[derive(Clone, Copy)]
struct LagrangeSolution {
    state: CartesianState,
    initial_radius: f64,
    final_radius: f64,
    energy: f64,
    sigma0: f64,
    semi_major_axis: f64,
    s0: f64,
    c0: f64,
    anomaly_difference: f64,
    f: f64,
    g: f64,
    f_dot: f64,
    g_dot: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relative_error(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / expected.abs().max(1.0)
    }

    fn energy(state: &CartesianState, mu: f64) -> f64 {
        let (position, velocity) = split(state);
        0.5 * dot(&velocity, &velocity) - mu / norm(&position)
    }

    fn angular_momentum(state: &CartesianState) -> [f64; 3] {
        let (position, velocity) = split(state);
        cross(&position, &velocity)
    }

    #[test]
    fn circular_orbit_and_grid_have_expected_semantics() {
        let initial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let quarter = propagate_lagrangian(&initial, core::f64::consts::FRAC_PI_2, 1.0).unwrap();
        for (actual, expected) in quarter.iter().zip([0.0, 1.0, 0.0, -1.0, 0.0, 0.0]) {
            assert!(relative_error(*actual, expected) < 2e-15);
        }
        let grid = propagate_lagrangian_grid(&initial, &[10.0, 10.0, 10.5], 1.0).unwrap();
        assert_eq!(grid[0], initial);
        assert_eq!(grid[1], initial);
        assert_eq!(
            propagate_lagrangian_grid(&initial, &[], 1.0).unwrap(),
            Vec::<CartesianState>::new()
        );
    }

    #[test]
    fn propagation_conserves_invariants_and_reverses_time() {
        let cases = [
            ([1.2, 0.3, -0.4, 0.06, 0.43, -0.87], 3.56, 1.24),
            ([1.2, 0.3, -0.4, -3.0, 4.4, -0.8], 1.3, 1.24),
            ([4.0, -2.0, 1.0, 0.1, 0.3, -0.2], 20.0, 2.0),
        ];
        for (initial, time, mu) in cases {
            for operation in [
                propagate_lagrangian as fn(&CartesianState, f64, f64) -> Result<CartesianState>,
                propagate_universal,
            ] {
                let final_state = operation(&initial, time, mu).unwrap();
                assert!(relative_error(energy(&final_state, mu), energy(&initial, mu)) < 2e-12);
                for (actual, expected) in angular_momentum(&final_state)
                    .iter()
                    .zip(angular_momentum(&initial))
                {
                    assert!(relative_error(*actual, expected) < 2e-12);
                }
                let recovered = operation(&final_state, -time, mu).unwrap();
                for (actual, expected) in recovered.iter().zip(initial) {
                    assert!(relative_error(*actual, expected) < 2e-11);
                }
            }
        }
    }

    #[test]
    fn universal_variables_cover_the_exact_parabolic_limit() {
        let initial = [1.0, 0.0, 0.0, 0.0, 2.0_f64.sqrt(), 0.0];
        let final_state = propagate_universal(&initial, 0.75, 1.0).unwrap();
        assert!(relative_error(energy(&final_state, 1.0), 0.0) < 2e-14);
        assert!(
            propagate_lagrangian(&initial, 0.75, 1.0)
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn analytic_stm_matches_central_differences_and_composes() {
        let initial = [1.223, 0.3123, -0.432, 0.06345, 0.43234, -0.874634];
        let mu = 1.24;
        let time = 0.8;
        let analytic = state_transition_matrix_lagrangian(&initial, time, mu).unwrap();
        for column in 0..6 {
            let step = 2e-6 * initial[column].abs().max(1.0);
            let mut plus = initial;
            let mut minus = initial;
            plus[column] += step;
            minus[column] -= step;
            let propagated_plus = propagate_lagrangian(&plus, time, mu).unwrap();
            let propagated_minus = propagate_lagrangian(&minus, time, mu).unwrap();
            for row in 0..6 {
                let numerical = (propagated_plus[row] - propagated_minus[row]) / (2.0 * step);
                assert!(
                    relative_error(analytic[row][column], numerical) < 2e-8,
                    "[{row}][{column}]: {} != {numerical}",
                    analytic[row][column]
                );
            }
        }

        let first_time = 0.3;
        let second_time = 0.5;
        let (middle, first) = propagate_lagrangian_with_stm(&initial, first_time, mu).unwrap();
        let (_, second) = propagate_lagrangian_with_stm(&middle, second_time, mu).unwrap();
        let composed = crate::math::linalg::matrix_multiply(&second, &first).unwrap();
        for row in 0..6 {
            for column in 0..6 {
                assert!(relative_error(composed[row][column], analytic[row][column]) < 2e-11);
            }
        }
    }
}
