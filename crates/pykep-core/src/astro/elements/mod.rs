// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Conversions among Cartesian, classical, and modified equinoctial states.
//!
//! Cartesian states are ordered `[x, y, z, vx, vy, vz]`. Jacobians are
//! row-major matrices whose rows correspond to outputs and columns to inputs.

mod automatic_differentiation;
mod classical;
mod equinoctial;
mod values;

pub use classical::{cartesian_to_classical, classical_to_cartesian};
pub use equinoctial::{
    cartesian_to_modified_equinoctial, cartesian_to_modified_equinoctial_jacobian,
    classical_to_modified_equinoctial, modified_equinoctial_to_cartesian,
    modified_equinoctial_to_cartesian_jacobian, modified_equinoctial_to_classical,
};
pub use values::{ClassicalElements, ModifiedEquinoctialElements};

use crate::error::{ensure_finite, ensure_finite_output};
use crate::{CartesianState, PykepError, Result};

fn validate_mu(mu: f64) -> Result<()> {
    ensure_finite("mu", mu)?;
    if mu > 0.0 {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter: "mu",
            reason: "must be greater than zero".into(),
        })
    }
}

fn validate_six(parameter: &'static str, values: &[f64; 6]) -> Result<()> {
    for &value in values {
        ensure_finite(parameter, value)?;
    }
    Ok(())
}

fn validate_state(state: &CartesianState) -> Result<()> {
    validate_six("state", state)
}

fn validate_output(operation: &'static str, values: [f64; 6]) -> Result<[f64; 6]> {
    for &value in &values {
        ensure_finite_output(operation, value)?;
    }
    Ok(values)
}

fn split_state(state: &CartesianState) -> ([f64; 3], [f64; 3]) {
    (
        [state[0], state[1], state[2]],
        [state[3], state[4], state[5]],
    )
}

fn join_state(position: [f64; 3], velocity: [f64; 3]) -> CartesianState {
    [
        position[0],
        position[1],
        position[2],
        velocity[0],
        velocity[1],
        velocity[2],
    ]
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
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn singular(operation: &'static str) -> PykepError {
    PykepError::SingularGeometry { operation }
}

#[cfg(test)]
mod tests {
    use core::f64::consts::PI;

    use super::*;
    use crate::math::linalg::matrix_multiply;

    struct Pcg32 {
        state: u64,
        increment: u64,
    }

    impl Pcg32 {
        fn new(seed: u64, sequence: u64) -> Self {
            let mut result = Self {
                state: 0,
                increment: (sequence << 1) | 1,
            };
            result.next();
            result.state = result.state.wrapping_add(seed);
            result.next();
            result
        }

        fn next(&mut self) -> u32 {
            let old_state = self.state;
            self.state = old_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(self.increment);
            let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
            xorshifted.rotate_right((old_state >> 59) as u32)
        }

        fn uniform(&mut self, lower: f64, upper: f64) -> f64 {
            lower + (upper - lower) * f64::from(self.next()) * 2.0_f64.powi(-32)
        }
    }

    fn state_close(left: &CartesianState, right: &CartesianState, tolerance: f64) {
        for index in 0..6 {
            let scale = left[index].abs().max(right[index].abs()).max(1.0);
            assert!(
                (left[index] - right[index]).abs() <= tolerance * scale,
                "component {index}: {} != {}",
                left[index],
                right[index]
            );
        }
    }

    #[test]
    fn classical_reference_cases_and_singularities_are_explicit() {
        let elements =
            ClassicalElements::new(1.265_822_784_810_126_9, 0.21, PI / 2.0, 0.0, 0.0, 0.0);
        let state = classical_to_cartesian(elements, 1.0).unwrap();
        state_close(&state, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.1], 2e-15);
        let reconstructed = cartesian_to_classical(&state, 1.0).unwrap();
        assert!((reconstructed.semi_major_axis - elements.semi_major_axis).abs() < 1e-15);
        assert!((reconstructed.eccentricity - elements.eccentricity).abs() < 1e-15);

        assert!(matches!(
            cartesian_to_classical(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 1.0),
            Err(PykepError::SingularGeometry { .. })
        ));
        assert!(
            classical_to_cartesian(ClassicalElements::new(1.0, 1.0, 0.2, 0.3, 0.4, 0.5), 1.0)
                .is_err()
        );
        assert!(
            classical_to_cartesian(ClassicalElements::new(-2.0, 1.5, 0.2, 0.3, 0.4, PI), 1.0)
                .is_err()
        );
    }

    #[test]
    fn seeded_elliptic_and_hyperbolic_states_round_trip() {
        let mut generator = Pcg32::new(122_012_203, 7);
        for index in 0..2_000 {
            let hyperbolic = index % 2 == 1;
            let semi_major_axis = if hyperbolic {
                -generator.uniform(1.1, 100.0)
            } else {
                generator.uniform(1.1, 100.0)
            };
            let eccentricity = if hyperbolic {
                generator.uniform(1.05, 2.5)
            } else {
                generator.uniform(0.001, 0.99)
            };
            let inclination = generator.uniform(0.001, PI - 0.001);
            let node = generator.uniform(0.0, 2.0 * PI);
            let periapsis = generator.uniform(0.0, 2.0 * PI);
            let true_anomaly = if hyperbolic {
                let limit = (-1.0 / eccentricity).acos() - 0.01;
                generator.uniform(-limit, limit)
            } else {
                generator.uniform(-PI, PI)
            };
            let elements = ClassicalElements::new(
                semi_major_axis,
                eccentricity,
                inclination,
                node,
                periapsis,
                true_anomaly,
            );
            let state = classical_to_cartesian(elements, 1.0).unwrap();
            let reconstructed =
                classical_to_cartesian(cartesian_to_classical(&state, 1.0).unwrap(), 1.0).unwrap();
            state_close(&state, &reconstructed, 1e-11);

            for retrograde in [false, true] {
                let mee = cartesian_to_modified_equinoctial(&state, 1.0, retrograde).unwrap();
                let mee_state = modified_equinoctial_to_cartesian(mee, 1.0, retrograde).unwrap();
                state_close(&state, &mee_state, 1e-11);
                let via_classical =
                    classical_to_modified_equinoctial(elements, retrograde).unwrap();
                let via_classical_state =
                    modified_equinoctial_to_cartesian(via_classical, 1.0, retrograde).unwrap();
                state_close(&state, &via_classical_state, 1e-11);
            }
        }
    }

    #[test]
    fn equinoctial_conventions_cover_both_circular_equatorial_poles() {
        let prograde = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let prograde_mee = cartesian_to_modified_equinoctial(&prograde, 1.0, false).unwrap();
        assert_eq!(
            prograde_mee,
            ModifiedEquinoctialElements::new(1.0, 0.0, 0.0, -0.0, 0.0, 0.0)
        );
        state_close(
            &modified_equinoctial_to_cartesian(prograde_mee, 1.0, false).unwrap(),
            &prograde,
            1e-15,
        );

        let retrograde = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let retrograde_mee = cartesian_to_modified_equinoctial(&retrograde, 1.0, true).unwrap();
        state_close(
            &modified_equinoctial_to_cartesian(retrograde_mee, 1.0, true).unwrap(),
            &retrograde,
            1e-15,
        );
        assert!(cartesian_to_modified_equinoctial(&retrograde, 1.0, false).is_err());
        assert!(cartesian_to_modified_equinoctial(&prograde, 1.0, true).is_err());
    }

    #[test]
    fn analytic_jacobians_match_finite_differences_and_inverse_identity() {
        let classical = ClassicalElements::new(3.0, 0.3, 0.7, 1.1, 0.4, -0.8);
        let state = classical_to_cartesian(classical, 1.0).unwrap();
        let mee = cartesian_to_modified_equinoctial(&state, 1.0, false).unwrap();
        let forward = cartesian_to_modified_equinoctial_jacobian(&state, 1.0, false).unwrap();
        let inverse = modified_equinoctial_to_cartesian_jacobian(mee, 1.0, false).unwrap();

        for column in 0..6 {
            let step = 1e-6 * state[column].abs().max(1.0);
            let mut plus = state;
            let mut minus = state;
            plus[column] += step;
            minus[column] -= step;
            let plus = cartesian_to_modified_equinoctial(&plus, 1.0, false)
                .unwrap()
                .to_array();
            let minus = cartesian_to_modified_equinoctial(&minus, 1.0, false)
                .unwrap()
                .to_array();
            for row in 0..6 {
                let finite_difference = (plus[row] - minus[row]) / (2.0 * step);
                let scale = finite_difference.abs().max(1.0);
                assert!(
                    (forward[row][column] - finite_difference).abs() <= 2e-8 * scale,
                    "forward Jacobian [{row}][{column}]"
                );
            }
        }

        let identity = matrix_multiply(&forward, &inverse).unwrap();
        for (row, values) in identity.iter().enumerate() {
            for (column, &value) in values.iter().enumerate() {
                let expected = f64::from(row == column);
                assert!(
                    (value - expected).abs() < 2e-13,
                    "inverse identity [{row}][{column}] = {value}"
                );
            }
        }
    }
}
