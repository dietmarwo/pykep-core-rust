// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Decision-gate validation for the pure-Rust adaptive integration backend.

use core::f64::consts::{PI, TAU};

use pykep_core::astro::propagation::{propagate_lagrangian, propagate_lagrangian_with_stm};
use pykep_core::integration::{
    DifferentiableDynamicsModel, Dop853, DynamicsModel, Event, EventDirection, InitialValueProblem,
    IntegratorOptions, SensitivityProblem, Termination,
};
use pykep_core::{PykepError, Result};

#[derive(Clone, Copy)]
struct Kepler;

impl DynamicsModel<6, 1> for Kepler {
    const NAME: &'static str = "phase10_kepler";

    fn validate(&self, _time: f64, state: &[f64; 6], parameters: &[f64; 1]) -> Result<()> {
        if parameters[0] <= 0.0 {
            return Err(PykepError::InvalidInput {
                parameter: "mu",
                reason: "must be greater than zero".into(),
            });
        }
        if state[..3].iter().map(|value| value * value).sum::<f64>() == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "phase10_kepler",
            });
        }
        Ok(())
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64; 6],
        parameters: &[f64; 1],
        derivative: &mut [f64; 6],
    ) -> Result<()> {
        let radius_squared = state[..3].iter().map(|value| value * value).sum::<f64>();
        if radius_squared == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "phase10_kepler",
            });
        }
        let acceleration_scale = -parameters[0] / (radius_squared * radius_squared.sqrt());
        derivative[..3].copy_from_slice(&state[3..]);
        for axis in 0..3 {
            derivative[axis + 3] = acceleration_scale * state[axis];
        }
        Ok(())
    }
}

impl DifferentiableDynamicsModel<6, 1> for Kepler {
    fn jacobians(
        &self,
        _time: f64,
        state: &[f64; 6],
        parameters: &[f64; 1],
        state_jacobian: &mut [[f64; 6]; 6],
        parameter_jacobian: &mut [[f64; 1]; 6],
    ) -> Result<()> {
        let radius_squared = state[..3].iter().map(|value| value * value).sum::<f64>();
        if radius_squared == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "phase10_kepler",
            });
        }
        let radius = radius_squared.sqrt();
        let inverse_radius_cubed = 1.0 / (radius_squared * radius);
        for axis in 0..3 {
            state_jacobian[axis][axis + 3] = 1.0;
            for column in 0..3 {
                let identity = f64::from(axis == column);
                state_jacobian[axis + 3][column] = parameters[0]
                    * inverse_radius_cubed
                    * (3.0 * state[axis] * state[column] / radius_squared - identity);
            }
            parameter_jacobian[axis + 3][0] = -state[axis] * inverse_radius_cubed;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Cr3bp;

impl DynamicsModel<6, 1> for Cr3bp {
    const NAME: &'static str = "phase10_cr3bp";

    fn validate(&self, _time: f64, state: &[f64; 6], parameters: &[f64; 1]) -> Result<()> {
        let mu = parameters[0];
        if !(0.0..=0.5).contains(&mu) {
            return Err(PykepError::InvalidInput {
                parameter: "mu",
                reason: "must lie in [0, 0.5]".into(),
            });
        }
        let primary_distance_squared =
            (state[0] + mu).powi(2) + state[1].powi(2) + state[2].powi(2);
        let secondary_distance_squared =
            (state[0] - 1.0 + mu).powi(2) + state[1].powi(2) + state[2].powi(2);
        if primary_distance_squared == 0.0 || secondary_distance_squared == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "phase10_cr3bp",
            });
        }
        Ok(())
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64; 6],
        parameters: &[f64; 1],
        derivative: &mut [f64; 6],
    ) -> Result<()> {
        let mu = parameters[0];
        let dx1 = state[0] + mu;
        let dx2 = state[0] - 1.0 + mu;
        let r1_squared = dx1 * dx1 + state[1] * state[1] + state[2] * state[2];
        let r2_squared = dx2 * dx2 + state[1] * state[1] + state[2] * state[2];
        if r1_squared == 0.0 || r2_squared == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "phase10_cr3bp",
            });
        }
        let primary = (1.0 - mu) / (r1_squared * r1_squared.sqrt());
        let secondary = mu / (r2_squared * r2_squared.sqrt());
        derivative[..3].copy_from_slice(&state[3..]);
        derivative[3] = state[0] + 2.0 * state[4] - primary * dx1 - secondary * dx2;
        derivative[4] = state[1] - 2.0 * state[3] - (primary + secondary) * state[1];
        derivative[5] = -(primary + secondary) * state[2];
        Ok(())
    }
}

fn energy(state: &[f64; 6], mu: f64) -> f64 {
    let radius = state[..3]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    0.5 * state[3..].iter().map(|value| value * value).sum::<f64>() - mu / radius
}

fn cr3bp_jacobi(state: &[f64; 6], mu: f64) -> f64 {
    let r1 = ((state[0] + mu).powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
    let r2 = ((state[0] - 1.0 + mu).powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
    let potential = 0.5 * (state[0] * state[0] + state[1] * state[1]) + (1.0 - mu) / r1 + mu / r2;
    2.0 * potential - state[3..].iter().map(|value| value * value).sum::<f64>()
}

#[test]
fn kepler_matches_closed_form_and_has_bounded_long_term_drift() {
    let initial = [0.5, 0.0, 0.0, 0.0, 3.0_f64.sqrt(), 0.0];
    let options = IntegratorOptions {
        relative_tolerance: 1e-13,
        absolute_tolerance: 1e-13,
        ..IntegratorOptions::default()
    };
    let numerical = Dop853
        .propagate(
            &Kepler,
            InitialValueProblem::new(0.0, initial, TAU * 8.0_f64.sqrt() / 2.0, [1.0]),
            options,
        )
        .unwrap();
    let reference = propagate_lagrangian(&initial, TAU * 8.0_f64.sqrt() / 2.0, 1.0).unwrap();
    for (actual, expected) in numerical.state.iter().zip(reference) {
        assert!((actual - expected).abs() < 2e-11);
    }

    let long = Dop853
        .propagate(
            &Kepler,
            InitialValueProblem::new(0.0, initial, 100.0 * TAU * 8.0_f64.sqrt(), [1.0]),
            options,
        )
        .unwrap();
    assert!((energy(&long.state, 1.0) - energy(&initial, 1.0)).abs() < 2e-10);
}

#[test]
fn cr3bp_close_approach_backward_solve_and_invariant_are_stable() {
    let mu = 0.012_150_585_609_624;
    let initial = [1.0 - mu + 0.02, 0.0, 0.003, 0.0, 0.7, 0.0];
    let options = IntegratorOptions {
        relative_tolerance: 1e-12,
        absolute_tolerance: 1e-12,
        maximum_step: Some(0.002),
        ..IntegratorOptions::default()
    };
    let forward = Dop853
        .propagate(
            &Cr3bp,
            InitialValueProblem::new(0.0, initial, 0.2, [mu]),
            options,
        )
        .unwrap();
    assert!((cr3bp_jacobi(&forward.state, mu) - cr3bp_jacobi(&initial, mu)).abs() < 2e-9);
    let backward = Dop853
        .propagate(
            &Cr3bp,
            InitialValueProblem::new(0.2, forward.state, 0.0, [mu]),
            options,
        )
        .unwrap();
    for (actual, expected) in backward.state.iter().zip(initial) {
        assert!((actual - expected).abs() < 2e-9);
    }
}

#[test]
fn rejected_steps_limits_and_reproducibility_are_observable() {
    let initial = [0.05, 0.0, 0.0, 0.0, 6.0, 0.0];
    let options = IntegratorOptions {
        relative_tolerance: 1e-14,
        absolute_tolerance: 1e-14,
        initial_step: Some(1.0),
        ..IntegratorOptions::default()
    };
    let first = Dop853
        .propagate(
            &Kepler,
            InitialValueProblem::new(0.0, initial, 1.0, [1.0]),
            options,
        )
        .unwrap();
    let second = Dop853
        .propagate(
            &Kepler,
            InitialValueProblem::new(0.0, initial, 1.0, [1.0]),
            options,
        )
        .unwrap();
    assert!(first.stats.rejected_steps > 0);
    assert_eq!(first, second);

    let limited = IntegratorOptions {
        maximum_steps: 1,
        ..options
    };
    assert!(matches!(
        Dop853.propagate(
            &Kepler,
            InitialValueProblem::new(0.0, initial, 1.0, [1.0]),
            limited
        ),
        Err(PykepError::IntegrationFailure { .. })
    ));
}

#[test]
fn state_and_parameter_sensitivities_match_independent_references() {
    let initial = [1.2, -0.3, 0.1, 0.2, 0.8, -0.1];
    let mut initial_seeds = [[0.0; 7]; 6];
    for (index, row) in initial_seeds.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    let mut parameter_seeds = [[0.0; 7]; 1];
    parameter_seeds[0][6] = 1.0;
    let options = IntegratorOptions {
        relative_tolerance: 2e-13,
        absolute_tolerance: 2e-13,
        ..IntegratorOptions::default()
    };
    let propagated = Dop853
        .propagate_with_sensitivities(
            &Kepler,
            SensitivityProblem {
                nominal: InitialValueProblem::new(0.0, initial, 0.75, [1.0]),
                initial_sensitivities: initial_seeds,
                parameter_seeds,
            },
            options,
        )
        .unwrap();

    let (_, reference_stm) = propagate_lagrangian_with_stm(&initial, 0.75, 1.0).unwrap();
    for (row, expected_row) in propagated.sensitivities.iter().zip(reference_stm) {
        for (actual, expected) in row[..6].iter().zip(expected_row) {
            assert!((actual - expected).abs() < 3e-9);
        }
    }

    let step = 2e-6;
    let plus = Dop853
        .propagate(
            &Kepler,
            InitialValueProblem::new(0.0, initial, 0.75, [1.0 + step]),
            options,
        )
        .unwrap();
    let minus = Dop853
        .propagate(
            &Kepler,
            InitialValueProblem::new(0.0, initial, 0.75, [1.0 - step]),
            options,
        )
        .unwrap();
    for component in 0..6 {
        let finite_difference = (plus.state[component] - minus.state[component]) / (2.0 * step);
        assert!((propagated.sensitivities[component][6] - finite_difference).abs() < 2e-8);
    }
}

#[derive(Clone, Copy)]
struct Oscillator;

impl DynamicsModel<2, 0> for Oscillator {
    const NAME: &'static str = "phase10_oscillator";

    fn rhs(
        &self,
        _time: f64,
        state: &[f64; 2],
        _parameters: &[f64; 0],
        derivative: &mut [f64; 2],
    ) -> Result<()> {
        derivative[0] = state[1];
        derivative[1] = -state[0];
        Ok(())
    }
}

struct DescendingZero;

impl Event<2> for DescendingZero {
    fn direction(&self) -> EventDirection {
        EventDirection::Decreasing
    }

    fn value(&self, _time: f64, state: &[f64; 2]) -> f64 {
        state[0]
    }
}

struct AnyZero;

impl Event<2> for AnyZero {
    fn value(&self, _time: f64, state: &[f64; 2]) -> f64 {
        state[0]
    }
}

struct IncreasingZero;

impl Event<2> for IncreasingZero {
    fn direction(&self) -> EventDirection {
        EventDirection::Increasing
    }

    fn value(&self, _time: f64, state: &[f64; 2]) -> f64 {
        state[0]
    }
}

struct NonFiniteEvent;

impl Event<2> for NonFiniteEvent {
    fn value(&self, _time: f64, _state: &[f64; 2]) -> f64 {
        f64::NAN
    }
}

#[test]
fn dense_output_and_terminal_events_are_located() {
    let times = [0.0, PI / 4.0, PI / 2.0];
    let options = IntegratorOptions {
        relative_tolerance: 1e-13,
        absolute_tolerance: 1e-13,
        maximum_step: Some(0.05),
        ..IntegratorOptions::default()
    };
    let dense = Dop853
        .propagate_dense(
            &Oscillator,
            InitialValueProblem::new(0.0, [1.0, 0.0], PI / 2.0, []),
            &times,
            options,
        )
        .unwrap();
    assert_eq!(dense.times, times);
    for ((&time, state), expected_time) in dense.times.iter().zip(&dense.states).zip(times) {
        assert_eq!(time, expected_time);
        assert!(
            (state[0] - time.cos()).abs() < 2e-11,
            "dense x at {time}: {} versus {}",
            state[0],
            time.cos()
        );
        assert!(
            (state[1] + time.sin()).abs() < 2e-11,
            "dense v at {time}: {} versus {}",
            state[1],
            -time.sin()
        );
    }

    let event = Dop853
        .propagate_until_event(
            &Oscillator,
            &DescendingZero,
            InitialValueProblem::new(0.0, [1.0, 0.0], PI, []),
            options,
        )
        .unwrap();
    assert_eq!(event.termination, Termination::Event);
    assert!(
        (event.time - PI / 2.0).abs() < 5e-11,
        "event time: {} versus {}",
        event.time,
        PI / 2.0
    );
    assert!(event.state[0].abs() < 5e-11);
}

#[test]
fn malformed_inputs_and_model_failures_are_typed_errors() {
    let invalid_options = IntegratorOptions {
        relative_tolerance: 0.0,
        ..IntegratorOptions::default()
    };
    assert!(matches!(
        Dop853.propagate(
            &Oscillator,
            InitialValueProblem::new(0.0, [1.0, 0.0], 1.0, []),
            invalid_options
        ),
        Err(PykepError::InvalidInput {
            parameter: "relative_tolerance",
            ..
        })
    ));
    assert!(matches!(
        Dop853.propagate_dense(
            &Oscillator,
            InitialValueProblem::new(0.0, [1.0, 0.0], 1.0, []),
            &[0.0, 0.7, 0.6],
            IntegratorOptions::default()
        ),
        Err(PykepError::InvalidInput {
            parameter: "evaluation_times",
            ..
        })
    ));
    assert!(matches!(
        Dop853.propagate(
            &Kepler,
            InitialValueProblem::new(0.0, [0.0; 6], 1.0, [1.0]),
            IntegratorOptions::default()
        ),
        Err(PykepError::SingularGeometry { .. })
    ));
}

#[test]
fn zero_duration_event_directions_and_dense_grid_edges_are_explicit() {
    let options = IntegratorOptions {
        maximum_step: Some(0.05),
        ..IntegratorOptions::default()
    };
    let zero = InitialValueProblem::new(2.0, [1.0, 0.0], 2.0, []);
    let nominal = Dop853.propagate(&Oscillator, zero, options).unwrap();
    assert_eq!(nominal.state, [1.0, 0.0]);
    assert_eq!(nominal.stats.accepted_steps, 0);
    let dense = Dop853
        .propagate_dense(&Oscillator, zero, &[2.0], options)
        .unwrap();
    assert_eq!(dense.states, [[1.0, 0.0]]);
    let event = Dop853
        .propagate_until_event(&Oscillator, &AnyZero, zero, options)
        .unwrap();
    assert_eq!(event.termination, Termination::FinalTime);

    let increasing = Dop853
        .propagate_until_event(
            &Oscillator,
            &IncreasingZero,
            InitialValueProblem::new(0.0, [-1.0, 0.0], PI, []),
            options,
        )
        .unwrap();
    assert_eq!(increasing.termination, Termination::Event);
    assert!((increasing.time - PI / 2.0).abs() < 2e-8);

    let no_crossing = Dop853
        .propagate_until_event(
            &Oscillator,
            &IncreasingZero,
            InitialValueProblem::new(0.0, [1.0, 0.0], PI / 4.0, []),
            options,
        )
        .unwrap();
    assert_eq!(no_crossing.termination, Termination::FinalTime);

    for times in [&[][..], &[2.0, 2.1][..], &[0.0, 1.1][..]] {
        assert!(matches!(
            Dop853.propagate_dense(
                &Oscillator,
                if times.first() == Some(&0.0) {
                    InitialValueProblem::new(0.0, [1.0, 0.0], 1.0, [])
                } else {
                    zero
                },
                times,
                options,
            ),
            Err(PykepError::InvalidInput {
                parameter: "evaluation_times",
                ..
            })
        ));
    }
    assert!(matches!(
        Dop853.propagate_until_event(
            &Oscillator,
            &NonFiniteEvent,
            InitialValueProblem::new(0.0, [1.0, 0.0], 1.0, []),
            options
        ),
        Err(PykepError::IntegrationFailure { .. })
    ));
}

struct EmptyModel;

impl DynamicsModel<0, 0> for EmptyModel {
    const NAME: &'static str = "empty";

    fn rhs(
        &self,
        _time: f64,
        _state: &[f64; 0],
        _parameters: &[f64; 0],
        _derivative: &mut [f64; 0],
    ) -> Result<()> {
        Ok(())
    }
}

struct BadDerivative;

impl DynamicsModel<1, 0> for BadDerivative {
    const NAME: &'static str = "bad_derivative";

    fn rhs(
        &self,
        _time: f64,
        _state: &[f64; 1],
        _parameters: &[f64; 0],
        derivative: &mut [f64; 1],
    ) -> Result<()> {
        derivative[0] = f64::NAN;
        Ok(())
    }
}

#[test]
fn all_option_nonfinite_dimension_and_sensitivity_edges_are_rejected() {
    let base = InitialValueProblem::new(0.0, [1.0, 0.0], 1.0, []);
    let invalid_options = [
        IntegratorOptions {
            absolute_tolerance: 0.0,
            ..IntegratorOptions::default()
        },
        IntegratorOptions {
            initial_step: Some(-1.0),
            ..IntegratorOptions::default()
        },
        IntegratorOptions {
            maximum_step: Some(f64::NAN),
            ..IntegratorOptions::default()
        },
        IntegratorOptions {
            maximum_steps: 0,
            ..IntegratorOptions::default()
        },
        IntegratorOptions {
            maximum_rejections: 0,
            ..IntegratorOptions::default()
        },
    ];
    for options in invalid_options {
        assert!(Dop853.propagate(&Oscillator, base, options).is_err());
    }
    assert!(matches!(
        Dop853.propagate(
            &Oscillator,
            InitialValueProblem::new(f64::NAN, [1.0, 0.0], 1.0, []),
            IntegratorOptions::default()
        ),
        Err(PykepError::NonFiniteInput {
            parameter: "initial_time"
        })
    ));
    assert!(matches!(
        Dop853.propagate(
            &Oscillator,
            InitialValueProblem::new(0.0, [f64::INFINITY, 0.0], 1.0, []),
            IntegratorOptions::default()
        ),
        Err(PykepError::NonFiniteInput {
            parameter: "initial_state"
        })
    ));
    assert!(matches!(
        Dop853.propagate(
            &EmptyModel,
            InitialValueProblem::new(0.0, [], 1.0, []),
            IntegratorOptions::default()
        ),
        Err(PykepError::InvalidInput {
            parameter: "state_dimension",
            ..
        })
    ));
    assert!(matches!(
        Dop853.propagate(
            &BadDerivative,
            InitialValueProblem::new(0.0, [1.0], 1.0, []),
            IntegratorOptions::default()
        ),
        Err(PykepError::IntegrationFailure { .. })
    ));

    let zero_width = SensitivityProblem {
        nominal: InitialValueProblem::new(0.0, [1.0; 6], 1.0, [1.0]),
        initial_sensitivities: [[0.0; 0]; 6],
        parameter_seeds: [[0.0; 0]; 1],
    };
    assert!(matches!(
        Dop853.propagate_with_sensitivities(&Kepler, zero_width, IntegratorOptions::default()),
        Err(PykepError::InvalidInput {
            parameter: "sensitivity_width",
            ..
        })
    ));
    let zero_time = SensitivityProblem {
        nominal: InitialValueProblem::new(0.0, [1.2, 0.0, 0.0, 0.0, 0.8, 0.0], 0.0, [1.0]),
        initial_sensitivities: [[1.0]; 6],
        parameter_seeds: [[0.0]],
    };
    let unchanged = Dop853
        .propagate_with_sensitivities(&Kepler, zero_time, IntegratorOptions::default())
        .unwrap();
    assert_eq!(unchanged.sensitivities, [[1.0]; 6]);

    let bad_seed = SensitivityProblem {
        nominal: InitialValueProblem::new(0.0, [1.2, 0.0, 0.0, 0.0, 0.8, 0.0], 1.0, [1.0]),
        initial_sensitivities: [[f64::NAN]; 6],
        parameter_seeds: [[0.0]],
    };
    assert!(matches!(
        Dop853.propagate_with_sensitivities(&Kepler, bad_seed, IntegratorOptions::default()),
        Err(PykepError::NonFiniteInput {
            parameter: "initial_sensitivities"
        })
    ));
}
