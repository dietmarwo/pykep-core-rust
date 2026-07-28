// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Internal implementation of adaptive Taylor integration.

#[cfg(test)]
mod series;
mod systems;
mod tape;

use crate::integration::{
    DenseTrajectory, DynamicsModel, InitialValueProblem, IntegrationStats, IntegratorOptions,
    Propagation, SensitivityProblem, SensitivityPropagation, Termination, ensure_finite_matrix,
    ensure_finite_state, validate_evaluation_times, validate_problem,
};
use crate::{PykepError, Result};

const MAX_ORDER: usize = 24;
const SAFETY_FACTOR: f64 = 0.75;
const MIN_ORDER: usize = 8;

pub(crate) trait TaylorCoefficientModel<const N: usize, const P: usize>:
    DynamicsModel<N, P>
{
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; N],
        parameters: &[f64; P],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; N],
    ) -> Result<()>;
}

pub(crate) fn propagate<M, const N: usize, const P: usize>(
    model: &M,
    problem: InitialValueProblem<N, P>,
    options: IntegratorOptions,
) -> Result<Propagation<N>>
where
    M: TaylorCoefficientModel<N, P>,
{
    validate_problem(model, &problem, options)?;
    let direction = propagation_direction(problem.initial_time, problem.final_time);
    if direction == 0.0 {
        return Ok(Propagation {
            time: problem.initial_time,
            state: problem.initial_state,
            stats: IntegrationStats::default(),
            termination: Termination::FinalTime,
        });
    }

    let mut workspace = Workspace::new(problem.initial_time, problem.initial_state);
    let mut stats = IntegrationStats::default();
    integrate_to(
        model,
        &problem.parameters,
        problem.final_time,
        options,
        direction,
        &mut workspace,
        &mut stats,
    )?;
    Ok(Propagation {
        time: workspace.time,
        state: workspace.state,
        stats,
        termination: Termination::FinalTime,
    })
}

pub(crate) fn propagate_dense<M, const N: usize, const P: usize>(
    model: &M,
    problem: InitialValueProblem<N, P>,
    evaluation_times: &[f64],
    options: IntegratorOptions,
) -> Result<DenseTrajectory<N>>
where
    M: TaylorCoefficientModel<N, P>,
{
    validate_problem(model, &problem, options)?;
    validate_evaluation_times(problem.initial_time, problem.final_time, evaluation_times)?;
    let direction = propagation_direction(problem.initial_time, problem.final_time);
    if direction == 0.0 {
        return Ok(DenseTrajectory {
            times: vec![problem.initial_time],
            states: vec![problem.initial_state],
            stats: IntegrationStats::default(),
        });
    }

    let mut workspace = Workspace::new(problem.initial_time, problem.initial_state);
    let mut stats = IntegrationStats::default();
    let mut states = Vec::with_capacity(evaluation_times.len());
    for &evaluation_time in evaluation_times {
        integrate_to(
            model,
            &problem.parameters,
            evaluation_time,
            options,
            direction,
            &mut workspace,
            &mut stats,
        )?;
        states.push(workspace.state);
    }
    integrate_to(
        model,
        &problem.parameters,
        problem.final_time,
        options,
        direction,
        &mut workspace,
        &mut stats,
    )?;
    Ok(DenseTrajectory {
        times: evaluation_times.to_vec(),
        states,
        stats,
    })
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn propagate_with_sensitivities<M, const N: usize, const P: usize, const W: usize>(
    model: &M,
    problem: SensitivityProblem<N, P, W>,
    options: IntegratorOptions,
) -> Result<SensitivityPropagation<N, W>>
where
    M: TaylorCoefficientModel<N, P>,
{
    validate_problem(model, &problem.nominal, options)?;
    if W == 0 {
        return Err(PykepError::InvalidInput {
            parameter: "sensitivity_width",
            reason: "must be greater than zero".into(),
        });
    }
    ensure_finite_matrix("initial_sensitivities", &problem.initial_sensitivities)?;
    ensure_finite_matrix("parameter_seeds", &problem.parameter_seeds)?;
    if problem.nominal.initial_time == problem.nominal.final_time {
        return Ok(SensitivityPropagation {
            time: problem.nominal.initial_time,
            state: problem.nominal.initial_state,
            sensitivities: problem.initial_sensitivities,
            stats: IntegrationStats::default(),
        });
    }

    let nominal = propagate(model, problem.nominal, options)?;
    let mut sensitivities = [[0.0; W]; N];
    let mut statistics = nominal.stats;
    for column in 0..W {
        let step = sensitivity_step(&problem, column);
        let mut plus = problem.nominal;
        let mut minus = problem.nominal;
        for row in 0..N {
            plus.initial_state[row] += step * problem.initial_sensitivities[row][column];
            minus.initial_state[row] -= step * problem.initial_sensitivities[row][column];
        }
        for row in 0..P {
            plus.parameters[row] += step * problem.parameter_seeds[row][column];
            minus.parameters[row] -= step * problem.parameter_seeds[row][column];
        }
        let plus = propagate(model, plus, options)?;
        let minus = propagate(model, minus, options)?;
        add_stats(&mut statistics, plus.stats);
        add_stats(&mut statistics, minus.stats);
        for row in 0..N {
            sensitivities[row][column] = (plus.state[row] - minus.state[row]) / (2.0 * step);
        }
    }
    ensure_finite_matrix("sensitivities", &sensitivities)?;
    Ok(SensitivityPropagation {
        time: nominal.time,
        state: nominal.state,
        sensitivities,
        stats: statistics,
    })
}

struct Workspace<const N: usize> {
    time: f64,
    state: [f64; N],
    jet: [[f64; MAX_ORDER + 1]; N],
}

impl<const N: usize> Workspace<N> {
    fn new(time: f64, state: [f64; N]) -> Self {
        Self {
            time,
            state,
            jet: [[0.0; MAX_ORDER + 1]; N],
        }
    }
}

fn integrate_to<M, const N: usize, const P: usize>(
    model: &M,
    parameters: &[f64; P],
    target_time: f64,
    options: IntegratorOptions,
    direction: f64,
    workspace: &mut Workspace<N>,
    stats: &mut IntegrationStats,
) -> Result<()>
where
    M: TaylorCoefficientModel<N, P>,
{
    let order = selected_order(options);
    while direction * (target_time - workspace.time) > 0.0 {
        if stats.accepted_steps + stats.rejected_steps >= options.maximum_steps {
            return integration_failure(M::NAME, "maximum step count exceeded");
        }
        model.coefficients(
            workspace.time,
            &workspace.state,
            parameters,
            order,
            &mut workspace.jet,
        )?;
        stats.rhs_evaluations += order;

        let mut step = suggested_step(&workspace.jet, &workspace.state, order, options);
        if stats.accepted_steps == 0 {
            step = step.min(options.initial_step.unwrap_or(f64::INFINITY));
        }
        if let Some(maximum) = options.maximum_step {
            step = step.min(maximum);
        }
        step = step.min((target_time - workspace.time).abs()) * direction;
        if !step.is_finite() || step == 0.0 || workspace.time + step == workspace.time {
            return integration_failure(M::NAME, "Taylor step size underflow");
        }

        workspace.state = evaluate(&workspace.jet, order, step);
        workspace.time += step;
        ensure_finite_state(M::NAME, &workspace.state)?;
        model.validate(workspace.time, &workspace.state, parameters)?;
        stats.accepted_steps += 1;
    }
    workspace.time = target_time;
    Ok(())
}

fn selected_order(options: IntegratorOptions) -> usize {
    let tolerance = options.relative_tolerance.min(options.absolute_tolerance);
    ((-0.5 * tolerance.ln()).ceil() as usize + 1).clamp(MIN_ORDER, MAX_ORDER)
}

fn suggested_step<const N: usize>(
    jet: &[[f64; MAX_ORDER + 1]; N],
    state: &[f64; N],
    order: usize,
    options: IntegratorOptions,
) -> f64 {
    let scaled_norm = |coefficient_order: usize| {
        (0..N)
            .map(|component| {
                let scale = options.absolute_tolerance
                    + options.relative_tolerance * state[component].abs().max(1.0);
                jet[component][coefficient_order].abs() / scale
            })
            .fold(0.0, f64::max)
    };
    let previous = scaled_norm(order - 1);
    let last = scaled_norm(order);
    let previous_step = if previous > 0.0 {
        previous.powf(-1.0 / (order - 1) as f64)
    } else {
        f64::INFINITY
    };
    let last_step = if last > 0.0 {
        last.powf(-1.0 / order as f64)
    } else {
        f64::INFINITY
    };
    SAFETY_FACTOR * previous_step.min(last_step)
}

fn evaluate<const N: usize>(jet: &[[f64; MAX_ORDER + 1]; N], order: usize, step: f64) -> [f64; N] {
    core::array::from_fn(|component| {
        (0..=order).rev().fold(0.0_f64, |value, coefficient| {
            value.mul_add(step, jet[component][coefficient])
        })
    })
}

fn propagation_direction(initial_time: f64, final_time: f64) -> f64 {
    (final_time - initial_time).signum()
}

fn sensitivity_step<const N: usize, const P: usize, const W: usize>(
    problem: &SensitivityProblem<N, P, W>,
    column: usize,
) -> f64 {
    let state_norm = (0..N)
        .map(|row| {
            problem.initial_sensitivities[row][column].abs()
                / problem.nominal.initial_state[row].abs().max(1.0)
        })
        .fold(0.0, f64::max);
    let parameter_norm = (0..P)
        .map(|row| {
            problem.parameter_seeds[row][column].abs()
                / problem.nominal.parameters[row].abs().max(1.0)
        })
        .fold(0.0, f64::max);
    3e-6 / state_norm.max(parameter_norm).max(1.0)
}

fn add_stats(total: &mut IntegrationStats, increment: IntegrationStats) {
    total.rhs_evaluations += increment.rhs_evaluations;
    total.accepted_steps += increment.accepted_steps;
    total.rejected_steps += increment.rejected_steps;
}

fn integration_failure<T>(model: &'static str, reason: &'static str) -> Result<T> {
    Err(PykepError::IntegrationFailure {
        model,
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::propagation::propagate_lagrangian;
    use crate::dynamics::pontryagin::{
        CartesianMassOptimal, CartesianTimeOptimal, EquinoctialMassOptimal, EquinoctialTimeOptimal,
    };
    use crate::dynamics::zoh::{
        ZohCr3bpDynamics, ZohEquinoctialDynamics, ZohKeplerDynamics, ZohSolarSailDynamics,
    };
    use crate::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
    use crate::integration::Dop853;

    fn options(tolerance: f64) -> IntegratorOptions {
        IntegratorOptions {
            relative_tolerance: tolerance,
            absolute_tolerance: tolerance,
            maximum_steps: 1_000_000,
            ..IntegratorOptions::default()
        }
    }

    #[test]
    fn kepler_matches_analytic_solution_forward_and_backward() {
        let initial = [0.5, 0.0, 0.0, 0.0, 3.0_f64.sqrt(), 0.0];
        for duration in [core::f64::consts::TAU, -core::f64::consts::TAU] {
            let result = propagate(
                &KeplerDynamics,
                InitialValueProblem::new(0.0, initial, duration, [1.0]),
                options(1e-12),
            )
            .unwrap();
            let expected = propagate_lagrangian(&initial, duration, 1.0).unwrap();
            for (actual, expected) in result.state.iter().zip(expected) {
                assert!((actual - expected).abs() < 3e-11);
            }
        }
    }

    #[test]
    fn dense_grid_matches_independent_propagations() {
        let initial = [0.8, 0.0, 0.0, 0.0, 1.2, 0.0];
        let times = [0.0, 0.25, 0.75, 1.5];
        let dense = propagate_dense(
            &KeplerDynamics,
            InitialValueProblem::new(0.0, initial, 1.5, [1.0]),
            &times,
            options(1e-12),
        )
        .unwrap();
        for (&time, state) in times.iter().zip(&dense.states) {
            let expected = propagate_lagrangian(&initial, time, 1.0).unwrap();
            for (&actual, expected) in state.iter().zip(expected) {
                assert!((actual - expected).abs() < 3e-11);
            }
        }
    }

    #[test]
    fn maximum_step_is_obeyed() {
        let options = IntegratorOptions {
            maximum_step: Some(0.01),
            ..options(1e-12)
        };
        let result = propagate(
            &KeplerDynamics,
            InitialValueProblem::new(0.0, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.1, [1.0]),
            options,
        )
        .unwrap();
        assert!(result.stats.accepted_steps >= 10);
    }

    #[test]
    fn zero_duration_invalid_options_and_step_exhaustion_are_explicit() {
        let initial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let zero = propagate(
            &KeplerDynamics,
            InitialValueProblem::new(2.0, initial, 2.0, [1.0]),
            options(1e-12),
        )
        .unwrap();
        assert_eq!(zero.state, initial);
        assert_eq!(zero.stats, IntegrationStats::default());

        let invalid = IntegratorOptions {
            relative_tolerance: 0.0,
            ..options(1e-12)
        };
        assert!(matches!(
            propagate(
                &KeplerDynamics,
                InitialValueProblem::new(0.0, initial, 1.0, [1.0]),
                invalid,
            ),
            Err(PykepError::InvalidInput {
                parameter: "relative_tolerance",
                ..
            })
        ));

        let exhausted = IntegratorOptions {
            maximum_step: Some(0.01),
            maximum_steps: 1,
            ..options(1e-12)
        };
        assert!(matches!(
            propagate(
                &KeplerDynamics,
                InitialValueProblem::new(0.0, initial, 1.0, [1.0]),
                exhausted,
            ),
            Err(PykepError::IntegrationFailure { .. })
        ));
        assert!(matches!(
            propagate(
                &KeplerDynamics,
                InitialValueProblem::new(0.0, [0.0; 6], 1.0, [1.0]),
                options(1e-12),
            ),
            Err(PykepError::SingularGeometry { .. })
        ));
    }

    #[test]
    fn dense_sampling_finishes_the_declared_problem_interval() {
        let initial = [0.8, 0.0, 0.0, 0.0, 1.2, 0.0];
        let problem = InitialValueProblem::new(0.0, initial, 1.5, [1.0]);
        let dense =
            propagate_dense(&KeplerDynamics, problem, &[0.25, 0.75], options(1e-12)).unwrap();
        let full = propagate(&KeplerDynamics, problem, options(1e-12)).unwrap();
        assert_eq!(dense.times, [0.25, 0.75]);
        assert!(dense.stats.accepted_steps >= full.stats.accepted_steps);
    }

    #[test]
    fn zero_width_sensitivities_are_rejected() {
        let problem = SensitivityProblem {
            nominal: InitialValueProblem::new(0.0, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 1.0, [1.0]),
            initial_sensitivities: [[]; 6],
            parameter_seeds: [[]],
        };
        assert!(matches!(
            propagate_with_sensitivities(&KeplerDynamics, problem, options(1e-12)),
            Err(PykepError::InvalidInput {
                parameter: "sensitivity_width",
                ..
            })
        ));
    }

    fn compare_with_dop853<M, const N: usize, const P: usize>(
        model: &M,
        initial: [f64; N],
        parameters: [f64; P],
        final_time: f64,
        tolerance: f64,
    ) where
        M: TaylorCoefficientModel<N, P>,
    {
        let problem = InitialValueProblem::new(0.0, initial, final_time, parameters);
        let taylor = propagate(model, problem, options(1e-13)).unwrap();
        let dop = Dop853.propagate(model, problem, options(1e-13)).unwrap();
        for (actual, expected) in taylor.state.iter().zip(dop.state) {
            let scale = actual.abs().max(expected.abs()).max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance * scale,
                "{}: Taylor={actual:.17e}, DOP853={expected:.17e}",
                M::NAME
            );
        }
    }

    #[test]
    fn rotating_models_match_dop853() {
        compare_with_dop853(
            &Cr3bpDynamics,
            [0.8, -0.2, 0.1, 0.03, -0.04, 0.02],
            [0.012_150_585_609_624_04],
            0.75,
            2e-11,
        );
        compare_with_dop853(
            &BcpDynamics,
            [0.8, -0.2, 0.1, 0.03, -0.04, 0.02],
            [
                0.012_150_585_609_624_04,
                328_900.56,
                389.172,
                -0.925_195_985_520_347,
            ],
            0.25,
            2e-10,
        );
    }

    #[test]
    fn zoh_models_match_dop853() {
        compare_with_dop853(
            &ZohKeplerDynamics,
            [0.8, -0.2, 0.1, 0.03, 1.0, 0.02, 1.1],
            [0.02, 0.3, -0.4, 0.5, 0.01],
            0.5,
            2e-11,
        );
        compare_with_dop853(
            &ZohCr3bpDynamics,
            [0.8, -0.2, 0.1, 0.03, -0.04, 0.02, 1.1],
            [0.02, 0.3, -0.4, 0.5, 0.01, 0.012_150_585_609_624_04],
            0.5,
            2e-10,
        );
        compare_with_dop853(
            &ZohEquinoctialDynamics,
            [1.1, 0.1, -0.05, 0.02, -0.03, 0.4, 1.1],
            [0.02, 0.3, -0.4, 0.5, 0.01],
            0.5,
            2e-10,
        );
        compare_with_dop853(
            &ZohSolarSailDynamics,
            [0.8, -0.4, 0.3, 0.2, 0.9, -0.1],
            [0.25, -1.1, 0.04],
            0.5,
            2e-10,
        );
    }

    #[test]
    fn pontryagin_models_match_dop853() {
        let cartesian = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        compare_with_dop853(
            &CartesianMassOptimal,
            cartesian,
            [1.0, 0.01, 1.0, 0.5, 1.0],
            0.1,
            5e-10,
        );
        compare_with_dop853(
            &CartesianTimeOptimal,
            cartesian,
            [1.0, 0.01, 1.0],
            0.1,
            5e-10,
        );

        let equinoctial = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ];
        compare_with_dop853(
            &EquinoctialMassOptimal,
            equinoctial,
            [1.0, 1e-4, 1.0, 1.0, 1e-4],
            0.1,
            2e-9,
        );
        compare_with_dop853(
            &EquinoctialTimeOptimal,
            equinoctial,
            [1.0, 1e-4, 1.0],
            0.1,
            2e-9,
        );
    }

    #[test]
    fn seeded_sensitivities_match_dop853_variational_equations() {
        let initial = [0.8, -0.2, 0.1, 0.03, 1.0, 0.02];
        let problem = SensitivityProblem {
            nominal: InitialValueProblem::new(0.0, initial, 0.5, [1.0]),
            initial_sensitivities: core::array::from_fn(|row| {
                core::array::from_fn(|column| if row == column { 1.0 } else { 0.0 })
            }),
            parameter_seeds: [[0.0; 6]],
        };
        let taylor =
            propagate_with_sensitivities(&KeplerDynamics, problem, options(1e-13)).unwrap();
        let dop = Dop853
            .propagate_with_sensitivities(&KeplerDynamics, problem, options(1e-13))
            .unwrap();
        for row in 0..6 {
            for column in 0..6 {
                assert!(
                    (taylor.sensitivities[row][column] - dop.sensitivities[row][column]).abs()
                        < 2e-7
                );
            }
        }
    }
}
