// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Internal implementation of adaptive Taylor integration.

pub(super) mod events;
mod generated;
mod kernel;
#[cfg(test)]
mod profiling;
#[cfg(test)]
mod series;
#[cfg(feature = "simd")]
mod simd;
mod systems;
#[cfg_attr(not(test), allow(dead_code))]
mod tape;
mod variational;

use crate::integration::{
    DenseTrajectory, DynamicsModel, InitialValueProblem, IntegrationStats, IntegratorOptions,
    Propagation, SensitivityProblem, SensitivityPropagation, Termination, ensure_finite_matrix,
    ensure_finite_state, validate_evaluation_times, validate_problem,
};
use crate::{PykepError, Result};

const MAX_ORDER: usize = 24;
const SAFETY_FACTOR: f64 = 0.75;
const MIN_ORDER: usize = 8;
const BASELINE_X86: bool = cfg!(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "fma")
));

pub(crate) trait TaylorCoefficientModel<const N: usize, const P: usize>:
    DynamicsModel<N, P>
{
    const CARTESIAN_EVENTS: bool = false;
    const NOMINAL_STEP_SAFETY: f64 = SAFETY_FACTOR;

    #[cfg(feature = "simd")]
    const SIMD_CAPABLE: bool = false;

    fn generated_coefficients<T: kernel::Coefficient>(
        &self,
        time: T,
        state: &[T; N],
        parameters: &[T; P],
        order: usize,
        jet: &mut [[T; MAX_ORDER + 1]; N],
    );

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

pub(crate) fn propagate_batch<M, const N: usize, const P: usize>(
    model: &M,
    problems: &[InitialValueProblem<N, P>],
    options: IntegratorOptions,
    workers: usize,
) -> Result<Vec<Propagation<N>>>
where
    M: TaylorCoefficientModel<N, P> + Sync,
{
    #[cfg(feature = "simd")]
    if M::SIMD_CAPABLE && problems.len() >= 4 {
        let (groups, remainder) = problems.as_chunks::<4>();
        let groups = crate::batch::try_map(groups, workers, |group| {
            simd::propagate_group(model, group, options)
        })?;
        let mut results = Vec::with_capacity(problems.len());
        results.extend(groups.into_iter().flatten());
        for &problem in remainder {
            results.push(propagate(model, problem, options)?);
        }
        return Ok(results);
    }
    crate::batch::try_map(problems, workers, |&problem| {
        propagate(model, problem, options)
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
    let order = selected_order(options);
    while direction * (problem.final_time - workspace.time) > 0.0 {
        let start = workspace.time;
        let step = advance(
            model,
            &problem.parameters,
            problem.final_time,
            options,
            order,
            &mut workspace,
            &mut stats,
        )?;
        let remaining = &evaluation_times[states.len()..];
        let count = remaining.partition_point(|&time| direction * (time - workspace.time) <= 0.0);
        for &time in &remaining[..count] {
            let state = if time == workspace.time {
                workspace.state
            } else if time == start {
                workspace.jet.each_ref().map(|row| row[0])
            } else {
                let offset = (time - start).abs().min(step.abs()).copysign(step);
                evaluate_dense(&workspace.jet, order, offset)
            };
            ensure_finite_state(M::NAME, &state)?;
            model.validate(time, &state, &problem.parameters)?;
            states.push(state);
        }
    }
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

    variational::propagate(model, problem, options)
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
        advance(
            model,
            parameters,
            target_time,
            options,
            order,
            workspace,
            stats,
        )?;
    }
    Ok(())
}

fn advance<M, const N: usize, const P: usize>(
    model: &M,
    parameters: &[f64; P],
    target_time: f64,
    options: IntegratorOptions,
    order: usize,
    workspace: &mut Workspace<N>,
    stats: &mut IntegrationStats,
) -> Result<f64>
where
    M: TaylorCoefficientModel<N, P>,
{
    if stats.accepted_steps >= options.maximum_steps {
        return integration_failure(M::NAME, "maximum step count exceeded");
    }
    model.coefficients(
        workspace.time,
        &workspace.state,
        parameters,
        order,
        &mut workspace.jet,
    )?;
    accept_jet(
        model,
        parameters,
        target_time,
        options,
        order,
        workspace,
        stats,
    )
}

fn accept_jet<M, const N: usize, const P: usize>(
    model: &M,
    parameters: &[f64; P],
    target_time: f64,
    options: IntegratorOptions,
    order: usize,
    workspace: &mut Workspace<N>,
    stats: &mut IntegrationStats,
) -> Result<f64>
where
    M: TaylorCoefficientModel<N, P>,
{
    let remaining = target_time - workspace.time;
    let step = select_step(
        M::NAME,
        remaining,
        workspace.time,
        M::NOMINAL_STEP_SAFETY * suggested_step(&workspace.jet, &workspace.state, order, options),
        options,
        stats,
    )?;
    workspace.state = evaluate(&workspace.jet, order, step);
    workspace.time = accept_state(
        model,
        parameters,
        workspace.time,
        target_time,
        step,
        &workspace.state,
    )?;
    stats.rhs_evaluations += order;
    stats.accepted_steps += 1;
    Ok(step)
}

fn accept_state<M, const N: usize, const P: usize>(
    model: &M,
    parameters: &[f64; P],
    time: f64,
    target: f64,
    step: f64,
    state: &[f64; N],
) -> Result<f64>
where
    M: TaylorCoefficientModel<N, P>,
{
    let next_time = if step == target - time {
        target
    } else {
        time + step
    };
    ensure_finite_state(M::NAME, state)?;
    model.validate(next_time, state, parameters)?;
    Ok(next_time)
}

fn select_step(
    model: &'static str,
    remaining: f64,
    time: f64,
    mut magnitude: f64,
    options: IntegratorOptions,
    stats: &IntegrationStats,
) -> Result<f64> {
    if stats.accepted_steps >= options.maximum_steps {
        return integration_failure(model, "maximum step count exceeded");
    }
    if stats.accepted_steps == 0 {
        magnitude = magnitude.min(options.initial_step.unwrap_or(f64::INFINITY));
    }
    if let Some(maximum) = options.maximum_step {
        magnitude = magnitude.min(maximum);
    }
    let step = magnitude.min(remaining.abs()).copysign(remaining);
    if !step.is_finite() || step == 0.0 || time + step == time {
        return integration_failure(model, "Taylor step size underflow");
    }
    Ok(step)
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
    suggested_step_from_coefficients(state, order, options, |component, degree| {
        jet[component][degree]
    })
}

fn suggested_step_from_coefficients<const N: usize>(
    state: &[f64; N],
    order: usize,
    options: IntegratorOptions,
    coefficient: impl Fn(usize, usize) -> f64,
) -> f64 {
    let scaled_norm = |coefficient_order: usize| {
        (0..N)
            .map(|component| {
                let scale = options.absolute_tolerance
                    + options.relative_tolerance * state[component].abs().max(1.0);
                coefficient(component, coefficient_order).abs() / scale
            })
            .fold(0.0, f64::max)
    };
    step_from_norms([scaled_norm(order - 1), scaled_norm(order)], order)
}

fn scaled_coefficient(value: f64, initial: f64, options: IntegratorOptions) -> Result<f64> {
    if !value.is_finite() || !initial.is_finite() {
        return Err(PykepError::NumericalOverflow {
            operation: "Taylor coefficients",
        });
    }
    Ok(value.abs()
        / (options.absolute_tolerance + options.relative_tolerance * initial.abs().max(1.0)))
}

fn step_from_norms(norms: [f64; 2], order: usize) -> f64 {
    let bound = |norm: f64, degree| {
        if norm > 0.0 {
            norm.powf(-1.0 / degree as f64)
        } else {
            f64::INFINITY
        }
    };
    bound(norms[0], order - 1).min(bound(norms[1], order))
}

fn evaluate<const N: usize>(jet: &[[f64; MAX_ORDER + 1]; N], order: usize, step: f64) -> [f64; N] {
    evaluate_horner::<N, true>(jet, order, step)
}

fn evaluate_dense<const N: usize>(
    jet: &[[f64; MAX_ORDER + 1]; N],
    order: usize,
    step: f64,
) -> [f64; N] {
    // On baseline x86 builds each scalar mul_add may call a runtime helper.
    // Keep the fused evaluation for accepted steps and for exceptional samples.
    if BASELINE_X86 {
        let state = evaluate_horner::<N, false>(jet, order, step);
        if state.iter().all(|value| value.is_finite()) {
            return state;
        }
    }
    evaluate(jet, order, step)
}

fn evaluate_horner<const N: usize, const FUSED: bool>(
    jet: &[[f64; MAX_ORDER + 1]; N],
    order: usize,
    step: f64,
) -> [f64; N] {
    core::array::from_fn(|component| {
        jet[component][..=order]
            .iter()
            .rev()
            .fold(0.0_f64, |value, &coefficient| {
                if FUSED {
                    value.mul_add(step, coefficient)
                } else {
                    value * step + coefficient
                }
            })
    })
}

fn propagation_direction(initial_time: f64, final_time: f64) -> f64 {
    if final_time == initial_time {
        0.0
    } else {
        (final_time - initial_time).signum()
    }
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
        assert_eq!(dense.stats, full.stats);
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
