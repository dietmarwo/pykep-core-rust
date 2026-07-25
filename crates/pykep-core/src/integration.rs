// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Adaptive integration for evaluated astrodynamics models.
//!
//! This module deliberately exposes pykep-owned model and result types rather
//! than the implementation types of the selected solver crate. The numerical
//! backend is DOP853: an explicit Runge-Kutta method of order 8 with embedded
//! error estimates and seventh-order dense output.

use core::cell::RefCell;

use differential_equations::control::ControlFlag;
use differential_equations::interpolate::Interpolation;
use differential_equations::ode::{ODE, solve_ode};
use differential_equations::prelude::ExplicitRungeKutta;
use differential_equations::solout::{
    CrossingDirection, DefaultSolout, Event as BackendEvent, EventConfig, Solout, TEvalSolout,
};
use differential_equations::solution::Solution;
use differential_equations::tolerance::Tolerance as BackendTolerance;
use differential_equations::traits::State as BackendState;

use crate::error::ensure_finite;
use crate::{PykepError, Result};

/// An evaluated first-order dynamics model with `N` states and `P` parameters.
///
/// Implementations must not allocate in [`Self::rhs`]. The integrator validates
/// dimensions and finite inputs before calling the model.
pub trait DynamicsModel<const N: usize, const P: usize> {
    /// Stable model name used in diagnostics.
    const NAME: &'static str;

    /// Validates state and parameter domains before a propagation.
    ///
    /// The default accepts every finite state and parameter vector.
    ///
    /// # Errors
    ///
    /// Returns an error when the initial state or parameters are outside the
    /// model's physical domain.
    fn validate(&self, _time: f64, _state: &[f64; N], _parameters: &[f64; P]) -> Result<()> {
        Ok(())
    }

    /// Evaluates `dstate/dt` into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns an error when the dynamics are singular or cannot be evaluated.
    fn rhs(
        &self,
        time: f64,
        state: &[f64; N],
        parameters: &[f64; P],
        derivative: &mut [f64; N],
    ) -> Result<()>;
}

/// A dynamics model that provides the Jacobians required by sensitivities.
pub trait DifferentiableDynamicsModel<const N: usize, const P: usize>: DynamicsModel<N, P> {
    /// Evaluates `df/dstate` and `df/dparameters` in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`DynamicsModel::rhs`].
    fn jacobians(
        &self,
        time: f64,
        state: &[f64; N],
        parameters: &[f64; P],
        state_jacobian: &mut [[f64; N]; N],
        parameter_jacobian: &mut [[f64; P]; N],
    ) -> Result<()>;
}

/// Direction in which an event function must cross zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EventDirection {
    /// Detect either crossing direction.
    #[default]
    Either,
    /// Detect a transition from negative to non-negative.
    Increasing,
    /// Detect a transition from positive to non-positive.
    Decreasing,
}

/// A scalar zero-crossing condition for a fixed-size state.
pub trait Event<const N: usize> {
    /// Crossing direction to detect.
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }

    /// Evaluates the scalar event function.
    fn value(&self, time: f64, state: &[f64; N]) -> f64;
}

/// Scalar error tolerances and safety limits for adaptive integration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegratorOptions {
    /// Relative local-error tolerance.
    pub relative_tolerance: f64,
    /// Absolute local-error tolerance.
    pub absolute_tolerance: f64,
    /// Optional initial step magnitude. The backend estimates it when absent.
    pub initial_step: Option<f64>,
    /// Optional maximum step magnitude.
    pub maximum_step: Option<f64>,
    /// Maximum accepted plus rejected steps.
    pub maximum_steps: usize,
    /// Maximum consecutive rejected steps.
    pub maximum_rejections: usize,
}

impl Default for IntegratorOptions {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-12,
            absolute_tolerance: 1e-12,
            initial_step: None,
            maximum_step: None,
            maximum_steps: 100_000,
            maximum_rejections: 100,
        }
    }
}

impl IntegratorOptions {
    fn validate(self) -> Result<()> {
        validate_positive_finite("relative_tolerance", self.relative_tolerance)?;
        validate_positive_finite("absolute_tolerance", self.absolute_tolerance)?;
        if let Some(step) = self.initial_step {
            validate_positive_finite("initial_step", step)?;
        }
        if let Some(step) = self.maximum_step {
            validate_positive_finite("maximum_step", step)?;
        }
        if self.maximum_steps == 0 {
            return Err(PykepError::InvalidInput {
                parameter: "maximum_steps",
                reason: "must be greater than zero".into(),
            });
        }
        if self.maximum_rejections == 0 {
            return Err(PykepError::InvalidInput {
                parameter: "maximum_rejections",
                reason: "must be greater than zero".into(),
            });
        }
        Ok(())
    }
}

fn validate_positive_finite(parameter: &'static str, value: f64) -> Result<()> {
    ensure_finite(parameter, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter,
            reason: "must be greater than zero".into(),
        })
    }
}

/// Initial-value problem for a fixed-size model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InitialValueProblem<const N: usize, const P: usize> {
    /// Initial integration time.
    pub initial_time: f64,
    /// State at [`Self::initial_time`].
    pub initial_state: [f64; N],
    /// Requested final integration time.
    pub final_time: f64,
    /// Constant model parameters for this propagation.
    pub parameters: [f64; P],
}

impl<const N: usize, const P: usize> InitialValueProblem<N, P> {
    /// Creates an initial-value problem.
    pub const fn new(
        initial_time: f64,
        initial_state: [f64; N],
        final_time: f64,
        parameters: [f64; P],
    ) -> Self {
        Self {
            initial_time,
            initial_state,
            final_time,
            parameters,
        }
    }
}

/// Initial-value problem augmented with arbitrary sensitivity seed directions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensitivityProblem<const N: usize, const P: usize, const W: usize> {
    /// Nominal initial-value problem.
    pub nominal: InitialValueProblem<N, P>,
    /// Initial `dstate/dseed` matrix.
    pub initial_sensitivities: [[f64; W]; N],
    /// Constant `dparameters/dseed` matrix.
    pub parameter_seeds: [[f64; W]; P],
}

/// Work counters reported by a completed propagation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntegrationStats {
    /// Right-hand-side evaluations.
    pub rhs_evaluations: usize,
    /// Accepted integration steps.
    pub accepted_steps: usize,
    /// Rejected integration steps.
    pub rejected_steps: usize,
}

/// Reason that integration stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// The requested final time was reached.
    FinalTime,
    /// A terminal event was located by dense interpolation.
    Event,
}

/// Final state and diagnostics from a propagation.
#[derive(Clone, Debug, PartialEq)]
pub struct Propagation<const N: usize> {
    /// Time at which integration stopped.
    pub time: f64,
    /// State at [`Self::time`].
    pub state: [f64; N],
    /// Integration work counters.
    pub stats: IntegrationStats,
    /// Reason that integration stopped.
    pub termination: Termination,
}

/// States evaluated at caller-requested times using dense output.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseTrajectory<const N: usize> {
    /// Evaluation times in propagation order.
    pub times: Vec<f64>,
    /// States corresponding one-to-one with [`Self::times`].
    pub states: Vec<[f64; N]>,
    /// Integration work counters.
    pub stats: IntegrationStats,
}

/// Final state and first-order sensitivities.
///
/// `sensitivities[i][j]` is the derivative of state component `i` with
/// respect to seed direction `j`.
#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityPropagation<const N: usize, const W: usize> {
    /// Time at the end of propagation.
    pub time: f64,
    /// Nominal propagated state.
    pub state: [f64; N],
    /// Row-major state sensitivity matrix.
    pub sensitivities: [[f64; W]; N],
    /// Integration work counters.
    pub stats: IntegrationStats,
}

/// Pure-Rust adaptive DOP853 integration facade.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dop853;

impl Dop853 {
    /// Propagates a model from `initial_time` to `final_time`.
    ///
    /// No heap allocation occurs per accepted step for fixed-size states.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid options, non-finite inputs, model failures,
    /// step-size underflow, stiffness detection, or exhausted step limits.
    pub fn propagate<M, const N: usize, const P: usize>(
        &self,
        model: &M,
        problem: InitialValueProblem<N, P>,
        options: IntegratorOptions,
    ) -> Result<Propagation<N>>
    where
        M: DynamicsModel<N, P>,
    {
        validate_problem(model, &problem, options)?;
        if problem.initial_time == problem.final_time {
            return Ok(Propagation {
                time: problem.initial_time,
                state: problem.initial_state,
                stats: IntegrationStats::default(),
                termination: Termination::FinalTime,
            });
        }

        let adapter = ModelAdapter::new(model, &problem.parameters);
        let mut solver = configured_solver(options);
        let mut output = FinalState::default();
        let solution = solve_ode(
            &mut solver,
            &adapter,
            problem.initial_time,
            problem.final_time,
            &problem.initial_state,
            &mut output,
        );
        finish_model_error(&adapter)?;
        let solution = solution.map_err(|error| integration_error(M::NAME, &error))?;
        let (time, state) = output.value.ok_or_else(|| PykepError::IntegrationFailure {
            model: M::NAME,
            reason: "backend returned no final state".into(),
        })?;
        ensure_finite_state(M::NAME, &state)?;

        Ok(Propagation {
            time,
            state,
            stats: stats(&solution),
            termination: Termination::FinalTime,
        })
    }

    /// Propagates and evaluates the dense solution at requested times.
    ///
    /// `evaluation_times` must be strictly monotone in the propagation
    /// direction and lie in the closed integration interval.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid grid or any error described by
    /// [`Self::propagate`].
    pub fn propagate_dense<M, const N: usize, const P: usize>(
        &self,
        model: &M,
        problem: InitialValueProblem<N, P>,
        evaluation_times: &[f64],
        options: IntegratorOptions,
    ) -> Result<DenseTrajectory<N>>
    where
        M: DynamicsModel<N, P>,
    {
        validate_problem(model, &problem, options)?;
        validate_evaluation_times(problem.initial_time, problem.final_time, evaluation_times)?;
        if problem.initial_time == problem.final_time {
            return Ok(DenseTrajectory {
                times: vec![problem.initial_time],
                states: vec![problem.initial_state],
                stats: IntegrationStats::default(),
            });
        }

        let adapter = ModelAdapter::new(model, &problem.parameters);
        let mut solver = configured_solver(options);
        let mut output =
            TEvalSolout::new(evaluation_times, problem.initial_time, problem.final_time);
        let solution = solve_ode(
            &mut solver,
            &adapter,
            problem.initial_time,
            problem.final_time,
            &problem.initial_state,
            &mut output,
        );
        finish_model_error(&adapter)?;
        let solution = solution.map_err(|error| integration_error(M::NAME, &error))?;
        for state in &solution.y {
            ensure_finite_state(M::NAME, state)?;
        }
        Ok(DenseTrajectory {
            times: solution.t.clone(),
            states: solution.y.clone(),
            stats: stats(&solution),
        })
    }

    /// Propagates until a terminal zero-crossing or the requested final time.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite event value or any error described by
    /// [`Self::propagate`].
    pub fn propagate_until_event<M, E, const N: usize, const P: usize>(
        &self,
        model: &M,
        event: &E,
        problem: InitialValueProblem<N, P>,
        options: IntegratorOptions,
    ) -> Result<Propagation<N>>
    where
        M: DynamicsModel<N, P>,
        E: Event<N>,
    {
        validate_problem(model, &problem, options)?;
        if problem.initial_time == problem.final_time {
            return Ok(Propagation {
                time: problem.initial_time,
                state: problem.initial_state,
                stats: IntegrationStats::default(),
                termination: Termination::FinalTime,
            });
        }

        let adapter = ModelAdapter::new(model, &problem.parameters);
        let event_adapter = EventAdapter::new(event);
        let mut solver = configured_solver(options);
        let mut output = differential_equations::solout::EventWrappedSolout::new(
            DefaultSolout::new(),
            &event_adapter,
            problem.initial_time,
            problem.final_time,
        );
        let solution = solve_ode(
            &mut solver,
            &adapter,
            problem.initial_time,
            problem.final_time,
            &problem.initial_state,
            &mut output,
        );
        finish_model_error(&adapter)?;
        if event_adapter.non_finite_value() {
            return Err(PykepError::IntegrationFailure {
                model: M::NAME,
                reason: "event function returned a non-finite value".into(),
            });
        }
        let solution = solution.map_err(|error| integration_error(M::NAME, &error))?;
        let (&time, &state) = solution
            .last()
            .map_err(|_| PykepError::IntegrationFailure {
                model: M::NAME,
                reason: "backend returned no event or final state".into(),
            })?;
        ensure_finite_state(M::NAME, &state)?;
        let termination = if matches!(
            solution.status,
            differential_equations::status::Status::Interrupted
        ) {
            Termination::Event
        } else {
            Termination::FinalTime
        };
        Ok(Propagation {
            time,
            state,
            stats: stats(&solution),
            termination,
        })
    }

    /// Propagates a nominal state and arbitrary first-order seed directions.
    ///
    /// The initial sensitivity matrix contains `dstate0/dseed`; the parameter
    /// seed matrix contains `dparameters/dseed`. This supports STMs, parameter
    /// sensitivities, and the state-plus-control variations required by ZOH
    /// legs without solver-generated expression graphs.
    ///
    /// # Errors
    ///
    /// Returns an error for `W == 0`, invalid seeds, Jacobian failures, or any
    /// error described by [`Self::propagate`].
    pub fn propagate_with_sensitivities<M, const N: usize, const P: usize, const W: usize>(
        &self,
        model: &M,
        problem: SensitivityProblem<N, P, W>,
        options: IntegratorOptions,
    ) -> Result<SensitivityPropagation<N, W>>
    where
        M: DifferentiableDynamicsModel<N, P>,
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

        let augmented = AugmentedModel {
            model,
            parameters: &problem.nominal.parameters,
            parameter_seeds: &problem.parameter_seeds,
            error: RefCell::new(None),
        };
        let initial = AugmentedState {
            state: problem.nominal.initial_state,
            sensitivities: problem.initial_sensitivities,
        };
        let mut solver = configured_solver(options);
        let mut output = FinalState::default();
        let solution = solve_ode(
            &mut solver,
            &augmented,
            problem.nominal.initial_time,
            problem.nominal.final_time,
            &initial,
            &mut output,
        );
        if let Some(error) = augmented.error.borrow_mut().take() {
            return Err(error);
        }
        let solution = solution.map_err(|error| integration_error(M::NAME, &error))?;
        let (time, final_state) = output.value.ok_or_else(|| PykepError::IntegrationFailure {
            model: M::NAME,
            reason: "backend returned no sensitivity state".into(),
        })?;
        ensure_finite_state(M::NAME, &final_state.state)?;
        ensure_finite_matrix("sensitivities", &final_state.sensitivities)?;
        Ok(SensitivityPropagation {
            time,
            state: final_state.state,
            sensitivities: final_state.sensitivities,
            stats: stats(&solution),
        })
    }
}

fn configured_solver<Y>(
    options: IntegratorOptions,
) -> differential_equations::methods::ExplicitRungeKutta<
    differential_equations::methods::Ordinary,
    differential_equations::methods::DormandPrince,
    f64,
    Y,
    8,
    12,
    16,
>
where
    Y: BackendState<f64>,
{
    ExplicitRungeKutta::dop853()
        .rtol(options.relative_tolerance)
        .atol(options.absolute_tolerance)
        .h0(options.initial_step.unwrap_or(0.0))
        .h_max(options.maximum_step.unwrap_or(f64::INFINITY))
        .max_steps(options.maximum_steps)
        .max_rejects(options.maximum_rejections)
}

fn validate_problem<M, const N: usize, const P: usize>(
    model: &M,
    problem: &InitialValueProblem<N, P>,
    options: IntegratorOptions,
) -> Result<()>
where
    M: DynamicsModel<N, P>,
{
    if N == 0 {
        return Err(PykepError::InvalidInput {
            parameter: "state_dimension",
            reason: "must be greater than zero".into(),
        });
    }
    ensure_finite("initial_time", problem.initial_time)?;
    ensure_finite("final_time", problem.final_time)?;
    ensure_finite_matrix("initial_state", &problem.initial_state)?;
    ensure_finite_matrix("parameters", &problem.parameters)?;
    options.validate()?;
    model.validate(
        problem.initial_time,
        &problem.initial_state,
        &problem.parameters,
    )
}

fn validate_evaluation_times(initial: f64, final_time: f64, times: &[f64]) -> Result<()> {
    if times.is_empty() {
        return Err(PykepError::InvalidInput {
            parameter: "evaluation_times",
            reason: "must contain at least one time".into(),
        });
    }
    if initial == final_time {
        if times.len() == 1 && times[0] == initial {
            return Ok(());
        }
        return Err(PykepError::InvalidInput {
            parameter: "evaluation_times",
            reason: "a zero-duration propagation accepts only the initial time".into(),
        });
    }
    let direction = (final_time - initial).signum();
    let lower = initial.min(final_time);
    let upper = initial.max(final_time);
    for (index, &time) in times.iter().enumerate() {
        ensure_finite("evaluation_time", time)?;
        if !(lower..=upper).contains(&time) {
            return Err(PykepError::InvalidInput {
                parameter: "evaluation_times",
                reason: "all times must lie in the closed propagation interval".into(),
            });
        }
        if index > 0 && (time - times[index - 1]) * direction <= 0.0 {
            return Err(PykepError::InvalidInput {
                parameter: "evaluation_times",
                reason: "times must be strictly monotone in the propagation direction".into(),
            });
        }
    }
    Ok(())
}

fn ensure_finite_matrix<const R: usize, T>(parameter: &'static str, values: &[T; R]) -> Result<()>
where
    T: FiniteValues,
{
    for value in values {
        if !value.all_finite() {
            return Err(PykepError::NonFiniteInput { parameter });
        }
    }
    Ok(())
}

trait FiniteValues {
    fn all_finite(&self) -> bool;
}

impl FiniteValues for f64 {
    fn all_finite(&self) -> bool {
        self.is_finite()
    }
}

impl<const N: usize> FiniteValues for [f64; N] {
    fn all_finite(&self) -> bool {
        self.iter().all(|value| value.is_finite())
    }
}

fn ensure_finite_state<const N: usize>(model: &'static str, state: &[f64; N]) -> Result<()> {
    if state.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PykepError::IntegrationFailure {
            model,
            reason: "propagation produced a non-finite state".into(),
        })
    }
}

fn integration_error<T: core::fmt::Debug>(model: &'static str, error: &T) -> PykepError {
    PykepError::IntegrationFailure {
        model,
        reason: format!("{error:?}"),
    }
}

fn stats<T, Y>(solution: &Solution<T, Y>) -> IntegrationStats
where
    T: differential_equations::traits::Real,
    Y: BackendState<T>,
{
    IntegrationStats {
        rhs_evaluations: solution.evals.function,
        accepted_steps: solution.steps.accepted,
        rejected_steps: solution.steps.rejected,
    }
}

struct ModelAdapter<'a, M, const N: usize, const P: usize> {
    model: &'a M,
    parameters: &'a [f64; P],
    error: RefCell<Option<PykepError>>,
}

impl<'a, M, const N: usize, const P: usize> ModelAdapter<'a, M, N, P> {
    fn new(model: &'a M, parameters: &'a [f64; P]) -> Self {
        Self {
            model,
            parameters,
            error: RefCell::new(None),
        }
    }
}

impl<M, const N: usize, const P: usize> ODE<f64, [f64; N]> for ModelAdapter<'_, M, N, P>
where
    M: DynamicsModel<N, P>,
{
    fn diff(&self, time: f64, state: &[f64; N], derivative: &mut [f64; N]) {
        if self.error.borrow().is_some() {
            derivative.fill(f64::NAN);
            return;
        }
        if let Err(error) = self.model.rhs(time, state, self.parameters, derivative) {
            *self.error.borrow_mut() = Some(error);
            derivative.fill(f64::NAN);
        } else if derivative.iter().any(|value| !value.is_finite()) {
            *self.error.borrow_mut() = Some(PykepError::IntegrationFailure {
                model: M::NAME,
                reason: "right-hand side returned a non-finite derivative".into(),
            });
            derivative.fill(f64::NAN);
        }
    }
}

fn finish_model_error<M, const N: usize, const P: usize>(
    adapter: &ModelAdapter<'_, M, N, P>,
) -> Result<()> {
    match adapter.error.borrow_mut().take() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[derive(Clone, Debug)]
struct FinalState<Y> {
    value: Option<(f64, Y)>,
}

impl<Y> Default for FinalState<Y> {
    fn default() -> Self {
        Self { value: None }
    }
}

impl<Y> Solout<f64, Y> for FinalState<Y>
where
    Y: BackendState<f64>,
{
    fn solout<I>(
        &mut self,
        time: f64,
        _previous_time: f64,
        state: &Y,
        _previous_state: &Y,
        _interpolator: &mut I,
        _solution: &mut Solution<f64, Y>,
    ) -> ControlFlag<f64, Y>
    where
        I: Interpolation<f64, Y> + ?Sized,
    {
        self.value = Some((time, state.clone()));
        ControlFlag::Continue
    }
}

struct EventAdapter<'a, E> {
    event: &'a E,
    non_finite: core::cell::Cell<bool>,
}

impl<'a, E> EventAdapter<'a, E> {
    fn new(event: &'a E) -> Self {
        Self {
            event,
            non_finite: core::cell::Cell::new(false),
        }
    }

    fn non_finite_value(&self) -> bool {
        self.non_finite.get()
    }
}

impl<E, const N: usize> BackendEvent<f64, [f64; N]> for EventAdapter<'_, E>
where
    E: Event<N>,
{
    fn config(&self) -> EventConfig {
        let direction = match self.event.direction() {
            EventDirection::Either => CrossingDirection::Both,
            EventDirection::Increasing => CrossingDirection::Positive,
            EventDirection::Decreasing => CrossingDirection::Negative,
        };
        EventConfig::new(direction, Some(1))
    }

    fn event(&self, time: f64, state: &[f64; N]) -> f64 {
        let value = self.event.value(time, state);
        if value.is_finite() {
            value
        } else {
            self.non_finite.set(true);
            f64::NAN
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct AugmentedState<const N: usize, const W: usize> {
    state: [f64; N],
    sensitivities: [[f64; W]; N],
}

impl<const N: usize, const W: usize> AugmentedState<N, W> {
    fn for_each(&self, mut operation: impl FnMut(usize, f64)) {
        for (index, &value) in self.state.iter().enumerate() {
            operation(index, value);
        }
        let mut index = N;
        for row in &self.sensitivities {
            for &value in row {
                operation(index, value);
                index += 1;
            }
        }
    }
}

impl<const N: usize, const W: usize> BackendState<f64> for AugmentedState<N, W> {
    fn len(&self) -> usize {
        N + N * W
    }

    fn get_component(&self, index: usize) -> f64 {
        if index < N {
            self.state[index]
        } else {
            let flat = index - N;
            self.sensitivities[flat / W][flat % W]
        }
    }

    fn set_component(&mut self, index: usize, value: f64) {
        if index < N {
            self.state[index] = value;
        } else {
            let flat = index - N;
            self.sensitivities[flat / W][flat % W] = value;
        }
    }

    fn map_components_mut<F>(&mut self, mut operation: F)
    where
        F: FnMut(usize, &mut f64),
    {
        for (index, value) in self.state.iter_mut().enumerate() {
            operation(index, value);
        }
        let mut index = N;
        for row in &mut self.sensitivities {
            for value in row {
                operation(index, value);
                index += 1;
            }
        }
    }

    fn zeros_like(&self) -> Self {
        Self::zeros()
    }

    fn zeros() -> Self {
        Self {
            state: [0.0; N],
            sensitivities: [[0.0; W]; N],
        }
    }

    fn mul_add_assign(&mut self, alpha: f64, other: &Self) {
        self.map_components_mut(|index, value| {
            *value += alpha * other.get_component(index);
        });
    }

    fn scale_mut(&mut self, alpha: f64) {
        self.map_components_mut(|_, value| *value *= alpha);
    }

    fn norm_squared(&self) -> f64 {
        let mut value = 0.0;
        self.for_each(|_, component| value += component * component);
        value
    }

    fn diff_norm_squared(&self, other: &Self) -> f64 {
        let mut value = 0.0;
        self.for_each(|index, component| {
            let difference = component - other.get_component(index);
            value += difference * difference;
        });
        value
    }

    fn error_norm(
        &self,
        new_state: &Self,
        error: &Self,
        absolute: &BackendTolerance<f64>,
        relative: &BackendTolerance<f64>,
    ) -> f64 {
        let mut value = 0.0;
        self.for_each(|index, component| {
            let scale = absolute[index]
                + relative[index] * component.abs().max(new_state.get_component(index).abs());
            let normalized = error.get_component(index) / scale;
            value += normalized * normalized;
        });
        value
    }

    fn error_norm_inf(
        &self,
        new_state: &Self,
        error: &Self,
        absolute: &BackendTolerance<f64>,
        relative: &BackendTolerance<f64>,
    ) -> f64 {
        let mut value: f64 = 0.0;
        self.for_each(|index, component| {
            let scale = absolute[index]
                + relative[index] * component.abs().max(new_state.get_component(index).abs());
            value = value.max((error.get_component(index) / scale).abs());
        });
        value
    }
}

struct AugmentedModel<'a, M, const N: usize, const P: usize, const W: usize> {
    model: &'a M,
    parameters: &'a [f64; P],
    parameter_seeds: &'a [[f64; W]; P],
    error: RefCell<Option<PykepError>>,
}

impl<M, const N: usize, const P: usize, const W: usize> ODE<f64, AugmentedState<N, W>>
    for AugmentedModel<'_, M, N, P, W>
where
    M: DifferentiableDynamicsModel<N, P>,
{
    fn diff(&self, time: f64, state: &AugmentedState<N, W>, derivative: &mut AugmentedState<N, W>) {
        if self.error.borrow().is_some() {
            derivative.fill(f64::NAN);
            return;
        }
        if let Err(error) =
            self.model
                .rhs(time, &state.state, self.parameters, &mut derivative.state)
        {
            *self.error.borrow_mut() = Some(error);
            derivative.fill(f64::NAN);
            return;
        }
        let mut state_jacobian = [[0.0; N]; N];
        let mut parameter_jacobian = [[0.0; P]; N];
        if let Err(error) = self.model.jacobians(
            time,
            &state.state,
            self.parameters,
            &mut state_jacobian,
            &mut parameter_jacobian,
        ) {
            *self.error.borrow_mut() = Some(error);
            derivative.fill(f64::NAN);
            return;
        }
        for row in 0..N {
            for seed in 0..W {
                let state_term = (0..N)
                    .map(|column| state_jacobian[row][column] * state.sensitivities[column][seed])
                    .sum::<f64>();
                let parameter_term = (0..P)
                    .map(|column| {
                        parameter_jacobian[row][column] * self.parameter_seeds[column][seed]
                    })
                    .sum::<f64>();
                derivative.sensitivities[row][seed] = state_term + parameter_term;
            }
        }
        if derivative
            .state
            .iter()
            .chain(derivative.sensitivities.iter().flat_map(|row| row.iter()))
            .any(|value| !value.is_finite())
        {
            *self.error.borrow_mut() = Some(PykepError::IntegrationFailure {
                model: M::NAME,
                reason: "variational right-hand side returned a non-finite derivative".into(),
            });
            derivative.fill(f64::NAN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AugmentedState, BackendState, BackendTolerance};

    #[test]
    fn augmented_state_backend_layout_and_arithmetic_are_consistent() {
        let mut state = AugmentedState {
            state: [1.0, 2.0],
            sensitivities: [[3.0, 4.0], [5.0, 6.0]],
        };
        assert_eq!(state.len(), 6);
        assert_eq!(state.get_component(0), 1.0);
        assert_eq!(state.get_component(5), 6.0);
        state.set_component(0, 2.0);
        state.set_component(5, 7.0);
        assert_eq!(state.state[0], 2.0);
        assert_eq!(state.sensitivities[1][1], 7.0);

        let zero = state.zeros_like();
        assert_eq!(zero, AugmentedState::zeros());
        let mut arithmetic = state.clone();
        arithmetic.mul_add_assign(2.0, &state);
        arithmetic.scale_mut(0.5);
        for index in 0..state.len() {
            assert_eq!(
                arithmetic.get_component(index),
                1.5 * state.get_component(index)
            );
        }
        assert!(state.norm_squared() > 0.0);
        assert_eq!(state.diff_norm_squared(&state), 0.0);

        let error = AugmentedState {
            state: [0.1, -0.2],
            sensitivities: [[0.3, -0.4], [0.5, -0.6]],
        };
        let absolute = BackendTolerance::Scalar(1.0);
        let relative = BackendTolerance::Scalar(0.0);
        let squared = state.error_norm(&state, &error, &absolute, &relative);
        assert!((squared - 0.91).abs() < 1e-15);
        assert!((state.error_norm_inf(&state, &error, &absolute, &relative) - 0.6).abs() < 1e-15);
    }
}
