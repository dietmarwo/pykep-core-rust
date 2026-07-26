// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                          Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Generic zero-order-hold low-thrust leg.
//!
//! Adapted from `src/leg/zoh.cpp` in pinned pykep/kep3 3.0.1.

use core::marker::PhantomData;

use crate::dynamics::zoh::{
    ControlSchedule, ZeroOrderHoldModel, ZohCr3bpDynamics, ZohEquinoctialDynamics,
    ZohKeplerDynamics, ZohSolarSailDynamics,
};
use crate::error::ensure_finite;
use crate::integration::{
    Dop853, DynamicsModel, InitialValueProblem, IntegratorOptions, SensitivityProblem,
};
use crate::{PykepError, Result};

/// Mismatch derivatives in output-by-input row order.
///
/// These derivatives compose segment sensitivities whose built-in model
/// Jacobians use centered differences. Validation against the pinned oracle
/// therefore uses scaled tolerances up to `3e-5`, independently of the
/// integrator tolerance.
#[derive(Clone, Debug, PartialEq)]
pub struct ZohLegMismatchJacobian<const N: usize> {
    /// Derivative with respect to the initial state, shape `N × N`.
    pub initial_state: [[f64; N]; N],
    /// Derivative with respect to the final state, shape `N × N`.
    pub final_state: [[f64; N]; N],
    /// Derivative with respect to chronological flattened controls, shape
    /// `N × (control_dimension * segment_count)`.
    pub controls: Vec<Vec<f64>>,
    /// Derivative with respect to every grid node, shape
    /// `N × (segment_count + 1)`.
    pub time_grid: Vec<Vec<f64>>,
}

/// Per-segment state samples from both sides of a ZOH leg.
///
/// Backward segments appear in propagation order: final segment first,
/// followed by successively earlier segments.
#[derive(Clone, Debug, PartialEq)]
pub struct ZohLegHistory<const N: usize> {
    /// Forward segment samples in chronological order.
    pub forward: Vec<Vec<[f64; N]>>,
    /// Backward segment samples in reverse chronological order.
    pub backward: Vec<Vec<[f64; N]>>,
}

/// Validated generic zero-order-hold low-thrust leg.
///
/// `W` is the local variational width and must equal `N + C`. The public
/// aliases for the four built-in ZOH models select the correct dimensions.
///
/// ```
/// use pykep_core::dynamics::zoh::ZohKeplerDynamics;
/// use pykep_core::integration::IntegratorOptions;
/// use pykep_core::leg::ZohKeplerLeg;
///
/// let state = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
/// let leg = ZohKeplerLeg::new(
///     ZohKeplerDynamics,
///     state,
///     vec![[0.0; 4]],
///     state,
///     vec![0.0, 0.1],
///     [0.0],
///     0.5,
///     IntegratorOptions::default(),
/// )?;
/// assert!(leg.mismatch_constraints()?.iter().all(|value| value.is_finite()));
/// # Ok::<(), pykep_core::PykepError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ZohLeg<M, const N: usize, const C: usize, const K: usize, const P: usize, const W: usize>
{
    model: M,
    initial_state: [f64; N],
    final_state: [f64; N],
    schedule: ControlSchedule<C>,
    constants: [f64; K],
    cut: f64,
    forward_segments: usize,
    options: IntegratorOptions,
    parameter_dimension: PhantomData<[(); P]>,
}

/// Seven-state, four-control normalized Kepler ZOH leg.
pub type ZohKeplerLeg = ZohLeg<ZohKeplerDynamics, 7, 4, 1, 5, 11>;
/// Seven-state, four-control CR3BP ZOH leg.
pub type ZohCr3bpLeg = ZohLeg<ZohCr3bpDynamics, 7, 4, 2, 6, 11>;
/// Seven-state, four-control modified-equinoctial ZOH leg.
pub type ZohEquinoctialLeg = ZohLeg<ZohEquinoctialDynamics, 7, 4, 1, 5, 11>;
/// Six-state, two-control ideal solar-sail ZOH leg.
pub type ZohSolarSailLeg = ZohLeg<ZohSolarSailDynamics, 6, 2, 1, 3, 8>;

struct ReverseTimeModel<'a, M> {
    model: &'a M,
    origin: f64,
}

impl<M, const N: usize, const P: usize> DynamicsModel<N, P> for ReverseTimeModel<'_, M>
where
    M: DynamicsModel<N, P>,
{
    const NAME: &'static str = M::NAME;

    fn validate(&self, time: f64, state: &[f64; N], parameters: &[f64; P]) -> Result<()> {
        self.model.validate(self.origin - time, state, parameters)
    }

    fn rhs(
        &self,
        time: f64,
        state: &[f64; N],
        parameters: &[f64; P],
        derivative: &mut [f64; N],
    ) -> Result<()> {
        self.model
            .rhs(self.origin - time, state, parameters, derivative)?;
        for value in derivative {
            *value = -*value;
        }
        Ok(())
    }
}

impl<M, const N: usize, const C: usize, const K: usize, const P: usize, const W: usize>
    ZohLeg<M, N, C, K, P, W>
where
    M: ZeroOrderHoldModel<N, C, K, P>,
{
    /// Constructs and validates a ZOH leg.
    ///
    /// Controls and grid nodes are chronological. The number propagated
    /// forward is `floor(segment_count * cut)`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, values, model endpoints,
    /// integration settings, or cuts outside `[0,1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: M,
        initial_state: [f64; N],
        controls: Vec<[f64; C]>,
        final_state: [f64; N],
        time_grid: Vec<f64>,
        constants: [f64; K],
        cut: f64,
        options: IntegratorOptions,
    ) -> Result<Self> {
        if W != N + C {
            return Err(PykepError::InvalidInput {
                parameter: "variational_width",
                reason: format!("must equal state + control dimensions ({})", N + C),
            });
        }
        ensure_finite("cut", cut)?;
        if !(0.0..=1.0).contains(&cut) {
            return Err(PykepError::InvalidInput {
                parameter: "cut",
                reason: "must lie in 0..=1".into(),
            });
        }
        options.validate()?;
        let schedule = ControlSchedule::new(time_grid, controls)?;
        model.validate(
            schedule.initial_time(),
            &initial_state,
            &M::parameters(schedule.controls()[0], constants),
        )?;
        model.validate(
            schedule.final_time(),
            &final_state,
            &M::parameters(schedule.controls()[schedule.len() - 1], constants),
        )?;
        let forward_segments = (schedule.len() as f64 * cut) as usize;
        Ok(Self {
            model,
            initial_state,
            final_state,
            schedule,
            constants,
            cut,
            forward_segments,
            options,
            parameter_dimension: PhantomData,
        })
    }

    /// Initial endpoint state.
    pub const fn initial_state(&self) -> [f64; N] {
        self.initial_state
    }

    /// Final endpoint state.
    pub const fn final_state(&self) -> [f64; N] {
        self.final_state
    }

    /// Chronological control schedule.
    pub const fn schedule(&self) -> &ControlSchedule<C> {
        &self.schedule
    }

    /// Constant model parameters in model-specific order.
    pub const fn constants(&self) -> [f64; K] {
        self.constants
    }

    /// Fractional cut in `[0,1]`.
    pub const fn cut(&self) -> f64 {
        self.cut
    }

    /// Integration settings applied independently to every segment.
    pub const fn integrator_options(&self) -> IntegratorOptions {
        self.options
    }

    /// Total number of segments.
    pub fn segment_count(&self) -> usize {
        self.schedule.len()
    }

    /// Number of segments propagated from the initial state.
    pub const fn forward_segment_count(&self) -> usize {
        self.forward_segments
    }

    /// Number of segments propagated backward from the final state.
    pub fn backward_segment_count(&self) -> usize {
        self.segment_count() - self.forward_segments
    }

    /// Evaluates `forward_state - backward_state` at the cut.
    ///
    /// # Errors
    ///
    /// Returns an integration error containing direction, segment index, and
    /// segment time interval.
    pub fn mismatch_constraints(&self) -> Result<[f64; N]> {
        let forward = self.propagate_nominal_half(true)?;
        let backward = self.propagate_nominal_half(false)?;
        Ok(core::array::from_fn(|row| forward[row] - backward[row]))
    }

    /// Evaluates all four first-order mismatch derivative groups.
    ///
    /// The matrices use output-by-input rows and chronological control
    /// blocks, matching the upstream ZOH-leg contract. Built-in model
    /// Jacobians are centered finite differences with a relative step of
    /// `3e-6`; the validation ceiling is a scaled `3e-5`, even when the
    /// propagation tolerance is tighter.
    ///
    /// # Errors
    ///
    /// Returns an integration error containing direction, segment index, and
    /// segment time interval.
    pub fn mismatch_jacobian(&self) -> Result<ZohLegMismatchJacobian<N>> {
        let segment_count = self.segment_count();
        let mut initial_state = identity();
        let mut final_state = identity();
        let mut controls = vec![vec![0.0; C * segment_count]; N];
        let mut time_grid = vec![vec![0.0; segment_count + 1]; N];

        let forward = self.segment_variations(true)?;
        let forward_maps = forward
            .iter()
            .map(|segment| segment.transition)
            .collect::<Vec<_>>();
        let forward_suffix = suffix_products(&forward_maps);
        if !forward.is_empty() {
            initial_state = forward_suffix[0];
            for (index, segment) in forward.iter().enumerate() {
                let block = matrix_times_columns(&forward_suffix[index + 1], &segment.control);
                write_control_block(&mut controls, index, &block, 1.0);
            }
            add_column(
                &mut time_grid,
                0,
                &matrix_times_vector(&forward_suffix[1], &forward[0].dynamics),
                -1.0,
            );
            let last = forward.len() - 1;
            add_column(
                &mut time_grid,
                forward.len(),
                &matrix_times_vector(&forward_suffix[forward.len()], &forward[last].dynamics),
                1.0,
            );
            for index in 1..forward.len() {
                let previous_at_next_start =
                    matrix_times_vector(&forward[index].transition, &forward[index - 1].dynamics);
                let jump = core::array::from_fn(|row| {
                    previous_at_next_start[row] - forward[index].dynamics[row]
                });
                let derivative = matrix_times_vector(&forward_suffix[index + 1], &jump);
                add_column(&mut time_grid, index, &derivative, 1.0);
            }
        }

        let backward = self.segment_variations(false)?;
        let backward_maps = backward
            .iter()
            .map(|segment| segment.transition)
            .collect::<Vec<_>>();
        let backward_suffix = suffix_products(&backward_maps);
        if !backward.is_empty() {
            final_state = backward_suffix[0].map(|row| row.map(|value| -value));
            for (index, segment) in backward.iter().enumerate() {
                let block = matrix_times_columns(&backward_suffix[index + 1], &segment.control);
                let chronological = segment_count - 1 - index;
                write_control_block(&mut controls, chronological, &block, -1.0);
            }
            add_column(
                &mut time_grid,
                segment_count,
                &matrix_times_vector(&backward_suffix[1], &backward[0].dynamics),
                1.0,
            );
            let last = backward.len() - 1;
            add_column(
                &mut time_grid,
                segment_count - backward.len(),
                &matrix_times_vector(&backward_suffix[backward.len()], &backward[last].dynamics),
                -1.0,
            );
            for index in 1..backward.len() {
                let previous_at_next_start =
                    matrix_times_vector(&backward[index].transition, &backward[index - 1].dynamics);
                let jump = core::array::from_fn(|row| {
                    previous_at_next_start[row] - backward[index].dynamics[row]
                });
                let derivative = matrix_times_vector(&backward_suffix[index + 1], &jump);
                add_column(&mut time_grid, segment_count - index, &derivative, -1.0);
            }
        } else {
            final_state = final_state.map(|row| row.map(|value| -value));
        }

        validate_rows("ZOH leg control Jacobian", &controls)?;
        validate_rows("ZOH leg time-grid Jacobian", &time_grid)?;
        Ok(ZohLegMismatchJacobian {
            initial_state,
            final_state,
            controls,
            time_grid,
        })
    }

    /// Samples every propagated segment at `samples_per_segment` uniformly
    /// spaced times, including both endpoints.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than two samples are requested or when a
    /// segment cannot be integrated.
    pub fn state_history(&self, samples_per_segment: usize) -> Result<ZohLegHistory<N>> {
        if samples_per_segment < 2 {
            return Err(PykepError::InvalidInput {
                parameter: "samples_per_segment",
                reason: "must be at least two".into(),
            });
        }
        Ok(ZohLegHistory {
            forward: self.sample_half(true, samples_per_segment)?,
            backward: self.sample_half(false, samples_per_segment)?,
        })
    }

    fn propagate_nominal_half(&self, forward: bool) -> Result<[f64; N]> {
        let mut state = if forward {
            self.initial_state
        } else {
            self.final_state
        };
        if forward {
            for index in 0..self.forward_segments {
                state = self.propagate_segment(index, state, true)?;
            }
        } else {
            for index in (self.forward_segments..self.segment_count()).rev() {
                state = self.propagate_segment(index, state, false)?;
            }
        }
        Ok(state)
    }

    fn propagate_segment(&self, index: usize, state: [f64; N], forward: bool) -> Result<[f64; N]> {
        let (start, end) = self.segment_interval(index, forward);
        Dop853
            .propagate(
                &self.model,
                InitialValueProblem::new(
                    start,
                    state,
                    end,
                    M::parameters(self.schedule.controls()[index], self.constants),
                ),
                self.options,
            )
            .map(|result| result.state)
            .map_err(|error| segment_error::<M, N, P>(forward, index, start, end, error))
    }

    fn segment_variations(&self, forward: bool) -> Result<Vec<SegmentVariation<N, C>>> {
        let mut state = if forward {
            self.initial_state
        } else {
            self.final_state
        };
        let indices: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(0..self.forward_segments)
        } else {
            Box::new((self.forward_segments..self.segment_count()).rev())
        };
        let mut output = Vec::with_capacity(if forward {
            self.forward_segment_count()
        } else {
            self.backward_segment_count()
        });
        for index in indices {
            let (start, end) = self.segment_interval(index, forward);
            let mut initial_sensitivities = [[0.0; W]; N];
            for (row, values) in initial_sensitivities.iter_mut().enumerate() {
                values[row] = 1.0;
            }
            let mut control_seeds = [[0.0; W]; C];
            for (row, values) in control_seeds.iter_mut().enumerate() {
                values[N + row] = 1.0;
            }
            let parameters = M::parameters(self.schedule.controls()[index], self.constants);
            let result = Dop853
                .propagate_with_sensitivities(
                    &self.model,
                    SensitivityProblem {
                        nominal: InitialValueProblem::new(start, state, end, parameters),
                        initial_sensitivities,
                        parameter_seeds: M::parameter_seeds(control_seeds, [[0.0; W]; K]),
                    },
                    self.options,
                )
                .map_err(|error| segment_error::<M, N, P>(forward, index, start, end, error))?;
            state = result.state;
            let transition = core::array::from_fn(|row| {
                core::array::from_fn(|column| result.sensitivities[row][column])
            });
            let control = core::array::from_fn(|row| {
                core::array::from_fn(|column| result.sensitivities[row][N + column])
            });
            let mut dynamics = [0.0; N];
            self.model
                .rhs(end, &state, &parameters, &mut dynamics)
                .map_err(|error| segment_error::<M, N, P>(forward, index, start, end, error))?;
            output.push(SegmentVariation {
                transition,
                control,
                dynamics,
            });
        }
        Ok(output)
    }

    fn sample_half(&self, forward: bool, samples: usize) -> Result<Vec<Vec<[f64; N]>>> {
        let mut state = if forward {
            self.initial_state
        } else {
            self.final_state
        };
        let indices: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(0..self.forward_segments)
        } else {
            Box::new((self.forward_segments..self.segment_count()).rev())
        };
        let mut result = Vec::new();
        for index in indices {
            let (start, end) = self.segment_interval(index, forward);
            let times = (0..samples)
                .map(|sample| {
                    start + (end - start) * sample as f64 / (samples.saturating_sub(1)) as f64
                })
                .collect::<Vec<_>>();
            let parameters = M::parameters(self.schedule.controls()[index], self.constants);
            let sample_states = if forward {
                Dop853
                    .propagate_dense(
                        &self.model,
                        InitialValueProblem::new(start, state, end, parameters),
                        &times,
                        self.options,
                    )
                    .map_err(|error| segment_error::<M, N, P>(forward, index, start, end, error))?
                    .states
            } else {
                // differential-equations 0.6.1 can panic when a decreasing
                // dense-output request rounds just outside the completed step,
                // so integrate the equivalent dx/dtau = -f(start - tau, x)
                // problem on an increasing interval.
                let evaluation_times = times
                    .iter()
                    .map(|&sample_time| start - sample_time)
                    .collect::<Vec<_>>();
                Dop853
                    .propagate_dense(
                        &ReverseTimeModel {
                            model: &self.model,
                            origin: start,
                        },
                        InitialValueProblem::new(0.0, state, start - end, parameters),
                        &evaluation_times,
                        self.options,
                    )
                    .map_err(|error| segment_error::<M, N, P>(forward, index, start, end, error))?
                    .states
            };
            state = sample_states.last().copied().ok_or_else(|| {
                segment_error::<M, N, P>(
                    forward,
                    index,
                    start,
                    end,
                    PykepError::IntegrationFailure {
                        model: M::NAME,
                        reason: "dense propagation returned no samples".into(),
                    },
                )
            })?;
            result.push(sample_states);
        }
        Ok(result)
    }

    fn segment_interval(&self, index: usize, forward: bool) -> (f64, f64) {
        if forward {
            (
                self.schedule.boundaries()[index],
                self.schedule.boundaries()[index + 1],
            )
        } else {
            (
                self.schedule.boundaries()[index + 1],
                self.schedule.boundaries()[index],
            )
        }
    }
}

/// Evaluates a batch of already validated ZOH legs in input order.
///
/// # Errors
///
/// Returns the first segment-contextual integration error.
pub fn evaluate_zoh_mismatch_batch<
    M,
    const N: usize,
    const C: usize,
    const K: usize,
    const P: usize,
    const W: usize,
>(
    legs: &[ZohLeg<M, N, C, K, P, W>],
) -> Result<Vec<[f64; N]>>
where
    M: ZeroOrderHoldModel<N, C, K, P>,
{
    legs.iter().map(ZohLeg::mismatch_constraints).collect()
}

#[derive(Clone, Copy)]
struct SegmentVariation<const N: usize, const C: usize> {
    transition: [[f64; N]; N],
    control: [[f64; C]; N],
    dynamics: [f64; N],
}

fn identity<const N: usize>() -> [[f64; N]; N] {
    core::array::from_fn(|row| core::array::from_fn(|column| f64::from(row == column)))
}

fn matrix_product<const N: usize>(left: &[[f64; N]; N], right: &[[f64; N]; N]) -> [[f64; N]; N] {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| {
            (0..N)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

fn suffix_products<const N: usize>(matrices: &[[[f64; N]; N]]) -> Vec<[[f64; N]; N]> {
    let mut suffix = vec![identity(); matrices.len() + 1];
    for index in (0..matrices.len()).rev() {
        suffix[index] = matrix_product(&suffix[index + 1], &matrices[index]);
    }
    suffix
}

fn matrix_times_columns<const N: usize, const C: usize>(
    matrix: &[[f64; N]; N],
    columns: &[[f64; C]; N],
) -> [[f64; C]; N] {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| {
            (0..N)
                .map(|inner| matrix[row][inner] * columns[inner][column])
                .sum()
        })
    })
}

fn matrix_times_vector<const N: usize>(matrix: &[[f64; N]; N], vector: &[f64; N]) -> [f64; N] {
    core::array::from_fn(|row| {
        (0..N)
            .map(|column| matrix[row][column] * vector[column])
            .sum()
    })
}

fn write_control_block<const N: usize, const C: usize>(
    destination: &mut [Vec<f64>],
    segment: usize,
    block: &[[f64; C]; N],
    scale: f64,
) {
    for row in 0..N {
        for column in 0..C {
            destination[row][C * segment + column] = scale * block[row][column];
        }
    }
}

fn add_column<const N: usize>(
    destination: &mut [Vec<f64>],
    column: usize,
    values: &[f64; N],
    scale: f64,
) {
    for row in 0..N {
        destination[row][column] += scale * values[row];
    }
}

fn validate_rows(operation: &'static str, rows: &[Vec<f64>]) -> Result<()> {
    if rows.iter().flatten().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PykepError::NumericalOverflow { operation })
    }
}

fn segment_error<M, const N: usize, const P: usize>(
    forward: bool,
    index: usize,
    start: f64,
    end: f64,
    error: PykepError,
) -> PykepError
where
    M: DynamicsModel<N, P>,
{
    let direction = if forward { "forward" } else { "backward" };
    PykepError::IntegrationFailure {
        model: M::NAME,
        reason: format!("{direction} segment {index} [{start}, {end}]: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{ZohKeplerDynamics, ZohLeg};
    use crate::integration::IntegratorOptions;

    #[test]
    fn cut_boundaries_have_exact_segment_counts() {
        let make = |cut| {
            ZohLeg::<ZohKeplerDynamics, 7, 4, 1, 5, 11>::new(
                ZohKeplerDynamics,
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                vec![[0.0; 4]; 5],
                [0.5, 0.8, 0.0, -0.8, 0.5, 0.0, 1.0],
                vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                [0.1],
                cut,
                IntegratorOptions::default(),
            )
            .unwrap()
        };
        assert_eq!(make(0.0).forward_segment_count(), 0);
        assert_eq!(make(0.5).forward_segment_count(), 2);
        assert_eq!(make(1.0).forward_segment_count(), 5);
    }
}
