// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                          Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Zero-order-hold Kepler, CR3BP, equinoctial, and solar-sail dynamics.
//!
//! This file adapts `src/ta/zoh_kep.cpp`, `zoh_cr3bp.cpp`, `zoh_eq.cpp`, and
//! `zoh_ss.cpp` from the pinned pykep/kep3 source into evaluated models.

use crate::error::ensure_finite;
use crate::integration::{
    DifferentiableDynamicsModel, Dop853, DynamicsModel, InitialValueProblem, IntegrationStats,
    IntegratorOptions, Propagation, SensitivityProblem, SensitivityPropagation, Termination,
};
use crate::{PykepError, Result};

/// A validated piecewise-constant control history.
///
/// `controls[i]` owns `[boundaries[i], boundaries[i + 1])`; the final
/// boundary is included in the final segment. Boundaries are strictly
/// increasing. Backward propagation traverses the same intervals in reverse,
/// using the interval to the left of each encountered switching time.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlSchedule<const C: usize> {
    boundaries: Vec<f64>,
    controls: Vec<[f64; C]>,
}

impl<const C: usize> ControlSchedule<C> {
    /// Constructs and validates a zero-order-hold control schedule.
    ///
    /// # Errors
    ///
    /// Returns an error unless there is at least one non-empty segment,
    /// `boundaries.len() == controls.len() + 1`, every value is finite, and
    /// the boundaries are strictly increasing.
    pub fn new(boundaries: Vec<f64>, controls: Vec<[f64; C]>) -> Result<Self> {
        if C == 0 {
            return Err(PykepError::InvalidInput {
                parameter: "control_dimension",
                reason: "must be greater than zero".into(),
            });
        }
        if controls.is_empty() {
            return Err(PykepError::InvalidInput {
                parameter: "controls",
                reason: "must contain at least one segment".into(),
            });
        }
        if boundaries.len() != controls.len() + 1 {
            return Err(PykepError::DimensionMismatch {
                expected: controls.len() + 1,
                actual: boundaries.len(),
            });
        }
        for &boundary in &boundaries {
            ensure_finite("boundaries", boundary)?;
        }
        if boundaries.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PykepError::InvalidInput {
                parameter: "boundaries",
                reason: "must be strictly increasing".into(),
            });
        }
        for control in &controls {
            for &value in control {
                ensure_finite("controls", value)?;
            }
        }
        Ok(Self {
            boundaries,
            controls,
        })
    }

    /// Returns the switching boundaries.
    pub fn boundaries(&self) -> &[f64] {
        &self.boundaries
    }

    /// Returns the controls in chronological segment order.
    pub fn controls(&self) -> &[[f64; C]] {
        &self.controls
    }

    /// Returns the number of control segments.
    pub fn len(&self) -> usize {
        self.controls.len()
    }

    /// Returns whether the schedule contains no segments.
    ///
    /// Valid schedules are never empty; this method is supplied alongside
    /// [`Self::len`] for collection-like APIs.
    pub fn is_empty(&self) -> bool {
        self.controls.is_empty()
    }

    /// Returns the first boundary.
    pub fn initial_time(&self) -> f64 {
        self.boundaries[0]
    }

    /// Returns the final boundary.
    pub fn final_time(&self) -> f64 {
        self.boundaries[self.boundaries.len() - 1]
    }

    /// Returns the right-continuous control at a time inside the schedule.
    ///
    /// At an internal switching time, the later segment owns the boundary.
    /// The final boundary returns the final segment's control.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite times or times outside the closed
    /// schedule interval.
    pub fn control_at(&self, time: f64) -> Result<[f64; C]> {
        ensure_finite("time", time)?;
        if time < self.initial_time() || time > self.final_time() {
            return Err(PykepError::InvalidInput {
                parameter: "time",
                reason: "must lie inside the control schedule".into(),
            });
        }
        if time == self.final_time() {
            return Ok(self.controls[self.controls.len() - 1]);
        }
        let index = self
            .boundaries
            .partition_point(|boundary| *boundary <= time)
            - 1;
        Ok(self.controls[index])
    }
}

/// Seed directions carried through a segmented ZOH propagation.
///
/// State sensitivities remain continuous at switches. Each segment has its
/// own control seed matrix, so a future segment's control columns are exactly
/// zero before that segment becomes active.
#[derive(Clone, Debug, PartialEq)]
pub struct ZohSensitivitySeeds<const N: usize, const C: usize, const K: usize, const W: usize> {
    /// Initial `dstate/dseed`.
    pub initial_state: [[f64; W]; N],
    /// One `dcontrol/dseed` matrix per schedule segment.
    pub segment_controls: Vec<[[f64; W]; C]>,
    /// Constant-model-parameter seed matrix.
    pub constants: [[f64; W]; K],
}

/// Model contract used by the common segmented propagators.
pub trait ZeroOrderHoldModel<const N: usize, const C: usize, const K: usize, const P: usize>:
    DifferentiableDynamicsModel<N, P>
{
    /// Combines a segment control and model constants in upstream parameter
    /// order.
    fn parameters(control: [f64; C], constants: [f64; K]) -> [f64; P];

    /// Combines control and constant seed matrices in upstream parameter
    /// order.
    fn parameter_seeds<const W: usize>(
        control: [[f64; W]; C],
        constants: [[f64; W]; K],
    ) -> [[f64; W]; P];
}

/// Propagates a ZOH schedule from its first boundary to its last.
///
/// Each segment is integrated exactly once and the active control is copied
/// into a fixed-size parameter array before integration. No control search or
/// allocation occurs inside right-hand-side evaluation.
///
/// # Errors
///
/// Returns a model-domain or integration error.
pub fn propagate_schedule<M, const N: usize, const C: usize, const K: usize, const P: usize>(
    model: &M,
    schedule: &ControlSchedule<C>,
    initial_state: [f64; N],
    constants: [f64; K],
    options: IntegratorOptions,
) -> Result<Propagation<N>>
where
    M: ZeroOrderHoldModel<N, C, K, P>,
{
    propagate_schedule_direction(model, schedule, initial_state, constants, options, false)
}

/// Propagates a ZOH schedule backward from its last boundary to its first.
///
/// # Errors
///
/// Returns a model-domain or integration error.
pub fn propagate_schedule_backward<
    M,
    const N: usize,
    const C: usize,
    const K: usize,
    const P: usize,
>(
    model: &M,
    schedule: &ControlSchedule<C>,
    final_state: [f64; N],
    constants: [f64; K],
    options: IntegratorOptions,
) -> Result<Propagation<N>>
where
    M: ZeroOrderHoldModel<N, C, K, P>,
{
    propagate_schedule_direction(model, schedule, final_state, constants, options, true)
}

/// Propagates arbitrary seed directions through a complete ZOH schedule.
///
/// Runtime is linear in the number of segments for a fixed sensitivity width:
/// each segment advances the current augmented state once.
///
/// # Errors
///
/// Returns an error for a seed/schedule length mismatch, invalid seeds, or a
/// model/integration failure.
pub fn propagate_schedule_with_sensitivities<
    M,
    const N: usize,
    const C: usize,
    const K: usize,
    const P: usize,
    const W: usize,
>(
    model: &M,
    schedule: &ControlSchedule<C>,
    initial_state: [f64; N],
    constants: [f64; K],
    seeds: &ZohSensitivitySeeds<N, C, K, W>,
    options: IntegratorOptions,
) -> Result<SensitivityPropagation<N, W>>
where
    M: ZeroOrderHoldModel<N, C, K, P>,
{
    propagate_schedule_sensitivities_direction(
        model,
        schedule,
        initial_state,
        constants,
        seeds,
        options,
        false,
    )
}

/// Propagates arbitrary seed directions backward through a ZOH schedule.
///
/// # Errors
///
/// Returns an error for a seed/schedule length mismatch, invalid seeds, or a
/// model/integration failure.
pub fn propagate_schedule_with_sensitivities_backward<
    M,
    const N: usize,
    const C: usize,
    const K: usize,
    const P: usize,
    const W: usize,
>(
    model: &M,
    schedule: &ControlSchedule<C>,
    final_state: [f64; N],
    constants: [f64; K],
    seeds: &ZohSensitivitySeeds<N, C, K, W>,
    options: IntegratorOptions,
) -> Result<SensitivityPropagation<N, W>>
where
    M: ZeroOrderHoldModel<N, C, K, P>,
{
    propagate_schedule_sensitivities_direction(
        model,
        schedule,
        final_state,
        constants,
        seeds,
        options,
        true,
    )
}

fn propagate_schedule_direction<M, const N: usize, const C: usize, const K: usize, const P: usize>(
    model: &M,
    schedule: &ControlSchedule<C>,
    initial_state: [f64; N],
    constants: [f64; K],
    options: IntegratorOptions,
    backward: bool,
) -> Result<Propagation<N>>
where
    M: ZeroOrderHoldModel<N, C, K, P>,
{
    let mut state = initial_state;
    let mut statistics = IntegrationStats::default();
    if backward {
        for index in (0..schedule.len()).rev() {
            let result = Dop853.propagate(
                model,
                InitialValueProblem::new(
                    schedule.boundaries[index + 1],
                    state,
                    schedule.boundaries[index],
                    M::parameters(schedule.controls[index], constants),
                ),
                options,
            )?;
            state = result.state;
            add_statistics(&mut statistics, result.stats);
        }
    } else {
        for index in 0..schedule.len() {
            let result = Dop853.propagate(
                model,
                InitialValueProblem::new(
                    schedule.boundaries[index],
                    state,
                    schedule.boundaries[index + 1],
                    M::parameters(schedule.controls[index], constants),
                ),
                options,
            )?;
            state = result.state;
            add_statistics(&mut statistics, result.stats);
        }
    }
    Ok(Propagation {
        time: if backward {
            schedule.initial_time()
        } else {
            schedule.final_time()
        },
        state,
        stats: statistics,
        termination: Termination::FinalTime,
    })
}

#[allow(clippy::too_many_arguments)]
fn propagate_schedule_sensitivities_direction<
    M,
    const N: usize,
    const C: usize,
    const K: usize,
    const P: usize,
    const W: usize,
>(
    model: &M,
    schedule: &ControlSchedule<C>,
    initial_state: [f64; N],
    constants: [f64; K],
    seeds: &ZohSensitivitySeeds<N, C, K, W>,
    options: IntegratorOptions,
    backward: bool,
) -> Result<SensitivityPropagation<N, W>>
where
    M: ZeroOrderHoldModel<N, C, K, P>,
{
    if seeds.segment_controls.len() != schedule.len() {
        return Err(PykepError::DimensionMismatch {
            expected: schedule.len(),
            actual: seeds.segment_controls.len(),
        });
    }
    let mut state = initial_state;
    let mut sensitivities = seeds.initial_state;
    let mut statistics = IntegrationStats::default();
    let mut run_segment = |index: usize, start: f64, end: f64| -> Result<()> {
        let result = Dop853.propagate_with_sensitivities(
            model,
            SensitivityProblem {
                nominal: InitialValueProblem::new(
                    start,
                    state,
                    end,
                    M::parameters(schedule.controls[index], constants),
                ),
                initial_sensitivities: sensitivities,
                parameter_seeds: M::parameter_seeds(seeds.segment_controls[index], seeds.constants),
            },
            options,
        )?;
        state = result.state;
        sensitivities = result.sensitivities;
        add_statistics(&mut statistics, result.stats);
        Ok(())
    };
    if backward {
        for index in (0..schedule.len()).rev() {
            run_segment(
                index,
                schedule.boundaries[index + 1],
                schedule.boundaries[index],
            )?;
        }
    } else {
        for index in 0..schedule.len() {
            run_segment(
                index,
                schedule.boundaries[index],
                schedule.boundaries[index + 1],
            )?;
        }
    }
    Ok(SensitivityPropagation {
        time: if backward {
            schedule.initial_time()
        } else {
            schedule.final_time()
        },
        state,
        sensitivities,
        stats: statistics,
    })
}

fn add_statistics(total: &mut IntegrationStats, segment: IntegrationStats) {
    total.rhs_evaluations += segment.rhs_evaluations;
    total.accepted_steps += segment.accepted_steps;
    total.rejected_steps += segment.rejected_steps;
}

/// Seven-state Cartesian low-thrust Kepler dynamics with normalized `mu = 1`.
///
/// State order is `[x, y, z, vx, vy, vz, mass]`; parameters are
/// `[thrust, ix, iy, iz, c]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZohKeplerDynamics;

impl DynamicsModel<7, 5> for ZohKeplerDynamics {
    const NAME: &'static str = "ZOH Kepler dynamics";

    fn validate(&self, time: f64, state: &[f64; 7], parameters: &[f64; 5]) -> Result<()> {
        validate_finite(time, state, parameters)?;
        validate_positive_state(state[6], "mass")?;
        validate_radius(&state[..3], "ZOH Kepler radius")?;
        Ok(())
    }

    fn rhs(
        &self,
        time: f64,
        state: &[f64; 7],
        parameters: &[f64; 5],
        derivative: &mut [f64; 7],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let radius_squared = squared_norm(&state[..3]);
        let gravity = -1.0 / (radius_squared * radius_squared.sqrt());
        let [thrust, ix, iy, iz, c] = *parameters;
        *derivative = [
            state[3],
            state[4],
            state[5],
            gravity * state[0] + thrust * ix / state[6],
            gravity * state[1] + thrust * iy / state[6],
            gravity * state[2] + thrust * iz / state[6],
            regularized_mass_flow(state[6], thrust, c),
        ];
        validate_rhs(Self::NAME, derivative)
    }
}

impl DifferentiableDynamicsModel<7, 5> for ZohKeplerDynamics {
    fn jacobians(
        &self,
        time: f64,
        state: &[f64; 7],
        parameters: &[f64; 5],
        state_jacobian: &mut [[f64; 7]; 7],
        parameter_jacobian: &mut [[f64; 5]; 7],
    ) -> Result<()> {
        numerical_jacobians(
            self,
            time,
            state,
            parameters,
            state_jacobian,
            parameter_jacobian,
            &[6],
        )
    }
}

impl ZeroOrderHoldModel<7, 4, 1, 5> for ZohKeplerDynamics {
    fn parameters(control: [f64; 4], constants: [f64; 1]) -> [f64; 5] {
        [control[0], control[1], control[2], control[3], constants[0]]
    }

    fn parameter_seeds<const W: usize>(
        control: [[f64; W]; 4],
        constants: [[f64; W]; 1],
    ) -> [[f64; W]; 5] {
        [control[0], control[1], control[2], control[3], constants[0]]
    }
}

/// Seven-state low-thrust CR3BP dynamics in the synodic frame.
///
/// Parameters are `[thrust, ix, iy, iz, c, mu]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZohCr3bpDynamics;

impl DynamicsModel<7, 6> for ZohCr3bpDynamics {
    const NAME: &'static str = "ZOH CR3BP dynamics";

    fn validate(&self, time: f64, state: &[f64; 7], parameters: &[f64; 6]) -> Result<()> {
        validate_finite(time, state, parameters)?;
        validate_positive_state(state[6], "mass")?;
        validate_mass_fraction(parameters[5])?;
        let mu = parameters[5];
        validate_radius(
            &[state[0] + mu, state[1], state[2]],
            "ZOH CR3BP primary-one distance",
        )?;
        validate_radius(
            &[state[0] + mu - 1.0, state[1], state[2]],
            "ZOH CR3BP primary-two distance",
        )?;
        Ok(())
    }

    fn rhs(
        &self,
        time: f64,
        state: &[f64; 7],
        parameters: &[f64; 6],
        derivative: &mut [f64; 7],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let [thrust, ix, iy, iz, c, mu] = *parameters;
        let d1 = [state[0] + mu, state[1], state[2]];
        let d2 = [state[0] + mu - 1.0, state[1], state[2]];
        let r1_squared = squared_norm(&d1);
        let r2_squared = squared_norm(&d2);
        let primary = (1.0 - mu) / (r1_squared * r1_squared.sqrt());
        let secondary = mu / (r2_squared * r2_squared.sqrt());
        *derivative = [
            state[3],
            state[4],
            state[5],
            2.0 * state[4] + state[0] - primary * d1[0] - secondary * d2[0]
                + thrust * ix / state[6],
            -2.0 * state[3] + state[1] - (primary + secondary) * state[1] + thrust * iy / state[6],
            -(primary + secondary) * state[2] + thrust * iz / state[6],
            regularized_mass_flow(state[6], thrust, c),
        ];
        validate_rhs(Self::NAME, derivative)
    }
}

impl DifferentiableDynamicsModel<7, 6> for ZohCr3bpDynamics {
    fn jacobians(
        &self,
        time: f64,
        state: &[f64; 7],
        parameters: &[f64; 6],
        state_jacobian: &mut [[f64; 7]; 7],
        parameter_jacobian: &mut [[f64; 6]; 7],
    ) -> Result<()> {
        numerical_jacobians(
            self,
            time,
            state,
            parameters,
            state_jacobian,
            parameter_jacobian,
            &[6],
        )
    }
}

impl ZeroOrderHoldModel<7, 4, 2, 6> for ZohCr3bpDynamics {
    fn parameters(control: [f64; 4], constants: [f64; 2]) -> [f64; 6] {
        [
            control[0],
            control[1],
            control[2],
            control[3],
            constants[0],
            constants[1],
        ]
    }

    fn parameter_seeds<const W: usize>(
        control: [[f64; W]; 4],
        constants: [[f64; W]; 2],
    ) -> [[f64; W]; 6] {
        [
            control[0],
            control[1],
            control[2],
            control[3],
            constants[0],
            constants[1],
        ]
    }
}

/// Seven-state modified-equinoctial low-thrust dynamics with normalized
/// `mu = 1`.
///
/// State order is `[p, f, g, h, k, L, mass]`; control is
/// `[thrust, i_r, i_t, i_n]` and the sole constant is mass-flow coefficient
/// `c`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZohEquinoctialDynamics;

impl DynamicsModel<7, 5> for ZohEquinoctialDynamics {
    const NAME: &'static str = "ZOH equinoctial dynamics";

    fn validate(&self, time: f64, state: &[f64; 7], parameters: &[f64; 5]) -> Result<()> {
        validate_finite(time, state, parameters)?;
        validate_positive_state(state[0], "semilatus_rectum")?;
        validate_positive_state(state[6], "mass")?;
        let w = 1.0 + state[1] * state[5].cos() + state[2] * state[5].sin();
        if w == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "ZOH equinoctial radial denominator",
            });
        }
        Ok(())
    }

    fn rhs(
        &self,
        time: f64,
        state: &[f64; 7],
        parameters: &[f64; 5],
        derivative: &mut [f64; 7],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let [p, f, g, h, k, longitude, mass] = *state;
        let [thrust, radial, transverse, normal, c] = *parameters;
        let sine = longitude.sin();
        let cosine = longitude.cos();
        let w = 1.0 + f * cosine + g * sine;
        let s2 = 1.0 + h * h + k * k;
        let hsk = h * sine - k * cosine;
        let sqrt_p = p.sqrt();
        let radial_thrust = radial * thrust;
        let transverse_thrust = transverse * thrust;
        let normal_thrust = normal * thrust;
        *derivative = [
            sqrt_p * (2.0 * p / w) * transverse_thrust / mass,
            sqrt_p
                * (radial_thrust * sine + ((1.0 + w) * cosine + f) / w * transverse_thrust
                    - g / w * hsk * normal_thrust)
                / mass,
            sqrt_p
                * (-radial_thrust * cosine
                    + ((1.0 + w) * sine + g) / w * transverse_thrust
                    + f / w * hsk * normal_thrust)
                / mass,
            sqrt_p * (s2 / w / 2.0) * cosine * normal_thrust / mass,
            sqrt_p * (s2 / w / 2.0) * sine * normal_thrust / mass,
            sqrt_p * hsk / w * normal_thrust / mass + w * w / p.powf(1.5),
            regularized_mass_flow(mass, thrust, c),
        ];
        validate_rhs(Self::NAME, derivative)
    }
}

impl DifferentiableDynamicsModel<7, 5> for ZohEquinoctialDynamics {
    fn jacobians(
        &self,
        time: f64,
        state: &[f64; 7],
        parameters: &[f64; 5],
        state_jacobian: &mut [[f64; 7]; 7],
        parameter_jacobian: &mut [[f64; 5]; 7],
    ) -> Result<()> {
        numerical_jacobians(
            self,
            time,
            state,
            parameters,
            state_jacobian,
            parameter_jacobian,
            &[0, 6],
        )
    }
}

impl ZeroOrderHoldModel<7, 4, 1, 5> for ZohEquinoctialDynamics {
    fn parameters(control: [f64; 4], constants: [f64; 1]) -> [f64; 5] {
        [control[0], control[1], control[2], control[3], constants[0]]
    }

    fn parameter_seeds<const W: usize>(
        control: [[f64; W]; 4],
        constants: [[f64; W]; 1],
    ) -> [[f64; W]; 5] {
        [control[0], control[1], control[2], control[3], constants[0]]
    }
}

/// Six-state ideal solar-sail dynamics with piecewise-constant cone and clock
/// angles.
///
/// Parameters are `[alpha, beta, c]`, where `c/r²` scales sail acceleration
/// in normalized heliocentric units.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZohSolarSailDynamics;

impl DynamicsModel<6, 3> for ZohSolarSailDynamics {
    const NAME: &'static str = "ZOH solar-sail dynamics";

    fn validate(&self, time: f64, state: &[f64; 6], parameters: &[f64; 3]) -> Result<()> {
        validate_finite(time, state, parameters)?;
        validate_radius(&state[..3], "ZOH solar-sail radius")?;
        let angular_momentum = cross(&state[..3], &state[3..]);
        validate_radius(&angular_momentum, "ZOH solar-sail angular momentum")?;
        Ok(())
    }

    fn rhs(
        &self,
        time: f64,
        state: &[f64; 6],
        parameters: &[f64; 3],
        derivative: &mut [f64; 6],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let [alpha, beta, c] = *parameters;
        let radius_squared = squared_norm(&state[..3]);
        let radius = radius_squared.sqrt();
        let angular_momentum = cross(&state[..3], &state[3..]);
        let angular_norm = squared_norm(&angular_momentum).sqrt();
        let radial = [state[0] / radius, state[1] / radius, state[2] / radius];
        let normal = [
            angular_momentum[0] / angular_norm,
            angular_momentum[1] / angular_norm,
            angular_momentum[2] / angular_norm,
        ];
        let transverse = cross(&normal, &radial);
        let thrust = c / radius_squared * alpha.cos().powi(2);
        let radial_acceleration = alpha.cos() * thrust;
        let transverse_acceleration = alpha.sin() * beta.sin() * thrust;
        let normal_acceleration = alpha.sin() * beta.cos() * thrust;
        let gravity = -1.0 / (radius_squared * radius);
        *derivative = [
            state[3],
            state[4],
            state[5],
            gravity * state[0]
                + radial_acceleration * radial[0]
                + transverse_acceleration * transverse[0]
                + normal_acceleration * normal[0],
            gravity * state[1]
                + radial_acceleration * radial[1]
                + transverse_acceleration * transverse[1]
                + normal_acceleration * normal[1],
            gravity * state[2]
                + radial_acceleration * radial[2]
                + transverse_acceleration * transverse[2]
                + normal_acceleration * normal[2],
        ];
        validate_rhs(Self::NAME, derivative)
    }
}

impl DifferentiableDynamicsModel<6, 3> for ZohSolarSailDynamics {
    fn jacobians(
        &self,
        time: f64,
        state: &[f64; 6],
        parameters: &[f64; 3],
        state_jacobian: &mut [[f64; 6]; 6],
        parameter_jacobian: &mut [[f64; 3]; 6],
    ) -> Result<()> {
        numerical_jacobians(
            self,
            time,
            state,
            parameters,
            state_jacobian,
            parameter_jacobian,
            &[],
        )
    }
}

impl ZeroOrderHoldModel<6, 2, 1, 3> for ZohSolarSailDynamics {
    fn parameters(control: [f64; 2], constants: [f64; 1]) -> [f64; 3] {
        [control[0], control[1], constants[0]]
    }

    fn parameter_seeds<const W: usize>(
        control: [[f64; W]; 2],
        constants: [[f64; W]; 1],
    ) -> [[f64; W]; 3] {
        [control[0], control[1], constants[0]]
    }
}

fn validate_finite<const N: usize, const P: usize>(
    time: f64,
    state: &[f64; N],
    parameters: &[f64; P],
) -> Result<()> {
    ensure_finite("time", time)?;
    for &value in state {
        ensure_finite("state", value)?;
    }
    for &value in parameters {
        ensure_finite("parameters", value)?;
    }
    Ok(())
}

fn validate_positive_state(value: f64, parameter: &'static str) -> Result<()> {
    if value > 0.0 {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter,
            reason: "must be greater than zero".into(),
        })
    }
}

fn validate_mass_fraction(mu: f64) -> Result<()> {
    if (0.0..=1.0).contains(&mu) {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter: "mu",
            reason: "must lie in the closed interval [0, 1]".into(),
        })
    }
}

fn validate_radius(vector: &[f64], operation: &'static str) -> Result<()> {
    let squared = squared_norm(vector);
    if squared == 0.0 {
        Err(PykepError::SingularGeometry { operation })
    } else if squared.is_finite() {
        Ok(())
    } else {
        Err(PykepError::NumericalOverflow { operation })
    }
}

fn squared_norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum()
}

fn cross(left: &[f64], right: &[f64]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn regularized_mass_flow(mass: f64, thrust: f64, coefficient: f64) -> f64 {
    -coefficient * thrust * (-1.0 / mass / 1e16).exp()
}

fn validate_rhs<const N: usize>(operation: &'static str, values: &[f64; N]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PykepError::NumericalOverflow { operation })
    }
}

fn numerical_jacobians<M, const N: usize, const P: usize>(
    model: &M,
    time: f64,
    state: &[f64; N],
    parameters: &[f64; P],
    state_jacobian: &mut [[f64; N]; N],
    parameter_jacobian: &mut [[f64; P]; N],
    positive_state_indices: &[usize],
) -> Result<()>
where
    M: DynamicsModel<N, P>,
{
    *state_jacobian = [[0.0; N]; N];
    *parameter_jacobian = [[0.0; P]; N];
    for column in 0..N {
        let mut step = 3e-6 * state[column].abs().max(1.0);
        if positive_state_indices.contains(&column) {
            step = step.min(state[column] * 0.25);
        }
        let mut plus = *state;
        let mut minus = *state;
        plus[column] += step;
        minus[column] -= step;
        let mut rhs_plus = [0.0; N];
        let mut rhs_minus = [0.0; N];
        model.rhs(time, &plus, parameters, &mut rhs_plus)?;
        model.rhs(time, &minus, parameters, &mut rhs_minus)?;
        for row in 0..N {
            state_jacobian[row][column] = (rhs_plus[row] - rhs_minus[row]) / (2.0 * step);
        }
    }
    for column in 0..P {
        let step = 3e-6 * parameters[column].abs().max(1.0);
        let mut plus = *parameters;
        let mut minus = *parameters;
        plus[column] += step;
        minus[column] -= step;
        let mut rhs_plus = [0.0; N];
        let mut rhs_minus = [0.0; N];
        model.rhs(time, state, &plus, &mut rhs_plus)?;
        model.rhs(time, state, &minus, &mut rhs_minus)?;
        for row in 0..N {
            parameter_jacobian[row][column] = (rhs_plus[row] - rhs_minus[row]) / (2.0 * step);
        }
    }
    Ok(())
}
