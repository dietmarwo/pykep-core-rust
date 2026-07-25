// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                          Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Sims–Flanagan impulsive low-thrust transcription.
//!
//! Adapted from `src/leg/sf_checks.cpp`, `sims_flanagan.cpp`, and
//! `sims_flanagan_alpha.cpp` in pinned pykep/kep3 3.0.1.

use crate::astro::propagation::{propagate_lagrangian, propagate_lagrangian_with_stm};
use crate::error::ensure_finite;
use crate::{CartesianState, PykepError, Result, Vector3};

const EXTENDED_DIMENSION: usize = 7;

/// Cartesian spacecraft endpoint with positive mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpacecraftEndpoint {
    /// Cartesian state `[x,y,z,vx,vy,vz]`.
    pub state: CartesianState,
    /// Spacecraft mass.
    pub mass: f64,
}

impl SpacecraftEndpoint {
    /// Constructs and validates an endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite state components, a central-body
    /// collision, or non-positive mass.
    pub fn new(state: CartesianState, mass: f64) -> Result<Self> {
        validate_state(&state)?;
        validate_positive("mass", mass)?;
        Ok(Self { state, mass })
    }

    fn extended(self) -> [f64; EXTENDED_DIMENSION] {
        [
            self.state[0],
            self.state[1],
            self.state[2],
            self.state[3],
            self.state[4],
            self.state[5],
            self.mass,
        ]
    }
}

/// Shared propulsion, gravity, duration, and cut settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimsFlanaganSettings {
    /// Complete leg time of flight.
    pub time_of_flight: f64,
    /// Maximum available thrust.
    pub maximum_thrust: f64,
    /// Effective exhaust velocity.
    pub exhaust_velocity: f64,
    /// Central-body gravitational parameter.
    pub mu: f64,
    /// Fractional cut in `[0,1]`; `floor(segment_count * cut)` segments are
    /// propagated forward.
    pub cut: f64,
}

impl SimsFlanaganSettings {
    /// Constructs validated leg settings.
    ///
    /// # Errors
    ///
    /// Returns an error unless time of flight and maximum thrust are
    /// non-negative, exhaust velocity and `mu` are positive, and `cut` lies
    /// in `[0,1]`.
    pub fn new(
        time_of_flight: f64,
        maximum_thrust: f64,
        exhaust_velocity: f64,
        mu: f64,
        cut: f64,
    ) -> Result<Self> {
        validate_non_negative("time_of_flight", time_of_flight)?;
        validate_non_negative("maximum_thrust", maximum_thrust)?;
        validate_positive("exhaust_velocity", exhaust_velocity)?;
        validate_positive("mu", mu)?;
        ensure_finite("cut", cut)?;
        if !(0.0..=1.0).contains(&cut) {
            return Err(PykepError::InvalidInput {
                parameter: "cut",
                reason: "must lie in 0..=1".into(),
            });
        }
        Ok(Self {
            time_of_flight,
            maximum_thrust,
            exhaust_velocity,
            mu,
            cut,
        })
    }
}

/// Mismatch derivatives in the upstream row/column order.
///
/// Rows always follow mismatch order `[rx,ry,rz,vx,vy,vz,m]`.
#[derive(Clone, Debug, PartialEq)]
pub struct SimsFlanaganMismatchJacobian {
    /// Derivatives with respect to departure `[r,v,m]`, as 7 × 7
    /// output-by-input rows.
    pub departure: [[f64; EXTENDED_DIMENSION]; EXTENDED_DIMENSION],
    /// Derivatives with respect to arrival `[r,v,m]`, as 7 × 7
    /// output-by-input rows.
    pub arrival: [[f64; EXTENDED_DIMENSION]; EXTENDED_DIMENSION],
    /// Derivatives with respect to flattened segment controls followed by
    /// time of flight. There are `3 * segment_count + 1` columns.
    pub controls_and_time: Vec<Vec<f64>>,
}

/// Fixed-duration Sims–Flanagan leg.
#[derive(Clone, Debug, PartialEq)]
pub struct SimsFlanaganLeg {
    departure: SpacecraftEndpoint,
    arrival: SpacecraftEndpoint,
    throttles: Vec<Vector3>,
    settings: SimsFlanaganSettings,
    forward_segments: usize,
}

impl SimsFlanaganLeg {
    /// Constructs a fully validated fixed-duration leg.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty control sequence or invalid endpoint,
    /// control, or setting.
    pub fn new(
        departure: SpacecraftEndpoint,
        throttles: Vec<Vector3>,
        arrival: SpacecraftEndpoint,
        settings: SimsFlanaganSettings,
    ) -> Result<Self> {
        validate_endpoint(departure)?;
        validate_endpoint(arrival)?;
        validate_settings(settings)?;
        validate_throttles(&throttles)?;
        let forward_segments = forward_count(throttles.len(), settings.cut);
        Ok(Self {
            departure,
            arrival,
            throttles,
            settings,
            forward_segments,
        })
    }

    /// Departure endpoint.
    pub const fn departure(&self) -> SpacecraftEndpoint {
        self.departure
    }

    /// Arrival endpoint.
    pub const fn arrival(&self) -> SpacecraftEndpoint {
        self.arrival
    }

    /// Segment throttle vectors in chronological order.
    pub fn throttles(&self) -> &[Vector3] {
        &self.throttles
    }

    /// Validated leg settings.
    pub const fn settings(&self) -> SimsFlanaganSettings {
        self.settings
    }

    /// Total number of segments.
    pub fn segment_count(&self) -> usize {
        self.throttles.len()
    }

    /// Number of segments propagated from departure.
    pub const fn forward_segment_count(&self) -> usize {
        self.forward_segments
    }

    /// Number of segments propagated backward from arrival.
    pub fn backward_segment_count(&self) -> usize {
        self.segment_count() - self.forward_segments
    }

    /// Evaluates `[delta_r, delta_v, delta_m]` at the cut.
    ///
    /// # Errors
    ///
    /// Returns a propagation, singularity, or numerical-overflow error.
    pub fn mismatch_constraints(&self) -> Result<[f64; EXTENDED_DIMENSION]> {
        let duration = self.settings.time_of_flight / self.segment_count() as f64;
        let durations = vec![duration; self.segment_count()];
        mismatch(
            self.departure,
            self.arrival,
            &self.throttles,
            &durations,
            self.settings,
            self.forward_segments,
        )
    }

    /// Evaluates `dot(u_i,u_i) - 1` for every segment.
    pub fn throttle_constraints(&self) -> Vec<f64> {
        throttle_constraints(&self.throttles)
    }

    /// Returns the analytic mismatch Jacobian.
    ///
    /// Columns are separated into departure `[r,v,m]`, arrival `[r,v,m]`,
    /// and `[u0x,u0y,u0z,...,tof]`, matching the upstream API.
    ///
    /// # Errors
    ///
    /// Returns a propagation, singularity, or numerical-overflow error.
    pub fn mismatch_jacobian(&self) -> Result<SimsFlanaganMismatchJacobian> {
        analytic_fixed_jacobian(self)
    }

    /// Returns the throttle-constraint Jacobian as
    /// `segment_count × (3 * segment_count)` output-by-input rows.
    pub fn throttle_jacobian(&self) -> Vec<Vec<f64>> {
        throttle_jacobian(&self.throttles)
    }
}

/// Variable-duration Sims–Flanagan leg.
///
/// `segment_durations` are direct time intervals in chronological order.
/// Their sum is not forced to equal `settings.time_of_flight`, preserving the
/// upstream class contract. Use [`Self::from_time_weights`] when normalized
/// positive weights are more convenient.
#[derive(Clone, Debug, PartialEq)]
pub struct SimsFlanaganAlphaLeg {
    departure: SpacecraftEndpoint,
    arrival: SpacecraftEndpoint,
    throttles: Vec<Vector3>,
    segment_durations: Vec<f64>,
    settings: SimsFlanaganSettings,
    forward_segments: usize,
}

impl SimsFlanaganAlphaLeg {
    /// Constructs a variable-duration leg from direct non-negative segment
    /// durations.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent dimensions, empty inputs, or invalid
    /// endpoint, control, duration, or setting values.
    pub fn new(
        departure: SpacecraftEndpoint,
        throttles: Vec<Vector3>,
        segment_durations: Vec<f64>,
        arrival: SpacecraftEndpoint,
        settings: SimsFlanaganSettings,
    ) -> Result<Self> {
        validate_endpoint(departure)?;
        validate_endpoint(arrival)?;
        validate_settings(settings)?;
        validate_throttles(&throttles)?;
        if segment_durations.len() != throttles.len() {
            return Err(PykepError::DimensionMismatch {
                expected: throttles.len(),
                actual: segment_durations.len(),
            });
        }
        for &duration in &segment_durations {
            validate_non_negative("segment_durations", duration)?;
        }
        let forward_segments = forward_count(throttles.len(), settings.cut);
        Ok(Self {
            departure,
            arrival,
            throttles,
            segment_durations,
            settings,
            forward_segments,
        })
    }

    /// Constructs a leg after normalizing strictly positive time weights to
    /// sum to `settings.time_of_flight`.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-positive/non-finite weight or the same
    /// invalid inputs as [`Self::new`].
    pub fn from_time_weights(
        departure: SpacecraftEndpoint,
        throttles: Vec<Vector3>,
        time_weights: Vec<f64>,
        arrival: SpacecraftEndpoint,
        settings: SimsFlanaganSettings,
    ) -> Result<Self> {
        for &weight in &time_weights {
            validate_positive("time_weights", weight)?;
        }
        let sum: f64 = time_weights.iter().sum();
        if !sum.is_finite() {
            return Err(PykepError::NumericalOverflow {
                operation: "Sims-Flanagan time-weight normalization",
            });
        }
        let durations = time_weights
            .into_iter()
            .map(|weight| weight / sum * settings.time_of_flight)
            .collect();
        Self::new(departure, throttles, durations, arrival, settings)
    }

    /// Departure endpoint.
    pub const fn departure(&self) -> SpacecraftEndpoint {
        self.departure
    }

    /// Arrival endpoint.
    pub const fn arrival(&self) -> SpacecraftEndpoint {
        self.arrival
    }

    /// Segment throttle vectors in chronological order.
    pub fn throttles(&self) -> &[Vector3] {
        &self.throttles
    }

    /// Direct segment durations in chronological order.
    pub fn segment_durations(&self) -> &[f64] {
        &self.segment_durations
    }

    /// Validated leg settings.
    pub const fn settings(&self) -> SimsFlanaganSettings {
        self.settings
    }

    /// Total number of segments.
    pub fn segment_count(&self) -> usize {
        self.throttles.len()
    }

    /// Number of segments propagated from departure.
    pub const fn forward_segment_count(&self) -> usize {
        self.forward_segments
    }

    /// Number of segments propagated backward from arrival.
    pub fn backward_segment_count(&self) -> usize {
        self.segment_count() - self.forward_segments
    }

    /// Evaluates `[delta_r, delta_v, delta_m]` at the cut.
    ///
    /// # Errors
    ///
    /// Returns a propagation, singularity, or numerical-overflow error.
    pub fn mismatch_constraints(&self) -> Result<[f64; EXTENDED_DIMENSION]> {
        mismatch(
            self.departure,
            self.arrival,
            &self.throttles,
            &self.segment_durations,
            self.settings,
            self.forward_segments,
        )
    }

    /// Evaluates `dot(u_i,u_i) - 1` for every segment.
    pub fn throttle_constraints(&self) -> Vec<f64> {
        throttle_constraints(&self.throttles)
    }

    /// Returns the throttle-constraint Jacobian as
    /// `segment_count × (3 * segment_count)` output-by-input rows.
    pub fn throttle_jacobian(&self) -> Vec<Vec<f64>> {
        throttle_jacobian(&self.throttles)
    }
}

fn validate_endpoint(endpoint: SpacecraftEndpoint) -> Result<()> {
    SpacecraftEndpoint::new(endpoint.state, endpoint.mass).map(|_| ())
}

fn validate_settings(settings: SimsFlanaganSettings) -> Result<()> {
    SimsFlanaganSettings::new(
        settings.time_of_flight,
        settings.maximum_thrust,
        settings.exhaust_velocity,
        settings.mu,
        settings.cut,
    )
    .map(|_| ())
}

fn validate_state(state: &CartesianState) -> Result<()> {
    for &value in state {
        ensure_finite("state", value)?;
    }
    let radius_squared = state[0] * state[0] + state[1] * state[1] + state[2] * state[2];
    if radius_squared == 0.0 {
        Err(PykepError::SingularGeometry {
            operation: "Sims-Flanagan endpoint radius",
        })
    } else if radius_squared.is_finite() {
        Ok(())
    } else {
        Err(PykepError::NumericalOverflow {
            operation: "Sims-Flanagan endpoint radius",
        })
    }
}

fn validate_throttles(throttles: &[Vector3]) -> Result<()> {
    if throttles.is_empty() {
        return Err(PykepError::InvalidInput {
            parameter: "throttles",
            reason: "at least one segment is required".into(),
        });
    }
    for throttle in throttles {
        for &value in throttle {
            ensure_finite("throttles", value)?;
        }
    }
    Ok(())
}

fn validate_positive(parameter: &'static str, value: f64) -> Result<()> {
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

fn validate_non_negative(parameter: &'static str, value: f64) -> Result<()> {
    ensure_finite(parameter, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter,
            reason: "must be greater than or equal to zero".into(),
        })
    }
}

fn forward_count(segment_count: usize, cut: f64) -> usize {
    (segment_count as f64 * cut) as usize
}

fn throttle_constraints(throttles: &[Vector3]) -> Vec<f64> {
    throttles
        .iter()
        .map(|control| control.iter().map(|value| value * value).sum::<f64>() - 1.0)
        .collect()
}

fn throttle_jacobian(throttles: &[Vector3]) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; throttles.len() * 3]; throttles.len()];
    for (segment, throttle) in throttles.iter().enumerate() {
        for component in 0..3 {
            result[segment][3 * segment + component] = 2.0 * throttle[component];
        }
    }
    result
}

fn mismatch(
    departure: SpacecraftEndpoint,
    arrival: SpacecraftEndpoint,
    throttles: &[Vector3],
    durations: &[f64],
    settings: SimsFlanaganSettings,
    forward_segments: usize,
) -> Result<[f64; EXTENDED_DIMENSION]> {
    let forward = nominal_half(
        departure,
        throttles,
        durations,
        settings,
        forward_segments,
        true,
    )?;
    let backward = nominal_half(
        arrival,
        throttles,
        durations,
        settings,
        forward_segments,
        false,
    )?;
    let result = core::array::from_fn(|index| forward[index] - backward[index]);
    validate_output("Sims-Flanagan mismatch", &result)?;
    Ok(result)
}

fn nominal_half(
    endpoint: SpacecraftEndpoint,
    throttles: &[Vector3],
    durations: &[f64],
    settings: SimsFlanaganSettings,
    forward_segments: usize,
    forward: bool,
) -> Result<[f64; EXTENDED_DIMENSION]> {
    let mut extended = endpoint.extended();
    let range = if forward {
        0..forward_segments
    } else {
        forward_segments..throttles.len()
    };
    if range.is_empty() {
        return Ok(extended);
    }
    let first = if forward { range.start } else { range.end - 1 };
    let direction = if forward { 1.0 } else { -1.0 };
    propagate_extended(
        &mut extended,
        direction * durations[first] / 2.0,
        settings.mu,
    )?;

    if forward {
        for segment in range.clone() {
            apply_impulse(
                &mut extended,
                throttles[segment],
                durations[segment],
                settings,
                1.0,
            )?;
            let coast = if segment + 1 == range.end {
                durations[segment] / 2.0
            } else {
                (durations[segment] + durations[segment + 1]) / 2.0
            };
            propagate_extended(&mut extended, coast, settings.mu)?;
        }
    } else {
        for segment in range.rev() {
            apply_impulse(
                &mut extended,
                throttles[segment],
                durations[segment],
                settings,
                -1.0,
            )?;
            let coast = if segment == forward_segments {
                -durations[segment] / 2.0
            } else {
                -(durations[segment] + durations[segment - 1]) / 2.0
            };
            propagate_extended(&mut extended, coast, settings.mu)?;
        }
    }
    validate_output("Sims-Flanagan half leg", &extended)?;
    Ok(extended)
}

fn propagate_extended(state: &mut [f64; EXTENDED_DIMENSION], duration: f64, mu: f64) -> Result<()> {
    let cartesian: CartesianState = state[..6].try_into().expect("fixed slice length");
    let propagated = propagate_lagrangian(&cartesian, duration, mu)?;
    state[..6].copy_from_slice(&propagated);
    Ok(())
}

fn apply_impulse(
    state: &mut [f64; EXTENDED_DIMENSION],
    throttle: Vector3,
    duration: f64,
    settings: SimsFlanaganSettings,
    direction: f64,
) -> Result<()> {
    let scale = settings.maximum_thrust * duration / state[6];
    let impulse = throttle.map(|value| scale * value);
    for component in 0..3 {
        state[component + 3] += direction * impulse[component];
    }
    let norm = impulse
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    state[6] *= (-direction * norm / settings.exhaust_velocity).exp();
    validate_output("Sims-Flanagan impulse", state)
}

fn validate_output(operation: &'static str, values: &[f64]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PykepError::NumericalOverflow { operation })
    }
}

fn analytic_fixed_jacobian(leg: &SimsFlanaganLeg) -> Result<SimsFlanaganMismatchJacobian> {
    let segment_count = leg.segment_count();
    let width = EXTENDED_DIMENSION + segment_count * 3 + 1;
    let time_column = width - 1;
    let duration = leg.settings.time_of_flight / segment_count as f64;
    let duration_derivative = 1.0 / segment_count as f64;
    let mut departure_seeds = vec![vec![0.0; width]; EXTENDED_DIMENSION];
    let mut arrival_seeds = vec![vec![0.0; width]; EXTENDED_DIMENSION];
    for index in 0..EXTENDED_DIMENSION {
        departure_seeds[index][index] = 1.0;
        arrival_seeds[index][index] = 1.0;
    }
    let forward = sensitivity_half(
        leg.departure,
        &leg.throttles,
        leg.settings,
        leg.forward_segments,
        true,
        duration,
        duration_derivative,
        departure_seeds,
    )?;
    let backward = sensitivity_half(
        leg.arrival,
        &leg.throttles,
        leg.settings,
        leg.forward_segments,
        false,
        duration,
        duration_derivative,
        arrival_seeds,
    )?;

    let departure =
        core::array::from_fn(|row| core::array::from_fn(|column| forward.1[row][column]));
    let arrival =
        core::array::from_fn(|row| core::array::from_fn(|column| -backward.1[row][column]));
    let mut controls_and_time = vec![vec![0.0; segment_count * 3 + 1]; EXTENDED_DIMENSION];
    for (row, values) in controls_and_time.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().take(segment_count * 3).enumerate() {
            *value = forward.1[row][EXTENDED_DIMENSION + column]
                - backward.1[row][EXTENDED_DIMENSION + column];
        }
        values[segment_count * 3] = forward.1[row][time_column] - backward.1[row][time_column];
    }
    validate_output(
        "Sims-Flanagan mismatch Jacobian",
        controls_and_time
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    Ok(SimsFlanaganMismatchJacobian {
        departure,
        arrival,
        controls_and_time,
    })
}

#[allow(clippy::too_many_arguments)]
fn sensitivity_half(
    endpoint: SpacecraftEndpoint,
    throttles: &[Vector3],
    settings: SimsFlanaganSettings,
    forward_segments: usize,
    forward: bool,
    duration: f64,
    duration_derivative: f64,
    mut sensitivities: Vec<Vec<f64>>,
) -> Result<([f64; EXTENDED_DIMENSION], Vec<Vec<f64>>)> {
    let mut state = endpoint.extended();
    let range = if forward {
        0..forward_segments
    } else {
        forward_segments..throttles.len()
    };
    if range.is_empty() {
        return Ok((state, sensitivities));
    }
    let direction = if forward { 1.0 } else { -1.0 };
    propagate_sensitivities(
        &mut state,
        &mut sensitivities,
        direction * duration / 2.0,
        direction * duration_derivative / 2.0,
        settings.mu,
    )?;
    if forward {
        for segment in range.clone() {
            impulse_sensitivities(
                &mut state,
                &mut sensitivities,
                throttles[segment],
                segment,
                duration,
                duration_derivative,
                settings,
                1.0,
            )?;
            let half_factor = if segment + 1 == range.end { 0.5 } else { 1.0 };
            propagate_sensitivities(
                &mut state,
                &mut sensitivities,
                duration * half_factor,
                duration_derivative * half_factor,
                settings.mu,
            )?;
        }
    } else {
        for segment in range.rev() {
            impulse_sensitivities(
                &mut state,
                &mut sensitivities,
                throttles[segment],
                segment,
                duration,
                duration_derivative,
                settings,
                -1.0,
            )?;
            let half_factor = if segment == forward_segments {
                0.5
            } else {
                1.0
            };
            propagate_sensitivities(
                &mut state,
                &mut sensitivities,
                -duration * half_factor,
                -duration_derivative * half_factor,
                settings.mu,
            )?;
        }
    }
    Ok((state, sensitivities))
}

fn propagate_sensitivities(
    state: &mut [f64; EXTENDED_DIMENSION],
    sensitivities: &mut [Vec<f64>],
    duration: f64,
    duration_derivative: f64,
    mu: f64,
) -> Result<()> {
    let initial: CartesianState = state[..6].try_into().expect("fixed slice length");
    let (propagated, transition) = propagate_lagrangian_with_stm(&initial, duration, mu)?;
    let radius_squared = propagated[0] * propagated[0]
        + propagated[1] * propagated[1]
        + propagated[2] * propagated[2];
    let gravity = -mu / radius_squared.powf(1.5);
    let dynamics = [
        propagated[3],
        propagated[4],
        propagated[5],
        gravity * propagated[0],
        gravity * propagated[1],
        gravity * propagated[2],
    ];
    let previous = sensitivities[..6].to_vec();
    let time_column = sensitivities[0].len() - 1;
    for row in 0..6 {
        for column in 0..sensitivities[row].len() {
            sensitivities[row][column] = (0..6)
                .map(|inner| transition[row][inner] * previous[inner][column])
                .sum();
        }
        sensitivities[row][time_column] += dynamics[row] * duration_derivative;
    }
    state[..6].copy_from_slice(&propagated);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn impulse_sensitivities(
    state: &mut [f64; EXTENDED_DIMENSION],
    sensitivities: &mut [Vec<f64>],
    throttle: Vector3,
    segment: usize,
    duration: f64,
    duration_derivative: f64,
    settings: SimsFlanaganSettings,
    direction: f64,
) -> Result<()> {
    let width = sensitivities[0].len();
    let time_column = width - 1;
    let coefficient = settings.maximum_thrust * duration;
    let scale = coefficient / state[6];
    let impulse = throttle.map(|value| scale * value);
    let norm = impulse
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let previous_mass_sensitivity = sensitivities[6].clone();
    let mut impulse_derivative = vec![[0.0; 3]; width];
    for (column, item) in impulse_derivative.iter_mut().enumerate() {
        let coefficient_derivative = if column == time_column {
            settings.maximum_thrust * duration_derivative
        } else {
            0.0
        };
        let scale_derivative = coefficient_derivative / state[6]
            - coefficient / (state[6] * state[6]) * previous_mass_sensitivity[column];
        for component in 0..3 {
            item[component] = throttle[component] * scale_derivative;
            if column == EXTENDED_DIMENSION + segment * 3 + component {
                item[component] += scale;
            }
            sensitivities[component + 3][column] += direction * item[component];
        }
    }
    let exponent = (-direction * norm / settings.exhaust_velocity).exp();
    let new_mass = state[6] * exponent;
    for column in 0..width {
        let norm_derivative = if norm == 0.0 {
            0.0
        } else {
            (0..3)
                .map(|component| impulse[component] * impulse_derivative[column][component])
                .sum::<f64>()
                / norm
        };
        sensitivities[6][column] = exponent * previous_mass_sensitivity[column]
            + new_mass * (-direction / settings.exhaust_velocity) * norm_derivative;
    }
    for component in 0..3 {
        state[component + 3] += direction * impulse[component];
    }
    state[6] = new_mass;
    validate_output("Sims-Flanagan impulse sensitivities", state)
}
