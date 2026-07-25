// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                          Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Evaluated Kepler, circular restricted three-body, and bicircular dynamics.
//!
//! The rotating-frame models use nondimensional CR3BP units and the state
//! order `[x, y, z, vx, vy, vz]`. Their primaries lie at `(-mu, 0, 0)` and
//! `(1 - mu, 0, 0)`. BCP adds a Sun whose rotating-frame position is
//! `rho_sun * [cos(omega_sun * t), sin(omega_sun * t), 0]`.
//!
//! This file is an evaluated Rust adaptation of the symbolic systems in
//! `src/ta/kep.cpp`, `src/ta/cr3bp.cpp`, and `src/ta/bcp.cpp` from the pinned
//! pykep/kep3 upstream source.

/// Cartesian and modified-equinoctial Pontryagin dynamics.
pub mod pontryagin;
/// Zero-order-hold low-thrust and solar-sail dynamics.
pub mod zoh;

use crate::error::ensure_finite;
use crate::integration::{
    DifferentiableDynamicsModel, Dop853, DynamicsModel, InitialValueProblem, IntegratorOptions,
    Propagation, SensitivityProblem, SensitivityPropagation,
};
use crate::{CartesianState, Matrix6, PykepError, Result};

const POSITION_DIMENSION: usize = 3;

/// Evaluated two-body Cartesian dynamics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeplerDynamics;

impl KeplerDynamics {
    /// Evaluates the two-body right-hand side for gravitational parameter
    /// `mu`.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite inputs, non-positive `mu`, collision
    /// with the central body, or a non-finite result.
    pub fn evaluate(&self, state: &CartesianState, mu: f64) -> Result<CartesianState> {
        let mut derivative = [0.0; 6];
        self.rhs(0.0, state, &[mu], &mut derivative)?;
        Ok(derivative)
    }

    /// Propagates a Cartesian state with adaptive DOP853 integration.
    ///
    /// # Errors
    ///
    /// Returns a model-domain or integration error.
    pub fn propagate(
        &self,
        initial_time: f64,
        initial_state: CartesianState,
        final_time: f64,
        mu: f64,
        options: IntegratorOptions,
    ) -> Result<Propagation<6>> {
        Dop853.propagate(
            self,
            InitialValueProblem::new(initial_time, initial_state, final_time, [mu]),
            options,
        )
    }

    /// Propagates a state and its row-major 6 × 6 state-transition matrix.
    ///
    /// # Errors
    ///
    /// Returns a model-domain, Jacobian, or integration error.
    pub fn propagate_with_stm(
        &self,
        initial_time: f64,
        initial_state: CartesianState,
        final_time: f64,
        mu: f64,
        options: IntegratorOptions,
    ) -> Result<SensitivityPropagation<6, 6>> {
        propagate_stm(self, initial_time, initial_state, final_time, [mu], options)
    }
}

impl DynamicsModel<6, 1> for KeplerDynamics {
    const NAME: &'static str = "Kepler dynamics";

    fn validate(&self, time: f64, state: &CartesianState, parameters: &[f64; 1]) -> Result<()> {
        validate_time_state(time, state)?;
        validate_positive("mu", parameters[0])?;
        radius_squared([state[0], state[1], state[2]], "Kepler dynamics radius")?;
        Ok(())
    }

    fn rhs(
        &self,
        time: f64,
        state: &CartesianState,
        parameters: &[f64; 1],
        derivative: &mut CartesianState,
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let position = [state[0], state[1], state[2]];
        let (acceleration, _) = point_mass_acceleration_and_gradient(
            position,
            parameters[0],
            "Kepler dynamics radius",
        )?;
        *derivative = [
            state[3],
            state[4],
            state[5],
            acceleration[0],
            acceleration[1],
            acceleration[2],
        ];
        validate_output(Self::NAME, derivative)
    }
}

impl DifferentiableDynamicsModel<6, 1> for KeplerDynamics {
    fn jacobians(
        &self,
        time: f64,
        state: &CartesianState,
        parameters: &[f64; 1],
        state_jacobian: &mut Matrix6,
        parameter_jacobian: &mut [[f64; 1]; 6],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let position = [state[0], state[1], state[2]];
        let (_, gradient) = point_mass_acceleration_and_gradient(
            position,
            parameters[0],
            "Kepler dynamics radius",
        )?;
        *state_jacobian = kinematic_jacobian();
        set_spatial_gradient(state_jacobian, &gradient);
        let radius = radius_squared(position, "Kepler dynamics radius")?.sqrt();
        let inverse_radius_cubed = 1.0 / (radius * radius * radius);
        *parameter_jacobian = [[0.0]; 6];
        for row in 0..POSITION_DIMENSION {
            parameter_jacobian[row + 3][0] = -position[row] * inverse_radius_cubed;
        }
        validate_jacobians(Self::NAME, state_jacobian, parameter_jacobian)
    }
}

/// Evaluated circular restricted three-body dynamics in the synodic frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cr3bpDynamics;

impl Cr3bpDynamics {
    /// Evaluates the CR3BP right-hand side.
    ///
    /// `mu` is the secondary mass divided by the total primary mass and must
    /// lie in `[0, 1]`; the conventional primary ordering uses `mu <= 0.5`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid inputs, collision with either primary, or
    /// a non-finite result.
    pub fn evaluate(&self, state: &CartesianState, mu: f64) -> Result<CartesianState> {
        let mut derivative = [0.0; 6];
        self.rhs(0.0, state, &[mu], &mut derivative)?;
        Ok(derivative)
    }

    /// Returns the positive effective potential
    /// `U = (x² + y²)/2 + (1-mu)/r1 + mu/r2`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid inputs or collision with a primary.
    pub fn effective_potential(&self, state: &CartesianState, mu: f64) -> Result<f64> {
        self.validate(0.0, state, &[mu])?;
        let (d1, d2) = primary_displacements(state, mu);
        let r1 = radius_squared(d1, "CR3BP primary-one distance")?.sqrt();
        let r2 = radius_squared(d2, "CR3BP primary-two distance")?.sqrt();
        let potential =
            0.5 * (state[0] * state[0] + state[1] * state[1]) + (1.0 - mu) / r1 + mu / r2;
        finite_output(Self::NAME, potential)
    }

    /// Returns the Jacobi constant `C = 2 U - (vx² + vy² + vz²)`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid inputs or collision with a primary.
    pub fn jacobi_constant(&self, state: &CartesianState, mu: f64) -> Result<f64> {
        let potential = self.effective_potential(state, mu)?;
        let velocity_squared = state[3] * state[3] + state[4] * state[4] + state[5] * state[5];
        finite_output(Self::NAME, 2.0 * potential - velocity_squared)
    }

    /// Propagates a CR3BP state with adaptive DOP853 integration.
    ///
    /// # Errors
    ///
    /// Returns a model-domain or integration error.
    pub fn propagate(
        &self,
        initial_time: f64,
        initial_state: CartesianState,
        final_time: f64,
        mu: f64,
        options: IntegratorOptions,
    ) -> Result<Propagation<6>> {
        Dop853.propagate(
            self,
            InitialValueProblem::new(initial_time, initial_state, final_time, [mu]),
            options,
        )
    }

    /// Propagates a CR3BP state and its row-major 6 × 6 STM.
    ///
    /// # Errors
    ///
    /// Returns a model-domain, Jacobian, or integration error.
    pub fn propagate_with_stm(
        &self,
        initial_time: f64,
        initial_state: CartesianState,
        final_time: f64,
        mu: f64,
        options: IntegratorOptions,
    ) -> Result<SensitivityPropagation<6, 6>> {
        propagate_stm(self, initial_time, initial_state, final_time, [mu], options)
    }
}

impl DynamicsModel<6, 1> for Cr3bpDynamics {
    const NAME: &'static str = "CR3BP dynamics";

    fn validate(&self, time: f64, state: &CartesianState, parameters: &[f64; 1]) -> Result<()> {
        validate_time_state(time, state)?;
        validate_mass_fraction(parameters[0])?;
        let (d1, d2) = primary_displacements(state, parameters[0]);
        radius_squared(d1, "CR3BP primary-one distance")?;
        radius_squared(d2, "CR3BP primary-two distance")?;
        Ok(())
    }

    fn rhs(
        &self,
        time: f64,
        state: &CartesianState,
        parameters: &[f64; 1],
        derivative: &mut CartesianState,
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let mu = parameters[0];
        let (d1, d2) = primary_displacements(state, mu);
        let (a1, _) =
            point_mass_acceleration_and_gradient(d1, 1.0 - mu, "CR3BP primary-one distance")?;
        let (a2, _) = point_mass_acceleration_and_gradient(d2, mu, "CR3BP primary-two distance")?;
        *derivative = [
            state[3],
            state[4],
            state[5],
            2.0 * state[4] + state[0] + a1[0] + a2[0],
            -2.0 * state[3] + state[1] + a1[1] + a2[1],
            a1[2] + a2[2],
        ];
        validate_output(Self::NAME, derivative)
    }
}

impl DifferentiableDynamicsModel<6, 1> for Cr3bpDynamics {
    fn jacobians(
        &self,
        time: f64,
        state: &CartesianState,
        parameters: &[f64; 1],
        state_jacobian: &mut Matrix6,
        parameter_jacobian: &mut [[f64; 1]; 6],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let mu = parameters[0];
        let (d1, d2) = primary_displacements(state, mu);
        let (_, gradient1) =
            point_mass_acceleration_and_gradient(d1, 1.0 - mu, "CR3BP primary-one distance")?;
        let (_, gradient2) =
            point_mass_acceleration_and_gradient(d2, mu, "CR3BP primary-two distance")?;
        *state_jacobian = rotating_kinematic_jacobian();
        add_spatial_gradient(state_jacobian, &gradient1);
        add_spatial_gradient(state_jacobian, &gradient2);

        *parameter_jacobian = [[0.0]; 6];
        let derivative1 =
            moving_mass_parameter_derivative(d1, 1.0 - mu, -1.0, "CR3BP primary-one distance")?;
        let derivative2 =
            moving_mass_parameter_derivative(d2, mu, 1.0, "CR3BP primary-two distance")?;
        for row in 0..POSITION_DIMENSION {
            parameter_jacobian[row + 3][0] = derivative1[row] + derivative2[row];
        }
        validate_jacobians(Self::NAME, state_jacobian, parameter_jacobian)
    }
}

/// Evaluated bicircular-problem dynamics in the Earth–Moon synodic frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BcpDynamics;

impl BcpDynamics {
    /// Evaluates the time-dependent BCP right-hand side.
    ///
    /// Parameters are `[mu, mu_sun, rho_sun, omega_sun]`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid parameters, collision with a massive body,
    /// or a non-finite result.
    pub fn evaluate(
        &self,
        time: f64,
        state: &CartesianState,
        parameters: [f64; 4],
    ) -> Result<CartesianState> {
        let mut derivative = [0.0; 6];
        self.rhs(time, state, &parameters, &mut derivative)?;
        Ok(derivative)
    }

    /// Propagates a BCP state with adaptive DOP853 integration.
    ///
    /// # Errors
    ///
    /// Returns a model-domain or integration error.
    pub fn propagate(
        &self,
        initial_time: f64,
        initial_state: CartesianState,
        final_time: f64,
        parameters: [f64; 4],
        options: IntegratorOptions,
    ) -> Result<Propagation<6>> {
        Dop853.propagate(
            self,
            InitialValueProblem::new(initial_time, initial_state, final_time, parameters),
            options,
        )
    }

    /// Propagates a BCP state and its row-major 6 × 6 STM.
    ///
    /// # Errors
    ///
    /// Returns a model-domain, Jacobian, or integration error.
    pub fn propagate_with_stm(
        &self,
        initial_time: f64,
        initial_state: CartesianState,
        final_time: f64,
        parameters: [f64; 4],
        options: IntegratorOptions,
    ) -> Result<SensitivityPropagation<6, 6>> {
        propagate_stm(
            self,
            initial_time,
            initial_state,
            final_time,
            parameters,
            options,
        )
    }
}

impl DynamicsModel<6, 4> for BcpDynamics {
    const NAME: &'static str = "BCP dynamics";

    fn validate(&self, time: f64, state: &CartesianState, parameters: &[f64; 4]) -> Result<()> {
        validate_time_state(time, state)?;
        validate_mass_fraction(parameters[0])?;
        validate_non_negative("mu_sun", parameters[1])?;
        validate_positive("rho_sun", parameters[2])?;
        ensure_finite("omega_sun", parameters[3])?;
        let (d1, d2) = primary_displacements(state, parameters[0]);
        radius_squared(d1, "BCP primary-one distance")?;
        radius_squared(d2, "BCP primary-two distance")?;
        let (_, _, sun_displacement) = sun_geometry(time, state, parameters[2], parameters[3]);
        radius_squared(sun_displacement, "BCP Sun distance")?;
        Ok(())
    }

    fn rhs(
        &self,
        time: f64,
        state: &CartesianState,
        parameters: &[f64; 4],
        derivative: &mut CartesianState,
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let [mu, mu_sun, rho_sun, omega_sun] = *parameters;
        let (d1, d2) = primary_displacements(state, mu);
        let (a1, _) =
            point_mass_acceleration_and_gradient(d1, 1.0 - mu, "BCP primary-one distance")?;
        let (a2, _) = point_mass_acceleration_and_gradient(d2, mu, "BCP primary-two distance")?;
        let (sun_direction, _, sun_displacement) = sun_geometry(time, state, rho_sun, omega_sun);
        let (sun_acceleration, _) =
            point_mass_acceleration_and_gradient(sun_displacement, mu_sun, "BCP Sun distance")?;
        let indirect_scale = -mu_sun / (rho_sun * rho_sun);
        *derivative = [
            state[3],
            state[4],
            state[5],
            2.0 * state[4]
                + state[0]
                + a1[0]
                + a2[0]
                + sun_acceleration[0]
                + indirect_scale * sun_direction[0],
            -2.0 * state[3]
                + state[1]
                + a1[1]
                + a2[1]
                + sun_acceleration[1]
                + indirect_scale * sun_direction[1],
            a1[2] + a2[2] + sun_acceleration[2],
        ];
        validate_output(Self::NAME, derivative)
    }
}

impl DifferentiableDynamicsModel<6, 4> for BcpDynamics {
    fn jacobians(
        &self,
        time: f64,
        state: &CartesianState,
        parameters: &[f64; 4],
        state_jacobian: &mut Matrix6,
        parameter_jacobian: &mut [[f64; 4]; 6],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        let [mu, mu_sun, rho_sun, omega_sun] = *parameters;
        let (d1, d2) = primary_displacements(state, mu);
        let (_, gradient1) =
            point_mass_acceleration_and_gradient(d1, 1.0 - mu, "BCP primary-one distance")?;
        let (_, gradient2) =
            point_mass_acceleration_and_gradient(d2, mu, "BCP primary-two distance")?;
        let (sun_direction, sun_direction_rate, sun_displacement) =
            sun_geometry(time, state, rho_sun, omega_sun);
        let (_, sun_gradient) =
            point_mass_acceleration_and_gradient(sun_displacement, mu_sun, "BCP Sun distance")?;
        *state_jacobian = rotating_kinematic_jacobian();
        add_spatial_gradient(state_jacobian, &gradient1);
        add_spatial_gradient(state_jacobian, &gradient2);
        add_spatial_gradient(state_jacobian, &sun_gradient);

        *parameter_jacobian = [[0.0; 4]; 6];
        let mu_derivative1 =
            moving_mass_parameter_derivative(d1, 1.0 - mu, -1.0, "BCP primary-one distance")?;
        let mu_derivative2 =
            moving_mass_parameter_derivative(d2, mu, 1.0, "BCP primary-two distance")?;
        let sun_radius = radius_squared(sun_displacement, "BCP Sun distance")?.sqrt();
        let inverse_sun_radius_cubed = 1.0 / sun_radius.powi(3);
        let inverse_rho_squared = 1.0 / (rho_sun * rho_sun);
        let inverse_rho_cubed = inverse_rho_squared / rho_sun;
        for row in 0..POSITION_DIMENSION {
            parameter_jacobian[row + 3][0] = mu_derivative1[row] + mu_derivative2[row];
            parameter_jacobian[row + 3][1] = -sun_displacement[row] * inverse_sun_radius_cubed
                - sun_direction[row] * inverse_rho_squared;
            let gradient_times_direction = dot_row(&sun_gradient, row, &sun_direction);
            parameter_jacobian[row + 3][2] =
                -gradient_times_direction + 2.0 * mu_sun * sun_direction[row] * inverse_rho_cubed;
            let gradient_times_rate = dot_row(&sun_gradient, row, &sun_direction_rate);
            parameter_jacobian[row + 3][3] = -rho_sun * time * gradient_times_rate
                - mu_sun * time * sun_direction_rate[row] * inverse_rho_squared;
        }
        validate_jacobians(Self::NAME, state_jacobian, parameter_jacobian)
    }
}

fn propagate_stm<M, const P: usize>(
    model: &M,
    initial_time: f64,
    initial_state: CartesianState,
    final_time: f64,
    parameters: [f64; P],
    options: IntegratorOptions,
) -> Result<SensitivityPropagation<6, 6>>
where
    M: DifferentiableDynamicsModel<6, P>,
{
    Dop853.propagate_with_sensitivities(
        model,
        SensitivityProblem {
            nominal: InitialValueProblem::new(initial_time, initial_state, final_time, parameters),
            initial_sensitivities: identity6(),
            parameter_seeds: [[0.0; 6]; P],
        },
        options,
    )
}

const fn identity6() -> Matrix6 {
    let mut matrix = [[0.0; 6]; 6];
    let mut index = 0;
    while index < 6 {
        matrix[index][index] = 1.0;
        index += 1;
    }
    matrix
}

fn validate_time_state(time: f64, state: &CartesianState) -> Result<()> {
    ensure_finite("time", time)?;
    for &value in state {
        ensure_finite("state", value)?;
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
            reason: "must be non-negative".into(),
        })
    }
}

fn validate_mass_fraction(mu: f64) -> Result<()> {
    ensure_finite("mu", mu)?;
    if (0.0..=1.0).contains(&mu) {
        Ok(())
    } else {
        Err(PykepError::InvalidInput {
            parameter: "mu",
            reason: "must lie in the closed interval [0, 1]".into(),
        })
    }
}

fn primary_displacements(state: &CartesianState, mu: f64) -> ([f64; 3], [f64; 3]) {
    (
        [state[0] + mu, state[1], state[2]],
        [state[0] + mu - 1.0, state[1], state[2]],
    )
}

fn sun_geometry(
    time: f64,
    state: &CartesianState,
    rho_sun: f64,
    omega_sun: f64,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let angle = omega_sun * time;
    let direction = [angle.cos(), angle.sin(), 0.0];
    let direction_rate = [-angle.sin(), angle.cos(), 0.0];
    let displacement = [
        state[0] - rho_sun * direction[0],
        state[1] - rho_sun * direction[1],
        state[2],
    ];
    (direction, direction_rate, displacement)
}

fn radius_squared(vector: [f64; 3], operation: &'static str) -> Result<f64> {
    let squared = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    if squared == 0.0 {
        return Err(PykepError::SingularGeometry { operation });
    }
    if squared.is_finite() {
        Ok(squared)
    } else {
        Err(PykepError::NumericalOverflow { operation })
    }
}

fn point_mass_acceleration_and_gradient(
    displacement: [f64; 3],
    mass: f64,
    operation: &'static str,
) -> Result<([f64; 3], [[f64; 3]; 3])> {
    let radius_squared = radius_squared(displacement, operation)?;
    let radius = radius_squared.sqrt();
    let inverse_radius_cubed = 1.0 / (radius_squared * radius);
    let inverse_radius_fifth = inverse_radius_cubed / radius_squared;
    let mut acceleration = [0.0; 3];
    let mut gradient = [[0.0; 3]; 3];
    for row in 0..POSITION_DIMENSION {
        acceleration[row] = -mass * displacement[row] * inverse_radius_cubed;
        for column in 0..POSITION_DIMENSION {
            let identity = f64::from(row == column);
            gradient[row][column] = mass
                * (3.0 * displacement[row] * displacement[column] * inverse_radius_fifth
                    - identity * inverse_radius_cubed);
        }
    }
    if acceleration
        .iter()
        .chain(gradient.iter().flatten())
        .all(|value| value.is_finite())
    {
        Ok((acceleration, gradient))
    } else {
        Err(PykepError::NumericalOverflow { operation })
    }
}

fn moving_mass_parameter_derivative(
    displacement: [f64; 3],
    mass: f64,
    mass_derivative: f64,
    operation: &'static str,
) -> Result<[f64; 3]> {
    let radius_squared = radius_squared(displacement, operation)?;
    let radius = radius_squared.sqrt();
    let inverse_radius_cubed = 1.0 / (radius_squared * radius);
    let inverse_radius_fifth = inverse_radius_cubed / radius_squared;
    let mut derivative = [0.0; 3];
    for row in 0..POSITION_DIMENSION {
        let displacement_derivative = f64::from(row == 0);
        derivative[row] = -mass_derivative * displacement[row] * inverse_radius_cubed
            - mass
                * (displacement_derivative * inverse_radius_cubed
                    - 3.0 * displacement[row] * displacement[0] * inverse_radius_fifth);
    }
    if derivative.iter().all(|value| value.is_finite()) {
        Ok(derivative)
    } else {
        Err(PykepError::NumericalOverflow { operation })
    }
}

const fn kinematic_jacobian() -> Matrix6 {
    let mut matrix = [[0.0; 6]; 6];
    matrix[0][3] = 1.0;
    matrix[1][4] = 1.0;
    matrix[2][5] = 1.0;
    matrix
}

const fn rotating_kinematic_jacobian() -> Matrix6 {
    let mut matrix = kinematic_jacobian();
    matrix[3][0] = 1.0;
    matrix[4][1] = 1.0;
    matrix[3][4] = 2.0;
    matrix[4][3] = -2.0;
    matrix
}

fn set_spatial_gradient(jacobian: &mut Matrix6, gradient: &[[f64; 3]; 3]) {
    for row in 0..POSITION_DIMENSION {
        for column in 0..POSITION_DIMENSION {
            jacobian[row + 3][column] = gradient[row][column];
        }
    }
}

fn add_spatial_gradient(jacobian: &mut Matrix6, gradient: &[[f64; 3]; 3]) {
    for row in 0..POSITION_DIMENSION {
        for column in 0..POSITION_DIMENSION {
            jacobian[row + 3][column] += gradient[row][column];
        }
    }
}

fn dot_row(matrix: &[[f64; 3]; 3], row: usize, vector: &[f64; 3]) -> f64 {
    matrix[row][0] * vector[0] + matrix[row][1] * vector[1] + matrix[row][2] * vector[2]
}

fn validate_output(model: &'static str, values: &CartesianState) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PykepError::NumericalOverflow { operation: model })
    }
}

fn finite_output(model: &'static str, value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PykepError::NumericalOverflow { operation: model })
    }
}

fn validate_jacobians<const P: usize>(
    model: &'static str,
    state_jacobian: &Matrix6,
    parameter_jacobian: &[[f64; P]; 6],
) -> Result<()> {
    if state_jacobian
        .iter()
        .flatten()
        .chain(parameter_jacobian.iter().flatten())
        .all(|value| value.is_finite())
    {
        Ok(())
    } else {
        Err(PykepError::NumericalOverflow { operation: model })
    }
}
