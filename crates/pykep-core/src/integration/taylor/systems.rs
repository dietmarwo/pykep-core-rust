// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use std::sync::OnceLock;

use crate::Result;
use crate::dynamics::pontryagin::{
    CartesianMassOptimal, CartesianTimeOptimal, EquinoctialMassOptimal, EquinoctialTimeOptimal,
};
use crate::dynamics::zoh::{
    ZohCr3bpDynamics, ZohEquinoctialDynamics, ZohKeplerDynamics, ZohSolarSailDynamics,
};
use crate::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
use crate::integration::{DynamicsModel, TaylorDynamicsModel};

#[cfg(test)]
use super::series::Series;
use super::tape::{Expression, IncrementalTape, TapeBuilder};
use super::{MAX_ORDER, TaylorCoefficientModel};

impl TaylorCoefficientModel<6, 1> for KeplerDynamics {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 6],
        parameters: &[f64; 1],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 6],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        *jet = [[0.0; MAX_ORDER + 1]; 6];
        for (component, &value) in jet.iter_mut().zip(state) {
            component[0] = value;
        }
        let mut radius_squared = [0.0; MAX_ORDER + 1];
        let mut inverse_radius_cubed = [0.0; MAX_ORDER + 1];
        for n in 0..order {
            radius_squared[n] = (0..=n)
                .map(|k| {
                    jet[0][k] * jet[0][n - k]
                        + jet[1][k] * jet[1][n - k]
                        + jet[2][k] * jet[2][n - k]
                })
                .sum();
            power_coefficient(&radius_squared, &mut inverse_radius_cubed, -1.5, n);
            let divisor = (n + 1) as f64;
            for axis in 0..3 {
                jet[axis][n + 1] = jet[axis + 3][n] / divisor;
                jet[axis + 3][n + 1] = -parameters[0]
                    * product_coefficient(&jet[axis], &inverse_radius_cubed, n)
                    / divisor;
            }
        }
        Ok(())
    }
}

impl TaylorDynamicsModel<6, 1> for KeplerDynamics {}

impl TaylorCoefficientModel<6, 1> for Cr3bpDynamics {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 6],
        parameters: &[f64; 1],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 6],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        cr3bp_tape().coefficients(time, state, parameters, order, jet);
        Ok(())
    }
}

impl TaylorDynamicsModel<6, 1> for Cr3bpDynamics {}

fn cr3bp_tape() -> &'static IncrementalTape<6> {
    static TAPE: OnceLock<IncrementalTape<6>> = OnceLock::new();
    TAPE.get_or_init(build_cr3bp_tape)
}

fn build_cr3bp_tape() -> IncrementalTape<6> {
    let builder = TapeBuilder::new();
    let outputs = {
        let state: [Expression<'_>; 6] = core::array::from_fn(|index| builder.state(index));
        let mu = builder.parameter(0);
        cr3bp_expression_rhs(&builder, state, mu).map(Expression::index)
    };
    builder.finish(outputs)
}

fn cr3bp_expression_rhs<'a>(
    builder: &'a TapeBuilder,
    state: [Expression<'a>; 6],
    mu: Expression<'a>,
) -> [Expression<'a>; 6] {
    let one = builder.constant(1.0);
    let d1 = [state[0] + mu, state[1], state[2]];
    let d2 = [state[0] + mu - 1.0, state[1], state[2]];
    let primary = (one - mu) * expression_squared_norm(d1).powf(-1.5);
    let secondary = mu * expression_squared_norm(d2).powf(-1.5);
    [
        state[3],
        state[4],
        state[5],
        state[4] * 2.0 + state[0] - primary * d1[0] - secondary * d2[0],
        -state[3] * 2.0 + state[1] - (primary + secondary) * state[1],
        -(primary + secondary) * state[2],
    ]
}

impl TaylorCoefficientModel<6, 4> for BcpDynamics {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 6],
        parameters: &[f64; 4],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 6],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        bcp_tape().coefficients(time, state, parameters, order, jet);
        Ok(())
    }
}

impl TaylorDynamicsModel<6, 4> for BcpDynamics {}

fn bcp_tape() -> &'static IncrementalTape<6> {
    static TAPE: OnceLock<IncrementalTape<6>> = OnceLock::new();
    TAPE.get_or_init(build_bcp_tape)
}

fn build_bcp_tape() -> IncrementalTape<6> {
    let builder = TapeBuilder::new();
    let outputs = {
        let state: [Expression<'_>; 6] = core::array::from_fn(|index| builder.state(index));
        let parameters: [Expression<'_>; 4] =
            core::array::from_fn(|index| builder.parameter(index));
        let mu = parameters[0];
        let mu_sun = parameters[1];
        let rho_sun = parameters[2];
        let angle = parameters[3] * builder.time();
        let (sine, cosine) = angle.sin_cos();
        let d1 = [state[0] + mu, state[1], state[2]];
        let d2 = [state[0] + mu - 1.0, state[1], state[2]];
        let sun_displacement = [
            state[0] - rho_sun * cosine,
            state[1] - rho_sun * sine,
            state[2],
        ];
        let primary = (builder.constant(1.0) - mu) * expression_squared_norm(d1).powf(-1.5);
        let secondary = mu * expression_squared_norm(d2).powf(-1.5);
        let sun_scale = mu_sun * expression_squared_norm(sun_displacement).powf(-1.5);
        let indirect_scale = -mu_sun / (rho_sun * rho_sun);
        [
            state[3],
            state[4],
            state[5],
            state[4] * 2.0 + state[0]
                - primary * d1[0]
                - secondary * d2[0]
                - sun_scale * sun_displacement[0]
                + indirect_scale * cosine,
            -state[3] * 2.0 + state[1]
                - primary * d1[1]
                - secondary * d2[1]
                - sun_scale * sun_displacement[1]
                + indirect_scale * sine,
            -primary * d1[2] - secondary * d2[2] - sun_scale * sun_displacement[2],
        ]
        .map(Expression::index)
    };
    builder.finish(outputs)
}

impl TaylorCoefficientModel<7, 6> for ZohCr3bpDynamics {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 7],
        parameters: &[f64; 6],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 7],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        zoh_cr3bp_tape().coefficients(time, state, parameters, order, jet);
        Ok(())
    }
}

impl TaylorDynamicsModel<7, 6> for ZohCr3bpDynamics {}

fn zoh_cr3bp_tape() -> &'static IncrementalTape<7> {
    static TAPE: OnceLock<IncrementalTape<7>> = OnceLock::new();
    TAPE.get_or_init(build_zoh_cr3bp_tape)
}

impl TaylorCoefficientModel<7, 5> for ZohEquinoctialDynamics {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 7],
        parameters: &[f64; 5],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 7],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        zoh_equinoctial_tape().coefficients(time, state, parameters, order, jet);
        Ok(())
    }
}

impl TaylorDynamicsModel<7, 5> for ZohEquinoctialDynamics {}

impl TaylorCoefficientModel<14, 5> for EquinoctialMassOptimal {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 14],
        parameters: &[f64; 5],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 14],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        equinoctial_mass_tape(state, parameters).coefficients(time, state, parameters, order, jet);
        Ok(())
    }
}

impl TaylorDynamicsModel<14, 5> for EquinoctialMassOptimal {}

impl TaylorCoefficientModel<14, 3> for EquinoctialTimeOptimal {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 14],
        parameters: &[f64; 3],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 14],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        equinoctial_time_tape().coefficients(time, state, parameters, order, jet);
        Ok(())
    }
}

impl TaylorDynamicsModel<14, 3> for EquinoctialTimeOptimal {}

fn zoh_equinoctial_tape() -> &'static IncrementalTape<7> {
    static TAPE: OnceLock<IncrementalTape<7>> = OnceLock::new();
    TAPE.get_or_init(build_zoh_equinoctial_tape)
}

fn build_zoh_equinoctial_tape() -> IncrementalTape<7> {
    let builder = TapeBuilder::new();
    let outputs = {
        let state: [Expression<'_>; 7] = core::array::from_fn(|index| builder.state(index));
        let parameters: [Expression<'_>; 5] =
            core::array::from_fn(|index| builder.parameter(index));
        let one = builder.constant(1.0);
        let p = state[0];
        let f = state[1];
        let g = state[2];
        let h = state[3];
        let k = state[4];
        let mass = state[6];
        let (sine, cosine) = state[5].sin_cos();
        let w = one + f * cosine + g * sine;
        let s2 = one + h * h + k * k;
        let hsk = h * sine - k * cosine;
        let sqrt_p = p.sqrt();
        let radial_thrust = parameters[1] * parameters[0];
        let transverse_thrust = parameters[2] * parameters[0];
        let normal_thrust = parameters[3] * parameters[0];
        let mass_regularization = (-one / mass / 1e16).exp();
        [
            sqrt_p * (p * 2.0 / w) * transverse_thrust / mass,
            sqrt_p
                * (radial_thrust * sine + ((one + w) * cosine + f) / w * transverse_thrust
                    - g / w * hsk * normal_thrust)
                / mass,
            sqrt_p
                * (-radial_thrust * cosine
                    + ((one + w) * sine + g) / w * transverse_thrust
                    + f / w * hsk * normal_thrust)
                / mass,
            sqrt_p * (s2 / w / 2.0) * cosine * normal_thrust / mass,
            sqrt_p * (s2 / w / 2.0) * sine * normal_thrust / mass,
            sqrt_p * hsk / w * normal_thrust / mass + w * w / p.powf(1.5),
            -parameters[4] * parameters[0] * mass_regularization,
        ]
        .map(Expression::index)
    };
    builder.finish(outputs)
}

impl TaylorCoefficientModel<6, 3> for ZohSolarSailDynamics {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 6],
        parameters: &[f64; 3],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 6],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        zoh_solar_sail_tape().coefficients(time, state, parameters, order, jet);
        Ok(())
    }
}

impl TaylorDynamicsModel<6, 3> for ZohSolarSailDynamics {}

fn zoh_solar_sail_tape() -> &'static IncrementalTape<6> {
    static TAPE: OnceLock<IncrementalTape<6>> = OnceLock::new();
    TAPE.get_or_init(build_zoh_solar_sail_tape)
}

fn build_zoh_solar_sail_tape() -> IncrementalTape<6> {
    let builder = TapeBuilder::new();
    let outputs = {
        let state: [Expression<'_>; 6] = core::array::from_fn(|index| builder.state(index));
        let parameters: [Expression<'_>; 3] =
            core::array::from_fn(|index| builder.parameter(index));
        let position = [state[0], state[1], state[2]];
        let velocity = [state[3], state[4], state[5]];
        let radius_squared = expression_squared_norm(position);
        let radius = radius_squared.sqrt();
        let angular_momentum = expression_cross(position, velocity);
        let angular_norm = expression_squared_norm(angular_momentum).sqrt();
        let radial = position.map(|item| item / radius);
        let normal = angular_momentum.map(|item| item / angular_norm);
        let transverse = expression_cross(normal, radial);
        let (alpha_sine, alpha_cosine) = parameters[0].sin_cos();
        let (beta_sine, beta_cosine) = parameters[1].sin_cos();
        let thrust = parameters[2] / radius_squared * alpha_cosine * alpha_cosine;
        let radial_acceleration = alpha_cosine * thrust;
        let transverse_acceleration = alpha_sine * beta_sine * thrust;
        let normal_acceleration = alpha_sine * beta_cosine * thrust;
        let gravity = -radius_squared.powf(-1.5);
        let acceleration: [Expression<'_>; 3] = core::array::from_fn(|axis| {
            gravity * position[axis]
                + radial_acceleration * radial[axis]
                + transverse_acceleration * transverse[axis]
                + normal_acceleration * normal[axis]
        });
        [
            state[3],
            state[4],
            state[5],
            acceleration[0],
            acceleration[1],
            acceleration[2],
        ]
        .map(Expression::index)
    };
    builder.finish(outputs)
}

fn build_zoh_cr3bp_tape() -> IncrementalTape<7> {
    let builder = TapeBuilder::new();
    let outputs = {
        let state: [Expression<'_>; 7] = core::array::from_fn(|index| builder.state(index));
        let parameters: [Expression<'_>; 6] =
            core::array::from_fn(|index| builder.parameter(index));
        let thrust_over_mass = parameters[0] / state[6];
        let mu = parameters[5];
        let d1 = [state[0] + mu, state[1], state[2]];
        let d2 = [state[0] + mu - 1.0, state[1], state[2]];
        let primary = (builder.constant(1.0) - mu) * expression_squared_norm(d1).powf(-1.5);
        let secondary = mu * expression_squared_norm(d2).powf(-1.5);
        let mass_regularization = (-builder.constant(1.0) / state[6] / 1e16).exp();
        [
            state[3],
            state[4],
            state[5],
            state[4] * 2.0 + state[0] - primary * d1[0] - secondary * d2[0]
                + thrust_over_mass * parameters[1],
            -state[3] * 2.0 + state[1] - (primary + secondary) * state[1]
                + thrust_over_mass * parameters[2],
            -(primary + secondary) * state[2] + thrust_over_mass * parameters[3],
            -parameters[4] * parameters[0] * mass_regularization,
        ]
        .map(Expression::index)
    };
    builder.finish(outputs)
}

impl TaylorCoefficientModel<7, 5> for ZohKeplerDynamics {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 7],
        parameters: &[f64; 5],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 7],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        *jet = [[0.0; MAX_ORDER + 1]; 7];
        for (component, &value) in jet.iter_mut().zip(state) {
            component[0] = value;
        }

        let [thrust, ix, iy, iz, mass_flow_coefficient] = *parameters;
        let thrust_direction = [thrust * ix, thrust * iy, thrust * iz];
        let mut radius_squared = [0.0; MAX_ORDER + 1];
        let mut inverse_radius_cubed = [0.0; MAX_ORDER + 1];
        let mut inverse_mass = [0.0; MAX_ORDER + 1];
        let mut mass_exponent = [0.0; MAX_ORDER + 1];
        let mut mass_regularization = [0.0; MAX_ORDER + 1];

        for n in 0..order {
            radius_squared[n] = (0..3)
                .map(|axis| product_coefficient(&jet[axis], &jet[axis], n))
                .sum();
            power_coefficient(&radius_squared, &mut inverse_radius_cubed, -1.5, n);
            power_coefficient(&jet[6], &mut inverse_mass, -1.0, n);
            mass_exponent[n] = -inverse_mass[n] / 1e16;
            exp_coefficient(&mass_exponent, &mut mass_regularization, n);

            let divisor = (n + 1) as f64;
            for axis in 0..3 {
                jet[axis][n + 1] = jet[axis + 3][n] / divisor;
                jet[axis + 3][n + 1] = (-product_coefficient(&jet[axis], &inverse_radius_cubed, n)
                    + thrust_direction[axis] * inverse_mass[n])
                    / divisor;
            }
            jet[6][n + 1] = -mass_flow_coefficient * thrust * mass_regularization[n] / divisor;
        }
        Ok(())
    }
}

impl TaylorDynamicsModel<7, 5> for ZohKeplerDynamics {}

impl TaylorCoefficientModel<14, 5> for CartesianMassOptimal {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 14],
        parameters: &[f64; 5],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 14],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        cartesian_mass_tape(state, parameters).coefficients(time, state, parameters, order, jet);
        Ok(())
    }
}

impl TaylorDynamicsModel<14, 5> for CartesianMassOptimal {}

impl TaylorCoefficientModel<14, 3> for CartesianTimeOptimal {
    fn coefficients(
        &self,
        time: f64,
        state: &[f64; 14],
        parameters: &[f64; 3],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; 14],
    ) -> Result<()> {
        self.validate(time, state, parameters)?;
        cartesian_time_tape().coefficients(time, state, parameters, order, jet);
        Ok(())
    }
}

impl TaylorDynamicsModel<14, 3> for CartesianTimeOptimal {}

#[cfg(test)]
fn coefficients_from_series_rhs<M, F, const N: usize, const P: usize>(
    model: &M,
    time: f64,
    state: &[f64; N],
    parameters: &[f64; P],
    order: usize,
    jet: &mut [[f64; MAX_ORDER + 1]; N],
    rhs: F,
) -> Result<()>
where
    M: DynamicsModel<N, P>,
    F: Fn(Series, &[Series; N], &[Series; P]) -> [Series; N],
{
    model.validate(time, state, parameters)?;
    *jet = [[0.0; MAX_ORDER + 1]; N];
    for (component, &value) in jet.iter_mut().zip(state) {
        component[0] = value;
    }
    for coefficient_order in 0..order {
        let series_state =
            core::array::from_fn(|index| Series::from_coefficients(&jet[index], coefficient_order));
        let series_parameters = parameters.map(|value| Series::constant(value, coefficient_order));
        let derivative = rhs(
            Series::variable(time, coefficient_order),
            &series_state,
            &series_parameters,
        );
        let divisor = (coefficient_order + 1) as f64;
        for index in 0..N {
            jet[index][coefficient_order + 1] =
                derivative[index].coefficient(coefficient_order) / divisor;
        }
    }
    Ok(())
}

#[cfg(test)]
fn cr3bp_rhs(_time: Series, state: &[Series; 6], parameters: &[Series; 1]) -> [Series; 6] {
    let one = Series::constant(1.0, state[0].order());
    let two = Series::constant(2.0, state[0].order());
    let mu = parameters[0];
    let d1 = [state[0] + mu, state[1], state[2]];
    let d2 = [state[0] + mu - 1.0, state[1], state[2]];
    let primary = (one - mu) * squared_norm(d1).powf(-1.5);
    let secondary = mu * squared_norm(d2).powf(-1.5);
    [
        state[3],
        state[4],
        state[5],
        two * state[4] + state[0] - primary * d1[0] - secondary * d2[0],
        -two * state[3] + state[1] - (primary + secondary) * state[1],
        -(primary + secondary) * state[2],
    ]
}

#[cfg(test)]
fn bcp_rhs(time: Series, state: &[Series; 6], parameters: &[Series; 4]) -> [Series; 6] {
    let order = state[0].order();
    let one = Series::constant(1.0, order);
    let two = Series::constant(2.0, order);
    let mu = parameters[0];
    let mu_sun = parameters[1];
    let rho_sun = parameters[2];
    let angle = parameters[3] * time;
    let (sine, cosine) = angle.sin_cos();
    let sun_direction = [cosine, sine, Series::constant(0.0, order)];
    let d1 = [state[0] + mu, state[1], state[2]];
    let d2 = [state[0] + mu - 1.0, state[1], state[2]];
    let sun_displacement = [
        state[0] - rho_sun * sun_direction[0],
        state[1] - rho_sun * sun_direction[1],
        state[2],
    ];
    let primary = (one - mu) * squared_norm(d1).powf(-1.5);
    let secondary = mu * squared_norm(d2).powf(-1.5);
    let sun_scale = mu_sun * squared_norm(sun_displacement).powf(-1.5);
    let indirect_scale = -mu_sun / (rho_sun * rho_sun);
    [
        state[3],
        state[4],
        state[5],
        two * state[4] + state[0]
            - primary * d1[0]
            - secondary * d2[0]
            - sun_scale * sun_displacement[0]
            + indirect_scale * sun_direction[0],
        -two * state[3] + state[1]
            - primary * d1[1]
            - secondary * d2[1]
            - sun_scale * sun_displacement[1]
            + indirect_scale * sun_direction[1],
        -primary * d1[2] - secondary * d2[2] - sun_scale * sun_displacement[2],
    ]
}

#[cfg(test)]
fn zoh_kepler_rhs(_time: Series, state: &[Series; 7], parameters: &[Series; 5]) -> [Series; 7] {
    let gravity = -squared_norm([state[0], state[1], state[2]]).powf(-1.5);
    let thrust_over_mass = parameters[0] / state[6];
    [
        state[3],
        state[4],
        state[5],
        gravity * state[0] + thrust_over_mass * parameters[1],
        gravity * state[1] + thrust_over_mass * parameters[2],
        gravity * state[2] + thrust_over_mass * parameters[3],
        regularized_mass_flow(state[6], parameters[0], parameters[4], 1e16),
    ]
}

#[cfg(test)]
fn zoh_cr3bp_rhs(_time: Series, state: &[Series; 7], parameters: &[Series; 6]) -> [Series; 7] {
    let order = state[0].order();
    let one = Series::constant(1.0, order);
    let two = Series::constant(2.0, order);
    let thrust_over_mass = parameters[0] / state[6];
    let mu = parameters[5];
    let d1 = [state[0] + mu, state[1], state[2]];
    let d2 = [state[0] + mu - 1.0, state[1], state[2]];
    let primary = (one - mu) * squared_norm(d1).powf(-1.5);
    let secondary = mu * squared_norm(d2).powf(-1.5);
    [
        state[3],
        state[4],
        state[5],
        two * state[4] + state[0] - primary * d1[0] - secondary * d2[0]
            + thrust_over_mass * parameters[1],
        -two * state[3] + state[1] - (primary + secondary) * state[1]
            + thrust_over_mass * parameters[2],
        -(primary + secondary) * state[2] + thrust_over_mass * parameters[3],
        regularized_mass_flow(state[6], parameters[0], parameters[4], 1e16),
    ]
}

#[cfg(test)]
fn zoh_equinoctial_rhs(
    _time: Series,
    state: &[Series; 7],
    parameters: &[Series; 5],
) -> [Series; 7] {
    let order = state[0].order();
    let one = Series::constant(1.0, order);
    let two = Series::constant(2.0, order);
    let p = state[0];
    let f = state[1];
    let g = state[2];
    let h = state[3];
    let k = state[4];
    let mass = state[6];
    let (sine, cosine) = state[5].sin_cos();
    let w = one + f * cosine + g * sine;
    let s2 = one + h * h + k * k;
    let hsk = h * sine - k * cosine;
    let sqrt_p = p.sqrt();
    let radial_thrust = parameters[1] * parameters[0];
    let transverse_thrust = parameters[2] * parameters[0];
    let normal_thrust = parameters[3] * parameters[0];
    [
        sqrt_p * (two * p / w) * transverse_thrust / mass,
        sqrt_p
            * (radial_thrust * sine + ((one + w) * cosine + f) / w * transverse_thrust
                - g / w * hsk * normal_thrust)
            / mass,
        sqrt_p
            * (-radial_thrust * cosine
                + ((one + w) * sine + g) / w * transverse_thrust
                + f / w * hsk * normal_thrust)
            / mass,
        sqrt_p * (s2 / w / 2.0) * cosine * normal_thrust / mass,
        sqrt_p * (s2 / w / 2.0) * sine * normal_thrust / mass,
        sqrt_p * hsk / w * normal_thrust / mass + w * w / p.powf(1.5),
        regularized_mass_flow(mass, parameters[0], parameters[4], 1e16),
    ]
}

#[cfg(test)]
fn zoh_solar_sail_rhs(_time: Series, state: &[Series; 6], parameters: &[Series; 3]) -> [Series; 6] {
    let radius_squared = squared_norm([state[0], state[1], state[2]]);
    let radius = radius_squared.sqrt();
    let angular_momentum = cross(
        [state[0], state[1], state[2]],
        [state[3], state[4], state[5]],
    );
    let angular_norm = squared_norm(angular_momentum).sqrt();
    let radial = [state[0] / radius, state[1] / radius, state[2] / radius];
    let normal = angular_momentum.map(|item| item / angular_norm);
    let transverse = cross(normal, radial);
    let alpha_sine = parameters[0].sin();
    let alpha_cosine = parameters[0].cos();
    let beta_sine = parameters[1].sin();
    let beta_cosine = parameters[1].cos();
    let thrust = parameters[2] / radius_squared * alpha_cosine * alpha_cosine;
    let radial_acceleration = alpha_cosine * thrust;
    let transverse_acceleration = alpha_sine * beta_sine * thrust;
    let normal_acceleration = alpha_sine * beta_cosine * thrust;
    let gravity = -radius_squared.powf(-1.5);
    [
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
    ]
}

#[cfg(test)]
fn squared_norm<const N: usize>(vector: [Series; N]) -> Series {
    vector
        .into_iter()
        .map(|item| item * item)
        .reduce(|left, right| left + right)
        .expect("dynamics vectors are nonempty")
}

#[cfg(test)]
fn cross(left: [Series; 3], right: [Series; 3]) -> [Series; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn expression_squared_norm<'a, const N: usize>(vector: [Expression<'a>; N]) -> Expression<'a> {
    vector
        .into_iter()
        .map(|item| item * item)
        .reduce(|left, right| left + right)
        .expect("dynamics vectors are nonempty")
}

fn expression_cross<'a>(
    left: [Expression<'a>; 3],
    right: [Expression<'a>; 3],
) -> [Expression<'a>; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[cfg(test)]
fn regularized_mass_flow(
    mass: Series,
    thrust: Series,
    coefficient: Series,
    regularization: f64,
) -> Series {
    -coefficient * thrust * (-Series::constant(1.0, mass.order()) / mass / regularization).exp()
}

#[derive(Clone, Copy)]
enum Objective {
    Mass,
    Time,
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct PontryaginParameters {
    mu: f64,
    maximum_thrust: f64,
    exhaust_velocity: f64,
    barrier: f64,
    lambda0: f64,
    objective: Objective,
}

#[cfg(test)]
fn cartesian_mass_rhs(
    _time: Series,
    state: &[Series; 14],
    parameters: &[Series; 5],
) -> [Series; 14] {
    cartesian_pontryagin_rhs(
        state,
        PontryaginParameters {
            mu: parameters[0].coefficient(0),
            maximum_thrust: parameters[1].coefficient(0),
            exhaust_velocity: parameters[2].coefficient(0),
            barrier: parameters[3].coefficient(0),
            lambda0: parameters[4].coefficient(0),
            objective: Objective::Mass,
        },
    )
}

fn cartesian_mass_tape(state: &[f64; 14], parameters: &[f64; 5]) -> &'static IncrementalTape<14> {
    static POSITIVE_TAPE: OnceLock<IncrementalTape<14>> = OnceLock::new();
    static NEGATIVE_TAPE: OnceLock<IncrementalTape<14>> = OnceLock::new();
    let primer_norm = (state[10].powi(2) + state[11].powi(2) + state[12].powi(2)).sqrt();
    let switching_function =
        1.0 - primer_norm * parameters[2] / state[6] / parameters[4] - state[13] / parameters[4];
    if switching_function >= 0.0 {
        POSITIVE_TAPE.get_or_init(|| build_cartesian_tape(Objective::Mass, true))
    } else {
        NEGATIVE_TAPE.get_or_init(|| build_cartesian_tape(Objective::Mass, false))
    }
}

fn cartesian_time_tape() -> &'static IncrementalTape<14> {
    static TAPE: OnceLock<IncrementalTape<14>> = OnceLock::new();
    TAPE.get_or_init(|| build_cartesian_tape(Objective::Time, false))
}

fn build_cartesian_tape(
    objective: Objective,
    positive_switching_function: bool,
) -> IncrementalTape<14> {
    let builder = TapeBuilder::new();
    let outputs = {
        let state: [Expression<'_>; 14] = core::array::from_fn(|index| builder.state(index));
        let zero = builder.constant(0.0);
        let one = builder.constant(1.0);
        let mu = builder.parameter(0);
        let maximum_thrust = builder.parameter(1);
        let exhaust_velocity = builder.parameter(2);
        let primer = [state[10], state[11], state[12]];
        let primer_norm =
            (primer[0] * primer[0] + primer[1] * primer[1] + primer[2] * primer[2]).sqrt();
        let direction = primer.map(|item| (-item / primer_norm).stop_gradient());
        let throttle = match objective {
            Objective::Time => one,
            Objective::Mass => {
                let barrier = builder.parameter(3);
                let lambda0 = builder.parameter(4);
                let switching_function =
                    one - primer_norm * exhaust_velocity / state[6] / lambda0 - state[13] / lambda0;
                let twice_barrier = barrier * 2.0;
                let root = (switching_function * switching_function
                    + twice_barrier * twice_barrier)
                    .sqrt();
                if positive_switching_function {
                    twice_barrier / (switching_function + twice_barrier + root)
                } else {
                    (root - switching_function) / (root - switching_function + twice_barrier)
                }
                .stop_gradient()
            }
        };

        let physical_state: [Expression<'_>; 7] = core::array::from_fn(|index| state[index]);
        let radius_squared = physical_state[0] * physical_state[0]
            + physical_state[1] * physical_state[1]
            + physical_state[2] * physical_state[2];
        let gravity_scale = -mu / radius_squared.powf(1.5);
        let thrust_scale = maximum_thrust * throttle / physical_state[6];
        let physical = [
            physical_state[3],
            physical_state[4],
            physical_state[5],
            gravity_scale * physical_state[0] + thrust_scale * direction[0],
            gravity_scale * physical_state[1] + thrust_scale * direction[1],
            gravity_scale * physical_state[2] + thrust_scale * direction[2],
            -maximum_thrust / exhaust_velocity * throttle,
        ];
        let hamiltonian = (0..7)
            .map(|index| state[index + 7] * physical[index])
            .reduce(|left, right| left + right)
            .expect("Cartesian Hamiltonian has seven terms");
        let hamiltonian_gradient = builder.gradient(hamiltonian, physical_state);
        let mut derivative = [zero; 14];
        for index in 0..7 {
            derivative[index] = physical[index];
            derivative[index + 7] = -hamiltonian_gradient[index];
        }
        derivative.map(Expression::index)
    };
    builder.finish(outputs)
}

fn equinoctial_mass_tape(state: &[f64; 14], parameters: &[f64; 5]) -> &'static IncrementalTape<14> {
    static POSITIVE_TAPE: OnceLock<IncrementalTape<14>> = OnceLock::new();
    static NEGATIVE_TAPE: OnceLock<IncrementalTape<14>> = OnceLock::new();
    let primer_norm = equinoctial_primer_norm(state, parameters[0]);
    let switching_function =
        1.0 - primer_norm * parameters[2] / state[6] / parameters[4] - state[13] / parameters[4];
    if switching_function >= 0.0 {
        POSITIVE_TAPE.get_or_init(|| build_equinoctial_tape(Objective::Mass, true))
    } else {
        NEGATIVE_TAPE.get_or_init(|| build_equinoctial_tape(Objective::Mass, false))
    }
}

fn equinoctial_time_tape() -> &'static IncrementalTape<14> {
    static TAPE: OnceLock<IncrementalTape<14>> = OnceLock::new();
    TAPE.get_or_init(|| build_equinoctial_tape(Objective::Time, false))
}

fn equinoctial_primer_norm(state: &[f64; 14], mu: f64) -> f64 {
    let [p, f, g, h, k, longitude, _mass, ..] = *state;
    let (sine, cosine) = longitude.sin_cos();
    let w = 1.0 + f * cosine + g * sine;
    let s2 = 1.0 + h * h + k * k;
    let hsk = h * sine - k * cosine;
    let scale = (p / mu).sqrt();
    let primer = [
        scale * (state[8] * sine - state[9] * cosine),
        scale
            * (state[7] * 2.0 * p / w
                + state[8] * ((1.0 + w) * cosine + f) / w
                + state[9] * ((1.0 + w) * sine + g) / w),
        scale
            * (-state[8] * g * hsk / w
                + state[9] * f * hsk / w
                + state[10] * s2 * cosine / (2.0 * w)
                + state[11] * s2 * sine / (2.0 * w)
                + state[12] * hsk / w),
    ];
    expression_norm_value(primer)
}

fn expression_norm_value<const N: usize>(vector: [f64; N]) -> f64 {
    vector
        .into_iter()
        .map(|item| item * item)
        .sum::<f64>()
        .sqrt()
}

fn build_equinoctial_tape(
    objective: Objective,
    positive_switching_function: bool,
) -> IncrementalTape<14> {
    let builder = TapeBuilder::new();
    let outputs = {
        let state: [Expression<'_>; 14] = core::array::from_fn(|index| builder.state(index));
        let physical_state: [Expression<'_>; 7] = core::array::from_fn(|index| state[index]);
        let zero = builder.constant(0.0);
        let one = builder.constant(1.0);
        let mu = builder.parameter(0);
        let maximum_thrust = builder.parameter(1);
        let exhaust_velocity = builder.parameter(2);
        let (matrix, w) = equinoctial_b_expression(&builder, physical_state, mu);
        let primer: [Expression<'_>; 3] = core::array::from_fn(|column| {
            (0..6)
                .map(|row| matrix[row][column] * state[row + 7])
                .reduce(|left, right| left + right)
                .expect("equinoctial matrix has six rows")
        });
        let primer_norm = expression_squared_norm(primer).sqrt();
        let direction = primer.map(|item| (-item / primer_norm).stop_gradient());
        let throttle = match objective {
            Objective::Time => one,
            Objective::Mass => {
                let barrier = builder.parameter(3);
                let lambda0 = builder.parameter(4);
                let switching_function =
                    one - primer_norm * exhaust_velocity / state[6] / lambda0 - state[13] / lambda0;
                let twice_barrier = barrier * 2.0;
                let root = (switching_function * switching_function
                    + twice_barrier * twice_barrier)
                    .sqrt();
                if positive_switching_function {
                    twice_barrier / (switching_function + twice_barrier + root)
                } else {
                    (root - switching_function) / (root - switching_function + twice_barrier)
                }
                .stop_gradient()
            }
        };
        let thrust_scale = maximum_thrust * throttle / physical_state[6];
        let mut physical = [zero; 7];
        for row in 0..6 {
            physical[row] = (matrix[row][0] * direction[0]
                + matrix[row][1] * direction[1]
                + matrix[row][2] * direction[2])
                * thrust_scale;
        }
        physical[5] = physical[5] + (mu / physical_state[0].powf(3.0)).sqrt() * w * w;
        physical[6] =
            -maximum_thrust / exhaust_velocity * throttle * (-one / physical_state[6] / 1e10).exp();

        let hamiltonian = (0..7)
            .map(|index| state[index + 7] * physical[index])
            .reduce(|left, right| left + right)
            .expect("equinoctial Hamiltonian has seven terms");
        let hamiltonian_gradient = builder.gradient(hamiltonian, physical_state);
        let mut derivative = [zero; 14];
        for index in 0..7 {
            derivative[index] = physical[index];
            derivative[index + 7] = -hamiltonian_gradient[index];
        }
        derivative.map(Expression::index)
    };
    builder.finish(outputs)
}

fn equinoctial_b_expression<'a>(
    builder: &'a TapeBuilder,
    state: [Expression<'a>; 7],
    mu: Expression<'a>,
) -> ([[Expression<'a>; 3]; 6], Expression<'a>) {
    let zero = builder.constant(0.0);
    let one = builder.constant(1.0);
    let (sine, cosine) = state[5].sin_cos();
    let w = one + state[1] * cosine + state[2] * sine;
    let s2 = one + state[3] * state[3] + state[4] * state[4];
    let hsk = state[3] * sine - state[4] * cosine;
    let scale = (state[0] / mu).sqrt();
    (
        [
            [zero, scale * state[0] * 2.0 / w, zero],
            [
                scale * sine,
                scale * ((one + w) * cosine + state[1]) / w,
                -scale * state[2] * hsk / w,
            ],
            [
                -scale * cosine,
                scale * ((one + w) * sine + state[2]) / w,
                scale * state[1] * hsk / w,
            ],
            [zero, zero, scale * s2 * cosine / (w * 2.0)],
            [zero, zero, scale * s2 * sine / (w * 2.0)],
            [zero, zero, scale * hsk / w],
        ],
        w,
    )
}

#[cfg(test)]
fn cartesian_time_rhs(
    _time: Series,
    state: &[Series; 14],
    parameters: &[Series; 3],
) -> [Series; 14] {
    cartesian_pontryagin_rhs(
        state,
        PontryaginParameters {
            mu: parameters[0].coefficient(0),
            maximum_thrust: parameters[1].coefficient(0),
            exhaust_velocity: parameters[2].coefficient(0),
            barrier: 0.0,
            lambda0: 1.0,
            objective: Objective::Time,
        },
    )
}

#[cfg(test)]
fn equinoctial_mass_rhs(
    _time: Series,
    state: &[Series; 14],
    parameters: &[Series; 5],
) -> [Series; 14] {
    equinoctial_pontryagin_rhs(
        state,
        PontryaginParameters {
            mu: parameters[0].coefficient(0),
            maximum_thrust: parameters[1].coefficient(0),
            exhaust_velocity: parameters[2].coefficient(0),
            barrier: parameters[3].coefficient(0),
            lambda0: parameters[4].coefficient(0),
            objective: Objective::Mass,
        },
    )
}

#[cfg(test)]
fn equinoctial_time_rhs(
    _time: Series,
    state: &[Series; 14],
    parameters: &[Series; 3],
) -> [Series; 14] {
    equinoctial_pontryagin_rhs(
        state,
        PontryaginParameters {
            mu: parameters[0].coefficient(0),
            maximum_thrust: parameters[1].coefficient(0),
            exhaust_velocity: parameters[2].coefficient(0),
            barrier: 0.0,
            lambda0: 1.0,
            objective: Objective::Time,
        },
    )
}

#[cfg(test)]
fn cartesian_pontryagin_rhs(
    state: &[Series; 14],
    parameters: PontryaginParameters,
) -> [Series; 14] {
    let order = state[0].order();
    let zero = Series::constant(0.0, order);
    let one = Series::constant(1.0, order);
    let primer = [state[10], state[11], state[12]];
    let primer_norm = squared_norm(primer).sqrt();
    let direction = primer.map(|item| -item / primer_norm);
    let switching_function = match parameters.objective {
        Objective::Mass => {
            one - primer_norm * parameters.exhaust_velocity / state[6] / parameters.lambda0
                - state[13] / parameters.lambda0
        }
        Objective::Time => {
            -primer_norm * parameters.exhaust_velocity / state[6] / parameters.lambda0
                - state[13] / parameters.lambda0
        }
    };
    let throttle = throttle(switching_function, parameters);

    let physical_state: [SeriesDual; 7] =
        core::array::from_fn(|index| SeriesDual::variable(state[index], index));
    let radius_squared = physical_state[0] * physical_state[0]
        + physical_state[1] * physical_state[1]
        + physical_state[2] * physical_state[2];
    let gravity_scale =
        SeriesDual::constant(Series::constant(-parameters.mu, order)) / radius_squared.powf(1.5);
    let thrust_scale =
        SeriesDual::constant(Series::constant(parameters.maximum_thrust, order) * throttle)
            / physical_state[6];
    let physical = [
        physical_state[3],
        physical_state[4],
        physical_state[5],
        gravity_scale * physical_state[0] + thrust_scale * SeriesDual::constant(direction[0]),
        gravity_scale * physical_state[1] + thrust_scale * SeriesDual::constant(direction[1]),
        gravity_scale * physical_state[2] + thrust_scale * SeriesDual::constant(direction[2]),
        SeriesDual::constant(
            Series::constant(
                -parameters.maximum_thrust / parameters.exhaust_velocity,
                order,
            ) * throttle,
        ),
    ];
    let mut hamiltonian = SeriesDual::constant(zero);
    let mut derivative = [zero; 14];
    for index in 0..7 {
        derivative[index] = physical[index].value;
        hamiltonian = hamiltonian + SeriesDual::constant(state[index + 7]) * physical[index];
    }
    for index in 0..7 {
        derivative[index + 7] = -hamiltonian.derivative[index];
    }
    derivative
}

#[cfg(test)]
fn equinoctial_pontryagin_rhs(
    state: &[Series; 14],
    parameters: PontryaginParameters,
) -> [Series; 14] {
    let order = state[0].order();
    let zero = Series::constant(0.0, order);
    let one = Series::constant(1.0, order);
    let matrix = equinoctial_b_series(state, parameters.mu);
    let primer: [Series; 3] = core::array::from_fn(|column| {
        (0..6)
            .map(|row| matrix[row][column] * state[row + 7])
            .reduce(|left, right| left + right)
            .expect("equinoctial matrix has six rows")
    });
    let primer_norm = squared_norm(primer).sqrt();
    let direction = primer.map(|item| -item / primer_norm);
    let switching_function = match parameters.objective {
        Objective::Mass => {
            one - primer_norm * parameters.exhaust_velocity / state[6] / parameters.lambda0
                - state[13] / parameters.lambda0
        }
        Objective::Time => {
            -primer_norm * parameters.exhaust_velocity / state[6] / parameters.lambda0
                - state[13] / parameters.lambda0
        }
    };
    let throttle = throttle(switching_function, parameters);

    let physical_state: [SeriesDual; 7] =
        core::array::from_fn(|index| SeriesDual::variable(state[index], index));
    let dual_matrix = equinoctial_b_dual(physical_state, parameters.mu);
    let thrust_scale =
        SeriesDual::constant(Series::constant(parameters.maximum_thrust, order) * throttle)
            / physical_state[6];
    let mut physical = [SeriesDual::constant(zero); 7];
    for row in 0..6 {
        physical[row] = (dual_matrix[row][0] * SeriesDual::constant(direction[0])
            + dual_matrix[row][1] * SeriesDual::constant(direction[1])
            + dual_matrix[row][2] * SeriesDual::constant(direction[2]))
            * thrust_scale;
    }
    let sine = physical_state[5].sin();
    let cosine = physical_state[5].cos();
    let w = SeriesDual::constant(one) + physical_state[1] * cosine + physical_state[2] * sine;
    physical[5] = physical[5]
        + (SeriesDual::constant(Series::constant(parameters.mu, order))
            / physical_state[0].powf(3.0))
        .sqrt()
            * w
            * w;
    physical[6] = scalar_dual(
        -parameters.maximum_thrust / parameters.exhaust_velocity,
        order,
    ) * SeriesDual::constant(throttle)
        * (-scalar_dual(1.0, order) / physical_state[6] / scalar_dual(1e10, order)).exp();

    let mut hamiltonian = SeriesDual::constant(zero);
    let mut derivative = [zero; 14];
    for index in 0..7 {
        derivative[index] = physical[index].value;
        hamiltonian = hamiltonian + SeriesDual::constant(state[index + 7]) * physical[index];
    }
    for index in 0..7 {
        derivative[index + 7] = -hamiltonian.derivative[index];
    }
    derivative
}

#[cfg(test)]
fn throttle(switching_function: Series, parameters: PontryaginParameters) -> Series {
    match parameters.objective {
        Objective::Time => Series::constant(1.0, switching_function.order()),
        Objective::Mass => {
            let twice_barrier = 2.0 * parameters.barrier;
            let root =
                (switching_function * switching_function + twice_barrier * twice_barrier).sqrt();
            if switching_function.coefficient(0) >= 0.0 {
                Series::constant(twice_barrier, switching_function.order())
                    / (switching_function + twice_barrier + root)
            } else {
                (root - switching_function) / (root - switching_function + twice_barrier)
            }
        }
    }
}

#[cfg(test)]
fn equinoctial_b_series(state: &[Series; 14], mu: f64) -> [[Series; 3]; 6] {
    let order = state[0].order();
    let zero = Series::constant(0.0, order);
    let one = Series::constant(1.0, order);
    let (sine, cosine) = state[5].sin_cos();
    let w = one + state[1] * cosine + state[2] * sine;
    let s2 = one + state[3] * state[3] + state[4] * state[4];
    let hsk = state[3] * sine - state[4] * cosine;
    let scale = (state[0] / mu).sqrt();
    [
        [zero, scale * 2.0 * state[0] / w, zero],
        [
            scale * sine,
            scale * ((one + w) * cosine + state[1]) / w,
            -scale * state[2] * hsk / w,
        ],
        [
            -scale * cosine,
            scale * ((one + w) * sine + state[2]) / w,
            scale * state[1] * hsk / w,
        ],
        [zero, zero, scale * s2 * cosine / (w * 2.0)],
        [zero, zero, scale * s2 * sine / (w * 2.0)],
        [zero, zero, scale * hsk / w],
    ]
}

#[cfg(test)]
fn equinoctial_b_dual(state: [SeriesDual; 7], mu: f64) -> [[SeriesDual; 3]; 6] {
    let order = state[0].value.order();
    let zero = SeriesDual::constant(Series::constant(0.0, order));
    let one = SeriesDual::constant(Series::constant(1.0, order));
    let sine = state[5].sin();
    let cosine = state[5].cos();
    let w = one + state[1] * cosine + state[2] * sine;
    let s2 = one + state[3] * state[3] + state[4] * state[4];
    let hsk = state[3] * sine - state[4] * cosine;
    let scale = (state[0] / SeriesDual::constant(Series::constant(mu, order))).sqrt();
    [
        [zero, scale * state[0] * scalar_dual(2.0, order) / w, zero],
        [
            scale * sine,
            scale * ((one + w) * cosine + state[1]) / w,
            -scale * state[2] * hsk / w,
        ],
        [
            -scale * cosine,
            scale * ((one + w) * sine + state[2]) / w,
            scale * state[1] * hsk / w,
        ],
        [
            zero,
            zero,
            scale * s2 * cosine / (scalar_dual(2.0, order) * w),
        ],
        [
            zero,
            zero,
            scale * s2 * sine / (scalar_dual(2.0, order) * w),
        ],
        [zero, zero, scale * hsk / w],
    ]
}

#[cfg(test)]
fn scalar_dual(value: f64, order: usize) -> SeriesDual {
    SeriesDual::constant(Series::constant(value, order))
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct SeriesDual {
    value: Series,
    derivative: [Series; 7],
}

#[cfg(test)]
impl SeriesDual {
    fn constant(value: Series) -> Self {
        Self {
            value,
            derivative: [Series::constant(0.0, value.order()); 7],
        }
    }

    fn variable(value: Series, index: usize) -> Self {
        let mut result = Self::constant(value);
        result.derivative[index] = Series::constant(1.0, value.order());
        result
    }

    fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        let scale = self.value.powf(exponent - 1.0) * exponent;
        Self {
            value,
            derivative: self.derivative.map(|item| item * scale),
        }
    }

    fn sqrt(self) -> Self {
        self.powf(0.5)
    }

    fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            derivative: self.derivative.map(|item| item * value),
        }
    }

    fn sin(self) -> Self {
        let value = self.value.sin();
        let scale = self.value.cos();
        Self {
            value,
            derivative: self.derivative.map(|item| item * scale),
        }
    }

    fn cos(self) -> Self {
        let value = self.value.cos();
        let scale = -self.value.sin();
        Self {
            value,
            derivative: self.derivative.map(|item| item * scale),
        }
    }
}

#[cfg(test)]
impl core::ops::Add for SeriesDual {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            derivative: core::array::from_fn(|index| {
                self.derivative[index] + rhs.derivative[index]
            }),
        }
    }
}

#[cfg(test)]
impl core::ops::Sub for SeriesDual {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            derivative: core::array::from_fn(|index| {
                self.derivative[index] - rhs.derivative[index]
            }),
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
#[cfg(test)]
impl core::ops::Mul for SeriesDual {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            derivative: core::array::from_fn(|index| {
                self.derivative[index] * rhs.value + self.value * rhs.derivative[index]
            }),
        }
    }
}

#[cfg(test)]
impl core::ops::Div for SeriesDual {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denominator = rhs.value * rhs.value;
        Self {
            value: self.value / rhs.value,
            derivative: core::array::from_fn(|index| {
                (self.derivative[index] * rhs.value - self.value * rhs.derivative[index])
                    / denominator
            }),
        }
    }
}

#[cfg(test)]
impl core::ops::Neg for SeriesDual {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            derivative: self.derivative.map(|item| -item),
        }
    }
}

fn power_coefficient(
    input: &[f64; MAX_ORDER + 1],
    output: &mut [f64; MAX_ORDER + 1],
    exponent: f64,
    order: usize,
) {
    if order == 0 {
        output[0] = input[0].powf(exponent);
        return;
    }
    let n = order as f64;
    let mut sum = 0.0;
    for k in 0..order {
        let kf = k as f64;
        sum += (n * exponent - kf * (exponent + 1.0)) * input[order - k] * output[k];
    }
    output[order] = sum / (n * input[0]);
}

fn product_coefficient(
    left: &[f64; MAX_ORDER + 1],
    right: &[f64; MAX_ORDER + 1],
    order: usize,
) -> f64 {
    (0..=order).map(|k| left[k] * right[order - k]).sum()
}

fn exp_coefficient(input: &[f64; MAX_ORDER + 1], output: &mut [f64; MAX_ORDER + 1], order: usize) {
    if order == 0 {
        output[0] = input[0].exp();
        return;
    }
    output[order] = (1..=order)
        .map(|k| k as f64 * input[k] * output[order - k])
        .sum::<f64>()
        / order as f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_jets_match<const N: usize>(
        incremental: &[[f64; MAX_ORDER + 1]; N],
        reference: &[[f64; MAX_ORDER + 1]; N],
        order: usize,
        tolerance: f64,
    ) {
        for component in 0..N {
            for coefficient in 0..=order {
                let scale = reference[component][coefficient].abs().max(1.0);
                assert!(
                    (incremental[component][coefficient] - reference[component][coefficient]).abs()
                        <= tolerance * scale,
                    "order={order}, state={component}, coefficient={coefficient}: \
                     incremental={:.17e}, reference={:.17e}",
                    incremental[component][coefficient],
                    reference[component][coefficient],
                );
            }
        }
    }

    #[test]
    fn incremental_cr3bp_matches_series_reference() {
        let state = [0.8, -0.2, 0.1, 0.03, 1.0, 0.02];
        let parameters = [0.012_150_585_609_624_04];
        assert!(cr3bp_tape().operation_count() < 50);

        for order in [8, 15, MAX_ORDER] {
            let mut incremental = [[0.0; MAX_ORDER + 1]; 6];
            Cr3bpDynamics
                .coefficients(0.3, &state, &parameters, order, &mut incremental)
                .unwrap();
            let mut reference = [[0.0; MAX_ORDER + 1]; 6];
            coefficients_from_series_rhs(
                &Cr3bpDynamics,
                0.3,
                &state,
                &parameters,
                order,
                &mut reference,
                cr3bp_rhs,
            )
            .unwrap();
            assert_jets_match(&incremental, &reference, order, 3e-13);
        }
    }

    #[test]
    fn incremental_cartesian_time_matches_series_reference() {
        let state = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let parameters = [1.0, 0.01, 1.0];
        assert!(cartesian_time_tape().operation_count() < 150);

        for order in [8, 15, MAX_ORDER] {
            let mut incremental = [[0.0; MAX_ORDER + 1]; 14];
            CartesianTimeOptimal
                .coefficients(0.0, &state, &parameters, order, &mut incremental)
                .unwrap();
            let mut reference = [[0.0; MAX_ORDER + 1]; 14];
            coefficients_from_series_rhs(
                &CartesianTimeOptimal,
                0.0,
                &state,
                &parameters,
                order,
                &mut reference,
                cartesian_time_rhs,
            )
            .unwrap();
            assert_jets_match(&incremental, &reference, order, 2e-11);
        }
    }

    #[test]
    fn incremental_equinoctial_time_matches_series_reference() {
        let state = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ];
        let parameters = [1.0, 1e-4, 1.0];
        assert!(equinoctial_time_tape().operation_count() < 1_000);

        for order in [8, 15, MAX_ORDER] {
            let mut incremental = [[0.0; MAX_ORDER + 1]; 14];
            EquinoctialTimeOptimal
                .coefficients(0.0, &state, &parameters, order, &mut incremental)
                .unwrap();
            let mut reference = [[0.0; MAX_ORDER + 1]; 14];
            coefficients_from_series_rhs(
                &EquinoctialTimeOptimal,
                0.0,
                &state,
                &parameters,
                order,
                &mut reference,
                equinoctial_time_rhs,
            )
            .unwrap();
            assert_jets_match(&incremental, &reference, order, 1e-5);
        }
    }

    #[test]
    fn incremental_equinoctial_mass_matches_series_reference() {
        let negative_switching_state = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ];
        let mut positive_switching_state = negative_switching_state;
        positive_switching_state[13] = -1.0;
        let parameters = [1.0, 1e-4, 1.0, 1.0, 1e-4];
        let switching_function = |state: &[f64; 14]| {
            1.0 - equinoctial_primer_norm(state, parameters[0]) * parameters[2]
                / state[6]
                / parameters[4]
                - state[13] / parameters[4]
        };
        assert!(switching_function(&negative_switching_state) < 0.0);
        assert!(switching_function(&positive_switching_state) >= 0.0);

        for state in [negative_switching_state, positive_switching_state] {
            let tape = equinoctial_mass_tape(&state, &parameters);
            assert!(tape.operation_count() < 1_100);

            for order in [8, 15, MAX_ORDER] {
                let mut incremental = [[0.0; MAX_ORDER + 1]; 14];
                EquinoctialMassOptimal
                    .coefficients(0.0, &state, &parameters, order, &mut incremental)
                    .unwrap();
                let mut reference = [[0.0; MAX_ORDER + 1]; 14];
                coefficients_from_series_rhs(
                    &EquinoctialMassOptimal,
                    0.0,
                    &state,
                    &parameters,
                    order,
                    &mut reference,
                    equinoctial_mass_rhs,
                )
                .unwrap();
                assert_jets_match(&incremental, &reference, order, 1e-4);
            }
        }
    }

    #[test]
    fn incremental_zoh_cr3bp_matches_series_reference() {
        let state = [0.9, -0.2, 0.1, 0.15, 0.95, -0.04, 1.2];
        let parameters = [0.07, 0.3, -0.4, 0.5, 0.025, 0.012_150_585_609_624_04];
        assert!(zoh_cr3bp_tape().operation_count() < 70);

        for order in [8, 15, MAX_ORDER] {
            let mut incremental = [[0.0; MAX_ORDER + 1]; 7];
            ZohCr3bpDynamics
                .coefficients(0.3, &state, &parameters, order, &mut incremental)
                .unwrap();
            let mut reference = [[0.0; MAX_ORDER + 1]; 7];
            coefficients_from_series_rhs(
                &ZohCr3bpDynamics,
                0.3,
                &state,
                &parameters,
                order,
                &mut reference,
                zoh_cr3bp_rhs,
            )
            .unwrap();
            assert_jets_match(&incremental, &reference, order, 5e-13);
        }
    }

    #[test]
    fn incremental_zoh_equinoctial_matches_series_reference() {
        let state = [1.1, 0.1, -0.05, 0.02, -0.03, 0.4, 1.1];
        let parameters = [0.02, 0.3, -0.4, 0.5, 0.01];
        assert!(zoh_equinoctial_tape().operation_count() < 110);

        for order in [8, 15, MAX_ORDER] {
            let mut incremental = [[0.0; MAX_ORDER + 1]; 7];
            ZohEquinoctialDynamics
                .coefficients(0.3, &state, &parameters, order, &mut incremental)
                .unwrap();
            let mut reference = [[0.0; MAX_ORDER + 1]; 7];
            coefficients_from_series_rhs(
                &ZohEquinoctialDynamics,
                0.3,
                &state,
                &parameters,
                order,
                &mut reference,
                zoh_equinoctial_rhs,
            )
            .unwrap();
            assert_jets_match(&incremental, &reference, order, 2e-12);
        }
    }

    #[test]
    fn incremental_bcp_matches_series_reference() {
        let state = [0.8, -0.2, 0.1, 0.03, 1.0, 0.02];
        let parameters = [
            0.012_150_585_609_624_04,
            328_900.56,
            389.172,
            -0.925_195_985_520_347,
        ];
        assert!(bcp_tape().operation_count() < 90);

        for order in [8, 15, MAX_ORDER] {
            let mut incremental = [[0.0; MAX_ORDER + 1]; 6];
            BcpDynamics
                .coefficients(0.3, &state, &parameters, order, &mut incremental)
                .unwrap();
            let mut reference = [[0.0; MAX_ORDER + 1]; 6];
            coefficients_from_series_rhs(
                &BcpDynamics,
                0.3,
                &state,
                &parameters,
                order,
                &mut reference,
                bcp_rhs,
            )
            .unwrap();
            assert_jets_match(&incremental, &reference, order, 8e-13);
        }
    }

    #[test]
    fn incremental_zoh_solar_sail_matches_series_reference() {
        let state = [0.8, -0.4, 0.3, 0.2, 0.9, -0.1];
        let parameters = [0.25, -1.1, 0.04];
        assert!(zoh_solar_sail_tape().operation_count() < 100);

        for order in [8, 15, MAX_ORDER] {
            let mut incremental = [[0.0; MAX_ORDER + 1]; 6];
            ZohSolarSailDynamics
                .coefficients(0.3, &state, &parameters, order, &mut incremental)
                .unwrap();
            let mut reference = [[0.0; MAX_ORDER + 1]; 6];
            coefficients_from_series_rhs(
                &ZohSolarSailDynamics,
                0.3,
                &state,
                &parameters,
                order,
                &mut reference,
                zoh_solar_sail_rhs,
            )
            .unwrap();
            assert_jets_match(&incremental, &reference, order, 2e-12);
        }
    }

    #[test]
    fn incremental_zoh_kepler_matches_series_reference() {
        let state = [0.9, -0.2, 0.1, 0.15, 0.95, -0.04, 1.2];
        let parameters = [0.07, 0.3, -0.4, 0.5, 0.025];

        for order in [8, 15, MAX_ORDER] {
            let mut incremental = [[0.0; MAX_ORDER + 1]; 7];
            ZohKeplerDynamics
                .coefficients(0.3, &state, &parameters, order, &mut incremental)
                .unwrap();

            let mut reference = [[0.0; MAX_ORDER + 1]; 7];
            coefficients_from_series_rhs(
                &ZohKeplerDynamics,
                0.3,
                &state,
                &parameters,
                order,
                &mut reference,
                zoh_kepler_rhs,
            )
            .unwrap();

            for component in 0..7 {
                for coefficient in 0..=order {
                    let scale = reference[component][coefficient].abs().max(1.0);
                    assert!(
                        (incremental[component][coefficient] - reference[component][coefficient])
                            .abs()
                            <= 3e-13 * scale,
                        "order={order}, state={component}, coefficient={coefficient}: \
                         incremental={:.17e}, reference={:.17e}",
                        incremental[component][coefficient],
                        reference[component][coefficient],
                    );
                }
            }
        }
    }

    #[test]
    fn cartesian_expression_tape_matches_series_reference() {
        let negative_switching_state = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let mut positive_switching_state = negative_switching_state;
        positive_switching_state[13] = -1.0;
        let parameters = [1.0, 0.01, 1.0, 0.5, 1.0];

        for state in [negative_switching_state, positive_switching_state] {
            let tape = cartesian_mass_tape(&state, &parameters);
            assert!(tape.operation_count() < 200);

            for order in [8, 15, MAX_ORDER] {
                let mut incremental = [[0.0; MAX_ORDER + 1]; 14];
                tape.coefficients(0.0, &state, &parameters, order, &mut incremental);

                let mut reference = [[0.0; MAX_ORDER + 1]; 14];
                coefficients_from_series_rhs(
                    &CartesianMassOptimal,
                    0.0,
                    &state,
                    &parameters,
                    order,
                    &mut reference,
                    cartesian_mass_rhs,
                )
                .unwrap();

                for component in 0..14 {
                    for coefficient in 0..=order {
                        let scale = reference[component][coefficient].abs().max(1.0);
                        assert!(
                            (incremental[component][coefficient]
                                - reference[component][coefficient])
                                .abs()
                                <= 2e-11 * scale,
                            "order={order}, state={component}, coefficient={coefficient}: \
                             incremental={:.17e}, reference={:.17e}",
                            incremental[component][coefficient],
                            reference[component][coefficient],
                        );
                    }
                }
            }
        }
    }
}
