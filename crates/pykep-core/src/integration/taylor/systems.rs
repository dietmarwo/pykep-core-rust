// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::Result;
use crate::dynamics::pontryagin::{
    CartesianMassOptimal, CartesianTimeOptimal, EquinoctialMassOptimal, EquinoctialTimeOptimal,
};
use crate::dynamics::zoh::{
    ZohCr3bpDynamics, ZohEquinoctialDynamics, ZohKeplerDynamics, ZohSolarSailDynamics,
};
use crate::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
use crate::integration::{DynamicsModel, TaylorDynamicsModel};

use super::series::Series;
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

macro_rules! implement_series_model {
    ($model:ty, $states:literal, $parameters:literal, $rhs:ident) => {
        impl TaylorCoefficientModel<$states, $parameters> for $model {
            fn coefficients(
                &self,
                time: f64,
                state: &[f64; $states],
                parameters: &[f64; $parameters],
                order: usize,
                jet: &mut [[f64; MAX_ORDER + 1]; $states],
            ) -> Result<()> {
                coefficients_from_series_rhs(self, time, state, parameters, order, jet, $rhs)
            }
        }

        impl TaylorDynamicsModel<$states, $parameters> for $model {}
    };
}

implement_series_model!(Cr3bpDynamics, 6, 1, cr3bp_rhs);
implement_series_model!(BcpDynamics, 6, 4, bcp_rhs);
implement_series_model!(ZohKeplerDynamics, 7, 5, zoh_kepler_rhs);
implement_series_model!(ZohCr3bpDynamics, 7, 6, zoh_cr3bp_rhs);
implement_series_model!(ZohEquinoctialDynamics, 7, 5, zoh_equinoctial_rhs);
implement_series_model!(ZohSolarSailDynamics, 6, 3, zoh_solar_sail_rhs);
implement_series_model!(CartesianMassOptimal, 14, 5, cartesian_mass_rhs);
implement_series_model!(CartesianTimeOptimal, 14, 3, cartesian_time_rhs);
implement_series_model!(EquinoctialMassOptimal, 14, 5, equinoctial_mass_rhs);
implement_series_model!(EquinoctialTimeOptimal, 14, 3, equinoctial_time_rhs);

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

fn squared_norm<const N: usize>(vector: [Series; N]) -> Series {
    vector
        .into_iter()
        .map(|item| item * item)
        .reduce(|left, right| left + right)
        .expect("dynamics vectors are nonempty")
}

fn cross(left: [Series; 3], right: [Series; 3]) -> [Series; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

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

#[derive(Clone, Copy)]
struct PontryaginParameters {
    mu: f64,
    maximum_thrust: f64,
    exhaust_velocity: f64,
    barrier: f64,
    lambda0: f64,
    objective: Objective,
}

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

fn scalar_dual(value: f64, order: usize) -> SeriesDual {
    SeriesDual::constant(Series::constant(value, order))
}

#[derive(Clone, Copy, Debug)]
struct SeriesDual {
    value: Series,
    derivative: [Series; 7],
}

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
