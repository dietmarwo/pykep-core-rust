// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                          Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Cartesian and modified-equinoctial Pontryagin dynamics.
//!
//! This evaluated implementation adapts
//! `src/ta/pontryagin_cartesian.cpp` and
//! `src/ta/pontryagin_equinoctial.cpp` from the pinned pykep/kep3 source.
//! Canonical costate rates use forward-mode differentiation. Full model
//! Jacobians for propagated sensitivities use fixed-size centered differences
//! with a relative step of `3e-6`; integrator tolerances do not imply the same
//! accuracy for those sensitivities.

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::error::ensure_finite;
use crate::integration::{DifferentiableDynamicsModel, DynamicsModel};
use crate::{PykepError, Result};

const PHYSICAL_DIMENSION: usize = 7;
const AUGMENTED_DIMENSION: usize = 14;

/// Supported indirect optimal-control objectives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Optimality {
    /// Maximize final mass using the upstream logarithmic throttle barrier.
    Mass,
    /// Minimize time with full throttle.
    Time,
}

/// Evaluated minimizing control and switching function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimalControl {
    /// Optimal throttle in `(0, 1)` for mass optimality and exactly `1` for
    /// time optimality.
    pub throttle: f64,
    /// Optimal Cartesian or RTN thrust direction.
    pub direction: [f64; 3],
    /// Switching function `rho`.
    pub switching_function: f64,
}

#[derive(Clone, Copy)]
struct Parameters {
    mu: f64,
    maximum_thrust: f64,
    exhaust_velocity: f64,
    barrier: Option<f64>,
    lambda0: f64,
    optimality: Optimality,
}

impl Parameters {
    fn mass(values: &[f64; 5]) -> Result<Self> {
        let parameters = Self {
            mu: values[0],
            maximum_thrust: values[1],
            exhaust_velocity: values[2],
            barrier: Some(values[3]),
            lambda0: values[4],
            optimality: Optimality::Mass,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    fn time(values: &[f64; 3]) -> Result<Self> {
        let parameters = Self {
            mu: values[0],
            maximum_thrust: values[1],
            exhaust_velocity: values[2],
            barrier: None,
            lambda0: 1.0,
            optimality: Optimality::Time,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    fn validate(self) -> Result<()> {
        validate_positive("mu", self.mu)?;
        validate_non_negative("maximum_thrust", self.maximum_thrust)?;
        validate_positive("exhaust_velocity", self.exhaust_velocity)?;
        validate_positive("lambda0", self.lambda0)?;
        if let Some(barrier) = self.barrier {
            validate_positive("barrier", barrier)?;
        }
        Ok(())
    }

    fn throttle(self, switching_function: f64) -> f64 {
        match self.optimality {
            Optimality::Mass => {
                let barrier = self.barrier.expect("mass mode has a barrier");
                let root = switching_function.hypot(2.0 * barrier);
                if switching_function >= 0.0 {
                    2.0 * barrier / (switching_function + 2.0 * barrier + root)
                } else {
                    (root - switching_function) / (root - switching_function + 2.0 * barrier)
                }
            }
            Optimality::Time => 1.0,
        }
    }

    fn running_cost(self, throttle: f64) -> f64 {
        let scale = self.lambda0 * self.maximum_thrust / self.exhaust_velocity;
        match self.optimality {
            Optimality::Mass => {
                let barrier = self.barrier.expect("mass mode has a barrier");
                scale * (throttle - barrier * (throttle * (1.0 - throttle)).ln())
            }
            Optimality::Time => scale,
        }
    }
}

/// Cartesian mass-optimal Pontryagin dynamics.
///
/// The 14-state order is
/// `[x,y,z,vx,vy,vz,m,lx,ly,lz,lvx,lvy,lvz,lm]`. Parameter order is
/// `[mu, maximum_thrust, exhaust_velocity, barrier, lambda0]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CartesianMassOptimal;

/// Cartesian time-optimal Pontryagin dynamics.
///
/// State order matches [`CartesianMassOptimal`]. Parameter order is
/// `[mu, maximum_thrust, exhaust_velocity]`; upstream time optimality uses
/// full throttle and an implicit `lambda0 = 1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CartesianTimeOptimal;

/// Modified-equinoctial mass-optimal Pontryagin dynamics.
///
/// The 14-state order is
/// `[p,f,g,h,k,L,m,lp,lf,lg,lh,lk,lL,lm]`. Parameter order is
/// `[mu, maximum_thrust, exhaust_velocity, barrier, lambda0]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EquinoctialMassOptimal;

/// Modified-equinoctial time-optimal Pontryagin dynamics.
///
/// State order matches [`EquinoctialMassOptimal`]. Parameter order is
/// `[mu, maximum_thrust, exhaust_velocity]`; `lambda0 = 1` is implicit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EquinoctialTimeOptimal;

macro_rules! implement_model {
    ($model:ty, $parameter_count:literal, $name:literal, $parameters:expr, $rhs:ident, $positive:expr) => {
        impl DynamicsModel<AUGMENTED_DIMENSION, $parameter_count> for $model {
            const NAME: &'static str = $name;

            fn validate(
                &self,
                time: f64,
                state: &[f64; AUGMENTED_DIMENSION],
                parameters: &[f64; $parameter_count],
            ) -> Result<()> {
                validate_augmented(time, state)?;
                let parameters = $parameters(parameters)?;
                validate_model_state(state, parameters, stringify!($rhs))
            }

            fn rhs(
                &self,
                time: f64,
                state: &[f64; AUGMENTED_DIMENSION],
                parameters: &[f64; $parameter_count],
                derivative: &mut [f64; AUGMENTED_DIMENSION],
            ) -> Result<()> {
                self.validate(time, state, parameters)?;
                $rhs(state, $parameters(parameters)?, derivative)
            }
        }

        impl DifferentiableDynamicsModel<AUGMENTED_DIMENSION, $parameter_count> for $model {
            fn jacobians(
                &self,
                time: f64,
                state: &[f64; AUGMENTED_DIMENSION],
                parameters: &[f64; $parameter_count],
                state_jacobian: &mut [[f64; AUGMENTED_DIMENSION]; AUGMENTED_DIMENSION],
                parameter_jacobian: &mut [[f64; $parameter_count]; AUGMENTED_DIMENSION],
            ) -> Result<()> {
                numerical_jacobians(
                    self,
                    time,
                    state,
                    parameters,
                    state_jacobian,
                    parameter_jacobian,
                    &[6],
                    $positive,
                )
            }
        }
    };
}

implement_model!(
    CartesianMassOptimal,
    5,
    "Cartesian mass-optimal Pontryagin dynamics",
    Parameters::mass,
    cartesian_rhs,
    &[0, 1, 2, 3, 4]
);
implement_model!(
    CartesianTimeOptimal,
    3,
    "Cartesian time-optimal Pontryagin dynamics",
    Parameters::time,
    cartesian_rhs,
    &[0, 1, 2]
);
implement_model!(
    EquinoctialMassOptimal,
    5,
    "equinoctial mass-optimal Pontryagin dynamics",
    Parameters::mass,
    equinoctial_rhs,
    &[0, 1, 2, 3, 4]
);
implement_model!(
    EquinoctialTimeOptimal,
    3,
    "equinoctial time-optimal Pontryagin dynamics",
    Parameters::time,
    equinoctial_rhs,
    &[0, 1, 2]
);

/// Evaluates the Cartesian minimizing control in mass-optimal mode.
///
/// # Errors
///
/// Returns an error for invalid inputs or a zero velocity-primer norm.
pub fn cartesian_control_mass(state: &[f64; 14], parameters: &[f64; 5]) -> Result<OptimalControl> {
    let parameters = Parameters::mass(parameters)?;
    validate_augmented(0.0, state)?;
    validate_model_state(state, parameters, "cartesian")?;
    cartesian_control(state, parameters)
}

/// Evaluates the Cartesian minimizing control in time-optimal mode.
///
/// # Errors
///
/// Returns an error for invalid inputs or a zero velocity-primer norm.
pub fn cartesian_control_time(state: &[f64; 14], parameters: &[f64; 3]) -> Result<OptimalControl> {
    let parameters = Parameters::time(parameters)?;
    validate_augmented(0.0, state)?;
    validate_model_state(state, parameters, "cartesian")?;
    cartesian_control(state, parameters)
}

/// Evaluates the equinoctial minimizing control in mass-optimal mode.
///
/// # Errors
///
/// Returns an error for invalid inputs or a zero transformed-primer norm.
pub fn equinoctial_control_mass(
    state: &[f64; 14],
    parameters: &[f64; 5],
) -> Result<OptimalControl> {
    let parameters = Parameters::mass(parameters)?;
    validate_augmented(0.0, state)?;
    validate_model_state(state, parameters, "equinoctial")?;
    equinoctial_control(state, parameters)
}

/// Evaluates the equinoctial minimizing control in time-optimal mode.
///
/// # Errors
///
/// Returns an error for invalid inputs or a zero transformed-primer norm.
pub fn equinoctial_control_time(
    state: &[f64; 14],
    parameters: &[f64; 3],
) -> Result<OptimalControl> {
    let parameters = Parameters::time(parameters)?;
    validate_augmented(0.0, state)?;
    validate_model_state(state, parameters, "equinoctial")?;
    equinoctial_control(state, parameters)
}

/// Evaluates the minimized Cartesian Hamiltonian in mass-optimal mode.
///
/// # Errors
///
/// Returns an error under the same conditions as
/// [`cartesian_control_mass`].
pub fn cartesian_hamiltonian_mass(state: &[f64; 14], parameters: &[f64; 5]) -> Result<f64> {
    hamiltonian_cartesian_checked(state, Parameters::mass(parameters)?)
}

/// Evaluates the minimized Cartesian Hamiltonian in time-optimal mode.
///
/// # Errors
///
/// Returns an error under the same conditions as
/// [`cartesian_control_time`].
pub fn cartesian_hamiltonian_time(state: &[f64; 14], parameters: &[f64; 3]) -> Result<f64> {
    hamiltonian_cartesian_checked(state, Parameters::time(parameters)?)
}

/// Evaluates the minimized equinoctial Hamiltonian in mass-optimal mode.
///
/// # Errors
///
/// Returns an error under the same conditions as
/// [`equinoctial_control_mass`].
pub fn equinoctial_hamiltonian_mass(state: &[f64; 14], parameters: &[f64; 5]) -> Result<f64> {
    hamiltonian_equinoctial_checked(state, Parameters::mass(parameters)?)
}

/// Evaluates the minimized equinoctial Hamiltonian in time-optimal mode.
///
/// # Errors
///
/// Returns an error under the same conditions as
/// [`equinoctial_control_time`].
pub fn equinoctial_hamiltonian_time(state: &[f64; 14], parameters: &[f64; 3]) -> Result<f64> {
    hamiltonian_equinoctial_checked(state, Parameters::time(parameters)?)
}

fn validate_augmented(time: f64, state: &[f64; 14]) -> Result<()> {
    ensure_finite("time", time)?;
    for &value in state {
        ensure_finite("state", value)?;
    }
    Ok(())
}

fn validate_model_state(state: &[f64; 14], parameters: Parameters, model: &str) -> Result<()> {
    validate_positive("mass", state[6])?;
    if model == "cartesian" || model == "cartesian_rhs" {
        let radius_squared = state[0] * state[0] + state[1] * state[1] + state[2] * state[2];
        if radius_squared == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "Cartesian Pontryagin radius",
            });
        }
        if !radius_squared.is_finite() {
            return Err(PykepError::NumericalOverflow {
                operation: "Cartesian Pontryagin radius",
            });
        }
    } else {
        validate_positive("semilatus_rectum", state[0])?;
        let w = 1.0 + state[1] * state[5].cos() + state[2] * state[5].sin();
        if w == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "equinoctial Pontryagin radial denominator",
            });
        }
    }
    let control = if model == "cartesian" || model == "cartesian_rhs" {
        cartesian_control(state, parameters)
    } else {
        equinoctial_control(state, parameters)
    }?;
    if control.throttle.is_finite()
        && control.switching_function.is_finite()
        && control.direction.iter().all(|value| value.is_finite())
    {
        Ok(())
    } else {
        Err(PykepError::NumericalOverflow {
            operation: "Pontryagin minimizing control",
        })
    }
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

fn cartesian_control(state: &[f64; 14], parameters: Parameters) -> Result<OptimalControl> {
    let primer = [state[10], state[11], state[12]];
    let norm = norm(primer);
    if norm == 0.0 {
        return Err(PykepError::SingularGeometry {
            operation: "Cartesian Pontryagin primer norm",
        });
    }
    let direction = [-primer[0] / norm, -primer[1] / norm, -primer[2] / norm];
    let switching_function = match parameters.optimality {
        Optimality::Mass => {
            1.0 - parameters.exhaust_velocity * norm / state[6] / parameters.lambda0
                - state[13] / parameters.lambda0
        }
        Optimality::Time => {
            -parameters.exhaust_velocity * norm / state[6] / parameters.lambda0
                - state[13] / parameters.lambda0
        }
    };
    Ok(OptimalControl {
        throttle: parameters.throttle(switching_function),
        direction,
        switching_function,
    })
}

fn cartesian_rhs(
    state: &[f64; 14],
    parameters: Parameters,
    derivative: &mut [f64; 14],
) -> Result<()> {
    let control = cartesian_control(state, parameters)?;
    let variables = core::array::from_fn(|index| Dual::variable(state[index], index));
    let physical = cartesian_physical_dual(variables, parameters, control);
    let mut hamiltonian = Dual::constant(parameters.running_cost(control.throttle));
    for index in 0..PHYSICAL_DIMENSION {
        derivative[index] = physical[index].value;
        hamiltonian = hamiltonian + Dual::constant(state[index + 7]) * physical[index];
    }
    for index in 0..PHYSICAL_DIMENSION {
        derivative[index + 7] = -hamiltonian.derivative[index];
    }
    validate_derivative("Cartesian Pontryagin dynamics", derivative)
}

fn hamiltonian_cartesian_checked(state: &[f64; 14], parameters: Parameters) -> Result<f64> {
    validate_augmented(0.0, state)?;
    validate_model_state(state, parameters, "cartesian")?;
    let control = cartesian_control(state, parameters)?;
    let variables = core::array::from_fn(|index| Dual::constant(state[index]));
    let physical = cartesian_physical_dual(variables, parameters, control);
    let mut hamiltonian = parameters.running_cost(control.throttle);
    for index in 0..PHYSICAL_DIMENSION {
        hamiltonian += state[index + 7] * physical[index].value;
    }
    finite("Cartesian Pontryagin Hamiltonian", hamiltonian)
}

fn cartesian_physical_dual(
    state: [Dual; 7],
    parameters: Parameters,
    control: OptimalControl,
) -> [Dual; 7] {
    let radius_squared = state[0] * state[0] + state[1] * state[1] + state[2] * state[2];
    let gravity_scale = Dual::constant(-parameters.mu) / radius_squared.powf(1.5);
    let thrust_scale = Dual::constant(parameters.maximum_thrust * control.throttle) / state[6];
    [
        state[3],
        state[4],
        state[5],
        gravity_scale * state[0] + thrust_scale * Dual::constant(control.direction[0]),
        gravity_scale * state[1] + thrust_scale * Dual::constant(control.direction[1]),
        gravity_scale * state[2] + thrust_scale * Dual::constant(control.direction[2]),
        Dual::constant(-parameters.maximum_thrust / parameters.exhaust_velocity * control.throttle),
    ]
}

fn equinoctial_control(state: &[f64; 14], parameters: Parameters) -> Result<OptimalControl> {
    let matrix = equinoctial_b_values(state, parameters.mu);
    let costate = &state[7..13];
    let primer = core::array::from_fn(|column| {
        (0..6)
            .map(|row| matrix[row][column] * costate[row])
            .sum::<f64>()
    });
    let norm = norm(primer);
    if norm == 0.0 {
        return Err(PykepError::SingularGeometry {
            operation: "equinoctial Pontryagin primer norm",
        });
    }
    let direction = [-primer[0] / norm, -primer[1] / norm, -primer[2] / norm];
    let switching_function = match parameters.optimality {
        Optimality::Mass => {
            1.0 - parameters.exhaust_velocity * norm / state[6] / parameters.lambda0
                - state[13] / parameters.lambda0
        }
        Optimality::Time => {
            -parameters.exhaust_velocity * norm / state[6] / parameters.lambda0
                - state[13] / parameters.lambda0
        }
    };
    Ok(OptimalControl {
        throttle: parameters.throttle(switching_function),
        direction,
        switching_function,
    })
}

fn equinoctial_rhs(
    state: &[f64; 14],
    parameters: Parameters,
    derivative: &mut [f64; 14],
) -> Result<()> {
    let control = equinoctial_control(state, parameters)?;
    let variables = core::array::from_fn(|index| Dual::variable(state[index], index));
    let physical = equinoctial_physical_dual(variables, parameters, control);
    let mut hamiltonian = Dual::constant(parameters.running_cost(control.throttle));
    for index in 0..PHYSICAL_DIMENSION {
        derivative[index] = physical[index].value;
        hamiltonian = hamiltonian + Dual::constant(state[index + 7]) * physical[index];
    }
    for index in 0..PHYSICAL_DIMENSION {
        derivative[index + 7] = -hamiltonian.derivative[index];
    }
    validate_derivative("equinoctial Pontryagin dynamics", derivative)
}

fn hamiltonian_equinoctial_checked(state: &[f64; 14], parameters: Parameters) -> Result<f64> {
    validate_augmented(0.0, state)?;
    validate_model_state(state, parameters, "equinoctial")?;
    let control = equinoctial_control(state, parameters)?;
    let variables = core::array::from_fn(|index| Dual::constant(state[index]));
    let physical = equinoctial_physical_dual(variables, parameters, control);
    let mut hamiltonian = parameters.running_cost(control.throttle);
    for index in 0..PHYSICAL_DIMENSION {
        hamiltonian += state[index + 7] * physical[index].value;
    }
    finite("equinoctial Pontryagin Hamiltonian", hamiltonian)
}

fn equinoctial_b_values(state: &[f64; 14], mu: f64) -> [[f64; 3]; 6] {
    let [p, f, g, h, k, longitude, ..] = *state;
    let sine = longitude.sin();
    let cosine = longitude.cos();
    let w = 1.0 + f * cosine + g * sine;
    let s2 = 1.0 + h * h + k * k;
    let hsk = h * sine - k * cosine;
    let scale = (p / mu).sqrt();
    [
        [0.0, scale * 2.0 * p / w, 0.0],
        [
            scale * sine,
            scale * ((1.0 + w) * cosine + f) / w,
            -scale * g * hsk / w,
        ],
        [
            -scale * cosine,
            scale * ((1.0 + w) * sine + g) / w,
            scale * f * hsk / w,
        ],
        [0.0, 0.0, scale * s2 * cosine / (2.0 * w)],
        [0.0, 0.0, scale * s2 * sine / (2.0 * w)],
        [0.0, 0.0, scale * hsk / w],
    ]
}

fn equinoctial_physical_dual(
    state: [Dual; 7],
    parameters: Parameters,
    control: OptimalControl,
) -> [Dual; 7] {
    let one = Dual::constant(1.0);
    let sine = state[5].sin();
    let cosine = state[5].cos();
    let w = one + state[1] * cosine + state[2] * sine;
    let s2 = one + state[3] * state[3] + state[4] * state[4];
    let hsk = state[3] * sine - state[4] * cosine;
    let scale = (state[0] / Dual::constant(parameters.mu)).sqrt();
    let matrix = [
        [
            Dual::constant(0.0),
            scale * Dual::constant(2.0) * state[0] / w,
            Dual::constant(0.0),
        ],
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
            Dual::constant(0.0),
            Dual::constant(0.0),
            scale * s2 * cosine / (Dual::constant(2.0) * w),
        ],
        [
            Dual::constant(0.0),
            Dual::constant(0.0),
            scale * s2 * sine / (Dual::constant(2.0) * w),
        ],
        [Dual::constant(0.0), Dual::constant(0.0), scale * hsk / w],
    ];
    let thrust_scale = Dual::constant(parameters.maximum_thrust * control.throttle) / state[6];
    let mut derivative = [Dual::constant(0.0); 7];
    for row in 0..6 {
        derivative[row] = (matrix[row][0] * Dual::constant(control.direction[0])
            + matrix[row][1] * Dual::constant(control.direction[1])
            + matrix[row][2] * Dual::constant(control.direction[2]))
            * thrust_scale;
    }
    derivative[5] =
        derivative[5] + (Dual::constant(parameters.mu) / state[0].powf(3.0)).sqrt() * w * w;
    derivative[6] = Dual::constant(-parameters.maximum_thrust / parameters.exhaust_velocity)
        * Dual::constant(control.throttle)
        * (Dual::constant(-1.0) / state[6] / Dual::constant(1e10)).exp();
    derivative
}

fn norm(vector: [f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn finite(operation: &'static str, value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PykepError::NumericalOverflow { operation })
    }
}

fn validate_derivative(operation: &'static str, derivative: &[f64; 14]) -> Result<()> {
    if derivative.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PykepError::NumericalOverflow { operation })
    }
}

#[allow(clippy::too_many_arguments)]
fn numerical_jacobians<M, const P: usize>(
    model: &M,
    time: f64,
    state: &[f64; 14],
    parameters: &[f64; P],
    state_jacobian: &mut [[f64; 14]; 14],
    parameter_jacobian: &mut [[f64; P]; 14],
    positive_state_indices: &[usize],
    positive_parameter_indices: &[usize],
) -> Result<()>
where
    M: DynamicsModel<14, P>,
{
    *state_jacobian = [[0.0; 14]; 14];
    *parameter_jacobian = [[0.0; P]; 14];
    for column in 0..14 {
        let mut step = 3e-6 * state[column].abs().max(1.0);
        if positive_state_indices.contains(&column) {
            step = step.min(state[column] * 0.25);
        }
        let mut plus = *state;
        let mut minus = *state;
        plus[column] += step;
        minus[column] -= step;
        let mut rhs_plus = [0.0; 14];
        let mut rhs_minus = [0.0; 14];
        model.rhs(time, &plus, parameters, &mut rhs_plus)?;
        model.rhs(time, &minus, parameters, &mut rhs_minus)?;
        for row in 0..14 {
            state_jacobian[row][column] = (rhs_plus[row] - rhs_minus[row]) / (2.0 * step);
        }
    }
    for column in 0..P {
        let mut step = 3e-6 * parameters[column].abs().max(1.0);
        if positive_parameter_indices.contains(&column) {
            step = step.min(parameters[column] * 0.25);
        }
        let mut plus = *parameters;
        let mut minus = *parameters;
        plus[column] += step;
        minus[column] -= step;
        let mut rhs_plus = [0.0; 14];
        let mut rhs_minus = [0.0; 14];
        model.rhs(time, state, &plus, &mut rhs_plus)?;
        model.rhs(time, state, &minus, &mut rhs_minus)?;
        for row in 0..14 {
            parameter_jacobian[row][column] = (rhs_plus[row] - rhs_minus[row]) / (2.0 * step);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Dual {
    value: f64,
    derivative: [f64; PHYSICAL_DIMENSION],
}

impl Dual {
    const fn constant(value: f64) -> Self {
        Self {
            value,
            derivative: [0.0; PHYSICAL_DIMENSION],
        }
    }

    fn variable(value: f64, index: usize) -> Self {
        let mut derivative = [0.0; PHYSICAL_DIMENSION];
        derivative[index] = 1.0;
        Self { value, derivative }
    }

    fn sqrt(self) -> Self {
        self.powf(0.5)
    }

    fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        let scale = exponent * self.value.powf(exponent - 1.0);
        Self {
            value,
            derivative: self.derivative.map(|item| item * scale),
        }
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

macro_rules! dual_binary {
    ($trait:ident, $method:ident, $value:expr, $derivative:expr) => {
        impl $trait for Dual {
            type Output = Self;

            fn $method(self, right: Self) -> Self {
                let mut derivative = [0.0; PHYSICAL_DIMENSION];
                for (index, item) in derivative.iter_mut().enumerate() {
                    *item = $derivative(self, right, index);
                }
                Self {
                    value: $value(self, right),
                    derivative,
                }
            }
        }
    };
}

dual_binary!(
    Add,
    add,
    |left: Dual, right: Dual| left.value + right.value,
    |left: Dual, right: Dual, index| left.derivative[index] + right.derivative[index]
);
dual_binary!(
    Sub,
    sub,
    |left: Dual, right: Dual| left.value - right.value,
    |left: Dual, right: Dual, index| left.derivative[index] - right.derivative[index]
);
dual_binary!(
    Mul,
    mul,
    |left: Dual, right: Dual| left.value * right.value,
    |left: Dual, right: Dual, index| left.derivative[index] * right.value
        + left.value * right.derivative[index]
);
dual_binary!(
    Div,
    div,
    |left: Dual, right: Dual| left.value / right.value,
    |left: Dual, right: Dual, index| {
        (left.derivative[index] * right.value - left.value * right.derivative[index])
            / (right.value * right.value)
    }
);

impl Neg for Dual {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivative: self.derivative.map(|item| -item),
        }
    }
}
