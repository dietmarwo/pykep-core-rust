// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Small forward-mode dual numbers used to evaluate analytic Jacobians.

use core::ops::{Add, Div, Mul, Neg, Sub};

use super::{ModifiedEquinoctialElements, singular, validate_mu, validate_six, validate_state};
use crate::{CartesianState, Matrix6, PykepError, Result};

const DIMENSION: usize = 6;

#[derive(Clone, Copy, Debug)]
struct Dual {
    value: f64,
    derivative: [f64; DIMENSION],
}

impl Dual {
    const fn constant(value: f64) -> Self {
        Self {
            value,
            derivative: [0.0; DIMENSION],
        }
    }

    fn variable(value: f64, index: usize) -> Self {
        let mut derivative = [0.0; DIMENSION];
        derivative[index] = 1.0;
        Self { value, derivative }
    }

    fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        let scale = 0.5 / value;
        Self {
            value,
            derivative: self.derivative.map(|item| item * scale),
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

    fn atan2(self, other: Self) -> Self {
        let denominator = self.value * self.value + other.value * other.value;
        let mut derivative = [0.0; DIMENSION];
        for (index, item) in derivative.iter_mut().enumerate() {
            *item = (other.value * self.derivative[index] - self.value * other.derivative[index])
                / denominator;
        }
        Self {
            value: self.value.atan2(other.value),
            derivative,
        }
    }
}

impl Add for Dual {
    type Output = Self;

    fn add(self, right: Self) -> Self {
        let mut derivative = [0.0; DIMENSION];
        for (index, item) in derivative.iter_mut().enumerate() {
            *item = self.derivative[index] + right.derivative[index];
        }
        Self {
            value: self.value + right.value,
            derivative,
        }
    }
}

impl Sub for Dual {
    type Output = Self;

    fn sub(self, right: Self) -> Self {
        let mut derivative = [0.0; DIMENSION];
        for (index, item) in derivative.iter_mut().enumerate() {
            *item = self.derivative[index] - right.derivative[index];
        }
        Self {
            value: self.value - right.value,
            derivative,
        }
    }
}

impl Mul for Dual {
    type Output = Self;

    fn mul(self, right: Self) -> Self {
        let mut derivative = [0.0; DIMENSION];
        for (index, item) in derivative.iter_mut().enumerate() {
            *item = self.derivative[index] * right.value + self.value * right.derivative[index];
        }
        Self {
            value: self.value * right.value,
            derivative,
        }
    }
}

impl Div for Dual {
    type Output = Self;

    fn div(self, right: Self) -> Self {
        let denominator = right.value * right.value;
        let mut derivative = [0.0; DIMENSION];
        for (index, item) in derivative.iter_mut().enumerate() {
            *item = (self.derivative[index] * right.value - self.value * right.derivative[index])
                / denominator;
        }
        Self {
            value: self.value / right.value,
            derivative,
        }
    }
}

impl Neg for Dual {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivative: self.derivative.map(|item| -item),
        }
    }
}

fn matrix(outputs: [Dual; DIMENSION], operation: &'static str) -> Result<Matrix6> {
    let mut result = [[0.0; DIMENSION]; DIMENSION];
    for (row, output) in outputs.iter().enumerate() {
        if !output.value.is_finite() {
            return Err(PykepError::NumericalOverflow { operation });
        }
        for (column, &value) in output.derivative.iter().enumerate() {
            if !value.is_finite() {
                return Err(PykepError::NumericalOverflow { operation });
            }
            result[row][column] = value;
        }
    }
    Ok(result)
}

pub(super) fn cartesian_to_modified_equinoctial_jacobian(
    state: &CartesianState,
    mu: f64,
    retrograde: bool,
) -> Result<Matrix6> {
    validate_state(state)?;
    validate_mu(mu)?;
    let variables = core::array::from_fn(|index| Dual::variable(state[index], index));
    let [x, y, z, vx, vy, vz] = variables;
    let one = Dual::constant(1.0);
    let two = Dual::constant(2.0);
    let mu = Dual::constant(mu);
    let direction = Dual::constant(if retrograde { -1.0 } else { 1.0 });

    let radius_squared = x * x + y * y + z * z;
    let velocity_squared = vx * vx + vy * vy + vz * vz;
    let radial_dot = x * vx + y * vy + z * vz;
    let radius = radius_squared.sqrt();
    let angular_x = y * vz - z * vy;
    let angular_y = z * vx - x * vz;
    let angular_z = x * vy - y * vx;
    let angular_squared = angular_x * angular_x + angular_y * angular_y + angular_z * angular_z;
    let angular_norm = angular_squared.sqrt();
    if radius.value == 0.0 || angular_norm.value == 0.0 {
        return Err(singular("cartesian_to_modified_equinoctial_jacobian"));
    }
    let semilatus_rectum = angular_squared / mu;
    let frame_denominator = angular_norm + direction * angular_z;
    if frame_denominator.value.abs() <= 8.0 * f64::EPSILON * angular_norm.value {
        return Err(singular("cartesian_to_modified_equinoctial_jacobian"));
    }
    let h = -angular_y / frame_denominator;
    let k = angular_x / frame_denominator;
    let h_squared = h * h;
    let k_squared = k * k;
    let hk = h * k;
    let frame_scale = one + h_squared + k_squared;

    let eccentric_x = x * (velocity_squared / mu - one / radius) - vx * radial_dot / mu;
    let eccentric_y = y * (velocity_squared / mu - one / radius) - vy * radial_dot / mu;
    let d_prefactor = (radius_squared * velocity_squared - radial_dot * radial_dot) / mu - radius;
    let d_x = (vx * d_prefactor + x * radial_dot / radius) / angular_norm;
    let d_y = (vy * d_prefactor + y * radial_dot / radius) / angular_norm;
    let f = frame_scale * (eccentric_x + direction * d_y) / two;
    let g = frame_scale * (eccentric_y - direction * d_x) / two;

    let cosine_longitude = ((one + h_squared - k_squared) * (x / radius) + two * hk * (y / radius)
        - two * direction * k * (z / radius))
        / frame_scale;
    let sine_longitude = (two * direction * hk * (x / radius)
        + direction * (one - h_squared + k_squared) * (y / radius)
        + two * h * (z / radius))
        / frame_scale;
    let longitude = sine_longitude.atan2(cosine_longitude);

    matrix(
        [semilatus_rectum, f, g, h, k, longitude],
        "cartesian_to_modified_equinoctial_jacobian",
    )
}

pub(super) fn modified_equinoctial_to_cartesian_jacobian(
    elements: ModifiedEquinoctialElements,
    mu: f64,
    retrograde: bool,
) -> Result<Matrix6> {
    validate_six("elements", &elements.to_array())?;
    validate_mu(mu)?;
    if elements.semilatus_rectum <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "semilatus_rectum",
            reason: "must be greater than zero".into(),
        });
    }
    let values = elements.to_array();
    let variables = core::array::from_fn(|index| Dual::variable(values[index], index));
    let [p, f, g, h, k, longitude] = variables;
    let one = Dual::constant(1.0);
    let two = Dual::constant(2.0);
    let direction = Dual::constant(if retrograde { -1.0 } else { 1.0 });
    let mu = Dual::constant(mu);

    let denominator = one + k * k + h * h;
    let frame_f_x = (one - k * k + h * h) / denominator;
    let frame_f_y = two * k * h / denominator;
    let frame_f_z = -two * direction * k / denominator;
    let frame_g_x = two * direction * k * h / denominator;
    let frame_g_y = direction * (one + k * k - h * h) / denominator;
    let frame_g_z = two * h / denominator;
    let sine_longitude = longitude.sin();
    let cosine_longitude = longitude.cos();
    let radial_denominator = one + g * sine_longitude + f * cosine_longitude;
    if radial_denominator.value <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "elements",
            reason: "true longitude is at or beyond an orbital asymptote".into(),
        });
    }
    let radius = p / radial_denominator;
    let x_equinoctial = radius * cosine_longitude;
    let y_equinoctial = radius * sine_longitude;
    let speed_scale = (mu / p).sqrt();
    let velocity_x = -speed_scale * (g + sine_longitude);
    let velocity_y = speed_scale * (f + cosine_longitude);

    matrix(
        [
            x_equinoctial * frame_f_x + y_equinoctial * frame_g_x,
            x_equinoctial * frame_f_y + y_equinoctial * frame_g_y,
            x_equinoctial * frame_f_z + y_equinoctial * frame_g_z,
            velocity_x * frame_f_x + velocity_y * frame_g_x,
            velocity_x * frame_f_y + velocity_y * frame_g_y,
            velocity_x * frame_f_z + velocity_y * frame_g_z,
        ],
        "modified_equinoctial_to_cartesian_jacobian",
    )
}
