// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Truncated power-series arithmetic used by the Taylor integrator.

use core::ops::{Add, Div, Mul, Neg, Sub};

use super::MAX_ORDER;

/// Coefficients of `sum(c[k] * t^k, k=0..order)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Series {
    coefficients: [f64; MAX_ORDER + 1],
    order: usize,
}

impl Series {
    pub(crate) fn constant(value: f64, order: usize) -> Self {
        let mut coefficients = [0.0; MAX_ORDER + 1];
        coefficients[0] = value;
        Self {
            coefficients,
            order,
        }
    }

    pub(crate) fn variable(value: f64, order: usize) -> Self {
        let mut result = Self::constant(value, order);
        if order > 0 {
            result.coefficients[1] = 1.0;
        }
        result
    }

    pub(crate) fn from_coefficients(coefficients: &[f64; MAX_ORDER + 1], order: usize) -> Self {
        Self {
            coefficients: *coefficients,
            order,
        }
    }

    pub(crate) fn coefficient(self, order: usize) -> f64 {
        self.coefficients[order]
    }

    pub(crate) fn order(self) -> usize {
        self.order
    }

    #[cfg(test)]
    pub(crate) fn evaluate(self, argument: f64) -> f64 {
        (0..=self.order).rev().fold(0.0, |value, index| {
            value.mul_add(argument, self.coefficients[index])
        })
    }

    pub(crate) fn powf(self, exponent: f64) -> Self {
        let mut result = Self::constant(self.coefficients[0].powf(exponent), self.order);
        for n in 1..=self.order {
            let nf = n as f64;
            let mut sum = 0.0;
            for k in 0..n {
                let kf = k as f64;
                sum += (nf * exponent - kf * (exponent + 1.0))
                    * self.coefficients[n - k]
                    * result.coefficients[k];
            }
            result.coefficients[n] = sum / (nf * self.coefficients[0]);
        }
        result
    }

    pub(crate) fn sqrt(self) -> Self {
        self.powf(0.5)
    }

    pub(crate) fn exp(self) -> Self {
        let mut result = Self::constant(self.coefficients[0].exp(), self.order);
        for n in 1..=self.order {
            let mut sum = 0.0;
            for k in 1..=n {
                sum += k as f64 * self.coefficients[k] * result.coefficients[n - k];
            }
            result.coefficients[n] = sum / n as f64;
        }
        result
    }

    pub(crate) fn sin_cos(self) -> (Self, Self) {
        let mut sine = Self::constant(self.coefficients[0].sin(), self.order);
        let mut cosine = Self::constant(self.coefficients[0].cos(), self.order);
        for n in 1..=self.order {
            let mut sine_sum = 0.0;
            let mut cosine_sum = 0.0;
            for k in 1..=n {
                let differentiated = k as f64 * self.coefficients[k];
                sine_sum += differentiated * cosine.coefficients[n - k];
                cosine_sum -= differentiated * sine.coefficients[n - k];
            }
            sine.coefficients[n] = sine_sum / n as f64;
            cosine.coefficients[n] = cosine_sum / n as f64;
        }
        (sine, cosine)
    }

    pub(crate) fn sin(self) -> Self {
        self.sin_cos().0
    }

    pub(crate) fn cos(self) -> Self {
        self.sin_cos().1
    }
}

impl Add for Series {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.order, rhs.order);
        for index in 0..=self.order {
            self.coefficients[index] += rhs.coefficients[index];
        }
        self
    }
}

impl Add<f64> for Series {
    type Output = Self;

    fn add(mut self, rhs: f64) -> Self::Output {
        self.coefficients[0] += rhs;
        self
    }
}

impl Sub for Series {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.order, rhs.order);
        for index in 0..=self.order {
            self.coefficients[index] -= rhs.coefficients[index];
        }
        self
    }
}

impl Sub<f64> for Series {
    type Output = Self;

    fn sub(mut self, rhs: f64) -> Self::Output {
        self.coefficients[0] -= rhs;
        self
    }
}

impl Mul for Series {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.order, rhs.order);
        let mut result = Self::constant(0.0, self.order);
        for n in 0..=self.order {
            result.coefficients[n] = (0..=n)
                .map(|k| self.coefficients[k] * rhs.coefficients[n - k])
                .sum();
        }
        result
    }
}

impl Mul<f64> for Series {
    type Output = Self;

    fn mul(mut self, rhs: f64) -> Self::Output {
        for index in 0..=self.order {
            self.coefficients[index] *= rhs;
        }
        self
    }
}

impl Div for Series {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.powf(-1.0)
    }
}

impl Div<f64> for Series {
    type Output = Self;

    fn div(mut self, rhs: f64) -> Self::Output {
        for index in 0..=self.order {
            self.coefficients[index] /= rhs;
        }
        self
    }
}

impl Neg for Series {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }

    #[test]
    fn product_matches_closed_form_polynomial() {
        let order = 6;
        let t = Series::variable(0.25, order);
        let polynomial = (t + 2.0) * (t - 3.0);
        for argument in [-0.2, 0.0, 0.3] {
            let shifted = 0.25 + argument;
            assert_close(
                polynomial.evaluate(argument),
                (shifted + 2.0) * (shifted - 3.0),
                2e-15,
            );
        }
    }

    #[test]
    fn elementary_functions_match_direct_evaluation() {
        let order = 20;
        let t = Series::variable(0.3, order);
        let expression = ((t.sin_cos().0 + 1.5) * t.exp()) / (t * t + 2.0).sqrt();
        for argument in [-0.05_f64, 0.02, 0.08] {
            let shifted = 0.3 + argument;
            let expected = (shifted.sin() + 1.5) * shifted.exp() / (shifted * shifted + 2.0).sqrt();
            assert_close(expression.evaluate(argument), expected, 8e-15);
        }
    }

    #[test]
    fn real_power_coefficients_match_binomial_series() {
        let order = 8;
        let t = Series::variable(1.0, order);
        let inverse_sqrt = t.powf(-0.5);
        let expected = [
            1.0,
            -0.5,
            0.375,
            -0.3125,
            0.2734375,
            -0.24609375,
            0.2255859375,
            -0.20947265625,
            0.196380615234375,
        ];
        for (index, expected) in expected.into_iter().enumerate() {
            assert_close(inverse_sqrt.coefficient(index), expected, 2e-15);
        }
    }
}
