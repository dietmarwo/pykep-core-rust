// SPDX-License-Identifier: MPL-2.0

use core::ops::{Add, Div, Mul, Neg, Sub};

pub(super) const WIDTH: usize = super::MAX_ORDER + 1;

pub(crate) trait Coefficient:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const ZERO: Self;
    fn primal(self) -> f64;
    fn splat(value: f64) -> Self;
    fn powf(self, exponent: f64) -> Self;
    fn exp(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
}

impl Coefficient for f64 {
    const ZERO: Self = 0.0;
    #[inline]
    fn primal(self) -> f64 {
        self
    }
    #[inline]
    fn splat(value: f64) -> Self {
        value
    }
    #[inline]
    fn powf(self, exponent: f64) -> Self {
        self.powf(exponent)
    }
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

#[inline]
pub(super) fn product<T: Coefficient>(a: &[T; WIDTH], b: &[T; WIDTH], n: usize) -> T {
    (0..=n).fold(T::ZERO, |sum, k| sum + a[k] * b[n - k])
}

#[inline]
pub(super) fn square<T: Coefficient>(a: &[T; WIDTH], n: usize) -> T {
    let half = (0..n.div_ceil(2)).fold(T::ZERO, |sum, k| sum + a[k] * a[n - k]);
    let doubled = half + half;
    if n % 2 == 0 {
        doubled + a[n / 2] * a[n / 2]
    } else {
        doubled
    }
}

#[inline]
pub(super) fn power<T: Coefficient>(a: &[T; WIDTH], b: &[T; WIDTH], exponent: f64, n: usize) -> T {
    if n == 0 {
        return a[0].powf(exponent);
    }
    let degree = n as f64;
    let sum = (0..n).fold(T::ZERO, |sum, k| {
        sum + T::splat(degree * exponent - k as f64 * (exponent + 1.0)) * a[n - k] * b[k]
    });
    sum / (T::splat(degree) * a[0])
}

#[inline]
fn derivative_product<T: Coefficient>(a: &[T; WIDTH], b: &[T; WIDTH], n: usize) -> T {
    (1..=n).fold(T::ZERO, |sum, k| sum + T::splat(k as f64) * a[k] * b[n - k]) / T::splat(n as f64)
}

#[inline]
pub(super) fn exponential<T: Coefficient>(a: &[T; WIDTH], b: &[T; WIDTH], n: usize) -> T {
    if n == 0 {
        a[0].exp()
    } else {
        derivative_product(a, b, n)
    }
}

#[inline]
pub(super) fn trigonometric<T: Coefficient>(
    a: &[T; WIDTH],
    b: &[T; WIDTH],
    sine: bool,
    n: usize,
) -> T {
    match (n, sine) {
        (0, true) => a[0].sin(),
        (0, false) => a[0].cos(),
        (_, true) => derivative_product(a, b, n),
        (_, false) => -derivative_product(a, b, n),
    }
}
