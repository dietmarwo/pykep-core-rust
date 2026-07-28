// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Incremental coefficient evaluation for traced scalar expressions.

use core::cell::RefCell;
use core::ops::{Add, Div, Mul, Neg, Sub};

use super::MAX_ORDER;

thread_local! {
    static COEFFICIENT_WORKSPACE: RefCell<Vec<[f64; MAX_ORDER + 1]>> =
        const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Operation {
    Constant(f64),
    Time,
    State(usize),
    Parameter(usize),
    Add(usize, usize),
    Subtract(usize, usize),
    Multiply(usize, usize),
    Negate(usize),
    Power(usize, f64),
    Exponential(usize),
    Sine(usize),
    Cosine(usize),
    StopGradient(usize),
}

#[derive(Debug, Default)]
pub(super) struct TapeBuilder {
    operations: RefCell<Vec<Operation>>,
}

impl TapeBuilder {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn constant(&self, value: f64) -> Expression<'_> {
        self.push(Operation::Constant(value))
    }

    pub(super) fn time(&self) -> Expression<'_> {
        self.push(Operation::Time)
    }

    pub(super) fn state(&self, index: usize) -> Expression<'_> {
        self.push(Operation::State(index))
    }

    pub(super) fn parameter(&self, index: usize) -> Expression<'_> {
        self.push(Operation::Parameter(index))
    }

    fn push(&self, operation: Operation) -> Expression<'_> {
        let operation = match operation {
            Operation::Add(left, right) if left > right => Operation::Add(right, left),
            Operation::Multiply(left, right) if left > right => Operation::Multiply(right, left),
            operation => operation,
        };
        let mut operations = self.operations.borrow_mut();
        if let Some(index) = operations
            .iter()
            .position(|candidate| *candidate == operation)
        {
            return Expression {
                builder: self,
                index,
            };
        }
        let index = operations.len();
        operations.push(operation);
        Expression {
            builder: self,
            index,
        }
    }

    fn constant_value(&self, expression: Expression<'_>) -> Option<f64> {
        match self.operations.borrow()[expression.index] {
            Operation::Constant(value) => Some(value),
            _ => None,
        }
    }

    fn add<'a>(&'a self, left: Expression<'a>, right: Expression<'a>) -> Expression<'a> {
        debug_assert!(core::ptr::eq(left.builder, right.builder));
        match (self.constant_value(left), self.constant_value(right)) {
            (Some(0.0), _) => right,
            (_, Some(0.0)) => left,
            (Some(left), Some(right)) => self.constant(left + right),
            _ => self.push(Operation::Add(left.index, right.index)),
        }
    }

    fn subtract<'a>(&'a self, left: Expression<'a>, right: Expression<'a>) -> Expression<'a> {
        debug_assert!(core::ptr::eq(left.builder, right.builder));
        match (self.constant_value(left), self.constant_value(right)) {
            (_, Some(0.0)) => left,
            (Some(left), Some(right)) => self.constant(left - right),
            _ if left.index == right.index => self.constant(0.0),
            _ => self.push(Operation::Subtract(left.index, right.index)),
        }
    }

    fn multiply<'a>(&'a self, left: Expression<'a>, right: Expression<'a>) -> Expression<'a> {
        debug_assert!(core::ptr::eq(left.builder, right.builder));
        match (self.constant_value(left), self.constant_value(right)) {
            (Some(0.0), _) | (_, Some(0.0)) => self.constant(0.0),
            (Some(1.0), _) => right,
            (_, Some(1.0)) => left,
            (Some(left), Some(right)) => self.constant(left * right),
            _ => self.push(Operation::Multiply(left.index, right.index)),
        }
    }

    fn negate<'a>(&'a self, expression: Expression<'a>) -> Expression<'a> {
        match self.constant_value(expression) {
            Some(value) => self.constant(-value),
            _ => self.push(Operation::Negate(expression.index)),
        }
    }

    fn power<'a>(&'a self, expression: Expression<'a>, exponent: f64) -> Expression<'a> {
        if exponent == 0.0 {
            return self.constant(1.0);
        }
        if exponent == 1.0 {
            return expression;
        }
        match self.constant_value(expression) {
            Some(value) => self.constant(value.powf(exponent)),
            None => self.push(Operation::Power(expression.index, exponent)),
        }
    }

    fn exp<'a>(&'a self, expression: Expression<'a>) -> Expression<'a> {
        match self.constant_value(expression) {
            Some(value) => self.constant(value.exp()),
            None => self.push(Operation::Exponential(expression.index)),
        }
    }

    fn sin_cos<'a>(&'a self, expression: Expression<'a>) -> (Expression<'a>, Expression<'a>) {
        match self.constant_value(expression) {
            Some(value) => {
                let (sine, cosine) = value.sin_cos();
                (self.constant(sine), self.constant(cosine))
            }
            None => (
                self.push(Operation::Sine(expression.index)),
                self.push(Operation::Cosine(expression.index)),
            ),
        }
    }

    fn stop_gradient<'a>(&'a self, expression: Expression<'a>) -> Expression<'a> {
        self.push(Operation::StopGradient(expression.index))
    }

    pub(super) fn gradient<'a, const W: usize>(
        &'a self,
        output: Expression<'a>,
        variables: [Expression<'a>; W],
    ) -> [Expression<'a>; W] {
        debug_assert!(
            variables
                .iter()
                .all(|variable| core::ptr::eq(output.builder, variable.builder))
        );
        let original_count = self.operations.borrow().len();
        let mut adjoints = vec![None; original_count];
        adjoints[output.index] = Some(self.constant(1.0));

        for index in (0..original_count).rev() {
            let Some(adjoint) = adjoints[index] else {
                continue;
            };
            let operation = self.operations.borrow()[index];
            let expression = |node| Expression {
                builder: self,
                index: node,
            };
            match operation {
                Operation::Add(left, right) => {
                    accumulate_adjoint(&mut adjoints, left, adjoint);
                    accumulate_adjoint(&mut adjoints, right, adjoint);
                }
                Operation::Subtract(left, right) => {
                    accumulate_adjoint(&mut adjoints, left, adjoint);
                    accumulate_adjoint(&mut adjoints, right, -adjoint);
                }
                Operation::Multiply(left, right) => {
                    accumulate_adjoint(&mut adjoints, left, adjoint * expression(right));
                    accumulate_adjoint(&mut adjoints, right, adjoint * expression(left));
                }
                Operation::Negate(input) => {
                    accumulate_adjoint(&mut adjoints, input, -adjoint);
                }
                Operation::Power(input, exponent) => {
                    accumulate_adjoint(
                        &mut adjoints,
                        input,
                        adjoint * expression(input).powf(exponent - 1.0) * exponent,
                    );
                }
                Operation::Exponential(input) => {
                    accumulate_adjoint(&mut adjoints, input, adjoint * expression(index));
                }
                Operation::Sine(input) => {
                    accumulate_adjoint(&mut adjoints, input, adjoint * expression(input).cos());
                }
                Operation::Cosine(input) => {
                    accumulate_adjoint(&mut adjoints, input, -adjoint * expression(input).sin());
                }
                Operation::Constant(_)
                | Operation::Time
                | Operation::State(_)
                | Operation::Parameter(_)
                | Operation::StopGradient(_) => {}
            }
        }

        variables.map(|variable| adjoints[variable.index].unwrap_or_else(|| self.constant(0.0)))
    }

    pub(super) fn finish<const N: usize>(self, outputs: [usize; N]) -> IncrementalTape<N> {
        let operations = self.operations.into_inner();
        let trig_companions = operations
            .iter()
            .map(|operation| match operation {
                Operation::Sine(input) => operations
                    .iter()
                    .position(|candidate| *candidate == Operation::Cosine(*input)),
                Operation::Cosine(input) => operations
                    .iter()
                    .position(|candidate| *candidate == Operation::Sine(*input)),
                _ => None,
            })
            .collect();
        IncrementalTape {
            operations,
            trig_companions,
            outputs,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Expression<'a> {
    builder: &'a TapeBuilder,
    index: usize,
}

impl Expression<'_> {
    pub(super) fn index(self) -> usize {
        self.index
    }

    pub(super) fn powf(self, exponent: f64) -> Self {
        self.builder.power(self, exponent)
    }

    pub(super) fn sqrt(self) -> Self {
        self.powf(0.5)
    }

    pub(super) fn exp(self) -> Self {
        self.builder.exp(self)
    }

    pub(super) fn sin(self) -> Self {
        self.builder.sin_cos(self).0
    }

    pub(super) fn cos(self) -> Self {
        self.builder.sin_cos(self).1
    }

    pub(super) fn sin_cos(self) -> (Self, Self) {
        self.builder.sin_cos(self)
    }

    pub(super) fn stop_gradient(self) -> Self {
        self.builder.stop_gradient(self)
    }
}

fn accumulate_adjoint<'a>(
    adjoints: &mut [Option<Expression<'a>>],
    index: usize,
    contribution: Expression<'a>,
) {
    adjoints[index] = Some(match adjoints[index] {
        Some(adjoint) => adjoint + contribution,
        None => contribution,
    });
}

impl Add for Expression<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.builder.add(self, rhs)
    }
}

impl Add<f64> for Expression<'_> {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        self + self.builder.constant(rhs)
    }
}

impl Sub for Expression<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.builder.subtract(self, rhs)
    }
}

impl Sub<f64> for Expression<'_> {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        self - self.builder.constant(rhs)
    }
}

impl Mul for Expression<'_> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.builder.multiply(self, rhs)
    }
}

impl Mul<f64> for Expression<'_> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self * self.builder.constant(rhs)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div for Expression<'_> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.powf(-1.0)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f64> for Expression<'_> {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self * self.builder.constant(rhs.recip())
    }
}

impl Neg for Expression<'_> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.builder.negate(self)
    }
}

#[derive(Debug)]
pub(super) struct IncrementalTape<const N: usize> {
    operations: Vec<Operation>,
    trig_companions: Vec<Option<usize>>,
    outputs: [usize; N],
}

impl<const N: usize> IncrementalTape<N> {
    pub(super) fn coefficients<const P: usize>(
        &self,
        time: f64,
        state: &[f64; N],
        parameters: &[f64; P],
        order: usize,
        jet: &mut [[f64; MAX_ORDER + 1]; N],
    ) {
        *jet = [[0.0; MAX_ORDER + 1]; N];
        for (component, &value) in jet.iter_mut().zip(state) {
            component[0] = value;
        }
        COEFFICIENT_WORKSPACE.with(|workspace| {
            let mut coefficients = workspace.borrow_mut();
            coefficients.resize(self.operations.len(), [0.0; MAX_ORDER + 1]);

            for coefficient_order in 0..order {
                for (index, operation) in self.operations.iter().copied().enumerate() {
                    let value = match operation {
                        Operation::Constant(value) => {
                            if coefficient_order == 0 {
                                value
                            } else {
                                0.0
                            }
                        }
                        Operation::Time => match coefficient_order {
                            0 => time,
                            1 => 1.0,
                            _ => 0.0,
                        },
                        Operation::State(state_index) => jet[state_index][coefficient_order],
                        Operation::Parameter(parameter_index) => {
                            if coefficient_order == 0 {
                                parameters[parameter_index]
                            } else {
                                0.0
                            }
                        }
                        Operation::Add(left, right) => {
                            coefficients[left][coefficient_order]
                                + coefficients[right][coefficient_order]
                        }
                        Operation::Subtract(left, right) => {
                            coefficients[left][coefficient_order]
                                - coefficients[right][coefficient_order]
                        }
                        Operation::Multiply(left, right) => (0..=coefficient_order)
                            .map(|k| {
                                coefficients[left][k] * coefficients[right][coefficient_order - k]
                            })
                            .sum(),
                        Operation::Negate(input) => -coefficients[input][coefficient_order],
                        Operation::Power(input, exponent) => {
                            if coefficient_order == 0 {
                                coefficients[input][0].powf(exponent)
                            } else {
                                let n = coefficient_order as f64;
                                let sum = (0..coefficient_order)
                                    .map(|k| {
                                        let kf = k as f64;
                                        (n * exponent - kf * (exponent + 1.0))
                                            * coefficients[input][coefficient_order - k]
                                            * coefficients[index][k]
                                    })
                                    .sum::<f64>();
                                sum / (n * coefficients[input][0])
                            }
                        }
                        Operation::Exponential(input) => {
                            if coefficient_order == 0 {
                                coefficients[input][0].exp()
                            } else {
                                (1..=coefficient_order)
                                    .map(|k| {
                                        k as f64
                                            * coefficients[input][k]
                                            * coefficients[index][coefficient_order - k]
                                    })
                                    .sum::<f64>()
                                    / coefficient_order as f64
                            }
                        }
                        Operation::Sine(input) => {
                            if coefficient_order == 0 {
                                coefficients[input][0].sin()
                            } else {
                                let cosine = self.trig_companions[index]
                                    .expect("sine tape nodes have a cosine companion");
                                (1..=coefficient_order)
                                    .map(|k| {
                                        k as f64
                                            * coefficients[input][k]
                                            * coefficients[cosine][coefficient_order - k]
                                    })
                                    .sum::<f64>()
                                    / coefficient_order as f64
                            }
                        }
                        Operation::Cosine(input) => {
                            if coefficient_order == 0 {
                                coefficients[input][0].cos()
                            } else {
                                let sine = self.trig_companions[index]
                                    .expect("cosine tape nodes have a sine companion");
                                -(1..=coefficient_order)
                                    .map(|k| {
                                        k as f64
                                            * coefficients[input][k]
                                            * coefficients[sine][coefficient_order - k]
                                    })
                                    .sum::<f64>()
                                    / coefficient_order as f64
                            }
                        }
                        Operation::StopGradient(input) => coefficients[input][coefficient_order],
                    };
                    coefficients[index][coefficient_order] = value;
                }

                let divisor = (coefficient_order + 1) as f64;
                for (component, &output) in self.outputs.iter().enumerate() {
                    jet[component][coefficient_order + 1] =
                        coefficients[output][coefficient_order] / divisor;
                }
            }
        });
    }

    #[cfg(test)]
    pub(super) fn operation_count(&self) -> usize {
        self.operations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tape_advances_a_nonlinear_equation_incrementally() {
        let builder = TapeBuilder::new();
        let state = builder.state(0);
        let derivative = (state * state + state.sqrt()) / 3.0;
        let output = derivative.index();
        let tape = builder.finish([output]);
        let mut jet = [[0.0; MAX_ORDER + 1]; 1];
        tape.coefficients(0.0, &[0.2], &[], 8, &mut jet);

        let mut reference = [0.0; MAX_ORDER + 1];
        reference[0] = 0.2;
        for n in 0..8 {
            let series = super::super::series::Series::from_coefficients(&reference, n);
            reference[n + 1] =
                ((series * series + series.sqrt()) / 3.0).coefficient(n) / (n + 1) as f64;
        }
        for coefficient in 0..=8 {
            assert!((jet[0][coefficient] - reference[coefficient]).abs() < 2e-15);
        }
    }

    #[test]
    fn tape_eliminates_common_subexpressions_and_generates_stopped_gradients() {
        let builder = TapeBuilder::new();
        let x = builder.state(0);
        let y = builder.state(1);
        assert_eq!((x * y).index(), (y * x).index());
        assert_eq!((x + y).index(), (y + x).index());

        let stopped_square = (x * x).stop_gradient();
        let expression = stopped_square * x;
        let gradient = builder.gradient(expression, [x])[0];
        let outputs = [expression.index(), gradient.index()];
        let tape = builder.finish(outputs);
        let mut jet = [[0.0; MAX_ORDER + 1]; 2];
        tape.coefficients(0.0, &[2.0, 0.0], &[], 1, &mut jet);
        assert_eq!(jet[0][1], 8.0);
        assert_eq!(jet[1][1], 4.0);
    }

    #[test]
    fn tape_advances_time_trigonometric_and_exponential_nodes() {
        let builder = TapeBuilder::new();
        let output = {
            let state = builder.state(0);
            let angle = builder.parameter(0) * builder.time();
            (state.exp() + angle.sin() - angle.cos()).index()
        };
        let tape = builder.finish([output]);
        let initial_time = 0.3;
        let initial_state = [0.2];
        let parameters = [1.7];
        let mut jet = [[0.0; MAX_ORDER + 1]; 1];
        tape.coefficients(initial_time, &initial_state, &parameters, 12, &mut jet);

        let mut reference = [0.0; MAX_ORDER + 1];
        reference[0] = initial_state[0];
        for n in 0..12 {
            let state = super::super::series::Series::from_coefficients(&reference, n);
            let time = super::super::series::Series::variable(initial_time, n);
            let angle = time * parameters[0];
            reference[n + 1] =
                (state.exp() + angle.sin() - angle.cos()).coefficient(n) / (n + 1) as f64;
        }
        for coefficient in 0..=12 {
            let scale = reference[coefficient].abs().max(1.0);
            assert!(
                (jet[0][coefficient] - reference[coefficient]).abs() <= 2e-14 * scale,
                "coefficient={coefficient}: incremental={:.17e}, reference={:.17e}",
                jet[0][coefficient],
                reference[coefficient],
            );
        }
    }
}
