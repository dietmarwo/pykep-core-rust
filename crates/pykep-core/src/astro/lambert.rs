// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/lambert_problem.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Izzo single- and multi-revolution Lambert solver.
//!
//! Iteration exhaustion is an explicit [`crate::PykepError::ConvergenceFailure`]
//! instead of returning the pinned C++ implementation's last unconverged
//! iterate. At the measure-zero `|x - 1| = 0.01` boundary, Rust selects the
//! Lagrange time-of-flight expression.

use core::f64::consts::PI;

use crate::error::{ensure_finite, ensure_finite_output};
use crate::math::linalg::{cross, norm, normalize};
use crate::{PykepError, Result, Vector3};

/// Deterministic position of a solution within its revolution family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LambertPath {
    /// The unique zero-revolution solution.
    ZeroRevolution,
    /// Multi-revolution solution on the left side of minimum time.
    Left,
    /// Multi-revolution solution on the right side of minimum time.
    Right,
}

/// One Lambert velocity solution.
#[derive(Clone, Debug, PartialEq)]
pub struct LambertSolution {
    /// Departure velocity.
    pub departure_velocity: Vector3,
    /// Arrival velocity.
    pub arrival_velocity: Vector3,
    /// Izzo solver variable.
    pub x: f64,
    /// Householder iterations used.
    pub iterations: usize,
    /// Complete revolutions.
    pub revolutions: usize,
    /// Path within the revolution family.
    pub path: LambertPath,
}

/// Inputs for one member of an ordered Lambert batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LambertRequest {
    /// Initial position.
    pub initial_position: Vector3,
    /// Final position.
    pub final_position: Vector3,
    /// Positive time of flight.
    pub time: f64,
    /// Positive gravitational parameter.
    pub mu: f64,
    /// Whether clockwise/retrograde motion is requested.
    pub clockwise: bool,
    /// Largest requested complete-revolution count.
    pub maximum_revolutions: usize,
}

impl LambertRequest {
    /// Creates one Lambert batch request.
    #[must_use]
    pub const fn new(
        initial_position: Vector3,
        final_position: Vector3,
        time: f64,
        mu: f64,
        clockwise: bool,
        maximum_revolutions: usize,
    ) -> Self {
        Self {
            initial_position,
            final_position,
            time,
            mu,
            clockwise,
            maximum_revolutions,
        }
    }

    fn solve(&self) -> Result<LambertProblem> {
        LambertProblem::new(
            self.initial_position,
            self.final_position,
            self.time,
            self.mu,
            self.clockwise,
            self.maximum_revolutions,
        )
    }
}

/// Solves an ordered batch of independent Lambert problems.
///
/// Zero workers uses Rayon's shared global pool, one executes serially, and
/// larger values use exactly that many cached worker threads. Both problem
/// order and each problem's branch order are deterministic.
///
/// # Errors
///
/// Returns an invalid worker count or the first Lambert error in input order.
pub fn solve_lambert_batch(
    requests: &[LambertRequest],
    workers: usize,
) -> Result<Vec<LambertProblem>> {
    crate::batch::try_map(requests, workers, LambertRequest::solve)
}

/// Solved Lambert boundary-value problem.
///
/// ```
/// use pykep_core::astro::lambert::{LambertPath, LambertProblem};
///
/// let problem = LambertProblem::new(
///     [1.0, 0.0, 0.0],
///     [0.2, 1.1, 0.3],
///     20.0,
///     1.0,
///     false,
///     2,
/// )?;
/// assert_eq!(problem.solutions()[0].path, LambertPath::ZeroRevolution);
/// # Ok::<(), pykep_core::PykepError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LambertProblem {
    initial_position: Vector3,
    final_position: Vector3,
    time: f64,
    mu: f64,
    clockwise: bool,
    chord: f64,
    semiperimeter: f64,
    lambda: f64,
    solutions: Vec<LambertSolution>,
}

impl LambertProblem {
    /// Solves a Lambert problem and all feasible branches through
    /// `maximum_revolutions`.
    ///
    /// Solution order is zero-revolution, then left/right for revolution
    /// counts `1..=maximum_revolutions`.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite input, non-positive time or `mu`, zero
    /// positions, coincident/collinear endpoints, or non-convergence.
    pub fn new(
        initial_position: Vector3,
        final_position: Vector3,
        time: f64,
        mu: f64,
        clockwise: bool,
        maximum_revolutions: usize,
    ) -> Result<Self> {
        for &value in initial_position.iter().chain(final_position.iter()) {
            ensure_finite("position", value)?;
        }
        ensure_finite("time", time)?;
        ensure_finite("mu", mu)?;
        if time <= 0.0 || mu <= 0.0 {
            return Err(PykepError::InvalidInput {
                parameter: "time/mu",
                reason: "must be greater than zero".into(),
            });
        }
        let initial_radius = norm(&initial_position)?;
        let final_radius = norm(&final_position)?;
        if initial_radius == 0.0 || final_radius == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "LambertProblem::new",
            });
        }
        let chord_vector = [
            final_position[0] - initial_position[0],
            final_position[1] - initial_position[1],
            final_position[2] - initial_position[2],
        ];
        let chord = norm(&chord_vector)?;
        if chord == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "LambertProblem::new",
            });
        }
        let initial_direction = normalize(&initial_position)?;
        let final_direction = normalize(&final_position)?;
        let angular = normalize(&cross(&initial_direction, &final_direction)?)?;
        if angular[2] == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "LambertProblem::automatic_direction",
            });
        }
        let semiperimeter = (chord + initial_radius + final_radius) / 2.0;
        let lambda_squared = 1.0 - chord / semiperimeter;
        let mut lambda = lambda_squared.sqrt();
        let mut initial_tangent = normalize(&cross(&angular, &initial_direction)?)?;
        let mut final_tangent = normalize(&cross(&angular, &final_direction)?)?;
        if angular[2] < 0.0 {
            lambda = -lambda;
            initial_tangent = initial_tangent.map(|value| -value);
            final_tangent = final_tangent.map(|value| -value);
        }
        if clockwise {
            lambda = -lambda;
            initial_tangent = initial_tangent.map(|value| -value);
            final_tangent = final_tangent.map(|value| -value);
        }
        let dimensionless_time = (2.0 * mu / semiperimeter.powi(3)).sqrt() * time;
        let mut problem = Self {
            initial_position,
            final_position,
            time,
            mu,
            clockwise,
            chord,
            semiperimeter,
            lambda,
            solutions: Vec::new(),
        };
        let mut maximum = ((dimensionless_time / PI) as usize).min(maximum_revolutions);
        let time_at_zero = lambda.acos() + lambda * (1.0 - lambda_squared).sqrt();
        let lambda_cubed = lambda * lambda_squared;
        let time_at_one = 2.0 / 3.0 * (1.0 - lambda_cubed);
        if maximum > 0 {
            let mut minimum_time = time_at_zero + maximum as f64 * PI;
            if dimensionless_time < minimum_time {
                let mut old_x = 0.0;
                for _ in 0..13 {
                    let (first, second, third) = problem.time_derivatives(old_x, minimum_time);
                    let new_x = if first != 0.0 {
                        old_x - first * second / (second * second - first * third / 2.0)
                    } else {
                        old_x
                    };
                    if (old_x - new_x).abs() < 1e-13 {
                        break;
                    }
                    minimum_time = problem.time_of_flight(new_x, maximum)?;
                    old_x = new_x;
                }
                if minimum_time > dimensionless_time {
                    maximum -= 1;
                }
            }
            maximum = maximum.min(maximum_revolutions);
        }

        let initial_x = if dimensionless_time >= time_at_zero {
            -(dimensionless_time - time_at_zero) / (dimensionless_time - time_at_zero + 4.0)
        } else if dimensionless_time <= time_at_one {
            time_at_one * (time_at_one - dimensionless_time)
                / (0.4 * (1.0 - lambda_squared * lambda_cubed) * dimensionless_time)
                + 1.0
        } else {
            (dimensionless_time / time_at_zero)
                .powf(core::f64::consts::LN_2 / (time_at_one / time_at_zero).ln())
                - 1.0
        };
        let (x, iterations) = problem.householder(dimensionless_time, initial_x, 0, 1e-5)?;
        problem.push_solution(
            x,
            iterations,
            0,
            LambertPath::ZeroRevolution,
            initial_radius,
            final_radius,
            &initial_direction,
            &final_direction,
            &initial_tangent,
            &final_tangent,
        )?;
        for revolution in 1..=maximum {
            let temporary =
                (((revolution + 1) as f64 * PI) / (8.0 * dimensionless_time)).powf(2.0 / 3.0);
            let initial_left = (temporary - 1.0) / (temporary + 1.0);
            let (left, iterations) =
                problem.householder(dimensionless_time, initial_left, revolution, 1e-8)?;
            problem.push_solution(
                left,
                iterations,
                revolution,
                LambertPath::Left,
                initial_radius,
                final_radius,
                &initial_direction,
                &final_direction,
                &initial_tangent,
                &final_tangent,
            )?;
            let temporary = (8.0 * dimensionless_time / (revolution as f64 * PI)).powf(2.0 / 3.0);
            let initial_right = (temporary - 1.0) / (temporary + 1.0);
            let (right, iterations) =
                problem.householder(dimensionless_time, initial_right, revolution, 1e-8)?;
            problem.push_solution(
                right,
                iterations,
                revolution,
                LambertPath::Right,
                initial_radius,
                final_radius,
                &initial_direction,
                &final_direction,
                &initial_tangent,
                &final_tangent,
            )?;
        }
        Ok(problem)
    }

    #[allow(clippy::too_many_arguments)]
    fn push_solution(
        &mut self,
        x: f64,
        iterations: usize,
        revolutions: usize,
        path: LambertPath,
        initial_radius: f64,
        final_radius: f64,
        initial_direction: &Vector3,
        final_direction: &Vector3,
        initial_tangent: &Vector3,
        final_tangent: &Vector3,
    ) -> Result<()> {
        let lambda_squared = self.lambda * self.lambda;
        let y = (1.0 - lambda_squared + lambda_squared * x * x).sqrt();
        let gamma = (self.mu * self.semiperimeter / 2.0).sqrt();
        let rho = (initial_radius - final_radius) / self.chord;
        let sigma = (1.0 - rho * rho).sqrt();
        let radial_initial =
            gamma * ((self.lambda * y - x) - rho * (self.lambda * y + x)) / initial_radius;
        let radial_final =
            -gamma * ((self.lambda * y - x) + rho * (self.lambda * y + x)) / final_radius;
        let tangential = gamma * sigma * (y + self.lambda * x);
        let tangential_initial = tangential / initial_radius;
        let tangential_final = tangential / final_radius;
        let departure_velocity = core::array::from_fn(|index| {
            radial_initial * initial_direction[index] + tangential_initial * initial_tangent[index]
        });
        let arrival_velocity = core::array::from_fn(|index| {
            radial_final * final_direction[index] + tangential_final * final_tangent[index]
        });
        for &value in departure_velocity.iter().chain(arrival_velocity.iter()) {
            ensure_finite_output("LambertProblem::velocity_reconstruction", value)?;
        }
        self.solutions.push(LambertSolution {
            departure_velocity,
            arrival_velocity,
            x,
            iterations,
            revolutions,
            path,
        });
        Ok(())
    }

    fn householder(
        &self,
        target_time: f64,
        mut x: f64,
        revolutions: usize,
        tolerance: f64,
    ) -> Result<(f64, usize)> {
        for iteration in 1..=15 {
            let time = self.time_of_flight(x, revolutions)?;
            let (first, second, third) = self.time_derivatives(x, time);
            let difference = time - target_time;
            let first_squared = first * first;
            let next = x - difference * (first_squared - difference * second / 2.0)
                / (first * (first_squared - difference * second)
                    + third * difference * difference / 6.0);
            if !next.is_finite() {
                return Err(PykepError::NumericalOverflow {
                    operation: "LambertProblem::householder",
                });
            }
            let error = (x - next).abs();
            x = next;
            if error <= tolerance {
                return Ok((x, iteration));
            }
        }
        Err(PykepError::ConvergenceFailure {
            operation: "LambertProblem::householder",
            iterations: 15,
        })
    }

    fn time_derivatives(&self, x: f64, time: f64) -> (f64, f64, f64) {
        let lambda_squared = self.lambda * self.lambda;
        let lambda_cubed = lambda_squared * self.lambda;
        let one_minus_x_squared = 1.0 - x * x;
        let y = (1.0 - lambda_squared * one_minus_x_squared).sqrt();
        let first = (3.0 * time * x - 2.0 + 2.0 * lambda_cubed * x / y) / one_minus_x_squared;
        let second = (3.0 * time
            + 5.0 * x * first
            + 2.0 * (1.0 - lambda_squared) * lambda_cubed / y.powi(3))
            / one_minus_x_squared;
        let third = (7.0 * x * second + 8.0 * first
            - 6.0 * (1.0 - lambda_squared) * lambda_squared * lambda_cubed * x / y.powi(5))
            / one_minus_x_squared;
        (first, second, third)
    }

    fn time_of_flight(&self, x: f64, revolutions: usize) -> Result<f64> {
        let distance = (x - 1.0).abs();
        if (0.01..0.2).contains(&distance) {
            return self.time_of_flight_lagrange(x, revolutions);
        }
        let lambda_squared = self.lambda * self.lambda;
        let energy = x * x - 1.0;
        let rho = energy.abs();
        let z = (1.0 + lambda_squared * energy).sqrt();
        let time = if distance < 0.01 {
            let eta = z - self.lambda * x;
            let series_argument = 0.5 * (1.0 - self.lambda - x * eta);
            let q = 4.0 / 3.0 * hypergeometric(series_argument)?;
            (eta.powi(3) * q + 4.0 * self.lambda * eta) / 2.0
                + revolutions as f64 * PI / rho.powf(1.5)
        } else {
            let y = rho.sqrt();
            let g = x * z - self.lambda * energy;
            let d = if energy < 0.0 {
                revolutions as f64 * PI + g.clamp(-1.0, 1.0).acos()
            } else {
                (y * (z - self.lambda * x) + g).ln()
            };
            (x - self.lambda * z - d / y) / energy
        };
        ensure_finite_output("LambertProblem::time_of_flight", time)
    }

    fn time_of_flight_lagrange(&self, x: f64, revolutions: usize) -> Result<f64> {
        let semi_axis = 1.0 / (1.0 - x * x);
        let time = if semi_axis > 0.0 {
            let alpha = 2.0 * x.clamp(-1.0, 1.0).acos();
            let mut beta = 2.0
                * (self.lambda.abs() / semi_axis.sqrt())
                    .clamp(-1.0, 1.0)
                    .asin();
            if self.lambda < 0.0 {
                beta = -beta;
            }
            semi_axis.powf(1.5)
                * ((alpha - alpha.sin()) - (beta - beta.sin()) + 2.0 * PI * revolutions as f64)
                / 2.0
        } else {
            let alpha = 2.0 * x.acosh();
            let mut beta = 2.0 * (-self.lambda * self.lambda / semi_axis).sqrt().asinh();
            if self.lambda < 0.0 {
                beta = -beta;
            }
            -semi_axis * (-semi_axis).sqrt() * ((beta - beta.sinh()) - (alpha - alpha.sinh())) / 2.0
        };
        ensure_finite_output("LambertProblem::time_of_flight_lagrange", time)
    }

    /// Initial position.
    #[must_use]
    pub const fn initial_position(&self) -> Vector3 {
        self.initial_position
    }

    /// Final position.
    #[must_use]
    pub const fn final_position(&self) -> Vector3 {
        self.final_position
    }

    /// Time of flight.
    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }

    /// Gravitational parameter.
    #[must_use]
    pub const fn mu(&self) -> f64 {
        self.mu
    }

    /// Whether clockwise/retrograde motion was requested.
    #[must_use]
    pub const fn clockwise(&self) -> bool {
        self.clockwise
    }

    /// Maximum feasible revolution count returned.
    #[must_use]
    pub fn maximum_revolutions(&self) -> usize {
        self.solutions
            .last()
            .map_or(0, |solution| solution.revolutions)
    }

    /// Ordered solution family.
    #[must_use]
    pub fn solutions(&self) -> &[LambertSolution] {
        &self.solutions
    }
}

fn hypergeometric(argument: f64) -> Result<f64> {
    let mut sum = 1.0;
    let mut coefficient = 1.0;
    for index in 0..10_000 {
        let index = index as f64;
        coefficient *= (3.0 + index) * (1.0 + index) / (2.5 + index) * argument / (index + 1.0);
        sum += coefficient;
        if coefficient.abs() <= 1e-11 {
            return ensure_finite_output("LambertProblem::hypergeometric", sum);
        }
    }
    Err(PykepError::ConvergenceFailure {
        operation: "LambertProblem::hypergeometric",
        iterations: 10_000,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::propagation::propagate_lagrangian;

    #[test]
    fn upstream_clockwise_reference_matches() {
        let problem = LambertProblem::new(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            3.0 * PI / 2.0,
            1.0,
            true,
            5,
        )
        .unwrap();
        assert_eq!(problem.maximum_revolutions(), 0);
        assert_eq!(problem.solutions()[0].iterations, 3);
        assert!((problem.solutions()[0].x + 0.382_683_432_365_089_6).abs() < 1e-14);
        for (actual, expected) in problem.solutions()[0]
            .departure_velocity
            .iter()
            .zip([0.0, -1.0, 0.0])
        {
            assert!((actual - expected).abs() < 3e-15);
        }
        for (actual, expected) in problem.solutions()[0]
            .arrival_velocity
            .iter()
            .zip([1.0, 0.0, 0.0])
        {
            assert!((actual - expected).abs() < 3e-15);
        }
    }

    #[test]
    fn all_solution_branches_reconstruct_the_endpoint() {
        let problem =
            LambertProblem::new([1.0, 0.0, 0.0], [0.2, 1.1, 0.3], 20.0, 1.0, false, 4).unwrap();
        assert!(problem.solutions().len() >= 3);
        for solution in problem.solutions() {
            let initial = [
                1.0,
                0.0,
                0.0,
                solution.departure_velocity[0],
                solution.departure_velocity[1],
                solution.departure_velocity[2],
            ];
            let propagated = propagate_lagrangian(&initial, 20.0, 1.0).unwrap();
            for (actual, expected) in propagated[..3].iter().zip([0.2, 1.1, 0.3]) {
                assert!((actual - expected).abs() < 2e-10);
            }
        }
    }

    #[test]
    fn time_of_flight_dispatches_cover_each_expression_family() {
        for clockwise in [false, true] {
            let problem =
                LambertProblem::new([1.0, 0.0, 0.0], [0.2, 1.1, 0.3], 20.0, 1.0, clockwise, 0)
                    .unwrap();
            for x in [0.995, 0.9, 0.5, 1.1, 2.0] {
                assert!(problem.time_of_flight(x, 0).unwrap().is_finite());
            }
            let target = problem.time_of_flight(0.2, 0).unwrap();
            assert!(matches!(
                problem.householder(target, 0.2, 0, -1.0),
                Err(PykepError::ConvergenceFailure {
                    operation: "LambertProblem::householder",
                    iterations: 15
                })
            ));
        }
        assert!(matches!(
            hypergeometric(f64::NAN),
            Err(PykepError::ConvergenceFailure {
                operation: "LambertProblem::hypergeometric",
                iterations: 10_000
            })
        ));
    }

    #[test]
    fn invalid_geometries_and_scales_are_typed_errors() {
        let valid = [1.0, 0.0, 0.0];
        for result in [
            LambertProblem::new([f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0, 1.0, false, 0),
            LambertProblem::new(valid, [0.0, 1.0, 0.0], 0.0, 1.0, false, 0),
            LambertProblem::new(valid, [0.0, 1.0, 0.0], 1.0, 0.0, false, 0),
            LambertProblem::new([0.0; 3], [0.0, 1.0, 0.0], 1.0, 1.0, false, 0),
            LambertProblem::new(valid, valid, 1.0, 1.0, false, 0),
            LambertProblem::new(valid, [0.0, 0.0, 1.0], 1.0, 1.0, false, 0),
            LambertProblem::new([1e308; 3], [-1e308; 3], 1.0, 1.0, false, 0),
        ] {
            assert!(result.is_err());
        }
    }

    #[test]
    fn ordered_parallel_batch_matches_scalar_problems_and_error_order() {
        let requests = [
            LambertRequest::new([1.0, 0.0, 0.0], [0.2, 1.1, 0.3], 20.0, 1.0, false, 2),
            LambertRequest::new([1.1, 0.1, 0.0], [0.1, 1.2, 0.2], 22.0, 1.0, true, 1),
        ];
        let batch = solve_lambert_batch(&requests, 2).unwrap();
        assert_eq!(
            batch,
            requests
                .iter()
                .map(LambertRequest::solve)
                .collect::<Result<Vec<_>>>()
                .unwrap()
        );

        let invalid = [
            LambertRequest::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0, 1.0, false, 0),
            LambertRequest::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0, 0.0, false, 0),
        ];
        assert_eq!(
            solve_lambert_batch(&invalid, 2).unwrap_err(),
            invalid[0].solve().unwrap_err()
        );
    }
}
