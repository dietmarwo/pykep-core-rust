// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Fixed-protocol DOP853/Taylor accuracy and wall-time comparison.

use std::hint::black_box;
use std::time::{Duration, Instant};

use pykep_core::astro::propagation::propagate_lagrangian;
use pykep_core::dynamics::KeplerDynamics;
use pykep_core::integration::{
    Dop853, InitialValueProblem, IntegrationStats, IntegratorOptions, Propagation, Taylor,
};

fn energy(state: &[f64; 6], mu: f64) -> f64 {
    let radius = state[..3]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    0.5 * state[3..].iter().map(|value| value * value).sum::<f64>() - mu / radius
}

fn max_error(actual: &[f64; 6], expected: &[f64; 6]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn median(samples: &mut [Duration]) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64()
}

fn timed(mut operation: impl FnMut() -> Propagation<6>, repeats: usize) -> (Propagation<6>, f64) {
    let value = operation();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        black_box(operation());
        samples.push(started.elapsed());
    }
    (value, median(&mut samples))
}

fn print_result(
    method: &str,
    tolerance: f64,
    revolutions: u32,
    result: &Propagation<6>,
    elapsed: f64,
    reference: &[f64; 6],
    initial_energy: f64,
) {
    let IntegrationStats {
        rhs_evaluations,
        accepted_steps,
        rejected_steps,
    } = result.stats;
    println!(
        "{method},{tolerance:.17e},{revolutions},{accepted_steps},{rejected_steps},\
         {rhs_evaluations},{elapsed:.17e},{:.17e},{:.17e}",
        max_error(&result.state, reference),
        (energy(&result.state, 1.0) - initial_energy).abs()
    );
}

fn main() {
    let initial = [0.5, 0.0, 0.0, 0.0, 3.0_f64.sqrt(), 0.0];
    let initial_energy = energy(&initial, 1.0);
    println!(
        "method,tolerance,revolutions,accepted_steps,rejected_steps,work_units,\
         median_seconds,max_state_error,energy_drift"
    );
    for tolerance in [1e-9, 1e-12, 1e-14, f64::EPSILON] {
        let options = IntegratorOptions {
            relative_tolerance: tolerance,
            absolute_tolerance: tolerance,
            maximum_steps: 2_000_000,
            ..IntegratorOptions::default()
        };
        for revolutions in [1_u32, 100, 1_000] {
            let final_time = f64::from(revolutions) * core::f64::consts::TAU;
            let problem = InitialValueProblem::new(0.0, initial, final_time, [1.0]);
            let reference = propagate_lagrangian(&initial, final_time, 1.0).unwrap();
            let repeats = match revolutions {
                1 => 101,
                100 => 21,
                _ => 7,
            };
            let (dop, elapsed) = timed(
                || Dop853.propagate(&KeplerDynamics, problem, options).unwrap(),
                repeats,
            );
            print_result(
                "dop853",
                tolerance,
                revolutions,
                &dop,
                elapsed,
                &reference,
                initial_energy,
            );
            let (taylor, elapsed) = timed(
                || Taylor.propagate(&KeplerDynamics, problem, options).unwrap(),
                repeats,
            );
            print_result(
                "taylor",
                tolerance,
                revolutions,
                &taylor,
                elapsed,
                &reference,
                initial_energy,
            );
        }
    }
}
