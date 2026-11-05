// SPDX-License-Identifier: MPL-2.0

use super::*;
use pykep_core::astro::propagation::propagate_lagrangian;

pub(super) fn enabled() -> bool {
    std::env::args().any(|arg| arg == "--accuracy-corpus")
}

pub(super) const HEADER: &str = "model,scenario,split,tolerance,steps,work_units,scaled_error,reference_convergence,limit,status";

pub(super) fn check<M, const N: usize, const P: usize>(
    name: &str,
    model: &M,
    state: [f64; N],
    parameters: [f64; P],
    duration: f64,
) where
    M: TaylorDynamicsModel<N, P>,
{
    let requested = argument("--tolerance=")
        .map_or(1e-12, |tol| tol.parse().expect("valid requested tolerance"));
    let shifted = std::array::from_fn(|i| state[i] * 1.003 + 0.0001 * (i + 1) as f64);
    let changed_parameters = parameters.map(|p| p * 1.001);
    let cases = [
        ("original", "tune", state, parameters, 0.0, duration),
        (
            "shifted",
            "holdout",
            shifted,
            changed_parameters,
            0.37,
            duration * 0.8,
        ),
        (
            "backward",
            "holdout",
            shifted,
            parameters,
            0.9,
            -duration * 0.45,
        ),
    ];
    for (scenario, split, state, parameters, time, span) in cases {
        endpoint(
            name,
            scenario,
            split,
            model,
            InitialValueProblem::new(time, state, time + span, parameters),
            requested,
            2e-11,
        );
    }
    if name.ends_with("_mass") && N == 14 {
        let mut positive = state;
        positive[13] = -20.0;
        endpoint(
            name,
            "positive_switch",
            "holdout",
            model,
            InitialValueProblem::new(0.0, positive, duration * 0.25, parameters),
            requested,
            2e-11,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn endpoint<M, const N: usize, const P: usize>(
    name: &str,
    scenario: &str,
    split: &str,
    model: &M,
    problem: InitialValueProblem<N, P>,
    requested: f64,
    limit: f64,
) where
    M: TaylorDynamicsModel<N, P>,
{
    let span = (problem.final_time - problem.initial_time).abs();
    let reference = |subdivisions: f64| {
        Dop853.propagate(
            model,
            problem,
            IntegratorOptions {
                relative_tolerance: 3e-14,
                absolute_tolerance: 3e-14,
                maximum_step: Some(span / subdivisions),
                ..IntegratorOptions::default()
            },
        )
    };
    let actual = Taylor.propagate(
        model,
        problem,
        IntegratorOptions {
            relative_tolerance: requested,
            absolute_tolerance: requested,
            ..IntegratorOptions::default()
        },
    );
    match (reference(128.0), reference(256.0), actual) {
        (Ok(coarse), Ok(fine), Ok(candidate)) => {
            let convergence = error(&coarse.state, &fine.state);
            let delta = error(&candidate.state, &fine.state);
            let status = if convergence > limit * 0.1 {
                "reference_uncertain"
            } else if delta > limit {
                "over_budget"
            } else {
                "pass"
            };
            println!(
                "{name},{scenario},{split},{requested},{},{},{delta:.17e},{convergence:.17e},{limit:.17e},{status}",
                candidate.stats.accepted_steps, candidate.stats.rhs_evaluations
            );
        }
        (coarse, fine, actual) => {
            eprintln!("{name}/{scenario}: {coarse:?}, {fine:?}, {actual:?}");
            println!("{name},{scenario},{split},{requested},0,0,,,{limit:.17e},failed");
        }
    }
}

pub(super) fn extra() {
    if selected("cr3bp") {
        let mu = 0.012_150_585_609_624_04;
        endpoint(
            "cr3bp",
            "primary_approach",
            "holdout",
            &Cr3bpDynamics,
            InitialValueProblem::new(0.2, [0.96 - mu, 0.02, 0.005, 0.1, -0.4, 0.02], 0.24, [mu]),
            1e-14,
            2e-10,
        );
        endpoint(
            "cr3bp",
            "long_arc",
            "holdout",
            &Cr3bpDynamics,
            InitialValueProblem::new(0.0, [0.8, -0.2, 0.1, 0.03, -0.04, 0.02], 3.0, [mu]),
            1e-14,
            5e-10,
        );
    }
    if selected("zoh_kepler") {
        endpoint(
            "zoh_kepler",
            "low_mass",
            "holdout",
            &ZohKeplerDynamics,
            InitialValueProblem::new(
                0.0,
                [0.8, -0.2, 0.1, 0.03, 1.0, 0.02, 0.01],
                0.03,
                [0.02, 0.3, -0.4, 0.5, 0.01],
            ),
            1e-14,
            2e-11,
        );
    }
    if selected("kepler") {
        let state = [0.2, 0.0, 0.0, 0.0, 3.0, 0.0];
        let period = std::f64::consts::TAU;
        for revolutions in [1, 100, 1000] {
            // The initial state has semimajor axis one. Include a non-integer phase.
            let duration = f64::from(revolutions) * period + 0.13;
            let reference = propagate_lagrangian(&state, 0.13, 1.0).unwrap();
            let requested = 1e-14;
            let candidate = Taylor
                .propagate(
                    &KeplerDynamics,
                    InitialValueProblem::new(0.0, state, duration, [1.0]),
                    IntegratorOptions {
                        absolute_tolerance: requested,
                        relative_tolerance: requested,
                        ..IntegratorOptions::default()
                    },
                )
                .unwrap();
            let delta = error(&candidate.state, &reference);
            // Allow accumulated phase drift over this high-eccentricity orbit.
            let limit = 1e-8 * f64::from(revolutions);
            let status = if delta <= limit {
                "pass"
            } else {
                "over_budget"
            };
            println!(
                "kepler,eccentric_{revolutions}_orbits,holdout,{requested},{},{},{delta:.17e},0,{limit:.17e},{status}",
                candidate.stats.accepted_steps, candidate.stats.rhs_evaluations
            );
        }
    }
}
