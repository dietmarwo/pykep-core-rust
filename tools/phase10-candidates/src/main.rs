// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use std::hint::black_box;
use std::time::{Duration, Instant};

use ode_solvers::{Dop853 as CandidateDop853, OutputType, System, Vector6};
use pykep_core::integration::{
    Dop853, DynamicsModel, InitialValueProblem, IntegratorOptions,
};
use pykep_core::{PykepError, Result};

const CALLS_PER_SAMPLE: usize = 200;
const SAMPLES: usize = 30;
const FINAL_TIME: f64 = 5.785_665_678_258_923;
const INITIAL_STATE: [f64; 6] = [1.2, -0.3, 0.1, 0.2, 0.8, -0.1];

struct Kepler;

fn kepler_rhs(state: &[f64; 6], mu: f64, derivative: &mut [f64; 6]) -> Result<()> {
    let radius_squared = state[..3].iter().map(|value| value * value).sum::<f64>();
    if radius_squared == 0.0 {
        return Err(PykepError::SingularGeometry {
            operation: "phase10_candidate_kepler",
        });
    }
    let scale = -mu / (radius_squared * radius_squared.sqrt());
    derivative[..3].copy_from_slice(&state[3..]);
    for axis in 0..3 {
        derivative[axis + 3] = scale * state[axis];
    }
    Ok(())
}

impl DynamicsModel<6, 1> for Kepler {
    const NAME: &'static str = "phase10_candidate_kepler";

    fn rhs(
        &self,
        _time: f64,
        state: &[f64; 6],
        parameters: &[f64; 1],
        derivative: &mut [f64; 6],
    ) -> Result<()> {
        kepler_rhs(state, parameters[0], derivative)
    }
}

#[derive(Clone, Copy)]
struct OdeSolversKepler;

impl System<f64, Vector6<f64>> for OdeSolversKepler {
    fn system(&self, _time: f64, state: &Vector6<f64>, derivative: &mut Vector6<f64>) {
        let mut output = [0.0; 6];
        kepler_rhs(
            &[
                state[0], state[1], state[2], state[3], state[4], state[5],
            ],
            1.0,
            &mut output,
        )
        .unwrap();
        derivative.copy_from_slice(&output);
    }
}

fn selected() -> f64 {
    let problem = InitialValueProblem::new(0.0, INITIAL_STATE, FINAL_TIME, [1.0]);
    let options = IntegratorOptions {
        relative_tolerance: 1e-12,
        absolute_tolerance: 1e-12,
        ..IntegratorOptions::default()
    };
    Dop853
        .propagate(&Kepler, problem, options)
        .unwrap()
        .state[0]
}

fn ode_solvers() -> f64 {
    let initial = Vector6::from_row_slice(&INITIAL_STATE);
    let mut solver = CandidateDop853::new(
        OdeSolversKepler,
        0.0,
        FINAL_TIME,
        1.0,
        initial,
        1e-12,
        1e-12,
    );
    solver.set_output(OutputType::Sparse);
    solver.integrate().unwrap();
    solver.y_out().last().unwrap()[0]
}

fn measure(label: &str, operation: impl Fn() -> f64) {
    let mut checksum = 0.0;
    for _ in 0..100 {
        checksum += black_box(operation());
    }

    let mut samples = [Duration::ZERO; SAMPLES];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..CALLS_PER_SAMPLE {
            checksum += black_box(operation());
        }
        *sample = start.elapsed() / u32::try_from(CALLS_PER_SAMPLE).unwrap();
    }
    samples.sort_unstable();
    let total = samples.iter().sum::<Duration>();
    let mean = total / u32::try_from(SAMPLES).unwrap();
    println!(
        "{label}: mean {:.3} us, median {:.3} us, min {:.3} us, max {:.3} us",
        mean.as_secs_f64() * 1e6,
        samples[SAMPLES / 2].as_secs_f64() * 1e6,
        samples[0].as_secs_f64() * 1e6,
        samples[SAMPLES - 1].as_secs_f64() * 1e6,
    );
    black_box(checksum);
}

fn main() {
    measure("pykep facade / differential-equations 0.6.1", selected);
    measure("ode_solvers 0.6.2", ode_solvers);
}
