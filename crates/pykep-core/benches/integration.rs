// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Criterion benchmarks for the selected DOP853 integration backend.

use criterion::{Criterion, criterion_group, criterion_main};
use pykep_core::integration::{
    DifferentiableDynamicsModel, Dop853, DynamicsModel, InitialValueProblem, IntegratorOptions,
    SensitivityProblem,
};
use pykep_core::{PykepError, Result};
use std::hint::black_box;

struct Kepler;

impl DynamicsModel<6, 1> for Kepler {
    const NAME: &'static str = "benchmark_kepler";

    fn rhs(
        &self,
        _time: f64,
        state: &[f64; 6],
        parameters: &[f64; 1],
        derivative: &mut [f64; 6],
    ) -> Result<()> {
        let radius_squared = state[..3].iter().map(|value| value * value).sum::<f64>();
        if radius_squared == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "benchmark_kepler",
            });
        }
        let scale = -parameters[0] / (radius_squared * radius_squared.sqrt());
        derivative[..3].copy_from_slice(&state[3..]);
        for axis in 0..3 {
            derivative[axis + 3] = scale * state[axis];
        }
        Ok(())
    }
}

impl DifferentiableDynamicsModel<6, 1> for Kepler {
    fn jacobians(
        &self,
        _time: f64,
        state: &[f64; 6],
        parameters: &[f64; 1],
        state_jacobian: &mut [[f64; 6]; 6],
        parameter_jacobian: &mut [[f64; 1]; 6],
    ) -> Result<()> {
        let radius_squared = state[..3].iter().map(|value| value * value).sum::<f64>();
        if radius_squared == 0.0 {
            return Err(PykepError::SingularGeometry {
                operation: "benchmark_kepler",
            });
        }
        let inverse_radius_cubed = 1.0 / (radius_squared * radius_squared.sqrt());
        for row in 0..3 {
            state_jacobian[row][row + 3] = 1.0;
            for column in 0..3 {
                let identity = f64::from(row == column);
                state_jacobian[row + 3][column] = parameters[0]
                    * inverse_radius_cubed
                    * (3.0 * state[row] * state[column] / radius_squared - identity);
            }
            parameter_jacobian[row + 3][0] = -state[row] * inverse_radius_cubed;
        }
        Ok(())
    }
}

fn integration(criterion: &mut Criterion) {
    let state = [1.2, -0.3, 0.1, 0.2, 0.8, -0.1];
    let problem = InitialValueProblem::new(0.0, state, 5.785_665_678_258_923, [1.0]);
    let options = IntegratorOptions {
        relative_tolerance: 1e-12,
        absolute_tolerance: 1e-12,
        ..IntegratorOptions::default()
    };
    criterion.bench_function("integration/dop853_kepler", |bencher| {
        bencher.iter(|| {
            Dop853
                .propagate(black_box(&Kepler), black_box(problem), black_box(options))
                .unwrap()
        });
    });

    let mut initial_sensitivities = [[0.0; 6]; 6];
    for (index, row) in initial_sensitivities.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    let sensitivity_problem = SensitivityProblem {
        nominal: problem,
        initial_sensitivities,
        parameter_seeds: [[0.0; 6]],
    };
    criterion.bench_function("integration/dop853_kepler_stm", |bencher| {
        bencher.iter(|| {
            Dop853
                .propagate_with_sensitivities(
                    black_box(&Kepler),
                    black_box(sensitivity_problem),
                    black_box(options),
                )
                .unwrap()
        });
    });
}

criterion_group!(benches, integration);
criterion_main!(benches);
