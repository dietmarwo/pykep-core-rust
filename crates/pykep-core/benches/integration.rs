// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Criterion benchmarks for the adaptive DOP853 and Taylor backends.

use criterion::{Criterion, criterion_group, criterion_main};
use pykep_core::dynamics::KeplerDynamics;
use pykep_core::integration::{
    Dop853, InitialValueProblem, IntegratorOptions, SensitivityProblem, Taylor,
};
use std::hint::black_box;

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
                .propagate(
                    black_box(&KeplerDynamics),
                    black_box(problem),
                    black_box(options),
                )
                .unwrap()
        });
    });
    criterion.bench_function("integration/taylor_kepler", |bencher| {
        bencher.iter(|| {
            Taylor
                .propagate(
                    black_box(&KeplerDynamics),
                    black_box(problem),
                    black_box(options),
                )
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
                    black_box(&KeplerDynamics),
                    black_box(sensitivity_problem),
                    black_box(options),
                )
                .unwrap()
        });
    });
    criterion.bench_function("integration/taylor_kepler_stm", |bencher| {
        bencher.iter(|| {
            Taylor
                .propagate_with_sensitivities(
                    black_box(&KeplerDynamics),
                    black_box(sensitivity_problem),
                    black_box(options),
                )
                .unwrap()
        });
    });
}

criterion_group!(benches, integration);
criterion_main!(benches);
