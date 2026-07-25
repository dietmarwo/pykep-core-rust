// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Criterion benchmarks for Phase 6 mission-design algorithms.

use criterion::{Criterion, criterion_group, criterion_main};
use pykep_core::astro::flyby::{flyby_constraints, flyby_delta_v};
use pykep_core::astro::lambert::LambertProblem;
use pykep_core::astro::transfers::hohmann;
use std::hint::black_box;

fn mission_design(criterion: &mut Criterion) {
    criterion.bench_function("mission/hohmann", |bencher| {
        bencher.iter(|| hohmann(black_box(1.0), black_box(2.0), black_box(1.0)).unwrap());
    });
    let incoming = [7200.0, -4567.7655, 1234.4233];
    let outgoing = [7100.0, 220.123, -144.432];
    criterion.bench_function("mission/flyby_constraints", |bencher| {
        bencher.iter(|| {
            flyby_constraints(
                black_box(&incoming),
                black_box(&outgoing),
                black_box(3.986e14),
                black_box(7.0e6),
            )
            .unwrap()
        });
    });
    criterion.bench_function("mission/flyby_delta_v", |bencher| {
        bencher.iter(|| {
            flyby_delta_v(
                black_box(&incoming),
                black_box(&outgoing),
                black_box(3.986e14),
                black_box(7.0e6),
            )
            .unwrap()
        });
    });
    criterion.bench_function("mission/lambert_zero_revolution", |bencher| {
        bencher.iter(|| {
            LambertProblem::new(
                black_box([1.0, 0.0, 0.0]),
                black_box([0.2, 1.1, 0.3]),
                black_box(3.0),
                black_box(1.0),
                black_box(false),
                black_box(0),
            )
            .unwrap()
        });
    });
    criterion.bench_function("mission/lambert_multi_revolution", |bencher| {
        bencher.iter(|| {
            LambertProblem::new(
                black_box([1.0, 0.0, 0.0]),
                black_box([0.2, 1.1, 0.3]),
                black_box(20.0),
                black_box(1.0),
                black_box(false),
                black_box(4),
            )
            .unwrap()
        });
    });
}

criterion_group!(benches, mission_design);
criterion_main!(benches);
