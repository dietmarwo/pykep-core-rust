// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Criterion benchmarks for two-body propagation and analytic STMs.

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use pykep_core::astro::propagation::{
    propagate_lagrangian, propagate_lagrangian_with_stm, propagate_universal,
};
use std::hint::black_box;

fn propagation(criterion: &mut Criterion) {
    let elliptic = [1.223, 0.3123, -0.432, 0.06345, 0.43234, -0.874634];
    let hyperbolic = [1.223, 0.3123, -0.432, -3.06345, 4.43234, -0.874634];
    criterion.bench_function("propagation/lagrangian_elliptic", |bencher| {
        bencher.iter(|| propagate_lagrangian(black_box(&elliptic), black_box(3.56), 1.24).unwrap());
    });
    criterion.bench_function("propagation/lagrangian_hyperbolic", |bencher| {
        bencher
            .iter(|| propagate_lagrangian(black_box(&hyperbolic), black_box(3.56), 1.24).unwrap());
    });
    criterion.bench_function("propagation/universal_elliptic", |bencher| {
        bencher.iter(|| propagate_universal(black_box(&elliptic), black_box(3.56), 1.24).unwrap());
    });
    criterion.bench_function("propagation/lagrangian_stm", |bencher| {
        bencher.iter(|| {
            propagate_lagrangian_with_stm(black_box(&elliptic), black_box(3.56), 1.24).unwrap()
        });
    });

    let mut group = criterion.benchmark_group("propagation/batch");
    group.throughput(Throughput::Elements(1024));
    group.bench_function("1024_states", |bencher| {
        bencher.iter_batched(
            || elliptic,
            |state| {
                for index in 0..1024 {
                    black_box(
                        propagate_lagrangian(
                            black_box(&state),
                            black_box(f64::from(index) * 0.001),
                            1.24,
                        )
                        .unwrap(),
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, propagation);
criterion_main!(benches);
