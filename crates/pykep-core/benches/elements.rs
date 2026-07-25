// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Criterion benchmarks for element and Cartesian conversions.

use criterion::{Criterion, criterion_group, criterion_main};
use pykep_core::astro::elements::{
    ClassicalElements, cartesian_to_modified_equinoctial,
    cartesian_to_modified_equinoctial_jacobian, classical_to_cartesian,
};
use std::hint::black_box;

fn element_conversions(criterion: &mut Criterion) {
    let elements = ClassicalElements::new(3.0, 0.3, 0.7, 1.1, 0.4, -0.8);
    let state = classical_to_cartesian(elements, 1.0).unwrap();
    criterion.bench_function("elements/classical_to_cartesian", |bencher| {
        bencher.iter(|| classical_to_cartesian(black_box(elements), black_box(1.0)).unwrap());
    });
    criterion.bench_function("elements/cartesian_to_equinoctial", |bencher| {
        bencher.iter(|| {
            cartesian_to_modified_equinoctial(black_box(&state), black_box(1.0), black_box(false))
                .unwrap()
        });
    });
    criterion.bench_function("elements/cartesian_to_equinoctial_jacobian", |bencher| {
        bencher.iter(|| {
            cartesian_to_modified_equinoctial_jacobian(
                black_box(&state),
                black_box(1.0),
                black_box(false),
            )
            .unwrap()
        });
    });
    criterion.bench_function("elements/classical_to_cartesian_64", |bencher| {
        bencher.iter(|| {
            for index in 0..64 {
                let mut sample = elements;
                sample.true_anomaly += f64::from(index) * 0.01;
                black_box(classical_to_cartesian(black_box(sample), 1.0).unwrap());
            }
        });
    });
}

criterion_group!(benches, element_conversions);
criterion_main!(benches);
