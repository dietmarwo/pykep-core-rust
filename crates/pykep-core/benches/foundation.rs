// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Criterion infrastructure smoke benchmark.

use criterion::{Criterion, criterion_group, criterion_main};
use pykep_core::math::linalg;
use pykep_core::math::stumpff;
use pykep_core::time::julian;
use std::hint::black_box;

fn foundation_functions(criterion: &mut Criterion) {
    criterion.bench_function("stumpff_c/near_zero", |bencher| {
        bencher.iter(|| stumpff::stumpff_c(black_box(1e-12)).unwrap());
    });
    criterion.bench_function("stumpff_s/hyperbolic", |bencher| {
        bencher.iter(|| stumpff::stumpff_s(black_box(-4.0)).unwrap());
    });
    criterion.bench_function("julian/jd_to_mjd2000", |bencher| {
        bencher.iter(|| julian::jd_to_mjd2000(black_box(2_451_544.5)).unwrap());
    });
    criterion.bench_function("linalg/cross", |bencher| {
        bencher.iter(|| {
            linalg::cross(black_box(&[1.0, 2.0, 3.0]), black_box(&[4.0, 5.0, 6.0])).unwrap()
        });
    });
}

criterion_group!(benches, foundation_functions);
criterion_main!(benches);
