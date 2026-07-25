// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Criterion benchmarks for evaluated phase-11 dynamics.

use criterion::{Criterion, criterion_group, criterion_main};
use pykep_core::constants::{
    BCP_MU_EARTH_MOON, BCP_MU_SUN, BCP_SUN_ANGULAR_VELOCITY, BCP_SUN_DISTANCE,
};
use pykep_core::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
use pykep_core::integration::IntegratorOptions;
use std::hint::black_box;

fn dynamics(criterion: &mut Criterion) {
    let state = [0.8, -0.2, 0.1, 0.03, -0.04, 0.02];
    let options = IntegratorOptions::default();
    criterion.bench_function("dynamics/kepler_rhs", |bencher| {
        bencher.iter(|| {
            KeplerDynamics
                .evaluate(black_box(&state), black_box(1.0))
                .unwrap()
        });
    });
    criterion.bench_function("dynamics/cr3bp_rhs", |bencher| {
        bencher.iter(|| {
            Cr3bpDynamics
                .evaluate(black_box(&state), black_box(BCP_MU_EARTH_MOON))
                .unwrap()
        });
    });
    criterion.bench_function("dynamics/bcp_rhs", |bencher| {
        bencher.iter(|| {
            BcpDynamics
                .evaluate(
                    black_box(0.4),
                    black_box(&state),
                    black_box([
                        BCP_MU_EARTH_MOON,
                        BCP_MU_SUN,
                        BCP_SUN_DISTANCE,
                        BCP_SUN_ANGULAR_VELOCITY,
                    ]),
                )
                .unwrap()
        });
    });
    criterion.bench_function("dynamics/cr3bp_propagation", |bencher| {
        bencher.iter(|| {
            Cr3bpDynamics
                .propagate(
                    black_box(0.0),
                    black_box(state),
                    black_box(1.0),
                    black_box(BCP_MU_EARTH_MOON),
                    black_box(options),
                )
                .unwrap()
        });
    });
    criterion.bench_function("dynamics/cr3bp_stm", |bencher| {
        bencher.iter(|| {
            Cr3bpDynamics
                .propagate_with_stm(
                    black_box(0.0),
                    black_box(state),
                    black_box(1.0),
                    black_box(BCP_MU_EARTH_MOON),
                    black_box(options),
                )
                .unwrap()
        });
    });
}

criterion_group!(benches, dynamics);
criterion_main!(benches);
