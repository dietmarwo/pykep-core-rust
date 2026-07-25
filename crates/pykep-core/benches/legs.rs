// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Criterion benchmarks for low-thrust leg constraints and gradients.

use criterion::{Criterion, criterion_group, criterion_main};
use pykep_core::constants::{ASTRONOMICAL_UNIT, EARTH_ORBITAL_VELOCITY, STANDARD_GRAVITY};
use pykep_core::dynamics::zoh::ZohKeplerDynamics;
use pykep_core::integration::IntegratorOptions;
use pykep_core::leg::{SimsFlanaganLeg, SimsFlanaganSettings, SpacecraftEndpoint, ZohKeplerLeg};
use std::hint::black_box;

fn representative_leg() -> SimsFlanaganLeg {
    let departure = SpacecraftEndpoint::new(
        [
            ASTRONOMICAL_UNIT,
            0.1 * ASTRONOMICAL_UNIT,
            -0.1 * ASTRONOMICAL_UNIT,
            0.2 * EARTH_ORBITAL_VELOCITY,
            EARTH_ORBITAL_VELOCITY,
            -0.2 * EARTH_ORBITAL_VELOCITY,
        ],
        1500.0,
    )
    .unwrap();
    let arrival = SpacecraftEndpoint::new(
        [
            1.2 * ASTRONOMICAL_UNIT,
            -0.1 * ASTRONOMICAL_UNIT,
            0.1 * ASTRONOMICAL_UNIT,
            -0.2 * EARTH_ORBITAL_VELOCITY,
            1.023 * EARTH_ORBITAL_VELOCITY,
            -0.44 * EARTH_ORBITAL_VELOCITY,
        ],
        1300.0,
    )
    .unwrap();
    SimsFlanaganLeg::new(
        departure,
        vec![
            [0.10, 0.11, 0.12],
            [0.13, 0.14, 0.15],
            [0.16, 0.17, 0.18],
            [0.19, 0.20, 0.21],
            [0.22, 0.23, 0.24],
        ],
        arrival,
        SimsFlanaganSettings::new(
            324.0 * 86_400.0,
            0.12,
            100.0 * STANDARD_GRAVITY,
            1.327_124_400_18e20,
            0.6,
        )
        .unwrap(),
    )
    .unwrap()
}

fn representative_zoh_leg() -> ZohKeplerLeg {
    let controls = (0..20)
        .map(|index| [0.01 + 0.0001 * index as f64, 1.0, 0.1, -0.05])
        .collect();
    let grid = (0..=20).map(|index| index as f64 / 20.0).collect();
    ZohKeplerLeg::new(
        ZohKeplerDynamics,
        [1.0, 0.1, -0.05, -0.1, 0.95, 0.03, 1.2],
        controls,
        [0.4, 0.9, 0.08, -0.8, 0.3, -0.04, 1.1],
        grid,
        [0.2],
        0.5,
        IntegratorOptions::default(),
    )
    .unwrap()
}

fn legs(criterion: &mut Criterion) {
    let leg = representative_leg();
    criterion.bench_function("legs/sims_flanagan_mismatch_5_segments", |bencher| {
        bencher.iter(|| black_box(&leg).mismatch_constraints().unwrap());
    });
    criterion.bench_function("legs/sims_flanagan_gradient_5_segments", |bencher| {
        bencher.iter(|| black_box(&leg).mismatch_jacobian().unwrap());
    });
    let zoh = representative_zoh_leg();
    criterion.bench_function("legs/zoh_mismatch_20_segments", |bencher| {
        bencher.iter(|| black_box(&zoh).mismatch_constraints().unwrap());
    });
    criterion.bench_function("legs/zoh_gradient_20_segments", |bencher| {
        bencher.iter(|| black_box(&zoh).mismatch_jacobian().unwrap());
    });
}

criterion_group!(benches, legs);
criterion_main!(benches);
