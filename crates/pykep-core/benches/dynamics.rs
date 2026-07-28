// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Criterion benchmarks for evaluated dynamics.

use criterion::{Criterion, criterion_group, criterion_main};
use pykep_core::constants::{
    BCP_MU_EARTH_MOON, BCP_MU_SUN, BCP_SUN_ANGULAR_VELOCITY, BCP_SUN_DISTANCE,
};
use pykep_core::dynamics::pontryagin::{
    CartesianMassOptimal, CartesianTimeOptimal, EquinoctialMassOptimal, EquinoctialTimeOptimal,
};
use pykep_core::dynamics::zoh::{
    ControlSchedule, ZohCr3bpDynamics, ZohEquinoctialDynamics, ZohKeplerDynamics,
    ZohSolarSailDynamics, propagate_schedule,
};
use pykep_core::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
use pykep_core::integration::{
    Dop853, DynamicsModel, InitialValueProblem, IntegratorOptions, Taylor,
};
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

    let zoh_state = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.5];
    let mut zoh_derivative = [0.0; 7];
    criterion.bench_function("dynamics/zoh_kepler_rhs", |bencher| {
        bencher.iter(|| {
            ZohKeplerDynamics
                .rhs(
                    black_box(0.0),
                    black_box(&zoh_state),
                    black_box(&[0.01, 1.0, 0.0, 0.0, 0.02]),
                    black_box(&mut zoh_derivative),
                )
                .unwrap()
        });
    });
    let boundaries: Vec<_> = (0..=32).map(|index| f64::from(index) * 0.02).collect();
    let controls = (0..32)
        .map(|index| {
            if index % 2 == 0 {
                [0.01, 1.0, 0.0, 0.0]
            } else {
                [0.01, 0.0, 1.0, 0.0]
            }
        })
        .collect();
    let schedule = ControlSchedule::new(boundaries, controls).unwrap();
    criterion.bench_function("dynamics/zoh_kepler_32_segments", |bencher| {
        bencher.iter(|| {
            propagate_schedule(
                black_box(&ZohKeplerDynamics),
                black_box(&schedule),
                black_box(zoh_state),
                black_box([0.02]),
                black_box(options),
            )
            .unwrap()
        });
    });

    let pontryagin_state = [
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let pontryagin_parameters = [1.0, 0.01, 1.0, 0.5, 1.0];
    let mut pontryagin_derivative = [0.0; 14];
    criterion.bench_function("dynamics/pontryagin_cartesian_mass_rhs", |bencher| {
        bencher.iter(|| {
            CartesianMassOptimal
                .rhs(
                    black_box(0.0),
                    black_box(&pontryagin_state),
                    black_box(&pontryagin_parameters),
                    black_box(&mut pontryagin_derivative),
                )
                .unwrap()
        });
    });
    let pontryagin_options = IntegratorOptions {
        maximum_step: Some(0.01),
        ..options
    };
    criterion.bench_function(
        "dynamics/pontryagin_cartesian_mass_propagation",
        |bencher| {
            bencher.iter(|| {
                Dop853
                    .propagate(
                        black_box(&CartesianMassOptimal),
                        black_box(InitialValueProblem::new(
                            0.0,
                            pontryagin_state,
                            1.2345,
                            pontryagin_parameters,
                        )),
                        black_box(pontryagin_options),
                    )
                    .unwrap()
            });
        },
    );
}

fn optimized_taylor_models(criterion: &mut Criterion) {
    let options = IntegratorOptions {
        relative_tolerance: 1e-12,
        absolute_tolerance: 1e-12,
        ..IntegratorOptions::default()
    };
    let rotating_state = [0.8, -0.2, 0.1, 0.03, 1.0, 0.02];
    criterion.bench_function("taylor/cr3bp", |bencher| {
        bencher.iter(|| {
            Taylor
                .propagate(
                    black_box(&Cr3bpDynamics),
                    black_box(InitialValueProblem::new(
                        0.0,
                        rotating_state,
                        1.0,
                        [BCP_MU_EARTH_MOON],
                    )),
                    black_box(options),
                )
                .unwrap()
        });
    });
    criterion.bench_function("taylor/bcp", |bencher| {
        bencher.iter(|| {
            Taylor
                .propagate(
                    black_box(&BcpDynamics),
                    black_box(InitialValueProblem::new(
                        0.0,
                        rotating_state,
                        0.5,
                        [
                            BCP_MU_EARTH_MOON,
                            BCP_MU_SUN,
                            BCP_SUN_DISTANCE,
                            BCP_SUN_ANGULAR_VELOCITY,
                        ],
                    )),
                    black_box(options),
                )
                .unwrap()
        });
    });

    let zoh_cr3bp_state = [0.8, -0.2, 0.1, 0.03, -0.04, 0.02, 1.1];
    criterion.bench_function("taylor/zoh_cr3bp", |bencher| {
        bencher.iter(|| {
            Taylor
                .propagate(
                    black_box(&ZohCr3bpDynamics),
                    black_box(InitialValueProblem::new(
                        0.0,
                        zoh_cr3bp_state,
                        0.5,
                        [0.02, 0.3, -0.4, 0.5, 0.01, BCP_MU_EARTH_MOON],
                    )),
                    black_box(options),
                )
                .unwrap()
        });
    });
    let zoh_equinoctial_state = [1.1, 0.1, -0.05, 0.02, -0.03, 0.4, 1.1];
    criterion.bench_function("taylor/zoh_equinoctial", |bencher| {
        bencher.iter(|| {
            Taylor
                .propagate(
                    black_box(&ZohEquinoctialDynamics),
                    black_box(InitialValueProblem::new(
                        0.0,
                        zoh_equinoctial_state,
                        0.5,
                        [0.02, 0.3, -0.4, 0.5, 0.01],
                    )),
                    black_box(options),
                )
                .unwrap()
        });
    });
    let solar_sail_state = [0.8, -0.4, 0.3, 0.2, 0.9, -0.1];
    criterion.bench_function("taylor/zoh_solar_sail", |bencher| {
        bencher.iter(|| {
            Taylor
                .propagate(
                    black_box(&ZohSolarSailDynamics),
                    black_box(InitialValueProblem::new(
                        0.0,
                        solar_sail_state,
                        0.5,
                        [0.25, -1.1, 0.04],
                    )),
                    black_box(options),
                )
                .unwrap()
        });
    });

    let cartesian_state = [
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    criterion.bench_function("taylor/cartesian_time", |bencher| {
        bencher.iter(|| {
            Taylor
                .propagate(
                    black_box(&CartesianTimeOptimal),
                    black_box(InitialValueProblem::new(
                        0.0,
                        cartesian_state,
                        0.1,
                        [1.0, 0.01, 1.0],
                    )),
                    black_box(options),
                )
                .unwrap()
        });
    });
    let equinoctial_state = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ];
    criterion.bench_function("taylor/equinoctial_mass", |bencher| {
        bencher.iter(|| {
            Taylor
                .propagate(
                    black_box(&EquinoctialMassOptimal),
                    black_box(InitialValueProblem::new(
                        0.0,
                        equinoctial_state,
                        0.1,
                        [1.0, 1e-4, 1.0, 1.0, 1e-4],
                    )),
                    black_box(options),
                )
                .unwrap()
        });
    });
    criterion.bench_function("taylor/equinoctial_time", |bencher| {
        bencher.iter(|| {
            Taylor
                .propagate(
                    black_box(&EquinoctialTimeOptimal),
                    black_box(InitialValueProblem::new(
                        0.0,
                        equinoctial_state,
                        0.1,
                        [1.0, 1e-4, 1.0],
                    )),
                    black_box(options),
                )
                .unwrap()
        });
    });
}

criterion_group!(benches, dynamics, optimized_taylor_models);
criterion_main!(benches);
